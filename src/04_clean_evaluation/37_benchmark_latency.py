from __future__ import annotations

from pathlib import Path
import csv
import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import models
from torchvision.transforms.functional import pil_to_tensor, normalize


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

CLEAN_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "clean_evaluation"
)

TEST_MANIFEST = (
    CLEAN_REPORT_DIR
    / "test_clean.csv"
)

CORESET_PATH = (
    PROJECT_ROOT
    / "models"
    / "clean_evaluation"
    / "acc_clean_full32_greedy10_coreset.pt"
)

LATENCY_CSV = (
    CLEAN_REPORT_DIR
    / "latency_per_image.csv"
)

LATENCY_JSON = (
    CLEAN_REPORT_DIR
    / "latency_benchmark.json"
)


# ============================================================
# 2. Configuration
# ============================================================

TARGET_SIZE = 256
MIN_VALID_RATIO = 0.50

WARMUP_IMAGES = 20

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# 3. CUDA helpers
# ============================================================

def sync_device():
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_cuda_memory_stats():
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# 4. Read test manifest
# ============================================================

def load_test_manifest():

    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(
            f"Test manifest not found:\n{TEST_MANIFEST}"
        )

    records = []

    with TEST_MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            records.append(
                {
                    "image_path":
                        PROJECT_ROOT
                        / row["image_path"],

                    "label":
                        int(row["label"]),

                    "defect_type":
                        row["defect_type"],
                }
            )

    return records


test_records = load_test_manifest()

print(
    f"Test images: {len(test_records):,}"
)


# ============================================================
# 5. Load ResNet18
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
).to(device)

model.eval()


# ============================================================
# 6. Load clean coreset
# ============================================================

if not CORESET_PATH.exists():
    raise FileNotFoundError(
        f"Coreset not found:\n{CORESET_PATH}"
    )

checkpoint = torch.load(
    CORESET_PATH,
    map_location="cpu",
    weights_only=True
)

memory_bank = (
    checkpoint["memory_bank"]
    .float()
    .to(device)
)

memory_bank = F.normalize(
    memory_bank,
    p=2,
    dim=1
)

print(
    "Coreset:",
    tuple(memory_bank.shape)
)


# ============================================================
# 7. Letterbox preprocessing
# ============================================================

def letterbox_image(
    image: Image.Image
):

    original_width, original_height = image.size

    scale = min(
        TARGET_SIZE / original_width,
        TARGET_SIZE / original_height
    )

    new_width = max(
        1,
        int(
            round(
                original_width * scale
            )
        )
    )

    new_height = max(
        1,
        int(
            round(
                original_height * scale
            )
        )
    )

    resized = image.resize(
        (
            new_width,
            new_height
        ),
        resample=Image.Resampling.BILINEAR
    )

    mean_rgb = (
        0.485,
        0.456,
        0.406
    )

    padding_color = tuple(
        int(round(v * 255))
        for v in mean_rgb
    )

    canvas = Image.new(
        "RGB",
        (
            TARGET_SIZE,
            TARGET_SIZE
        ),
        color=padding_color
    )

    left = (
        TARGET_SIZE - new_width
    ) // 2

    top = (
        TARGET_SIZE - new_height
    ) // 2

    canvas.paste(
        resized,
        (
            left,
            top
        )
    )

    valid_region = torch.zeros(
        (
            1,
            TARGET_SIZE,
            TARGET_SIZE
        ),
        dtype=torch.float32
    )

    valid_region[
        :,
        top:top + new_height,
        left:left + new_width
    ] = 1.0

    tensor = (
        pil_to_tensor(canvas)
        .float()
        / 255.0
    )

    tensor = normalize(
        tensor,
        mean=[
            0.485,
            0.456,
            0.406
        ],
        std=[
            0.229,
            0.224,
            0.225
        ]
    )

    return (
        tensor,
        valid_region
    )


# ============================================================
# 8. ResNet18 Full32 features
# ============================================================

def extract_highres_features(
    tensor
):

    x = model.conv1(tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)

    layer2 = model.layer2(x)

    layer3 = model.layer3(layer2)

    layer3_upsampled = F.interpolate(
        layer3,
        size=layer2.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    combined = torch.cat(
        [
            layer2,
            layer3_upsampled
        ],
        dim=1
    )

    return combined


# ============================================================
# 9. Feature map -> patches
# ============================================================

def feature_map_to_patches(
    feature_map
):

    _, channels, h, w = (
        feature_map.shape
    )

    patches = (
        feature_map
        .permute(
            0,
            2,
            3,
            1
        )
        .reshape(
            h * w,
            channels
        )
    )

    patches = F.normalize(
        patches,
        p=2,
        dim=1
    )

    return (
        patches,
        h,
        w
    )


# ============================================================
# 10. Valid-region patch mask
# ============================================================

def valid_region_to_patch_mask(
    valid_region,
    h,
    w
):

    valid_region = (
        valid_region
        .unsqueeze(0)
        .to(device)
    )

    valid_ratio = (
        F.adaptive_avg_pool2d(
            valid_region,
            output_size=(
                h,
                w
            )
        )
        .reshape(-1)
    )

    return (
        valid_ratio
        >=
        MIN_VALID_RATIO
    )


# ============================================================
# 11. Single-image benchmark
# ============================================================

def benchmark_single_image(
    image_path: Path
):

    # --------------------------------------------------------
    # A. Image loading + preprocessing
    # --------------------------------------------------------

    t0 = time.perf_counter()

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor, valid_region = (
        letterbox_image(image)
    )

    tensor = (
        tensor
        .unsqueeze(0)
        .to(device)
    )

    sync_device()

    t1 = time.perf_counter()


    # --------------------------------------------------------
    # B. CNN feature extraction
    # --------------------------------------------------------

    sync_device()

    t2 = time.perf_counter()

    with torch.no_grad():

        feature_map = (
            extract_highres_features(
                tensor
            )
        )

    sync_device()

    t3 = time.perf_counter()


    # --------------------------------------------------------
    # C. Patch processing
    # --------------------------------------------------------

    sync_device()

    t4 = time.perf_counter()

    with torch.no_grad():

        patches, h, w = (
            feature_map_to_patches(
                feature_map
            )
        )

        valid_patch_mask = (
            valid_region_to_patch_mask(
                valid_region,
                h,
                w
            )
        )

        valid_patches = (
            patches[
                valid_patch_mask
            ]
        )

    sync_device()

    t5 = time.perf_counter()


    # --------------------------------------------------------
    # D. Nearest-neighbor search
    # --------------------------------------------------------

    sync_device()

    t6 = time.perf_counter()

    with torch.no_grad():

        similarity_matrix = (
            valid_patches
            @
            memory_bank.T
        )

        nearest_similarity = (
            similarity_matrix
            .max(
                dim=1
            )
            .values
        )

        distance_squared = (
            2.0
            -
            2.0
            *
            nearest_similarity
        )

        patch_distances = torch.sqrt(
            torch.clamp(
                distance_squared,
                min=0.0
            )
        )

        image_score = (
            patch_distances
            .max()
            .item()
        )

    sync_device()

    t7 = time.perf_counter()


    preprocessing_ms = (
        t1 - t0
    ) * 1000

    feature_ms = (
        t3 - t2
    ) * 1000

    patch_ms = (
        t5 - t4
    ) * 1000

    nearest_neighbor_ms = (
        t7 - t6
    ) * 1000

    total_ms = (
        preprocessing_ms
        +
        feature_ms
        +
        patch_ms
        +
        nearest_neighbor_ms
    )


    return {

        "score":
            image_score,

        "preprocessing_ms":
            preprocessing_ms,

        "feature_extraction_ms":
            feature_ms,

        "patch_processing_ms":
            patch_ms,

        "nearest_neighbor_ms":
            nearest_neighbor_ms,

        "total_ms":
            total_ms,

        "valid_patches":
            int(
                valid_patch_mask
                .sum()
                .item()
            ),

        "feature_h":
            h,

        "feature_w":
            w,
    }


# ============================================================
# 12. Warm-up
# ============================================================

def run_warmup():

    warmup_count = min(
        WARMUP_IMAGES,
        len(test_records)
    )

    print()

    print(
        "=" * 72
    )

    print(
        "GPU / MODEL WARM-UP"
    )

    print(
        "=" * 72
    )

    for i in range(
        warmup_count
    ):

        benchmark_single_image(
            test_records[i][
                "image_path"
            ]
        )

        print(
            f"Warm-up "
            f"{i + 1}/"
            f"{warmup_count}"
        )


# ============================================================
# 13. Summary helper
# ============================================================

def summarize(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    return {

        "mean":
            float(
                np.mean(values)
            ),

        "median":
            float(
                np.median(values)
            ),

        "p95":
            float(
                np.percentile(
                    values,
                    95
                )
            ),

        "min":
            float(
                np.min(values)
            ),

        "max":
            float(
                np.max(values)
            ),
    }


# ============================================================
# 14. Full benchmark
# ============================================================

def run_benchmark():

    print()

    print(
        "=" * 72
    )

    print(
        "CLEAN V2C LATENCY BENCHMARK"
    )

    print(
        "=" * 72
    )


    reset_cuda_memory_stats()


    rows = []

    preprocessing_times = []
    feature_times = []
    patch_times = []
    nearest_neighbor_times = []
    total_times = []

    valid_patch_counts = []


    benchmark_start = (
        time.perf_counter()
    )


    for index, record in enumerate(
        test_records,
        start=1
    ):

        result = (
            benchmark_single_image(
                record[
                    "image_path"
                ]
            )
        )


        preprocessing_times.append(
            result[
                "preprocessing_ms"
            ]
        )

        feature_times.append(
            result[
                "feature_extraction_ms"
            ]
        )

        patch_times.append(
            result[
                "patch_processing_ms"
            ]
        )

        nearest_neighbor_times.append(
            result[
                "nearest_neighbor_ms"
            ]
        )

        total_times.append(
            result[
                "total_ms"
            ]
        )

        valid_patch_counts.append(
            result[
                "valid_patches"
            ]
        )


        rows.append(
            {
                "image_path":
                    str(
                        record[
                            "image_path"
                        ]
                        .relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "label":
                    record[
                        "label"
                    ],

                "defect_type":
                    record[
                        "defect_type"
                    ],

                "score":
                    result[
                        "score"
                    ],

                "preprocessing_ms":
                    result[
                        "preprocessing_ms"
                    ],

                "feature_extraction_ms":
                    result[
                        "feature_extraction_ms"
                    ],

                "patch_processing_ms":
                    result[
                        "patch_processing_ms"
                    ],

                "nearest_neighbor_ms":
                    result[
                        "nearest_neighbor_ms"
                    ],

                "total_ms":
                    result[
                        "total_ms"
                    ],

                "valid_patches":
                    result[
                        "valid_patches"
                    ],
            }
        )


        if (
            index % 50 == 0
            or
            index == len(test_records)
        ):

            elapsed = (
                time.perf_counter()
                -
                benchmark_start
            )

            print(
                f"Processed "
                f"{index:,}/"
                f"{len(test_records):,}"
                f" | elapsed="
                f"{elapsed:.1f}s"
            )


    # ========================================================
    # Summary
    # ========================================================

    preprocessing_summary = (
        summarize(
            preprocessing_times
        )
    )

    feature_summary = (
        summarize(
            feature_times
        )
    )

    patch_summary = (
        summarize(
            patch_times
        )
    )

    nearest_neighbor_summary = (
        summarize(
            nearest_neighbor_times
        )
    )

    total_summary = (
        summarize(
            total_times
        )
    )


    mean_total_seconds = (
        total_summary[
            "mean"
        ]
        /
        1000.0
    )

    throughput = (
        1.0
        /
        mean_total_seconds
        if mean_total_seconds > 0
        else 0.0
    )


    peak_gpu_memory_mb = None

    if device.type == "cuda":

        peak_gpu_memory_mb = (
            torch.cuda
            .max_memory_allocated()
            /
            (1024 ** 2)
        )


    avg_valid_patches = float(
        np.mean(
            valid_patch_counts
        )
    )


    summary = {

        "experiment":
            "clean_v2c_latency_benchmark",

        "device":
            str(device),

        "images_benchmarked":
            len(test_records),

        "warmup_images":
            min(
                WARMUP_IMAGES,
                len(test_records)
            ),

        "batch_size":
            1,

        "feature_resolution":
            "32x32",

        "coreset_shape":
            list(
                memory_bank.shape
            ),

        "preprocessing_ms":
            preprocessing_summary,

        "feature_extraction_ms":
            feature_summary,

        "patch_processing_ms":
            patch_summary,

        "nearest_neighbor_ms":
            nearest_neighbor_summary,

        "total_ms":
            total_summary,

        "throughput_images_per_second":
            throughput,

        "average_valid_patches":
            avg_valid_patches,

        "peak_gpu_memory_mb":
            peak_gpu_memory_mb,
    }


    # ========================================================
    # Save CSV
    # ========================================================

    with LATENCY_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "label",
                "defect_type",
                "score",
                "preprocessing_ms",
                "feature_extraction_ms",
                "patch_processing_ms",
                "nearest_neighbor_ms",
                "total_ms",
                "valid_patches",
            ]
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # ========================================================
    # Save JSON
    # ========================================================

    LATENCY_JSON.write_text(
        json.dumps(
            summary,
            indent=2
        ),
        encoding="utf-8"
    )


    # ========================================================
    # Print summary
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "LATENCY RESULTS"
    )

    print(
        "=" * 72
    )


    print(
        f"Images benchmarked: "
        f"{len(test_records):,}"
    )


    print(
        "\nPreprocessing"
    )

    print(
        f"  Mean:   "
        f"{preprocessing_summary['mean']:.3f} ms"
    )

    print(
        f"  Median: "
        f"{preprocessing_summary['median']:.3f} ms"
    )

    print(
        f"  P95:    "
        f"{preprocessing_summary['p95']:.3f} ms"
    )


    print(
        "\nFeature extraction"
    )

    print(
        f"  Mean:   "
        f"{feature_summary['mean']:.3f} ms"
    )

    print(
        f"  Median: "
        f"{feature_summary['median']:.3f} ms"
    )

    print(
        f"  P95:    "
        f"{feature_summary['p95']:.3f} ms"
    )


    print(
        "\nPatch processing"
    )

    print(
        f"  Mean:   "
        f"{patch_summary['mean']:.3f} ms"
    )

    print(
        f"  Median: "
        f"{patch_summary['median']:.3f} ms"
    )

    print(
        f"  P95:    "
        f"{patch_summary['p95']:.3f} ms"
    )


    print(
        "\nNearest-neighbor search"
    )

    print(
        f"  Mean:   "
        f"{nearest_neighbor_summary['mean']:.3f} ms"
    )

    print(
        f"  Median: "
        f"{nearest_neighbor_summary['median']:.3f} ms"
    )

    print(
        f"  P95:    "
        f"{nearest_neighbor_summary['p95']:.3f} ms"
    )


    print(
        "\nEnd-to-end"
    )

    print(
        f"  Mean:   "
        f"{total_summary['mean']:.3f} ms"
    )

    print(
        f"  Median: "
        f"{total_summary['median']:.3f} ms"
    )

    print(
        f"  P95:    "
        f"{total_summary['p95']:.3f} ms"
    )


    print(
        f"\nThroughput: "
        f"{throughput:.2f} images/sec"
    )


    print(
        f"Average valid patches/image: "
        f"{avg_valid_patches:.1f}"
    )


    if peak_gpu_memory_mb is not None:

        print(
            f"Peak CUDA memory: "
            f"{peak_gpu_memory_mb:.2f} MB"
        )


    print(
        "\nSaved:"
    )

    print(
        LATENCY_CSV
    )

    print(
        LATENCY_JSON
    )


# ============================================================
# 15. Main
# ============================================================

def main():

    run_warmup()

    run_benchmark()


if __name__ == "__main__":
    main()