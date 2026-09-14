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

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "clean_evaluation"
)

TEST_MANIFEST = (
    REPORT_DIR
    / "test_clean.csv"
)

CORESET_PATH = (
    PROJECT_ROOT
    / "models"
    / "clean_evaluation"
    / "acc_clean_full32_greedy10_coreset.pt"
)

PROFILE_CSV = (
    REPORT_DIR
    / "detailed_latency_per_image.csv"
)

PROFILE_JSON = (
    REPORT_DIR
    / "detailed_latency_profile.json"
)


# ============================================================
# 2. Config
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


def reset_cuda_memory():
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# 4. Load manifest
# ============================================================

def load_test_manifest():

    if not TEST_MANIFEST.exists():
        raise FileNotFoundError(
            f"Missing test manifest:\n{TEST_MANIFEST}"
        )

    records = []

    with TEST_MANIFEST.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as f:

        reader = csv.DictReader(f)

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
# 5. Load model
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
).to(device)

model.eval()


# ============================================================
# 6. Load coreset
# ============================================================

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
# 7. CNN feature extraction
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

    layer3_up = F.interpolate(
        layer3,
        size=layer2.shape[-2:],
        mode="bilinear",
        align_corners=False
    )

    fused = torch.cat(
        [
            layer2,
            layer3_up
        ],
        dim=1
    )

    return fused


# ============================================================
# 8. Patch helpers
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

    return patches, h, w


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
# 9. Detailed single-image profiling
# ============================================================

def profile_single_image(
    image_path: Path
):

    timings = {}

    total_start = time.perf_counter()


    # --------------------------------------------------------
    # A. Disk read / Image.open
    # --------------------------------------------------------

    t0 = time.perf_counter()

    image = Image.open(
        image_path
    )

    # Force actual image decode/load now,
    # instead of leaving PIL lazy-loaded.
    image.load()

    t1 = time.perf_counter()

    timings["disk_read_ms"] = (
        t1 - t0
    ) * 1000


    # --------------------------------------------------------
    # B. RGB conversion
    # --------------------------------------------------------

    t0 = time.perf_counter()

    image = image.convert("RGB")

    t1 = time.perf_counter()

    timings["rgb_conversion_ms"] = (
        t1 - t0
    ) * 1000


    # --------------------------------------------------------
    # C. Resize
    # --------------------------------------------------------

    original_width, original_height = (
        image.size
    )

    scale = min(
        TARGET_SIZE / original_width,
        TARGET_SIZE / original_height
    )

    new_width = max(
        1,
        int(
            round(
                original_width
                *
                scale
            )
        )
    )

    new_height = max(
        1,
        int(
            round(
                original_height
                *
                scale
            )
        )
    )

    t0 = time.perf_counter()

    resized = image.resize(
        (
            new_width,
            new_height
        ),
        resample=Image.Resampling.BILINEAR
    )

    t1 = time.perf_counter()

    timings["resize_ms"] = (
        t1 - t0
    ) * 1000


    # --------------------------------------------------------
    # D. Canvas creation + paste
    # --------------------------------------------------------

    mean_rgb = (
        0.485,
        0.456,
        0.406
    )

    padding_color = tuple(
        int(round(v * 255))
        for v in mean_rgb
    )

    left = (
        TARGET_SIZE - new_width
    ) // 2

    top = (
        TARGET_SIZE - new_height
    ) // 2


    t0 = time.perf_counter()

    canvas = Image.new(
        "RGB",
        (
            TARGET_SIZE,
            TARGET_SIZE
        ),
        color=padding_color
    )

    canvas.paste(
        resized,
        (
            left,
            top
        )
    )

    t1 = time.perf_counter()

    timings["canvas_paste_ms"] = (
        t1 - t0
    ) * 1000


    # --------------------------------------------------------
    # E. Valid mask creation
    # --------------------------------------------------------

    t0 = time.perf_counter()

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

    t1 = time.perf_counter()

    timings["valid_mask_ms"] = (
        t1 - t0
    ) * 1000


    # --------------------------------------------------------
    # F. PIL -> Tensor
    # --------------------------------------------------------

    t0 = time.perf_counter()

    tensor = (
        pil_to_tensor(canvas)
        .float()
        /
        255.0
    )

    t1 = time.perf_counter()

    timings["pil_to_tensor_ms"] = (
        t1 - t0
    ) * 1000


    # --------------------------------------------------------
    # G. Normalize
    # --------------------------------------------------------

    t0 = time.perf_counter()

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

    t1 = time.perf_counter()

    timings["normalize_ms"] = (
        t1 - t0
    ) * 1000


    # --------------------------------------------------------
    # H. CPU -> GPU transfer
    # --------------------------------------------------------

    sync_device()

    t0 = time.perf_counter()

    tensor = (
        tensor
        .unsqueeze(0)
        .to(device)
    )

    sync_device()

    t1 = time.perf_counter()

    timings["cpu_to_gpu_ms"] = (
        t1 - t0
    ) * 1000


    # --------------------------------------------------------
    # I. Feature extraction
    # --------------------------------------------------------

    sync_device()

    t0 = time.perf_counter()

    with torch.no_grad():

        feature_map = (
            extract_highres_features(
                tensor
            )
        )

    sync_device()

    t1 = time.perf_counter()

    timings["feature_extraction_ms"] = (
        t1 - t0
    ) * 1000


    # --------------------------------------------------------
    # J. Patch processing
    # --------------------------------------------------------

    sync_device()

    t0 = time.perf_counter()

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

    t1 = time.perf_counter()

    timings["patch_processing_ms"] = (
        t1 - t0
    ) * 1000


    # --------------------------------------------------------
    # K. Nearest-neighbor search
    # --------------------------------------------------------

    sync_device()

    t0 = time.perf_counter()

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

    t1 = time.perf_counter()

    timings["nearest_neighbor_ms"] = (
        t1 - t0
    ) * 1000


    # --------------------------------------------------------
    # Total
    # --------------------------------------------------------

    total_end = time.perf_counter()

    timings["total_ms"] = (
        total_end
        -
        total_start
    ) * 1000

    timings["score"] = (
        image_score
    )

    timings["valid_patches"] = int(
        valid_patch_mask
        .sum()
        .item()
    )

    return timings


# ============================================================
# 10. Warm-up
# ============================================================

def run_warmup():

    count = min(
        WARMUP_IMAGES,
        len(test_records)
    )

    print()

    print(
        "=" * 72
    )

    print(
        "WARM-UP"
    )

    print(
        "=" * 72
    )

    for i in range(count):

        profile_single_image(
            test_records[i][
                "image_path"
            ]
        )

        print(
            f"Warm-up "
            f"{i + 1}/"
            f"{count}"
        )


# ============================================================
# 11. Summary helper
# ============================================================

def summarize(
    values
):

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
# 12. Full benchmark
# ============================================================

def run_profile():

    reset_cuda_memory()

    print()

    print(
        "=" * 72
    )

    print(
        "DETAILED INFERENCE PIPELINE PROFILING"
    )

    print(
        "=" * 72
    )


    timing_keys = [
        "disk_read_ms",
        "rgb_conversion_ms",
        "resize_ms",
        "canvas_paste_ms",
        "valid_mask_ms",
        "pil_to_tensor_ms",
        "normalize_ms",
        "cpu_to_gpu_ms",
        "feature_extraction_ms",
        "patch_processing_ms",
        "nearest_neighbor_ms",
        "total_ms",
    ]


    timing_values = {
        key: []
        for key in timing_keys
    }


    rows = []

    start = time.perf_counter()


    for index, record in enumerate(
        test_records,
        start=1
    ):

        result = profile_single_image(
            record[
                "image_path"
            ]
        )


        row = {
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

            "valid_patches":
                result[
                    "valid_patches"
                ],
        }


        for key in timing_keys:

            value = result[key]

            timing_values[key].append(
                value
            )

            row[key] = value


        rows.append(row)


        if (
            index % 50 == 0
            or
            index == len(test_records)
        ):

            elapsed = (
                time.perf_counter()
                -
                start
            )

            print(
                f"Processed "
                f"{index:,}/"
                f"{len(test_records):,}"
                f" | elapsed="
                f"{elapsed:.1f}s"
            )


    summary = {
        key:
            summarize(
                timing_values[key]
            )
        for key in timing_keys
    }


    peak_cuda_mb = None

    if device.type == "cuda":

        peak_cuda_mb = (
            torch.cuda
            .max_memory_allocated()
            /
            (1024 ** 2)
        )


    # ========================================================
    # Save CSV
    # ========================================================

    with PROFILE_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        fieldnames = [
            "image_path",
            "label",
            "defect_type",
            "score",
            "valid_patches",
        ] + timing_keys

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


    # ========================================================
    # Save JSON
    # ========================================================

    output = {
        "experiment":
            "clean_v2c_detailed_latency_profile",

        "device":
            str(device),

        "images":
            len(test_records),

        "coreset_shape":
            list(
                memory_bank.shape
            ),

        "timings_ms":
            summary,

        "peak_cuda_memory_mb":
            peak_cuda_mb,
    }


    PROFILE_JSON.write_text(
        json.dumps(
            output,
            indent=2
        ),
        encoding="utf-8"
    )


    # ========================================================
    # Print results
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "DETAILED LATENCY RESULTS"
    )

    print(
        "=" * 72
    )


    display_names = {
        "disk_read_ms":
            "Disk read / decode",

        "rgb_conversion_ms":
            "RGB conversion",

        "resize_ms":
            "Resize",

        "canvas_paste_ms":
            "Canvas + paste",

        "valid_mask_ms":
            "Valid-mask creation",

        "pil_to_tensor_ms":
            "PIL -> Tensor",

        "normalize_ms":
            "Normalize",

        "cpu_to_gpu_ms":
            "CPU -> GPU",

        "feature_extraction_ms":
            "ResNet18 features",

        "patch_processing_ms":
            "Patch processing",

        "nearest_neighbor_ms":
            "Nearest-neighbor",

        "total_ms":
            "End-to-end",
    }


    for key in timing_keys:

        stats = summary[key]

        print()

        print(
            display_names[key]
        )

        print(
            f"  Mean:   "
            f"{stats['mean']:.3f} ms"
        )

        print(
            f"  Median: "
            f"{stats['median']:.3f} ms"
        )

        print(
            f"  P95:    "
            f"{stats['p95']:.3f} ms"
        )


    if peak_cuda_mb is not None:

        print()

        print(
            f"Peak CUDA memory: "
            f"{peak_cuda_mb:.2f} MB"
        )


    print()

    print(
        "Saved:"
    )

    print(
        PROFILE_CSV
    )

    print(
        PROFILE_JSON
    )


# ============================================================
# 13. Main
# ============================================================

def main():

    run_warmup()

    run_profile()


if __name__ == "__main__":

    main()