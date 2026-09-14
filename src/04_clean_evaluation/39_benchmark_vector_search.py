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
# 1. Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

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

FINAL_RESULTS_PATH = (
    REPORT_DIR
    / "clean_v2c_final_results.json"
)

OUTPUT_JSON = (
    REPORT_DIR
    / "vector_search_benchmark.json"
)

OUTPUT_CSV = (
    REPORT_DIR
    / "vector_search_per_image.csv"
)


TARGET_SIZE = 256
MIN_VALID_RATIO = 0.50

# Use all 719 for final benchmark.
# If you want a quick test first, change to 100.
MAX_IMAGES = None

# Memory chunk size for chunked GPU search.
CHUNK_SIZE = 8192

WARMUP_IMAGES = 20

SCORE_TOLERANCE = 1e-5


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Device:", device)


def sync():
    if device.type == "cuda":
        torch.cuda.synchronize()


def reset_gpu_memory():
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


# ============================================================
# 3. Optional FAISS
# ============================================================

try:
    import faiss

    FAISS_AVAILABLE = True

except ImportError:

    faiss = None
    FAISS_AVAILABLE = False


print(
    "FAISS available:",
    FAISS_AVAILABLE
)


# ============================================================
# 4. Load validation-selected threshold
# ============================================================

if not FINAL_RESULTS_PATH.exists():
    raise FileNotFoundError(
        f"Missing:\n{FINAL_RESULTS_PATH}"
    )


final_results = json.loads(
    FINAL_RESULTS_PATH.read_text(
        encoding="utf-8"
    )
)


THRESHOLD = float(
    final_results[
        "threshold_selection"
    ][
        "selected_threshold"
    ]
)


print(
    f"Validation-selected threshold: "
    f"{THRESHOLD:.6f}"
)


# ============================================================
# 5. Load test manifest
# ============================================================

def load_test_manifest():

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

    if MAX_IMAGES is not None:
        records = records[:MAX_IMAGES]

    return records


test_records = load_test_manifest()

print(
    f"Images used: {len(test_records):,}"
)


# ============================================================
# 6. Load ResNet18
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
).to(device)

model.eval()


# ============================================================
# 7. Load coreset
# ============================================================

checkpoint = torch.load(
    CORESET_PATH,
    map_location="cpu",
    weights_only=True
)


memory_cpu = (
    checkpoint["memory_bank"]
    .float()
)

memory_cpu = F.normalize(
    memory_cpu,
    p=2,
    dim=1
)


memory_gpu = (
    memory_cpu
    .to(device)
)


print(
    "Coreset:",
    tuple(memory_cpu.shape)
)


# ============================================================
# 8. Preprocessing
# ============================================================

def letterbox_image(image):

    width, height = image.size

    scale = min(
        TARGET_SIZE / width,
        TARGET_SIZE / height
    )

    new_width = max(
        1,
        int(round(width * scale))
    )

    new_height = max(
        1,
        int(round(height * scale))
    )

    resized = image.resize(
        (new_width, new_height),
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
        TARGET_SIZE
        -
        new_width
    ) // 2

    top = (
        TARGET_SIZE
        -
        new_height
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

    return tensor, valid_region


# ============================================================
# 9. Feature extraction
# ============================================================

def extract_features(tensor):

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

    return torch.cat(
        [
            layer2,
            layer3_up
        ],
        dim=1
    )


# ============================================================
# 10. Extract valid test patches
# ============================================================

def extract_test_patches(
    image_path
):

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

    with torch.no_grad():

        feature_map = extract_features(
            tensor
        )

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

        valid_region_gpu = (
            valid_region
            .unsqueeze(0)
            .to(device)
        )

        valid_ratio = (
            F.adaptive_avg_pool2d(
                valid_region_gpu,
                output_size=(h, w)
            )
            .reshape(-1)
        )

        mask = (
            valid_ratio
            >=
            MIN_VALID_RATIO
        )

        patches = patches[mask]

    return patches


# ============================================================
# 11A. Existing full-matrix GPU search
# ============================================================

def search_full_gpu(
    query_patches
):

    sync()

    start = time.perf_counter()

    with torch.no_grad():

        similarity_matrix = (
            query_patches
            @
            memory_gpu.T
        )

        nearest_similarity = (
            similarity_matrix
            .max(dim=1)
            .values
        )

        distances = torch.sqrt(
            torch.clamp(
                2.0
                -
                2.0
                *
                nearest_similarity,
                min=0.0
            )
        )

        image_score = (
            distances
            .max()
            .item()
        )

    sync()

    latency_ms = (
        time.perf_counter()
        -
        start
    ) * 1000

    return (
        image_score,
        latency_ms
    )


# ============================================================
# 11B. Chunked GPU exact search
# ============================================================

def search_chunked_gpu(
    query_patches
):

    sync()

    start = time.perf_counter()

    with torch.no_grad():

        # Keep only the best similarity found
        # for every query patch.
        best_similarity = torch.full(
            (
                query_patches.shape[0],
            ),
            -float("inf"),
            device=device
        )

        total_memory = (
            memory_gpu.shape[0]
        )

        for chunk_start in range(
            0,
            total_memory,
            CHUNK_SIZE
        ):

            chunk_end = min(
                chunk_start + CHUNK_SIZE,
                total_memory
            )

            memory_chunk = (
                memory_gpu[
                    chunk_start:
                    chunk_end
                ]
            )

            similarities = (
                query_patches
                @
                memory_chunk.T
            )

            chunk_best = (
                similarities
                .max(dim=1)
                .values
            )

            best_similarity = (
                torch.maximum(
                    best_similarity,
                    chunk_best
                )
            )

        distances = torch.sqrt(
            torch.clamp(
                2.0
                -
                2.0
                *
                best_similarity,
                min=0.0
            )
        )

        image_score = (
            distances
            .max()
            .item()
        )

    sync()

    latency_ms = (
        time.perf_counter()
        -
        start
    ) * 1000

    return (
        image_score,
        latency_ms
    )


# ============================================================
# 12. Build FAISS exact index
# ============================================================

faiss_index = None
faiss_gpu_index = None
FAISS_GPU_AVAILABLE = False


if FAISS_AVAILABLE:

    # --------------------------------------------------------
    # Exact cosine similarity search.
    #
    # Features are already L2 normalized,
    # so Inner Product = Cosine Similarity.
    # --------------------------------------------------------

    memory_numpy = (
        memory_cpu
        .numpy()
        .astype(
            np.float32,
            copy=False
        )
    )

    faiss_index = faiss.IndexFlatIP(
        memory_numpy.shape[1]
    )

    faiss_index.add(
        memory_numpy
    )


    # --------------------------------------------------------
    # Optional GPU FAISS
    # --------------------------------------------------------

    try:

        if hasattr(
            faiss,
            "StandardGpuResources"
        ):

            resources = (
                faiss.StandardGpuResources()
            )

            faiss_gpu_index = (
                faiss.index_cpu_to_gpu(
                    resources,
                    0,
                    faiss_index
                )
            )

            FAISS_GPU_AVAILABLE = True

    except Exception as exc:

        print(
            "FAISS GPU unavailable:",
            exc
        )


print(
    "FAISS GPU available:",
    FAISS_GPU_AVAILABLE
)


# ============================================================
# 13A. FAISS CPU exact search
# ============================================================

def search_faiss_cpu(
    query_patches
):

    if faiss_index is None:
        return None, None

    # Important:
    # CNN output currently lives on GPU,
    # therefore this includes GPU -> CPU transfer.
    sync()

    start = time.perf_counter()

    query_numpy = (
        query_patches
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False
        )
    )

    similarities, _ = (
        faiss_index.search(
            query_numpy,
            1
        )
    )

    nearest_similarity = (
        similarities[:, 0]
    )

    distances = np.sqrt(
        np.clip(
            2.0
            -
            2.0
            *
            nearest_similarity,
            0.0,
            None
        )
    )

    image_score = float(
        distances.max()
    )

    latency_ms = (
        time.perf_counter()
        -
        start
    ) * 1000

    return (
        image_score,
        latency_ms
    )


# ============================================================
# 13B. Optional FAISS GPU exact search
# ============================================================

def search_faiss_gpu(
    query_patches
):

    if not FAISS_GPU_AVAILABLE:
        return None, None

    # Standard FAISS Python GPU path usually
    # receives NumPy input here.
    query_numpy = (
        query_patches
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32,
            copy=False
        )
    )

    sync()

    start = time.perf_counter()

    similarities, _ = (
        faiss_gpu_index.search(
            query_numpy,
            1
        )
    )

    sync()

    nearest_similarity = (
        similarities[:, 0]
    )

    distances = np.sqrt(
        np.clip(
            2.0
            -
            2.0
            *
            nearest_similarity,
            0.0,
            None
        )
    )

    image_score = float(
        distances.max()
    )

    latency_ms = (
        time.perf_counter()
        -
        start
    ) * 1000

    return (
        image_score,
        latency_ms
    )


# ============================================================
# 14. Statistics
# ============================================================

def summarize(values):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    return {
        "mean_ms":
            float(np.mean(values)),

        "median_ms":
            float(np.median(values)),

        "p95_ms":
            float(
                np.percentile(
                    values,
                    95
                )
            ),

        "min_ms":
            float(np.min(values)),

        "max_ms":
            float(np.max(values)),
    }


# ============================================================
# 15. Warmup
# ============================================================

def warmup():

    count = min(
        WARMUP_IMAGES,
        len(test_records)
    )

    print()
    print("=" * 72)
    print("VECTOR SEARCH WARM-UP")
    print("=" * 72)

    for i in range(count):

        patches = extract_test_patches(
            test_records[i][
                "image_path"
            ]
        )

        search_full_gpu(
            patches
        )

        search_chunked_gpu(
            patches
        )

        if FAISS_AVAILABLE:
            search_faiss_cpu(
                patches
            )

        if FAISS_GPU_AVAILABLE:
            search_faiss_gpu(
                patches
            )

        print(
            f"Warm-up {i + 1}/{count}"
        )


# ============================================================
# 16. Benchmark
# ============================================================

def main():

    warmup()

    print()
    print("=" * 72)
    print("VECTOR SEARCH BENCHMARK")
    print("=" * 72)


    full_times = []
    chunk_times = []

    faiss_cpu_times = []
    faiss_gpu_times = []

    score_diff_chunk = []
    score_diff_faiss_cpu = []
    score_diff_faiss_gpu = []

    prediction_match_chunk = []
    prediction_match_faiss_cpu = []
    prediction_match_faiss_gpu = []

    rows = []

    reset_gpu_memory()

    start_total = time.perf_counter()


    for index, record in enumerate(
        test_records,
        start=1
    ):

        query_patches = (
            extract_test_patches(
                record["image_path"]
            )
        )


        # ====================================================
        # Baseline
        # ====================================================

        baseline_score, baseline_ms = (
            search_full_gpu(
                query_patches
            )
        )

        full_times.append(
            baseline_ms
        )


        baseline_prediction = (
            baseline_score
            >=
            THRESHOLD
        )


        # ====================================================
        # Chunked GPU
        # ====================================================

        chunk_score, chunk_ms = (
            search_chunked_gpu(
                query_patches
            )
        )

        chunk_times.append(
            chunk_ms
        )


        chunk_diff = abs(
            baseline_score
            -
            chunk_score
        )

        score_diff_chunk.append(
            chunk_diff
        )


        chunk_prediction = (
            chunk_score
            >=
            THRESHOLD
        )

        prediction_match_chunk.append(
            baseline_prediction
            ==
            chunk_prediction
        )


        # ====================================================
        # FAISS CPU
        # ====================================================

        faiss_cpu_score = None
        faiss_cpu_ms = None
        faiss_cpu_diff = None
        faiss_cpu_match = None


        if FAISS_AVAILABLE:

            (
                faiss_cpu_score,
                faiss_cpu_ms

            ) = search_faiss_cpu(
                query_patches
            )

            faiss_cpu_times.append(
                faiss_cpu_ms
            )

            faiss_cpu_diff = abs(
                baseline_score
                -
                faiss_cpu_score
            )

            score_diff_faiss_cpu.append(
                faiss_cpu_diff
            )

            faiss_cpu_match = (
                baseline_prediction
                ==
                (
                    faiss_cpu_score
                    >=
                    THRESHOLD
                )
            )

            prediction_match_faiss_cpu.append(
                faiss_cpu_match
            )


        # ====================================================
        # Optional FAISS GPU
        # ====================================================

        faiss_gpu_score = None
        faiss_gpu_ms = None
        faiss_gpu_diff = None
        faiss_gpu_match = None


        if FAISS_GPU_AVAILABLE:

            (
                faiss_gpu_score,
                faiss_gpu_ms

            ) = search_faiss_gpu(
                query_patches
            )

            faiss_gpu_times.append(
                faiss_gpu_ms
            )

            faiss_gpu_diff = abs(
                baseline_score
                -
                faiss_gpu_score
            )

            score_diff_faiss_gpu.append(
                faiss_gpu_diff
            )

            faiss_gpu_match = (
                baseline_prediction
                ==
                (
                    faiss_gpu_score
                    >=
                    THRESHOLD
                )
            )

            prediction_match_faiss_gpu.append(
                faiss_gpu_match
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

                "valid_patches":
                    int(
                        query_patches.shape[0]
                    ),

                "baseline_score":
                    baseline_score,

                "baseline_ms":
                    baseline_ms,

                "chunked_score":
                    chunk_score,

                "chunked_ms":
                    chunk_ms,

                "chunked_score_diff":
                    chunk_diff,

                "chunked_prediction_match":
                    chunk_prediction
                    ==
                    baseline_prediction,

                "faiss_cpu_score":
                    faiss_cpu_score,

                "faiss_cpu_ms":
                    faiss_cpu_ms,

                "faiss_cpu_score_diff":
                    faiss_cpu_diff,

                "faiss_cpu_prediction_match":
                    faiss_cpu_match,

                "faiss_gpu_score":
                    faiss_gpu_score,

                "faiss_gpu_ms":
                    faiss_gpu_ms,

                "faiss_gpu_score_diff":
                    faiss_gpu_diff,

                "faiss_gpu_prediction_match":
                    faiss_gpu_match,
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
                start_total
            )

            print(
                f"Processed "
                f"{index:,}/"
                f"{len(test_records):,}"
                f" | elapsed="
                f"{elapsed:.1f}s"
            )


    # ========================================================
    # Peak GPU memory
    # ========================================================

    peak_gpu_mb = None

    if device.type == "cuda":

        peak_gpu_mb = (
            torch.cuda
            .max_memory_allocated()
            /
            (1024 ** 2)
        )


    # ========================================================
    # Results
    # ========================================================

    results = {

        "experiment":
            "vector_search_exact_benchmark",

        "device":
            str(device),

        "images":
            len(test_records),

        "coreset_shape":
            list(memory_gpu.shape),

        "chunk_size":
            CHUNK_SIZE,

        "threshold":
            THRESHOLD,

        "full_gpu":
            {
                "latency":
                    summarize(
                        full_times
                    )
            },

        "chunked_gpu":
            {
                "latency":
                    summarize(
                        chunk_times
                    ),

                "max_score_difference":
                    float(
                        max(
                            score_diff_chunk
                        )
                    ),

                "prediction_consistency":
                    float(
                        np.mean(
                            prediction_match_chunk
                        )
                    ),
            },

        "faiss_available":
            FAISS_AVAILABLE,

        "faiss_gpu_available":
            FAISS_GPU_AVAILABLE,

        "peak_gpu_memory_mb":
            peak_gpu_mb,
    }


    if FAISS_AVAILABLE:

        results[
            "faiss_cpu"
        ] = {

            "latency":
                summarize(
                    faiss_cpu_times
                ),

            "max_score_difference":
                float(
                    max(
                        score_diff_faiss_cpu
                    )
                ),

            "prediction_consistency":
                float(
                    np.mean(
                        prediction_match_faiss_cpu
                    )
                ),
        }


    if FAISS_GPU_AVAILABLE:

        results[
            "faiss_gpu"
        ] = {

            "latency":
                summarize(
                    faiss_gpu_times
                ),

            "max_score_difference":
                float(
                    max(
                        score_diff_faiss_gpu
                    )
                ),

            "prediction_consistency":
                float(
                    np.mean(
                        prediction_match_faiss_gpu
                    )
                ),
        }


    # ========================================================
    # Save CSV
    # ========================================================

    with OUTPUT_CSV.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=
                rows[0].keys()
        )

        writer.writeheader()
        writer.writerows(rows)


    # ========================================================
    # Save JSON
    # ========================================================

    OUTPUT_JSON.write_text(
        json.dumps(
            results,
            indent=2
        ),
        encoding="utf-8"
    )


    # ========================================================
    # Print
    # ========================================================

    print()
    print("=" * 72)
    print("VECTOR SEARCH RESULTS")
    print("=" * 72)


    baseline_stats = (
        results[
            "full_gpu"
        ][
            "latency"
        ]
    )

    print(
        "\nFull GPU brute-force"
    )

    print(
        f"  Mean:   "
        f"{baseline_stats['mean_ms']:.3f} ms"
    )

    print(
        f"  Median: "
        f"{baseline_stats['median_ms']:.3f} ms"
    )

    print(
        f"  P95:    "
        f"{baseline_stats['p95_ms']:.3f} ms"
    )


    chunk_stats = (
        results[
            "chunked_gpu"
        ][
            "latency"
        ]
    )

    print(
        "\nChunked GPU exact"
    )

    print(
        f"  Mean:   "
        f"{chunk_stats['mean_ms']:.3f} ms"
    )

    print(
        f"  Median: "
        f"{chunk_stats['median_ms']:.3f} ms"
    )

    print(
        f"  P95:    "
        f"{chunk_stats['p95_ms']:.3f} ms"
    )

    print(
        f"  Max score diff: "
        f"{results['chunked_gpu']['max_score_difference']:.8f}"
    )

    print(
        f"  Prediction consistency: "
        f"{results['chunked_gpu']['prediction_consistency']:.2%}"
    )


    if FAISS_AVAILABLE:

        stats = (
            results[
                "faiss_cpu"
            ][
                "latency"
            ]
        )

        print(
            "\nFAISS CPU exact"
        )

        print(
            f"  Mean:   "
            f"{stats['mean_ms']:.3f} ms"
        )

        print(
            f"  Median: "
            f"{stats['median_ms']:.3f} ms"
        )

        print(
            f"  P95:    "
            f"{stats['p95_ms']:.3f} ms"
        )

        print(
            f"  Max score diff: "
            f"{results['faiss_cpu']['max_score_difference']:.8f}"
        )

        print(
            f"  Prediction consistency: "
            f"{results['faiss_cpu']['prediction_consistency']:.2%}"
        )


    if FAISS_GPU_AVAILABLE:

        stats = (
            results[
                "faiss_gpu"
            ][
                "latency"
            ]
        )

        print(
            "\nFAISS GPU exact"
        )

        print(
            f"  Mean:   "
            f"{stats['mean_ms']:.3f} ms"
        )

        print(
            f"  Median: "
            f"{stats['median_ms']:.3f} ms"
        )

        print(
            f"  P95:    "
            f"{stats['p95_ms']:.3f} ms"
        )

        print(
            f"  Max score diff: "
            f"{results['faiss_gpu']['max_score_difference']:.8f}"
        )

        print(
            f"  Prediction consistency: "
            f"{results['faiss_gpu']['prediction_consistency']:.2%}"
        )


    print()

    if peak_gpu_mb is not None:

        print(
            f"Peak CUDA memory: "
            f"{peak_gpu_mb:.2f} MB"
        )


    # ========================================================
    # Safety check
    # ========================================================

    if max(
        score_diff_chunk
    ) > SCORE_TOLERANCE:

        print(
            "\nWARNING: Chunked search "
            "score difference exceeded tolerance."
        )

    else:

        print(
            "\nChunked exact search matches "
            "baseline within tolerance."
        )


    print(
        "\nSaved:"
    )

    print(OUTPUT_CSV)
    print(OUTPUT_JSON)


if __name__ == "__main__":
    main()