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
from torchvision.transforms.functional import (
    pil_to_tensor,
    normalize,
)

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)


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

TRAIN_MANIFEST = (
    CLEAN_REPORT_DIR
    / "train_clean.csv"
)

VAL_MANIFEST = (
    CLEAN_REPORT_DIR
    / "val_clean.csv"
)

TEST_MANIFEST = (
    CLEAN_REPORT_DIR
    / "test_clean.csv"
)


MODEL_DIR = (
    PROJECT_ROOT
    / "models"
    / "clean_evaluation"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FULL_MEMORY_PATH = (
    MODEL_DIR
    / "acc_clean_full32_memory_bank.pt"
)

CORESET_PATH = (
    MODEL_DIR
    / "acc_clean_full32_greedy10_coreset.pt"
)


VAL_SCORE_CSV = (
    CLEAN_REPORT_DIR
    / "val_clean_scores.csv"
)

TEST_SCORE_CSV = (
    CLEAN_REPORT_DIR
    / "test_clean_scores.csv"
)

FINAL_RESULTS_JSON = (
    CLEAN_REPORT_DIR
    / "clean_v2c_final_results.json"
)


# ============================================================
# 2. Configuration
# ============================================================

TARGET_SIZE = 256

MIN_VALID_RATIO = 0.50

CORESET_RATIO = 0.10

PROJECTION_DIM = 64

RANDOM_SEED = 42


# ------------------------------------------------------------
# First run:
# True / True
#
# If memory bank and coreset are already built,
# change to False / False to skip rebuilding.
# ------------------------------------------------------------

REBUILD_MEMORY_BANK = True

REBUILD_CORESET = True


# ============================================================
# 3. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using device:",
    device
)


# ============================================================
# 4. Read manifest
# ============================================================

def load_manifest(
    csv_path: Path
):

    if not csv_path.exists():

        raise FileNotFoundError(
            f"Manifest not found:\n{csv_path}"
        )


    records = []


    with csv_path.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(
            file
        )


        for row in reader:

            records.append(
                {
                    "image_path":
                        PROJECT_ROOT
                        /
                        row["image_path"],

                    "label":
                        int(
                            row["label"]
                        ),

                    "defect_type":
                        row["defect_type"],

                    "sha256":
                        row["sha256"],
                }
            )


    return records


train_records = load_manifest(
    TRAIN_MANIFEST
)

val_records = load_manifest(
    VAL_MANIFEST
)

test_records = load_manifest(
    TEST_MANIFEST
)


print()

print(
    "Clean manifests:"
)

print(
    f"  Train: {len(train_records):,}"
)

print(
    f"  Val:   {len(val_records):,}"
)

print(
    f"  Test:  {len(test_records):,}"
)


# ============================================================
# 5. Safety checks
# ============================================================

if any(
    row["label"] != 0
    for row in train_records
):

    raise RuntimeError(
        "Training manifest contains defect samples. "
        "This experiment must remain normal-only."
    )


train_hashes = {
    row["sha256"]
    for row in train_records
}

val_hashes = {
    row["sha256"]
    for row in val_records
}

test_hashes = {
    row["sha256"]
    for row in test_records
}


if train_hashes & val_hashes:

    raise RuntimeError(
        "Leakage detected between train and validation."
    )


if train_hashes & test_hashes:

    raise RuntimeError(
        "Leakage detected between train and test."
    )


if val_hashes & test_hashes:

    raise RuntimeError(
        "Leakage detected between validation and test."
    )


print(
    "\nExact-hash leakage check: PASS"
)


# ============================================================
# 6. ResNet18
# ============================================================

weights = (
    models.ResNet18_Weights.DEFAULT
)

model = models.resnet18(
    weights=weights
).to(device)

model.eval()


# ============================================================
# 7. Letterbox preprocessing
# ============================================================

def letterbox_image(
    image: Image.Image
):

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


    resized = image.resize(
        (
            new_width,
            new_height
        ),
        resample=Image.Resampling.BILINEAR
    )


    # ImageNet RGB mean
    mean_rgb = (
        0.485,
        0.456,
        0.406
    )


    padding_color = tuple(
        int(
            round(
                value
                *
                255
            )
        )
        for value in mean_rgb
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
        pil_to_tensor(
            canvas
        )
        .float()
        /
        255.0
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
# 8. Full32 feature extraction
#
# layer2:
# 32 x 32 x 128
#
# layer3:
# 16 x 16 x 256
#
# Upsample layer3 -> 32 x 32
#
# Concatenate:
# 32 x 32 x 384
# ============================================================

def extract_highres_features(
    tensor
):

    x = model.conv1(
        tensor
    )

    x = model.bn1(
        x
    )

    x = model.relu(
        x
    )

    x = model.maxpool(
        x
    )


    x = model.layer1(
        x
    )


    layer2 = model.layer2(
        x
    )


    layer3 = model.layer3(
        layer2
    )


    layer3_upsampled = (
        F.interpolate(
            layer3,
            size=(
                layer2.shape[-2],
                layer2.shape[-1]
            ),
            mode="bilinear",
            align_corners=False
        )
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


    valid_patch_mask = (
        valid_ratio
        >=
        MIN_VALID_RATIO
    )


    return valid_patch_mask


# ============================================================
# 11. Build CLEAN full32 memory bank
# ============================================================

def build_memory_bank():

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "BUILD CLEAN FULL32 MEMORY BANK"
    )

    print(
        "=" * 72
    )


    all_features = []

    total_candidate_patches = 0

    total_kept_patches = 0


    start_time = (
        time.perf_counter()
    )


    for index, record in enumerate(
        train_records,
        start=1
    ):

        image_path = (
            record["image_path"]
        )


        image = Image.open(
            image_path
        ).convert(
            "RGB"
        )


        (
            tensor,
            valid_region

        ) = letterbox_image(
            image
        )


        tensor = (
            tensor
            .unsqueeze(0)
            .to(device)
        )


        with torch.no_grad():

            feature_map = (
                extract_highres_features(
                    tensor
                )
            )


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


            total_candidate_patches += (
                patches.shape[0]
            )


            valid_patches = (
                patches[
                    valid_patch_mask
                ]
            )


            total_kept_patches += (
                valid_patches.shape[0]
            )


            all_features.append(
                valid_patches.cpu()
            )


        if (
            index % 50 == 0
            or
            index == len(train_records)
        ):

            elapsed = (
                time.perf_counter()
                -
                start_time
            )


            print(
                f"Processed "
                f"{index:,}/"
                f"{len(train_records):,}"
                f" | kept="
                f"{total_kept_patches:,}"
                f" | elapsed="
                f"{elapsed:.1f}s"
            )


    if not all_features:

        raise RuntimeError(
            "No training features collected."
        )


    memory_bank = torch.cat(
        all_features,
        dim=0
    )


    keep_ratio = (
        total_kept_patches
        /
        total_candidate_patches
    )


    memory_size_mb = (
        memory_bank.numel()
        *
        memory_bank.element_size()
        /
        (1024 ** 2)
    )


    checkpoint = {

        "memory_bank":
            memory_bank,

        "target_size":
            TARGET_SIZE,

        "min_valid_ratio":
            MIN_VALID_RATIO,

        "feature_dim":
            int(
                memory_bank.shape[1]
            ),

        "num_training_images":
            len(train_records),

        "candidate_patches":
            total_candidate_patches,

        "kept_patches":
            total_kept_patches,

        "keep_ratio":
            keep_ratio,

        "preprocessing":
            "letterbox256",

        "feature_layers":
            [
                "layer2",
                "layer3"
            ],

        "feature_resolution":
            "32x32",

        "spatial_sampling":
            "full32_no_stride",

        "split":
            "clean_train_manifest",

        "experiment":
            "clean_v2c_full32",
    }


    torch.save(
        checkpoint,
        FULL_MEMORY_PATH
    )


    print(
        "\nMemory bank shape:",
        tuple(
            memory_bank.shape
        )
    )

    print(
        f"Memory size: "
        f"{memory_size_mb:.2f} MB"
    )

    print(
        f"Valid patch keep ratio: "
        f"{keep_ratio:.2%}"
    )

    print(
        "\nSaved:"
    )

    print(
        FULL_MEMORY_PATH
    )


# ============================================================
# 12. Build 10% traditional greedy coreset
# ============================================================

def build_coreset():

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "BUILD CLEAN FULL32 10% GREEDY CORESET"
    )

    print(
        "=" * 72
    )


    checkpoint = torch.load(
        FULL_MEMORY_PATH,
        map_location="cpu",
        weights_only=True
    )


    memory_bank = (
        checkpoint["memory_bank"]
        .float()
    )


    memory_bank = F.normalize(
        memory_bank,
        p=2,
        dim=1
    )


    num_features = (
        memory_bank.shape[0]
    )

    feature_dim = (
        memory_bank.shape[1]
    )


    target_size = max(
        1,
        int(
            round(
                num_features
                *
                CORESET_RATIO
            )
        )
    )


    print(
        "Memory bank:",
        tuple(
            memory_bank.shape
        )
    )

    print(
        f"Target coreset: "
        f"{target_size:,}"
    )


    # --------------------------------------------------------
    # Random projection 384 -> 64
    # --------------------------------------------------------

    torch.manual_seed(
        RANDOM_SEED
    )


    projection_matrix = (
        torch.randn(
            feature_dim,
            PROJECTION_DIM,
            dtype=torch.float32
        )
        /
        np.sqrt(
            PROJECTION_DIM
        )
    )


    print(
        "\nProjecting features..."
    )


    projection_start = (
        time.perf_counter()
    )


    projected = (
        memory_bank
        @
        projection_matrix
    )


    projected = F.normalize(
        projected,
        p=2,
        dim=1
    )


    print(
        "Projected shape:",
        tuple(
            projected.shape
        )
    )

    print(
        f"Projection time: "
        f"{time.perf_counter() - projection_start:.2f}s"
    )


    # --------------------------------------------------------
    # Traditional greedy farthest-first
    # --------------------------------------------------------

    print(
        "\nStarting traditional greedy selection..."
    )


    start_time = (
        time.perf_counter()
    )


    selected_indices = [
        0
    ]


    first_feature = (
        projected[
            0:1
        ]
    )


    max_similarity = (
        projected
        @
        first_feature.T
    ).squeeze(1)


    max_similarity[
        0
    ] = 1.0


    for step in range(
        1,
        target_size
    ):

        next_index = int(
            torch.argmin(
                max_similarity
            )
        )


        selected_indices.append(
            next_index
        )


        new_feature = (
            projected[
                next_index:
                next_index + 1
            ]
        )


        similarity_to_new = (
            projected
            @
            new_feature.T
        ).squeeze(1)


        max_similarity = (
            torch.maximum(
                max_similarity,
                similarity_to_new
            )
        )


        max_similarity[
            selected_indices
        ] = 1.0


        if (
            (step + 1) % 100 == 0
            or
            step + 1 == target_size
        ):

            elapsed = (
                time.perf_counter()
                -
                start_time
            )


            avg_time = (
                elapsed
                /
                (step + 1)
            )


            remaining_steps = (
                target_size
                -
                (step + 1)
            )


            eta = (
                avg_time
                *
                remaining_steps
            )


            print(
                f"Selected "
                f"{step + 1:,}/"
                f"{target_size:,}"
                f" | elapsed="
                f"{elapsed:.1f}s"
                f" | ETA="
                f"{eta / 60:.1f} min"
            )


    selected_indices_tensor = (
        torch.tensor(
            selected_indices,
            dtype=torch.long
        )
    )


    coreset = (
        memory_bank[
            selected_indices_tensor
        ]
    )


    full_size_mb = (
        memory_bank.numel()
        *
        memory_bank.element_size()
        /
        (1024 ** 2)
    )


    coreset_size_mb = (
        coreset.numel()
        *
        coreset.element_size()
        /
        (1024 ** 2)
    )


    runtime = (
        time.perf_counter()
        -
        start_time
    )


    output_checkpoint = {

        "memory_bank":
            coreset,

        "selected_indices":
            selected_indices_tensor,

        "coreset_ratio":
            CORESET_RATIO,

        "projection_dim":
            PROJECTION_DIM,

        "selection_method":
            "traditional_greedy_farthest_first",

        "target_size":
            checkpoint["target_size"],

        "min_valid_ratio":
            checkpoint["min_valid_ratio"],

        "feature_dim":
            checkpoint["feature_dim"],

        "num_training_images":
            checkpoint["num_training_images"],

        "preprocessing":
            checkpoint["preprocessing"],

        "feature_layers":
            checkpoint["feature_layers"],

        "feature_resolution":
            checkpoint["feature_resolution"],

        "spatial_sampling":
            checkpoint["spatial_sampling"],

        "split":
            "clean_train_manifest",

        "experiment":
            "clean_v2c_full32_greedy10",
    }


    torch.save(
        output_checkpoint,
        CORESET_PATH
    )


    print()

    print(
        "Original:",
        tuple(
            memory_bank.shape
        )
    )

    print(
        "Coreset:",
        tuple(
            coreset.shape
        )
    )

    print(
        f"Original memory: "
        f"{full_size_mb:.2f} MB"
    )

    print(
        f"Coreset memory: "
        f"{coreset_size_mb:.2f} MB"
    )

    print(
        f"Reduction: "
        f"{1 - len(coreset) / len(memory_bank):.2%}"
    )

    print(
        f"Greedy runtime: "
        f"{runtime:.2f}s"
    )

    print(
        "\nSaved:"
    )

    print(
        CORESET_PATH
    )


# ============================================================
# 13. Load final clean coreset
# ============================================================

def load_coreset():

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


    return (
        memory_bank,
        checkpoint
    )


# ============================================================
# 14. Inference
# ============================================================

def infer_image_score(
    image_path: Path,
    memory_bank
):

    image = Image.open(
        image_path
    ).convert(
        "RGB"
    )


    (
        tensor,
        valid_region

    ) = letterbox_image(
        image
    )


    tensor = (
        tensor
        .unsqueeze(0)
        .to(device)
    )


    with torch.no_grad():

        feature_map = (
            extract_highres_features(
                tensor
            )
        )


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


        # ----------------------------------------------
        # Features are L2 normalized.
        #
        # cosine similarity:
        # q @ memory.T
        #
        # squared Euclidean distance:
        # 2 - 2*cosine_similarity
        # ----------------------------------------------

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


        distance_squared = (
            torch.clamp(
                distance_squared,
                min=0.0
            )
        )


        patch_distances = (
            torch.sqrt(
                distance_squared
            )
        )


        image_score = (
            patch_distances
            .max()
            .item()
        )


    return (
        image_score,
        int(
            valid_patch_mask
            .sum()
            .item()
        ),
        h,
        w
    )


# ============================================================
# 15. Run inference on manifest
# ============================================================

def run_manifest_inference(
    records,
    memory_bank,
    output_csv: Path,
    split_name: str
):

    print(
        "\n"
        +
        "=" * 72
    )

    print(
        f"{split_name.upper()} INFERENCE"
    )

    print(
        "=" * 72
    )


    results = []

    y_true = []

    y_score = []


    start_time = (
        time.perf_counter()
    )


    for index, record in enumerate(
        records,
        start=1
    ):

        (
            score,
            valid_patches,
            h,
            w

        ) = infer_image_score(
            record["image_path"],
            memory_bank
        )


        y_true.append(
            record["label"]
        )

        y_score.append(
            score
        )


        results.append(
            {
                "image_path":
                    str(
                        record["image_path"]
                        .relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "defect_type":
                    record["defect_type"],

                "true_label":
                    record["label"],

                "score":
                    score,

                "valid_patches":
                    valid_patches,
            }
        )


        if (
            index % 50 == 0
            or
            index == len(records)
        ):

            elapsed = (
                time.perf_counter()
                -
                start_time
            )


            print(
                f"Processed "
                f"{index:,}/"
                f"{len(records):,}"
                f" | feature="
                f"{h}x{w}"
                f" | elapsed="
                f"{elapsed:.1f}s"
            )


    with output_csv.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_path",
                "defect_type",
                "true_label",
                "score",
                "valid_patches",
            ]
        )

        writer.writeheader()

        writer.writerows(
            results
        )


    return (
        np.asarray(
            y_true,
            dtype=np.uint8
        ),

        np.asarray(
            y_score,
            dtype=np.float64
        ),

        results
    )


# ============================================================
# 16. Threshold-free metrics
# ============================================================

def calculate_ranking_metrics(
    y_true,
    y_score
):

    if len(
        np.unique(
            y_true
        )
    ) < 2:

        raise RuntimeError(
            "Evaluation split must contain both good and defect samples."
        )


    auroc = roc_auc_score(
        y_true,
        y_score
    )


    ap = average_precision_score(
        y_true,
        y_score
    )


    return {
        "auroc":
            float(
                auroc
            ),

        "average_precision":
            float(
                ap
            ),
    }


# ============================================================
# 17. Choose threshold ONLY on validation
# ============================================================

def choose_best_f1_threshold(
    y_true,
    y_score
):

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_true,
            y_score
        )
    )


    # Last precision / recall value has no threshold.
    p = (
        precision_values[:-1]
    )

    r = (
        recall_values[:-1]
    )


    f1_values = (
        2.0
        *
        p
        *
        r
        /
        np.maximum(
            p + r,
            1e-12
        )
    )


    best_index = int(
        np.argmax(
            f1_values
        )
    )


    threshold = float(
        thresholds[
            best_index
        ]
    )


    return {
        "threshold":
            threshold,

        "validation_precision":
            float(
                p[
                    best_index
                ]
            ),

        "validation_recall":
            float(
                r[
                    best_index
                ]
            ),

        "validation_f1":
            float(
                f1_values[
                    best_index
                ]
            ),
    }


# ============================================================
# 18. Optional:
# threshold for minimum recall target
# ============================================================

def choose_threshold_for_recall(
    y_true,
    y_score,
    recall_target=0.95
):

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_true,
            y_score
        )
    )


    candidates = []


    for i in range(
        len(thresholds)
    ):

        recall = (
            recall_values[i]
        )


        precision = (
            precision_values[i]
        )


        if recall >= recall_target:

            candidates.append(
                (
                    float(
                        precision
                    ),
                    float(
                        thresholds[i]
                    ),
                    float(
                        recall
                    )
                )
            )


    if not candidates:

        return None


    # Among thresholds satisfying recall target,
    # prefer highest precision.
    candidates.sort(
        reverse=True
    )


    precision, threshold, recall = (
        candidates[0]
    )


    return {
        "threshold":
            threshold,

        "precision":
            precision,

        "recall":
            recall,
    }


# ============================================================
# 19. Threshold-dependent metrics
# ============================================================

def calculate_threshold_metrics(
    y_true,
    y_score,
    threshold
):

    y_pred = (
        y_score
        >=
        threshold
    ).astype(
        np.uint8
    )


    tn, fp, fn, tp = (
        confusion_matrix(
            y_true,
            y_pred,
            labels=[
                0,
                1
            ]
        )
        .ravel()
    )


    precision = (
        tp
        /
        (tp + fp)
        if
        (tp + fp) > 0
        else
        0.0
    )


    recall = (
        tp
        /
        (tp + fn)
        if
        (tp + fn) > 0
        else
        0.0
    )


    f1 = (
        2
        *
        precision
        *
        recall
        /
        (
            precision
            +
            recall
        )
        if
        (
            precision
            +
            recall
        ) > 0
        else
        0.0
    )


    fpr = (
        fp
        /
        (fp + tn)
        if
        (fp + tn) > 0
        else
        0.0
    )


    fnr = (
        fn
        /
        (fn + tp)
        if
        (fn + tp) > 0
        else
        0.0
    )


    return {

        "threshold":
            float(
                threshold
            ),

        "precision":
            float(
                precision
            ),

        "recall":
            float(
                recall
            ),

        "f1":
            float(
                f1
            ),

        "tn":
            int(
                tn
            ),

        "fp":
            int(
                fp
            ),

        "fn":
            int(
                fn
            ),

        "tp":
            int(
                tp
            ),

        "false_positive_rate":
            float(
                fpr
            ),

        "false_negative_rate":
            float(
                fnr
            ),
    }


# ============================================================
# 20. Score statistics
# ============================================================

def score_statistics(
    y_true,
    y_score
):

    good_scores = (
        y_score[
            y_true == 0
        ]
    )


    defect_scores = (
        y_score[
            y_true == 1
        ]
    )


    def summarize(
        values
    ):

        return {

            "count":
                int(
                    len(values)
                ),

            "mean":
                float(
                    np.mean(
                        values
                    )
                ),

            "median":
                float(
                    np.median(
                        values
                    )
                ),

            "min":
                float(
                    np.min(
                        values
                    )
                ),

            "max":
                float(
                    np.max(
                        values
                    )
                ),
        }


    return {

        "good":
            summarize(
                good_scores
            ),

        "defect":
            summarize(
                defect_scores
            ),
    }


# ============================================================
# 21. Print evaluation
# ============================================================

def print_evaluation(
    split_name,
    ranking_metrics,
    threshold_metrics,
    score_stats
):

    print()

    print(
        "=" * 72
    )

    print(
        f"{split_name.upper()} RESULTS"
    )

    print(
        "=" * 72
    )


    print(
        f"AUROC: "
        f"{ranking_metrics['auroc']:.4f}"
    )

    print(
        f"Average Precision: "
        f"{ranking_metrics['average_precision']:.4f}"
    )


    print(
        f"\nThreshold: "
        f"{threshold_metrics['threshold']:.6f}"
    )

    print(
        f"Precision: "
        f"{threshold_metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{threshold_metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{threshold_metrics['f1']:.4f}"
    )


    print(
        "\nConfusion matrix"
    )

    print(
        f"TN: "
        f"{threshold_metrics['tn']:,}"
    )

    print(
        f"FP: "
        f"{threshold_metrics['fp']:,}"
    )

    print(
        f"FN: "
        f"{threshold_metrics['fn']:,}"
    )

    print(
        f"TP: "
        f"{threshold_metrics['tp']:,}"
    )


    print(
        "\nGOOD scores"
    )

    print(
        f"mean:   "
        f"{score_stats['good']['mean']:.4f}"
    )

    print(
        f"median: "
        f"{score_stats['good']['median']:.4f}"
    )


    print(
        "\nDEFECT scores"
    )

    print(
        f"mean:   "
        f"{score_stats['defect']['mean']:.4f}"
    )

    print(
        f"median: "
        f"{score_stats['defect']['median']:.4f}"
    )


# ============================================================
# 22. Main
# ============================================================

def main():

    # --------------------------------------------------------
    # Build clean training reference
    # --------------------------------------------------------

    if REBUILD_MEMORY_BANK:

        build_memory_bank()

    elif not FULL_MEMORY_PATH.exists():

        raise FileNotFoundError(
            "REBUILD_MEMORY_BANK=False but memory bank does not exist:\n"
            f"{FULL_MEMORY_PATH}"
        )


    if REBUILD_CORESET:

        build_coreset()

    elif not CORESET_PATH.exists():

        raise FileNotFoundError(
            "REBUILD_CORESET=False but coreset does not exist:\n"
            f"{CORESET_PATH}"
        )


    # --------------------------------------------------------
    # Load clean coreset
    # --------------------------------------------------------

    memory_bank, checkpoint = (
        load_coreset()
    )


    print(
        "\nLoaded clean coreset:",
        tuple(
            memory_bank.shape
        )
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    (
        val_y_true,
        val_y_score,
        val_results

    ) = run_manifest_inference(

        val_records,
        memory_bank,
        VAL_SCORE_CSV,
        "validation"
    )


    val_ranking = (
        calculate_ranking_metrics(
            val_y_true,
            val_y_score
        )
    )


    # ----------------------------------------------
    # This is the ONLY place where threshold is chosen.
    # ----------------------------------------------

    threshold_selection = (
        choose_best_f1_threshold(
            val_y_true,
            val_y_score
        )
    )


    selected_threshold = (
        threshold_selection[
            "threshold"
        ]
    )


    val_threshold_metrics = (
        calculate_threshold_metrics(
            val_y_true,
            val_y_score,
            selected_threshold
        )
    )


    val_score_stats = (
        score_statistics(
            val_y_true,
            val_y_score
        )
    )


    print_evaluation(
        "validation",
        val_ranking,
        val_threshold_metrics,
        val_score_stats
    )


    # ----------------------------------------------
    # Also report an exploratory 95%-recall
    # operating point on validation.
    #
    # We do NOT use this for final test unless
    # you deliberately choose it later.
    # ----------------------------------------------

    recall95_threshold = (
        choose_threshold_for_recall(
            val_y_true,
            val_y_score,
            recall_target=0.95
        )
    )


    if recall95_threshold is not None:

        print(
            "\nValidation operating point "
            "for Recall >= 95%:"
        )

        print(
            json.dumps(
                recall95_threshold,
                indent=2
            )
        )


    # ========================================================
    # FINAL UNTOUCHED TEST
    #
    # IMPORTANT:
    # no threshold optimization here.
    # ========================================================

    (
        test_y_true,
        test_y_score,
        test_results

    ) = run_manifest_inference(

        test_records,
        memory_bank,
        TEST_SCORE_CSV,
        "final test"
    )


    test_ranking = (
        calculate_ranking_metrics(
            test_y_true,
            test_y_score
        )
    )


    test_threshold_metrics = (
        calculate_threshold_metrics(
            test_y_true,
            test_y_score,
            selected_threshold
        )
    )


    test_score_stats = (
        score_statistics(
            test_y_true,
            test_y_score
        )
    )


    print_evaluation(
        "final test",
        test_ranking,
        test_threshold_metrics,
        test_score_stats
    )


    # ========================================================
    # Save final JSON
    # ========================================================

    final_summary = {

        "experiment":
            "clean_v2c_full32_greedy10",

        "product":
            "Aluminum_Camera_Cover",

        "dataset_split":
            {
                "train_images":
                    len(
                        train_records
                    ),

                "validation_images":
                    len(
                        val_records
                    ),

                "test_images":
                    len(
                        test_records
                    ),

                "zero_exact_hash_overlap":
                    True,
            },

        "model":
            {
                "backbone":
                    "ResNet18",

                "pretrained":
                    "ImageNet",

                "feature_layers":
                    [
                        "layer2",
                        "layer3"
                    ],

                "feature_resolution":
                    "32x32",

                "feature_dim":
                    int(
                        checkpoint[
                            "feature_dim"
                        ]
                    ),

                "preprocessing":
                    "Letterbox256",

                "memory":
                    "normal_only",

                "coreset_ratio":
                    CORESET_RATIO,

                "coreset_size":
                    int(
                        memory_bank.shape[0]
                    ),
            },

        "threshold_selection":
            {
                "source":
                    "validation_only",

                "method":
                    "best_f1",

                "selected_threshold":
                    selected_threshold,

                "validation_at_selected_threshold":
                    val_threshold_metrics,

                "validation_recall95_operating_point":
                    recall95_threshold,
            },

        "validation":
            {
                "ranking_metrics":
                    val_ranking,

                "threshold_metrics":
                    val_threshold_metrics,

                "score_statistics":
                    val_score_stats,
            },

        "final_test":
            {
                "ranking_metrics":
                    test_ranking,

                "threshold_metrics_using_validation_threshold":
                    test_threshold_metrics,

                "score_statistics":
                    test_score_stats,
            },

        "notes":
            [
                (
                    "Exact duplicate leakage was removed "
                    "before evaluation."
                ),

                (
                    "Threshold was selected using validation only."
                ),

                (
                    "Final test was not used for threshold optimization."
                ),

                (
                    "Near-duplicate pHash candidates were not "
                    "automatically removed."
                ),
            ],
    }


    FINAL_RESULTS_JSON.write_text(
        json.dumps(
            final_summary,
            indent=2
        ),
        encoding="utf-8"
    )


    print(
        "\n"
        +
        "=" * 72
    )

    print(
        "CLEAN BENCHMARK COMPLETE"
    )

    print(
        "=" * 72
    )


    print(
        "\nValidation-selected threshold:"
    )

    print(
        f"{selected_threshold:.6f}"
    )


    print(
        "\nFINAL TEST:"
    )

    print(
        f"AUROC: "
        f"{test_ranking['auroc']:.4f}"
    )

    print(
        f"AP: "
        f"{test_ranking['average_precision']:.4f}"
    )

    print(
        f"Precision: "
        f"{test_threshold_metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{test_threshold_metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{test_threshold_metrics['f1']:.4f}"
    )

    print(
        f"FP: "
        f"{test_threshold_metrics['fp']:,}"
    )

    print(
        f"FN: "
        f"{test_threshold_metrics['fn']:,}"
    )


    print(
        "\nSaved:"
    )

    print(
        VAL_SCORE_CSV
    )

    print(
        TEST_SCORE_CSV
    )

    print(
        FINAL_RESULTS_JSON
    )


if __name__ == "__main__":

    main()