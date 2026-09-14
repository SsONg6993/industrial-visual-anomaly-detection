from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRODUCT_DIR = (
    PROJECT_ROOT
    / "data"
    / "3CAD"
    / "Aluminum_Camera_Cover"
)

TEST_DIR = (
    PRODUCT_DIR
    / "test"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CORESET_PATH = (
    MODEL_DIR
    / "aluminum_camera_cover_multilayer_coreset.pt"
)


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. Load coreset
# ============================================================

checkpoint = torch.load(
    CORESET_PATH,
    map_location="cpu",
    weights_only=True,
)

memory_bank = (
    checkpoint["memory_bank"]
    .float()
    .to(device)
)

print(
    "Memory bank shape:",
    tuple(memory_bank.shape)
)


# ============================================================
# 4. Load ResNet18
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
).to(device)

model.eval()

preprocess = weights.transforms()


# ============================================================
# 5. Multi-layer feature extraction
# ============================================================

def extract_multilayer_features(
    image_tensor
):

    x = model.conv1(
        image_tensor
    )

    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)

    layer2 = model.layer2(x)

    layer3 = model.layer3(
        layer2
    )


    # Align layer2 spatial size:
    #
    # 28x28 -> 14x14

    layer2_downsampled = (
        F.adaptive_avg_pool2d(
            layer2,
            output_size=(14, 14)
        )
    )


    # Concatenate channels:
    #
    # 128 + 256 = 384

    combined = torch.cat(
        [
            layer2_downsampled,
            layer3,
        ],
        dim=1
    )

    return combined


# ============================================================
# 6. Feature map -> patch vectors
# ============================================================

def feature_map_to_patches(
    feature_map
):

    _, channels, h, w = (
        feature_map.shape
    )


    feature_map = (
        feature_map.permute(
            0,
            2,
            3,
            1
        )
    )


    patches = feature_map.reshape(
        h * w,
        channels
    )


    patches = F.normalize(
        patches,
        p=2,
        dim=1
    )

    return patches


# ============================================================
# 7. Image inference
# ============================================================

def infer_image_score(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")


    tensor = preprocess(
        image
    )


    tensor = (
        tensor
        .unsqueeze(0)
        .to(device)
    )


    with torch.no_grad():

        feature_map = (
            extract_multilayer_features(
                tensor
            )
        )


        patches = (
            feature_map_to_patches(
                feature_map
            )
        )


        similarity_matrix = (
            patches
            @
            memory_bank.T
        )


        nearest_similarity = (
            similarity_matrix.max(
                dim=1
            ).values
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


        # Baseline image-level score:
        #
        # most anomalous patch

        image_score = (
            patch_distances.max().item()
        )


    return image_score


# ============================================================
# 8. Collect test samples
# ============================================================

samples = []


for defect_dir in sorted(
    TEST_DIR.iterdir()
):

    if not defect_dir.is_dir():
        continue


    defect_type = (
        defect_dir.name
    )


    label = (
        0
        if defect_type == "good"
        else 1
    )


    for image_path in sorted(
        defect_dir.glob("*.png")
    ):

        samples.append(
            (
                image_path,
                defect_type,
                label,
            )
        )


print(
    f"\nTest images: "
    f"{len(samples):,}"
)


# ============================================================
# 9. Run entire test set
# ============================================================

labels = []
scores = []

start_time = time.perf_counter()


for index, (
    image_path,
    defect_type,
    label,

) in enumerate(
    samples,
    start=1
):


    score = infer_image_score(
        image_path
    )


    labels.append(
        label
    )

    scores.append(
        score
    )


    if (
        index % 50 == 0
        or
        index == len(samples)
    ):

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        print(
            f"Processed "
            f"{index:,}/"
            f"{len(samples):,}"
            f" | elapsed="
            f"{elapsed:.1f}s"
        )


labels = np.array(
    labels,
    dtype=np.int64
)

scores = np.array(
    scores,
    dtype=np.float32
)


# ============================================================
# 10. AUROC + Average Precision
# ============================================================

auroc = roc_auc_score(
    labels,
    scores
)


average_precision = (
    average_precision_score(
        labels,
        scores
    )
)


print(
    f"\nImage AUROC: "
    f"{auroc:.4f}"
)

print(
    f"Average Precision / PR-AUC: "
    f"{average_precision:.4f}"
)


# ============================================================
# 11. ROC curve
# ============================================================

fpr, tpr, roc_thresholds = (
    roc_curve(
        labels,
        scores
    )
)


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    fpr,
    tpr,
    label=f"AUROC = {auroc:.4f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate / Recall"
)

plt.title(
    "Image-Level ROC Curve"
)

plt.legend()

plt.tight_layout()


roc_path = (
    REPORT_DIR
    / "image_roc_curve.png"
)

plt.savefig(
    roc_path,
    dpi=150
)

plt.close()


# ============================================================
# 12. Precision-Recall curve
# ============================================================

precision, recall, pr_thresholds = (
    precision_recall_curve(
        labels,
        scores
    )
)


plt.figure(
    figsize=(7, 6)
)

plt.plot(
    recall,
    precision,
    label=(
        f"AP = "
        f"{average_precision:.4f}"
    )
)

plt.xlabel(
    "Recall"
)

plt.ylabel(
    "Precision"
)

plt.title(
    "Image-Level Precision-Recall Curve"
)

plt.legend()

plt.tight_layout()


pr_path = (
    REPORT_DIR
    / "image_pr_curve.png"
)

plt.savefig(
    pr_path,
    dpi=150
)

plt.close()


# ============================================================
# 13. Threshold sweep
# ============================================================

threshold_candidates = np.linspace(
    scores.min(),
    scores.max(),
    500
)


results = []


for threshold in threshold_candidates:

    predictions = (
        scores >= threshold
    ).astype(
        np.int64
    )


    precision_value = (
        precision_score(
            labels,
            predictions,
            zero_division=0
        )
    )


    recall_value = (
        recall_score(
            labels,
            predictions,
            zero_division=0
        )
    )


    f1_value = (
        f1_score(
            labels,
            predictions,
            zero_division=0
        )
    )


    tn, fp, fn, tp = (
        confusion_matrix(
            labels,
            predictions,
            labels=[0, 1]
        ).ravel()
    )


    false_positive_rate = (
        fp
        /
        (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )


    false_negative_rate = (
        fn
        /
        (fn + tp)
        if (fn + tp) > 0
        else 0.0
    )


    results.append(
        {
            "threshold":
                threshold,

            "precision":
                precision_value,

            "recall":
                recall_value,

            "f1":
                f1_value,

            "tn":
                tn,

            "fp":
                fp,

            "fn":
                fn,

            "tp":
                tp,

            "fpr":
                false_positive_rate,

            "fnr":
                false_negative_rate,
        }
    )


# ============================================================
# 14. Best F1 threshold
# ============================================================

best_f1_result = max(
    results,
    key=lambda x: x["f1"]
)


# ============================================================
# 15. High-recall operating point
# ============================================================
#
# Semiconductor inspection often cares strongly
# about missed defects.
#
# Here we search for:
#
# recall >= 95%
#
# Among those thresholds we choose the one
# with highest precision.
#
# This is an example operating policy,
# NOT a universal production rule.
# ============================================================

HIGH_RECALL_TARGET = 0.95


high_recall_candidates = [
    result
    for result in results
    if result["recall"]
    >=
    HIGH_RECALL_TARGET
]


if high_recall_candidates:

    high_recall_result = max(
        high_recall_candidates,
        key=lambda x:
        x["precision"]
    )

else:

    high_recall_result = None


# ============================================================
# 16. Helper printer
# ============================================================

def print_operating_point(
    name,
    result
):

    print(
        "\n"
        +
        "-" * 60
    )

    print(name)

    print(
        "-" * 60
    )


    print(
        f"Threshold : "
        f"{result['threshold']:.4f}"
    )

    print(
        f"Precision : "
        f"{result['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{result['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{result['f1']:.4f}"
    )


    print(
        "\nConfusion Matrix"
    )

    print(
        f"TN: {result['tn']:,}"
    )

    print(
        f"FP: {result['fp']:,}"
    )

    print(
        f"FN: {result['fn']:,}"
    )

    print(
        f"TP: {result['tp']:,}"
    )


    print(
        f"\nFalse Positive Rate: "
        f"{result['fpr']:.4f}"
    )

    print(
        f"False Negative Rate: "
        f"{result['fnr']:.4f}"
    )


# ============================================================
# 17. Print operating points
# ============================================================

print_operating_point(
    "BEST F1 OPERATING POINT",
    best_f1_result
)


if high_recall_result is not None:

    print_operating_point(
        "HIGH-RECALL OPERATING POINT "
        "(Recall >= 95%)",

        high_recall_result
    )


# ============================================================
# 18. Threshold trade-off plot
# ============================================================

threshold_values = np.array(
    [
        result["threshold"]
        for result in results
    ]
)


precision_values = np.array(
    [
        result["precision"]
        for result in results
    ]
)


recall_values = np.array(
    [
        result["recall"]
        for result in results
    ]
)


f1_values = np.array(
    [
        result["f1"]
        for result in results
    ]
)


plt.figure(
    figsize=(8, 6)
)


plt.plot(
    threshold_values,
    precision_values,
    label="Precision"
)


plt.plot(
    threshold_values,
    recall_values,
    label="Recall"
)


plt.plot(
    threshold_values,
    f1_values,
    label="F1"
)


plt.axvline(
    best_f1_result[
        "threshold"
    ],
    linestyle="--",
    label="Best F1 Threshold"
)


plt.xlabel(
    "Anomaly Threshold"
)

plt.ylabel(
    "Metric"
)

plt.title(
    "Threshold Trade-off"
)

plt.legend()

plt.tight_layout()


tradeoff_path = (
    REPORT_DIR
    / "threshold_tradeoff.png"
)


plt.savefig(
    tradeoff_path,
    dpi=150
)

plt.close()


# ============================================================
# 19. Final report
# ============================================================

print(
    "\n"
    +
    "=" * 70
)

print(
    "THRESHOLD ANALYSIS COMPLETE"
)

print(
    "=" * 70
)


print(
    "\nSaved:"
)

print(
    roc_path
)

print(
    pr_path
)

print(
    tradeoff_path
)


total_time = (
    time.perf_counter()
    -
    start_time
)


print(
    f"\nTotal runtime: "
    f"{total_time:.2f}s"
)