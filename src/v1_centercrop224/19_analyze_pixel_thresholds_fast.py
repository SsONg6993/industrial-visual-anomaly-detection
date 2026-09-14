from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from torchvision import models
from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    ToTensor,
    Normalize,
    InterpolationMode,
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

TEST_DIR = PRODUCT_DIR / "test"

GROUND_TRUTH_DIR = (
    PRODUCT_DIR
    / "ground_truth"
)

CORESET_PATH = (
    PROJECT_ROOT
    / "models"
    / "aluminum_camera_cover_multilayer_coreset.pt"
)

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PIXEL_SCORES_PATH = (
    REPORT_DIR
    / "pixel_scores.npy"
)

PIXEL_LABELS_PATH = (
    REPORT_DIR
    / "pixel_labels.npy"
)


# ============================================================
# 2. Configuration
# ============================================================

HISTOGRAM_BINS = 4096


# ============================================================
# 3. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# 4. Load coreset
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
# 5. ResNet18
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
).to(device)

model.eval()


# ============================================================
# 6. Image transform
# ============================================================

image_transform = Compose(
    [
        Resize(
            256,
            interpolation=InterpolationMode.BILINEAR
        ),

        CenterCrop(224),

        ToTensor(),

        Normalize(
            mean=[
                0.485,
                0.456,
                0.406,
            ],
            std=[
                0.229,
                0.224,
                0.225,
            ],
        ),
    ]
)


# ============================================================
# 7. Mask transform
# ============================================================

mask_transform = Compose(
    [
        Resize(
            256,
            interpolation=InterpolationMode.NEAREST
        ),

        CenterCrop(224),
    ]
)


# ============================================================
# 8. Feature extraction
# ============================================================

def extract_multilayer_features(
    tensor
):

    x = model.conv1(tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)

    layer2 = model.layer2(x)

    layer3 = model.layer3(
        layer2
    )

    layer2_downsampled = (
        F.adaptive_avg_pool2d(
            layer2,
            output_size=(14, 14)
        )
    )

    combined = torch.cat(
        [
            layer2_downsampled,
            layer3,
        ],
        dim=1
    )

    return combined


# ============================================================
# 9. Feature map -> patch vectors
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

    return (
        patches,
        h,
        w
    )


# ============================================================
# 10. Raw anomaly map
# ============================================================

def infer_anomaly_map(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    tensor = image_transform(
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

        patches, h, w = (
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
            similarity_matrix
            .max(dim=1)
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

        patch_scores = torch.sqrt(
            distance_squared
        )

        anomaly_map = (
            patch_scores.reshape(
                1,
                1,
                h,
                w
            )
        )

        anomaly_map = F.interpolate(
            anomaly_map,
            size=(224, 224),
            mode="bilinear",
            align_corners=False,
        )


    return (
        anomaly_map
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.float32)
    )


# ============================================================
# 11. Ground truth
# ============================================================

def load_ground_truth(
    image_path,
    defect_type
):

    mask_path = (
        GROUND_TRUTH_DIR
        / defect_type
        / image_path.name
    )

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = mask_transform(
        mask
    )

    mask = np.array(
        mask
    )

    return (
        mask > 127
    ).astype(
        np.uint8
    )


# ============================================================
# 12. Collect defect samples
# ============================================================

samples = []

for defect_dir in sorted(
    TEST_DIR.iterdir()
):

    if not defect_dir.is_dir():
        continue

    if defect_dir.name == "good":
        continue

    for image_path in sorted(
        defect_dir.glob("*.png")
    ):

        samples.append(
            (
                image_path,
                defect_dir.name,
            )
        )


print(
    f"\nDefect images: "
    f"{len(samples):,}"
)


# ============================================================
# 13. Check cached raw pixel data
# ============================================================

if (
    PIXEL_SCORES_PATH.exists()
    and
    PIXEL_LABELS_PATH.exists()
):

    print(
        "\nCached pixel data found."
    )

    print(
        "Skipping CNN inference."
    )

    all_scores = np.load(
        PIXEL_SCORES_PATH,
        mmap_mode="r"
    )

    all_labels = np.load(
        PIXEL_LABELS_PATH,
        mmap_mode="r"
    )


else:

    # ========================================================
    # 14. Run inference
    # ========================================================

    print(
        "\nNo cache found."
    )

    print(
        "Running CNN inference..."
    )


    score_list = []
    label_list = []


    start_time = time.perf_counter()


    for index, (
        image_path,
        defect_type,

    ) in enumerate(
        samples,
        start=1
    ):

        anomaly_map = (
            infer_anomaly_map(
                image_path
            )
        )

        ground_truth = (
            load_ground_truth(
                image_path,
                defect_type
            )
        )


        score_list.append(
            anomaly_map.reshape(-1)
        )

        label_list.append(
            ground_truth.reshape(-1)
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


    all_scores = np.concatenate(
        score_list
    ).astype(
        np.float32
    )

    all_labels = np.concatenate(
        label_list
    ).astype(
        np.uint8
    )


    print(
        "\nSaving raw pixel data..."
    )


    np.save(
        PIXEL_SCORES_PATH,
        all_scores
    )

    np.save(
        PIXEL_LABELS_PATH,
        all_labels
    )


    print(
        "Saved:"
    )

    print(
        PIXEL_SCORES_PATH
    )

    print(
        PIXEL_LABELS_PATH
    )


# ============================================================
# 15. Basic statistics
# ============================================================

total_pixels = len(
    all_scores
)

positive_pixels = int(
    all_labels.sum()
)

negative_pixels = (
    total_pixels
    -
    positive_pixels
)


print(
    "\nTotal pixels:",
    f"{total_pixels:,}"
)

print(
    "Defect pixels:",
    f"{positive_pixels:,}"
)

print(
    "Normal pixels:",
    f"{negative_pixels:,}"
)

print(
    "Defect pixel ratio:",
    f"{positive_pixels / total_pixels:.4%}"
)


score_min = float(
    np.min(all_scores)
)

score_max = float(
    np.max(all_scores)
)


print(
    "\nPixel score range:"
)

print(
    f"min = {score_min:.6f}"
)

print(
    f"max = {score_max:.6f}"
)


# ============================================================
# 16. Build separate histograms
# ============================================================

print(
    "\nBuilding histograms..."
)


positive_scores = (
    all_scores[
        all_labels == 1
    ]
)

negative_scores = (
    all_scores[
        all_labels == 0
    ]
)


hist_range = (
    score_min,
    score_max
)


positive_hist, bin_edges = (
    np.histogram(
        positive_scores,
        bins=HISTOGRAM_BINS,
        range=hist_range,
    )
)


negative_hist, _ = (
    np.histogram(
        negative_scores,
        bins=HISTOGRAM_BINS,
        range=hist_range,
    )
)


# ============================================================
# 17. Threshold sweep using cumulative histograms
# ============================================================

#
# Prediction rule:
#
# score >= threshold
# -> defect
#
# Starting from high score bins:
#
# TP = cumulative positive pixels
# FP = cumulative negative pixels
#

tp = np.cumsum(
    positive_hist[::-1]
)[::-1]

fp = np.cumsum(
    negative_hist[::-1]
)[::-1]


fn = (
    positive_pixels
    -
    tp
)

tn = (
    negative_pixels
    -
    fp
)


precision = (
    tp
    /
    np.maximum(
        tp + fp,
        1
    )
)

recall = (
    tp
    /
    np.maximum(
        tp + fn,
        1
    )
)


f1 = (
    2
    *
    precision
    *
    recall
    /
    np.maximum(
        precision + recall,
        1e-12
    )
)


best_index = int(
    np.argmax(f1)
)


best_threshold = float(
    bin_edges[
        best_index
    ]
)


print(
    "\nApproximate best threshold:"
)

print(
    f"{best_threshold:.6f}"
)

print(
    "Histogram F1:"
)

print(
    f"{f1[best_index]:.6f}"
)


# ============================================================
# 18. Exact metrics at best threshold
# ============================================================

print(
    "\nCalculating exact metrics..."
)


predictions = (
    all_scores
    >=
    best_threshold
)


labels = (
    all_labels
    ==
    1
)


exact_tp = int(
    np.sum(
        predictions
        &
        labels
    )
)

exact_fp = int(
    np.sum(
        predictions
        &
        ~labels
    )
)

exact_fn = int(
    np.sum(
        ~predictions
        &
        labels
    )
)

exact_tn = int(
    np.sum(
        ~predictions
        &
        ~labels
    )
)


precision_exact = (
    exact_tp
    /
    (
        exact_tp
        +
        exact_fp
    )
    if
    (
        exact_tp
        +
        exact_fp
    ) > 0
    else 0.0
)


recall_exact = (
    exact_tp
    /
    (
        exact_tp
        +
        exact_fn
    )
    if
    (
        exact_tp
        +
        exact_fn
    ) > 0
    else 0.0
)


f1_exact = (
    2
    *
    precision_exact
    *
    recall_exact
    /
    (
        precision_exact
        +
        recall_exact
    )
    if
    (
        precision_exact
        +
        recall_exact
    ) > 0
    else 0.0
)


iou = (
    exact_tp
    /
    (
        exact_tp
        +
        exact_fp
        +
        exact_fn
    )
)


dice = (
    2
    *
    exact_tp
    /
    (
        2
        *
        exact_tp
        +
        exact_fp
        +
        exact_fn
    )
)


# ============================================================
# 19. Results
# ============================================================

print(
    "\n"
    +
    "=" * 70
)

print(
    "FAST PIXEL THRESHOLD ANALYSIS"
)

print(
    "=" * 70
)


print(
    f"\nBest Pixel Threshold: "
    f"{best_threshold:.6f}"
)

print(
    f"Pixel Precision: "
    f"{precision_exact:.4f}"
)

print(
    f"Pixel Recall: "
    f"{recall_exact:.4f}"
)

print(
    f"Pixel F1: "
    f"{f1_exact:.4f}"
)

print(
    f"Pixel IoU: " #intersection over Union (IoU)
    f"{iou:.4f}"
)

print(
    f"Pixel Dice: "
    f"{dice:.4f}"
)


print(
    "\nPixel Confusion Matrix:"
)

print(
    f"TP: {exact_tp:,}"
)

print(
    f"FP: {exact_fp:,}"
)

print(
    f"FN: {exact_fn:,}"
)

print(
    f"TN: {exact_tn:,}"
)


print(
    "\nCache files:"
)

print(
    PIXEL_SCORES_PATH
)

print(
    PIXEL_LABELS_PATH
)