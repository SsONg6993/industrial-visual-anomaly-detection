from pathlib import Path
import csv
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

CORESET_PATH = (
    PROJECT_ROOT
    / "models"
    / "aluminum_camera_cover_letterbox256_multilayer_coreset.pt"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    REPORT_DIR
    / "v2_letterbox256_image_scores.csv"
)


# ============================================================
# 2. Device
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

TARGET_SIZE = int(
    checkpoint["target_size"]
)

MIN_VALID_RATIO = float(
    checkpoint["min_valid_ratio"]
)


print(
    "Memory bank shape:",
    tuple(memory_bank.shape)
)

print(
    "Target size:",
    TARGET_SIZE
)

print(
    "Minimum valid ratio:",
    MIN_VALID_RATIO
)


# ============================================================
# 4. ResNet18
# ============================================================

weights = (
    models.ResNet18_Weights.DEFAULT
)

model = models.resnet18(
    weights=weights
).to(device)

model.eval()


# ============================================================
# 5. Letterbox preprocessing
# ============================================================

def letterbox_image(
    image: Image.Image,
    target_size: int
):

    original_width, original_height = (
        image.size
    )

    scale = min(
        target_size / original_width,
        target_size / original_height
    )

    new_width = max(
        1,
        int(round(
            original_width * scale
        ))
    )

    new_height = max(
        1,
        int(round(
            original_height * scale
        ))
    )

    resized = image.resize(
        (
            new_width,
            new_height
        ),
        resample=Image.Resampling.BILINEAR
    )


    # ImageNet mean as padding.
    # After normalization, padding is near zero.

    mean_rgb = (
        0.485,
        0.456,
        0.406
    )

    padding_color = tuple(
        int(round(
            value * 255
        ))
        for value in mean_rgb
    )

    canvas = Image.new(
        "RGB",
        (
            target_size,
            target_size
        ),
        color=padding_color
    )


    left = (
        target_size
        -
        new_width
    ) // 2

    top = (
        target_size
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
            target_size,
            target_size
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
# 6. Feature extraction
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
            output_size=(
                layer3.shape[-2],
                layer3.shape[-1],
            )
        )
    )


    combined = torch.cat(
        [
            layer2_downsampled,
            layer3
        ],
        dim=1
    )


    return combined


# ============================================================
# 7. Feature map -> patches
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
# 8. Valid region -> valid patch mask
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


    patch_valid_ratio = (
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
        patch_valid_ratio
        >=
        MIN_VALID_RATIO
    )


    return valid_patch_mask


# ============================================================
# 9. Image inference
# ============================================================

def infer_image_score(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")


    (
        tensor,
        valid_region

    ) = letterbox_image(
        image,
        TARGET_SIZE
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


        valid_patch_mask = (
            valid_region_to_patch_mask(
                valid_region,
                h,
                w
            )
        )


        similarity_matrix = (
            patches
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


        distance_squared = torch.clamp(
            distance_squared,
            min=0.0
        )


        patch_distances = torch.sqrt(
            distance_squared
        )


        valid_distances = (
            patch_distances[
                valid_patch_mask
            ]
        )


        image_score = (
            valid_distances
            .max()
            .item()
        )


    return image_score


# ============================================================
# 10. Collect test images
# ============================================================

samples = []


for class_dir in sorted(
    TEST_DIR.iterdir()
):

    if not class_dir.is_dir():
        continue


    defect_type = (
        class_dir.name
    )


    true_label = (
        0
        if defect_type == "good"
        else 1
    )


    for image_path in sorted(
        class_dir.glob("*.png")
    ):

        samples.append(
            (
                image_path,
                defect_type,
                true_label,
            )
        )


print(
    f"\nTest images: "
    f"{len(samples):,}"
)


# ============================================================
# 11. Run inference
# ============================================================

results = []

y_true = []
y_score = []


start_time = (
    time.perf_counter()
)


for index, (
    image_path,
    defect_type,
    true_label,

) in enumerate(
    samples,
    start=1
):

    score = infer_image_score(
        image_path
    )


    y_true.append(
        true_label
    )

    y_score.append(
        score
    )


    results.append(
        {
            "image_path":
                str(
                    image_path.relative_to(
                        PRODUCT_DIR
                    )
                ),

            "defect_type":
                defect_type,

            "true_label":
                true_label,

            "score":
                score,
        }
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


y_true = np.array(
    y_true,
    dtype=np.uint8
)

y_score = np.array(
    y_score,
    dtype=np.float64
)


# ============================================================
# 12. Save raw scores
# ============================================================

with open(
    OUTPUT_CSV,
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
        ]
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# 13. Threshold-free metrics
# ============================================================

image_auroc = roc_auc_score(
    y_true,
    y_score
)


average_precision = (
    average_precision_score(
        y_true,
        y_score
    )
)


# ============================================================
# 14. Precision-recall curve
# ============================================================

precision_values, recall_values, thresholds = (
    precision_recall_curve(
        y_true,
        y_score
    )
)


#
# precision / recall have one more item
# than thresholds.
#

precision_for_thresholds = (
    precision_values[:-1]
)

recall_for_thresholds = (
    recall_values[:-1]
)


f1_values = (
    2
    *
    precision_for_thresholds
    *
    recall_for_thresholds
    /
    np.maximum(
        precision_for_thresholds
        +
        recall_for_thresholds,
        1e-12
    )
)


best_index = int(
    np.argmax(
        f1_values
    )
)


best_threshold = float(
    thresholds[
        best_index
    ]
)


# ============================================================
# 15. Exact predictions
# ============================================================

y_pred = (
    y_score
    >=
    best_threshold
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
    else 0.0
)


recall = (
    tp
    /
    (tp + fn)
    if
    (tp + fn) > 0
    else 0.0
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
    else 0.0
)


# ============================================================
# 16. Score statistics
# ============================================================

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


# ============================================================
# 17. Results
# ============================================================

print(
    "\n"
    +
    "=" * 70
)

print(
    "V2 LETTERBOX256 IMAGE-LEVEL EVALUATION"
)

print(
    "=" * 70
)


print(
    f"\nImage AUROC: "
    f"{image_auroc:.4f}"
)

print(
    f"Average Precision: "
    f"{average_precision:.4f}"
)


print(
    f"\nBest-F1 threshold: "
    f"{best_threshold:.6f}"
)

print(
    f"Precision: "
    f"{precision:.4f}"
)

print(
    f"Recall: "
    f"{recall:.4f}"
)

print(
    f"F1: "
    f"{f1:.4f}"
)


print(
    "\nConfusion Matrix:"
)

print(
    f"TN: {tn:,}"
)

print(
    f"FP: {fp:,}"
)

print(
    f"FN: {fn:,}"
)

print(
    f"TP: {tp:,}"
)


print(
    "\nGOOD score statistics:"
)

print(
    f"mean   = "
    f"{good_scores.mean():.4f}"
)

print(
    f"median = "
    f"{np.median(good_scores):.4f}"
)

print(
    f"min    = "
    f"{good_scores.min():.4f}"
)

print(
    f"max    = "
    f"{good_scores.max():.4f}"
)


print(
    "\nDEFECT score statistics:"
)

print(
    f"mean   = "
    f"{defect_scores.mean():.4f}"
)

print(
    f"median = "
    f"{np.median(defect_scores):.4f}"
)

print(
    f"min    = "
    f"{defect_scores.min():.4f}"
)

print(
    f"max    = "
    f"{defect_scores.max():.4f}"
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


print(
    "\nRaw score report saved to:"
)

print(
    OUTPUT_CSV
)