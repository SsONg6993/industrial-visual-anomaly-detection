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
    / "aluminum_camera_cover_letterbox256_full32_greedy10_coreset.pt"
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
    / "v2c_full32_image_scores.csv"
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

memory_bank = F.normalize(
    memory_bank,
    p=2,
    dim=1
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
    "Feature resolution:",
    checkpoint["feature_resolution"]
)

print(
    "Spatial sampling:",
    checkpoint["spatial_sampling"]
)

print(
    "Selection method:",
    checkpoint["selection_method"]
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
    image
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
# 6. Full 32x32 high-resolution fusion
# ============================================================

def extract_highres_features(
    tensor
):

    x = model.conv1(tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)

    # 32x32 x 128
    layer2 = model.layer2(x)

    # 16x16 x 256
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
            align_corners=False,
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
# 7. Feature map -> all patches
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
# 8. Valid patch mask
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


        distance_squared = torch.clamp(
            distance_squared,
            min=0.0
        )


        patch_distances = torch.sqrt(
            distance_squared
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
        w,
    )


# ============================================================
# 10. Collect test samples
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

    (
        score,
        valid_patches,
        h,
        w,

    ) = infer_image_score(
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

            "valid_patches":
                valid_patches,
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
            f" | feature="
            f"{h}x{w}"
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
            "valid_patches",
        ]
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# 13. Threshold-free metrics
# ============================================================

image_auroc = (
    roc_auc_score(
        y_true,
        y_score
    )
)


average_precision = (
    average_precision_score(
        y_true,
        y_score
    )
)


# ============================================================
# 14. Exploratory best-F1 threshold
# ============================================================

precision_values, recall_values, thresholds = (
    precision_recall_curve(
        y_true,
        y_score
    )
)


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
# 15. Confusion matrix
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
# 16. Score stats
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
# 17. Print results
# ============================================================

total_time = (
    time.perf_counter()
    -
    start_time
)


print(
    "\n"
    +
    "=" * 70
)

print(
    "V2C FULL32 IMAGE-LEVEL EVALUATION"
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


print(
    f"\nTotal runtime: "
    f"{total_time:.2f}s"
)


print(
    "\nSaved scores:"
)

print(
    OUTPUT_CSV
)