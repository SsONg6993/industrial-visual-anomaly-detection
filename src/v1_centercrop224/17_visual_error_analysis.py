from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models


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

GROUND_TRUTH_DIR = (
    PRODUCT_DIR
    / "ground_truth"
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

ERROR_CSV_PATH = (
    REPORT_DIR
    / "error_analysis.csv"
)

FP_OUTPUT_PATH = (
    REPORT_DIR
    / "top_false_positives.png"
)

FN_OUTPUT_PATH = (
    REPORT_DIR
    / "top_false_negatives.png"
)

RECALL_OUTPUT_PATH = (
    REPORT_DIR
    / "per_defect_recall.csv"
)


# ============================================================
# 2. Configuration
# ============================================================

TOP_N = 10

THRESHOLD = 0.5408


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
# 5. Load ResNet18
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
).to(device)

model.eval()

preprocess = weights.transforms()


# ============================================================
# 6. Multi-layer feature extractor
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
# 7. Feature map -> patches
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
# 8. Inference + anomaly heatmap
# ============================================================

def infer_image(
    image_path: Path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    original_array = np.array(
        image
    )

    original_height, original_width = (
        original_array.shape[:2]
    )

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

        patch_scores = torch.sqrt(
            distance_squared
        )

        image_score = (
            patch_scores.max().item()
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
            size=(
                original_height,
                original_width
            ),
            mode="bilinear",
            align_corners=False,
        )

    anomaly_map = (
        anomaly_map
        .squeeze()
        .cpu()
        .numpy()
    )

    # Normalize for visualization only.

    visual_map = (
        anomaly_map
        -
        anomaly_map.min()
    )

    visual_map = (
        visual_map
        /
        (
            visual_map.max()
            +
            1e-8
        )
    )

    return (
        original_array,
        visual_map,
        image_score
    )


# ============================================================
# 9. Load ground-truth mask
# ============================================================

def load_ground_truth_mask(
    image_path: Path,
    defect_type: str
):

    mask_path = (
        GROUND_TRUTH_DIR
        / defect_type
        / image_path.name
    )

    if not mask_path.exists():

        return None

    mask = Image.open(
        mask_path
    ).convert("L")

    mask = np.array(
        mask
    )

    mask = (
        mask > 127
    ).astype(
        np.uint8
    )

    return mask


# ============================================================
# 10. Load error-analysis CSV
# ============================================================

results = []

with open(
    ERROR_CSV_PATH,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(
        file
    )

    for row in reader:

        row["score"] = float(
            row["score"]
        )

        row["true_label"] = int(
            row["true_label"]
        )

        row["predicted_label"] = int(
            row["predicted_label"]
        )

        results.append(
            row
        )


# ============================================================
# 11. Extract FP / FN
# ============================================================

false_positives = [
    result
    for result in results
    if result["error_type"]
    ==
    "false_positive"
]


false_negatives = [
    result
    for result in results
    if result["error_type"]
    ==
    "false_negative"
]


# Most confidently wrong FP first.

false_positives.sort(
    key=lambda x: x["score"],
    reverse=True
)


# Most confidently missed defect first.

false_negatives.sort(
    key=lambda x: x["score"]
)


false_positives = (
    false_positives[:TOP_N]
)

false_negatives = (
    false_negatives[:TOP_N]
)


# ============================================================
# 12. False Positive board
# ============================================================

print(
    "\nGenerating False Positive board..."
)


fig, axes = plt.subplots(
    TOP_N,
    2,
    figsize=(10, TOP_N * 4)
)


for index, result in enumerate(
    false_positives
):

    image_path = (
        PRODUCT_DIR
        /
        result["image_path"]
    )

    (
        image,
        heatmap,
        score,

    ) = infer_image(
        image_path
    )


    # Original image

    axes[index, 0].imshow(
        image
    )

    axes[index, 0].set_title(
        f"GOOD image\n"
        f"{image_path.name}"
    )

    axes[index, 0].axis(
        "off"
    )


    # Heatmap overlay

    axes[index, 1].imshow(
        image
    )

    axes[index, 1].imshow(
        heatmap,
        alpha=0.45
    )

    axes[index, 1].set_title(
        f"FALSE POSITIVE\n"
        f"score={score:.4f}"
    )

    axes[index, 1].axis(
        "off"
    )


plt.tight_layout()

plt.savefig(
    FP_OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 13. False Negative board
# ============================================================

print(
    "Generating False Negative board..."
)


fig, axes = plt.subplots(
    TOP_N,
    3,
    figsize=(15, TOP_N * 4)
)


for index, result in enumerate(
    false_negatives
):

    image_path = (
        PRODUCT_DIR
        /
        result["image_path"]
    )

    defect_type = (
        result["defect_type"]
    )


    (
        image,
        heatmap,
        score,

    ) = infer_image(
        image_path
    )


    mask = load_ground_truth_mask(
        image_path,
        defect_type
    )


    # --------------------------------------------------------
    # Original image
    # --------------------------------------------------------

    axes[index, 0].imshow(
        image
    )

    axes[index, 0].set_title(
        f"{defect_type}\n"
        f"{image_path.name}"
    )

    axes[index, 0].axis(
        "off"
    )


    # --------------------------------------------------------
    # Ground-truth mask
    # --------------------------------------------------------

    if mask is not None:

        axes[index, 1].imshow(
            mask
        )

    axes[index, 1].set_title(
        "Ground Truth"
    )

    axes[index, 1].axis(
        "off"
    )


    # --------------------------------------------------------
    # Predicted heatmap
    # --------------------------------------------------------

    axes[index, 2].imshow(
        image
    )

    axes[index, 2].imshow(
        heatmap,
        alpha=0.45
    )

    axes[index, 2].set_title(
        f"FALSE NEGATIVE\n"
        f"score={score:.4f}"
    )

    axes[index, 2].axis(
        "off"
    )


plt.tight_layout()

plt.savefig(
    FN_OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 14. Per-defect recall calculation
# ============================================================

defect_stats = {}


for result in results:

    defect_type = (
        result["defect_type"]
    )

    # Ignore GOOD.

    if defect_type == "good":
        continue


    if defect_type not in defect_stats:

        defect_stats[
            defect_type
        ] = {
            "total": 0,
            "tp": 0,
            "fn": 0,
        }


    defect_stats[
        defect_type
    ]["total"] += 1


    if (
        result["error_type"]
        ==
        "true_positive"
    ):

        defect_stats[
            defect_type
        ]["tp"] += 1


    elif (
        result["error_type"]
        ==
        "false_negative"
    ):

        defect_stats[
            defect_type
        ]["fn"] += 1


# ============================================================
# 15. Save recall CSV
# ============================================================

with open(
    RECALL_OUTPUT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.writer(
        file
    )

    writer.writerow(
        [
            "defect_type",
            "total",
            "true_positive",
            "false_negative",
            "recall",
            "miss_rate",
        ]
    )


    rows = []


    for defect_type, stats in (
        defect_stats.items()
    ):

        total = stats["total"]
        tp = stats["tp"]
        fn = stats["fn"]

        recall = (
            tp / total
            if total > 0
            else 0.0
        )

        miss_rate = (
            fn / total
            if total > 0
            else 0.0
        )


        rows.append(
            (
                defect_type,
                total,
                tp,
                fn,
                recall,
                miss_rate,
            )
        )


    # Lowest recall first.

    rows.sort(
        key=lambda x: x[4]
    )


    for row in rows:

        writer.writerow(
            [
                row[0],
                row[1],
                row[2],
                row[3],
                f"{row[4]:.4f}",
                f"{row[5]:.4f}",
            ]
        )


# ============================================================
# 16. Print recall table
# ============================================================

print(
    "\nPer-defect recall:"
)

for row in rows:

    print(
        f"{row[0]:15s}"
        f" | total={row[1]:3d}"
        f" | FN={row[3]:2d}"
        f" | recall={row[4]:.4f}"
    )


print(
    "\nSaved:"
)

print(
    FP_OUTPUT_PATH
)

print(
    FN_OUTPUT_PATH
)

print(
    RECALL_OUTPUT_PATH
)

print(
    "\nDone."
)