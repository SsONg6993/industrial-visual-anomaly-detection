from pathlib import Path
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F

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

TEST_DIR = PRODUCT_DIR / "test"

MODEL_DIR = PROJECT_ROOT / "models"

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CORESET_PATH = (
    MODEL_DIR
    / "aluminum_camera_cover_multilayer_coreset.pt"
)

ERROR_REPORT_PATH = (
    REPORT_DIR
    / "error_analysis.csv"
)


# ============================================================
# 2. Threshold
# ============================================================

# Best-F1 exploratory threshold
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
# 6. Multi-layer feature extraction
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
# 7. Feature map -> patch features
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
# 8. Inference
# ============================================================

def infer_image_score(
    image_path: Path
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

    return image_score


# ============================================================
# 9. Collect test samples
# ============================================================

samples = []


for defect_dir in sorted(
    TEST_DIR.iterdir()
):

    if not defect_dir.is_dir():
        continue

    defect_type = defect_dir.name

    true_label = (
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
                true_label,
            )
        )


print(
    f"\nTest images: "
    f"{len(samples):,}"
)

print(
    f"Threshold: "
    f"{THRESHOLD:.4f}"
)


# ============================================================
# 10. Run inference
# ============================================================

results = []

start_time = time.perf_counter()


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

    predicted_label = (
        1
        if score >= THRESHOLD
        else 0
    )


    if (
        true_label == 0
        and
        predicted_label == 1
    ):

        error_type = (
            "false_positive"
        )

    elif (
        true_label == 1
        and
        predicted_label == 0
    ):

        error_type = (
            "false_negative"
        )

    elif (
        true_label == 1
        and
        predicted_label == 1
    ):

        error_type = (
            "true_positive"
        )

    else:

        error_type = (
            "true_negative"
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

            "predicted_label":
                predicted_label,

            "error_type":
                error_type,
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


# ============================================================
# 11. Save full CSV
# ============================================================

with open(
    ERROR_REPORT_PATH,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    fieldnames = [
        "image_path",
        "defect_type",
        "true_label",
        "score",
        "predicted_label",
        "error_type",
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    writer.writerows(
        results
    )


# ============================================================
# 12. Separate error groups
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


# ============================================================
# 13. Sort by severity
# ============================================================

# False Positive:
#
# higher score = more confidently wrong

false_positives.sort(
    key=lambda x: x["score"],
    reverse=True
)


# False Negative:
#
# lower score = more confidently missed

false_negatives.sort(
    key=lambda x: x["score"]
)


# ============================================================
# 14. Print summary
# ============================================================

print(
    "\n"
    +
    "=" * 70
)

print(
    "ERROR ANALYSIS SUMMARY"
)

print(
    "=" * 70
)


print(
    f"\nFalse Positives: "
    f"{len(false_positives):,}"
)

print(
    f"False Negatives: "
    f"{len(false_negatives):,}"
)


# ============================================================
# 15. Top False Positives
# ============================================================

print(
    "\nTop 15 False Positives:"
)


for result in false_positives[:15]:

    print(
        f"{result['score']:.4f}"
        f" | "
        f"{result['image_path']}"
    )


# ============================================================
# 16. Top False Negatives
# ============================================================

print(
    "\nTop 15 False Negatives:"
)


for result in false_negatives[:15]:

    print(
        f"{result['score']:.4f}"
        f" | "
        f"{result['defect_type']}"
        f" | "
        f"{result['image_path']}"
    )


# ============================================================
# 17. FN count by defect type
# ============================================================

fn_by_type = {}


for result in false_negatives:

    defect_type = (
        result["defect_type"]
    )

    fn_by_type[
        defect_type
    ] = (
        fn_by_type.get(
            defect_type,
            0
        )
        +
        1
    )


print(
    "\nFalse Negatives by defect type:"
)


for defect_type, count in sorted(
    fn_by_type.items(),
    key=lambda item:
    item[1],
    reverse=True
):

    print(
        f"{defect_type:20s}"
        f" : "
        f"{count}"
    )


# ============================================================
# 18. Runtime
# ============================================================

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
    "\nFull report saved to:"
)

print(
    ERROR_REPORT_PATH
)
