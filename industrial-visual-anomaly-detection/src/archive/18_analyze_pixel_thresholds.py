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

from sklearn.metrics import (
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


# ============================================================
# 5. Image transform
# ============================================================

image_transform = Compose(
    [
        Resize(
            256,
            interpolation=
            InterpolationMode.BILINEAR
        ),

        CenterCrop(
            224
        ),

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
# 6. Mask transform
# ============================================================

mask_transform = Compose(
    [
        Resize(
            256,
            interpolation=
            InterpolationMode.NEAREST
        ),

        CenterCrop(
            224
        ),
    ]
)


# ============================================================
# 7. Feature extraction
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
# 8. Feature map -> patch vectors
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
# 9. Produce raw anomaly map
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


    anomaly_map = (
        anomaly_map
        .squeeze()
        .cpu()
        .numpy()
        .astype(np.float32)
    )

    return anomaly_map


# ============================================================
# 10. Ground-truth loader
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

    mask = (
        mask > 127
    ).astype(
        np.uint8
    )

    return mask


# ============================================================
# 11. Collect defect images only
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
# 12. Collect all pixels
# ============================================================

all_scores = []
all_labels = []


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

    all_scores.append(
        anomaly_map.reshape(-1)
    )

    all_labels.append(
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
    all_scores
)

all_labels = np.concatenate(
    all_labels
)


print(
    "\nTotal evaluated pixels:",
    f"{len(all_scores):,}"
)

print(
    "Positive defect pixels:",
    f"{all_labels.sum():,}"
)


# ============================================================
# 13. Threshold candidates
# ============================================================

thresholds = np.linspace(
    float(all_scores.min()),
    float(all_scores.max()),
    300
)


best_result = None


# ============================================================
# 14. Sweep thresholds
# ============================================================

for threshold in thresholds:

    predictions = (
        all_scores
        >=
        threshold
    ).astype(
        np.uint8
    )


    precision = precision_score(
        all_labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        predictions,
        zero_division=0
    )


    tn, fp, fn, tp = (
        confusion_matrix(
            all_labels,
            predictions,
            labels=[0, 1]
        ).ravel()
    )


    # IoU / Jaccard

    iou = (
        tp
        /
        (tp + fp + fn)
        if
        (tp + fp + fn) > 0
        else 0.0
    )


    # Dice

    dice = (
        2 * tp
        /
        (2 * tp + fp + fn)
        if
        (2 * tp + fp + fn) > 0
        else 0.0
    )


    result = {
        "threshold":
            threshold,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "iou":
            iou,

        "dice":
            dice,

        "tp":
            tp,

        "fp":
            fp,

        "fn":
            fn,

        "tn":
            tn,
    }


    if (
        best_result is None
        or
        result["f1"]
        >
        best_result["f1"]
    ):

        best_result = (
            result
        )


# ============================================================
# 15. Results
# ============================================================

print(
    "\n"
    +
    "=" * 70
)

print(
    "PIXEL THRESHOLD ANALYSIS"
)

print(
    "=" * 70
)


print(
    f"\nBest Pixel Threshold: "
    f"{best_result['threshold']:.4f}"
)


print(
    f"Pixel Precision: "
    f"{best_result['precision']:.4f}"
)

print(
    f"Pixel Recall: "
    f"{best_result['recall']:.4f}"
)

print(
    f"Pixel F1: "
    f"{best_result['f1']:.4f}"
)

print(
    f"Pixel IoU: "
    f"{best_result['iou']:.4f}"
)

print(
    f"Pixel Dice: "
    f"{best_result['dice']:.4f}"
)


print(
    "\nPixel Confusion Matrix:"
)

print(
    f"TP: "
    f"{best_result['tp']:,}"
)

print(
    f"FP: "
    f"{best_result['fp']:,}"
)

print(
    f"FN: "
    f"{best_result['fn']:,}"
)

print(
    f"TN: "
    f"{best_result['tn']:,}"
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