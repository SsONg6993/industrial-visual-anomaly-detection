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

from sklearn.metrics import roc_auc_score


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
# 3. Load PatchCore coreset
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
# 5. Explicit image preprocessing
# ============================================================
#
# Equivalent spatial transformation to the
# standard pretrained ResNet inference:
#
# resize shortest side -> 256
# center crop -> 224
#
# Then convert to tensor and normalize.
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
# 6. Ground-truth mask transformation
# ============================================================
#
# IMPORTANT:
#
# For masks we use NEAREST interpolation.
#
# We do NOT want bilinear interpolation to
# create artificial values between 0 and 1.
# ============================================================

mask_spatial_transform = Compose(
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
# 7. Multi-layer feature extraction
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


    # layer2:
    # [B, 128, 28, 28]
    #
    # ->
    #
    # [B, 128, 14, 14]

    layer2_downsampled = (
        F.adaptive_avg_pool2d(
            layer2,
            output_size=(14, 14)
        )
    )


    # combine layer2 + layer3

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


    patches = (
        feature_map.reshape(
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
# 9. Inference
# ============================================================

def infer_image(
    image_path: Path
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


        # [196, 384]
        #
        # @
        #
        # [384, 15366]
        #
        # ->
        #
        # [196, 15366]

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


        # Upsample directly into
        # MODEL SPACE: 224×224

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


    # Simplest image-level score:
    # maximum patch anomaly

    image_score = (
        patch_scores
        .max()
        .item()
    )


    return (
        image_score,
        anomaly_map
    )


# ============================================================
# 10. Ground-truth mask loader
# ============================================================

def load_ground_truth(
    image_path: Path,
    defect_type: str
):

    # GOOD image:
    #
    # no defect mask exists.
    #
    # ground truth = all-zero mask.

    if defect_type == "good":

        return np.zeros(
            (224, 224),
            dtype=np.uint8
        )


    mask_path = (
        GROUND_TRUTH_DIR
        / defect_type
        / image_path.name
    )


    if not mask_path.exists():

        raise FileNotFoundError(
            f"Ground truth not found:\n"
            f"{mask_path}"
        )


    mask = Image.open(
        mask_path
    ).convert("L")


    mask = mask_spatial_transform(
        mask
    )


    mask = np.array(
        mask
    )


    # Binary:
    #
    # black -> 0
    # white -> 1

    mask = (
        mask > 127
    ).astype(
        np.uint8
    )


    return mask


# ============================================================
# 11. Collect test images
# ============================================================

samples = []


for defect_dir in sorted(
    TEST_DIR.iterdir()
):

    if not defect_dir.is_dir():
        continue


    defect_type = defect_dir.name


    for image_path in sorted(
        defect_dir.glob("*.png")
    ):

        # Image-level label:
        #
        # good   -> 0
        # defect -> 1

        image_label = (
            0
            if defect_type == "good"
            else 1
        )


        samples.append(
            (
                image_path,
                defect_type,
                image_label,
            )
        )


print(
    f"\nTest images found: "
    f"{len(samples):,}"
)


# ============================================================
# 12. Evaluation storage
# ============================================================

image_labels = []
image_scores = []


# For pixel AUROC:
#
# Storing every pixel from every image can consume
# a lot of RAM.
#
# Instead, we calculate per-defect-image AUROC
# first and average them.
#
# This is NOT identical to global pixel AUROC,
# so we report it explicitly as:
#
# mean per-image Pixel AUROC.

pixel_aurocs = []


# ============================================================
# 13. Evaluate entire test set
# ============================================================

start_time = time.perf_counter()


for index, (
    image_path,
    defect_type,
    image_label,

) in enumerate(
    samples,
    start=1
):


    image_score, anomaly_map = (
        infer_image(
            image_path
        )
    )


    image_labels.append(
        image_label
    )

    image_scores.append(
        image_score
    )


    # --------------------------------------------------------
    # Pixel-level evaluation
    # --------------------------------------------------------

    ground_truth = (
        load_ground_truth(
            image_path,
            defect_type
        )
    )


    # A completely good image has only label=0.
    #
    # ROC-AUC cannot be calculated on an image
    # containing only one class.
    #
    # Therefore per-image pixel AUROC is calculated
    # only for defect images containing both:
    #
    # normal pixels
    # defect pixels

    if (
        ground_truth.min()
        !=
        ground_truth.max()
    ):

        pixel_auc = roc_auc_score(
            ground_truth.reshape(-1),
            anomaly_map.reshape(-1),
        )

        pixel_aurocs.append(
            pixel_auc
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
            f"{len(samples):,} "
            f"| elapsed "
            f"{elapsed:.1f}s"
        )


# ============================================================
# 14. Image-level AUROC
# ============================================================

image_labels = np.array(
    image_labels
)

image_scores = np.array(
    image_scores
)


image_auroc = roc_auc_score(
    image_labels,
    image_scores
)


# ============================================================
# 15. Pixel-level AUROC
# ============================================================

mean_pixel_auroc = np.mean(
    pixel_aurocs
)


median_pixel_auroc = np.median(
    pixel_aurocs
)


# ============================================================
# 16. Good vs defect score statistics
# ============================================================

good_scores = (
    image_scores[
        image_labels == 0
    ]
)


defect_scores = (
    image_scores[
        image_labels == 1
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
    "PATCHCORE EVALUATION RESULTS"
)

print(
    "=" * 70
)


print(
    f"\nImage-level AUROC: "
    f"{image_auroc:.4f}"
)


print(
    f"\nMean per-image Pixel AUROC: "
    f"{mean_pixel_auroc:.4f}"
)

print(
    f"Median per-image Pixel AUROC: "
    f"{median_pixel_auroc:.4f}"
)


print(
    "\nImage score statistics:"
)


print(
    f"GOOD   "
    f"| mean={good_scores.mean():.4f}"
    f" median={np.median(good_scores):.4f}"
    f" min={good_scores.min():.4f}"
    f" max={good_scores.max():.4f}"
)


print(
    f"DEFECT "
    f"| mean={defect_scores.mean():.4f}"
    f" median={np.median(defect_scores):.4f}"
    f" min={defect_scores.min():.4f}"
    f" max={defect_scores.max():.4f}"
)


print(
    f"\nPixel AUROC evaluated on "
    f"{len(pixel_aurocs):,} defect images."
)


total_time = (
    time.perf_counter()
    -
    start_time
)


print(
    f"\nTotal evaluation time: "
    f"{total_time:.2f} seconds"
)