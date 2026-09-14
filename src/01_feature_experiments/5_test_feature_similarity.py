from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "3CAD"


# ============================================================
# 2. Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ============================================================
# 3. Load pretrained ResNet18
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)


# Remove the final classification layer
#
# Original ResNet18:
#
# CNN backbone
#      ↓
# average pooling
#      ↓
# fully connected layer
#      ↓
# 1000 ImageNet classes
#
# We DON'T need the classifier.
# We only want the learned visual feature vector.

feature_extractor = torch.nn.Sequential(
    *list(model.children())[:-1]
)

feature_extractor = feature_extractor.to(device)

feature_extractor.eval()


# ============================================================
# 4. Image preprocessing
# ============================================================

preprocess = weights.transforms()


# ============================================================
# 5. Extract image feature
# ============================================================

def extract_feature(image_path: Path):

    image = Image.open(image_path).convert("RGB")

    tensor = preprocess(image)

    # Add batch dimension:
    #
    # [C, H, W]
    #
    # becomes
    #
    # [1, C, H, W]

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)


    # We only run inference.
    # No gradient needed.
    with torch.no_grad():

        feature = feature_extractor(
            tensor
        )


    # ResNet output:
    #
    # [1, 512, 1, 1]
    #
    # flatten to:
    #
    # [512]

    feature = feature.flatten()


    # Normalize feature vector
    feature = F.normalize(
        feature,
        dim=0
    )

    return feature


# ============================================================
# 6. Cosine similarity
# ============================================================

def cosine_similarity(
    feature_a,
    feature_b
):

    similarity = torch.dot(
        feature_a,
        feature_b
    )

    return similarity.item()


# ============================================================
# 7. Helper function
# ============================================================

def compare_images(
    image_a,
    image_b,
    title
):

    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print("Image A:")
    print(image_a)

    print("\nImage B:")
    print(image_b)


    feature_a = extract_feature(
        image_a
    )

    feature_b = extract_feature(
        image_b
    )


    similarity = cosine_similarity(
        feature_a,
        feature_b
    )


    print(
        f"\nCosine similarity: "
        f"{similarity:.6f}"
    )


# ============================================================
# 8. Example image pairs
# ============================================================


# ------------------------------------------------------------
# GROUP A
# Exact duplicate from your SHA audit
# ------------------------------------------------------------

exact_a = (
    DATA_DIR
    / "Aluminum_Camera_Cover"
    / "test"
    / "good"
    / "000049.png"
)

exact_b = (
    DATA_DIR
    / "Aluminum_Camera_Cover"
    / "train"
    / "good"
    / "000395.png"
)


# ------------------------------------------------------------
# GROUP B
# pHash distance = 0,
# but you manually confirmed these are different images
# ------------------------------------------------------------

phash_a = (
    DATA_DIR
    / "Aluminum_Camera_Cover"
    / "train"
    / "good"
    / "000004.png"
)

phash_b = (
    DATA_DIR
    / "Aluminum_Camera_Cover"
    / "test"
    / "aotu"
    / "000028.png"
)


# ------------------------------------------------------------
# GROUP C
# Clearly different products
# ------------------------------------------------------------

different_a = (
    DATA_DIR
    / "Aluminum_Camera_Cover"
    / "train"
    / "good"
    / "000004.png"
)

different_b = (
    DATA_DIR
    / "Iron_Stator"
    / "test"
    / "neiqiaopian"
    / "000001.png"
)


# ============================================================
# 9. Run experiment
# ============================================================

compare_images(
    exact_a,
    exact_b,
    "GROUP A - Exact Duplicate"
)


compare_images(
    phash_a,
    phash_b,
    "GROUP B - pHash=0 but Different Images"
)


compare_images(
    different_a,
    different_b,
    "GROUP C - Clearly Different Images"
)