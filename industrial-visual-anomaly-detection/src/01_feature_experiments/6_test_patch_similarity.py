from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models


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

model = models.resnet18(weights=weights)

model = model.to(device)
model.eval()


# ============================================================
# 4. Build feature extractor
# ============================================================

# We stop BEFORE:
# - global average pooling
# - fully connected classifier
#
# ResNet18 structure:
#
# conv1
# bn1
# relu
# maxpool
# layer1
# layer2
# layer3
# layer4
# avgpool
# fc
#
# We want the output of layer4.

feature_extractor = torch.nn.Sequential(
    model.conv1,
    model.bn1,
    model.relu,
    model.maxpool,
    model.layer1,
    model.layer2,
    model.layer3,
    model.layer4,
)

feature_extractor = feature_extractor.to(device)
feature_extractor.eval()


# ============================================================
# 5. Preprocessing
# ============================================================

preprocess = weights.transforms()


# ============================================================
# 6. Extract feature map
# ============================================================

def extract_feature_map(image_path: Path):

    image = Image.open(image_path).convert("RGB")

    tensor = preprocess(image)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(device)

    with torch.no_grad():
        feature_map = feature_extractor(tensor)

    return feature_map


# ============================================================
# 7. Convert feature map to patch features
# ============================================================

def feature_map_to_patches(feature_map):

    """
    Input shape:
        [1, C, H, W]

    Example:
        [1, 512, 7, 7]

    Output shape:
        [H*W, C]

    Example:
        [49, 512]

    Each row becomes one local patch feature.
    """

    feature_map = feature_map.squeeze(0)

    # [C, H, W]
    #
    # becomes
    #
    # [H, W, C]

    feature_map = feature_map.permute(
        1,
        2,
        0
    )

    # flatten spatial positions:
    #
    # [H, W, C]
    #
    # becomes
    #
    # [H*W, C]

    patches = feature_map.reshape(
        -1,
        feature_map.shape[-1]
    )

    # Normalize every patch feature independently

    patches = F.normalize(
        patches,
        dim=1
    )

    return patches


# ============================================================
# 8. Compare patch features
# ============================================================

def compare_patch_features(
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

    fmap_a = extract_feature_map(image_a)
    fmap_b = extract_feature_map(image_b)

    print("\nFeature map A shape:", tuple(fmap_a.shape))
    print("Feature map B shape:", tuple(fmap_b.shape))

    patches_a = feature_map_to_patches(fmap_a)
    patches_b = feature_map_to_patches(fmap_b)

    print("Patch feature shape A:", tuple(patches_a.shape))
    print("Patch feature shape B:", tuple(patches_b.shape))


    # ========================================================
    # Cosine similarity matrix
    # ========================================================
    #
    # patches_a shape:
    # [N, C]
    #
    # patches_b shape:
    # [M, C]
    #
    # matrix multiply:
    #
    # [N, C] @ [C, M]
    #
    # gives:
    #
    # [N, M]
    #
    # Every value:
    # cosine similarity between one patch in A
    # and one patch in B.

    similarity_matrix = (
        patches_a
        @
        patches_b.T
    )


    # ========================================================
    # For every patch in B,
    # find its MOST similar patch in A
    # ========================================================

    best_similarity_for_b, best_match_indices = (
        similarity_matrix.max(dim=0)
    )


    print(
        "\nPatch similarity statistics"
    )

    print(
        f"Highest similarity : "
        f"{best_similarity_for_b.max().item():.6f}"
    )

    print(
        f"Average similarity : "
        f"{best_similarity_for_b.mean().item():.6f}"
    )

    print(
        f"Lowest similarity  : "
        f"{best_similarity_for_b.min().item():.6f}"
    )


    # ========================================================
    # Find most unusual patches in B
    # ========================================================

    sorted_values, sorted_indices = torch.sort(
        best_similarity_for_b
    )

    print(
        "\n5 most different patches in Image B:"
    )

    for rank in range(
        min(5, len(sorted_values))
    ):

        patch_index = sorted_indices[rank].item()
        similarity = sorted_values[rank].item()

        print(
            f"Patch {patch_index:02d}"
            f" -> best similarity "
            f"{similarity:.6f}"
        )


# ============================================================
# 9. Images
# ============================================================

# Exact duplicate

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


# pHash=0 but different images

similar_a = (
    DATA_DIR
    / "Aluminum_Camera_Cover"
    / "train"
    / "good"
    / "000004.png"
)

similar_b = (
    DATA_DIR
    / "Aluminum_Camera_Cover"
    / "test"
    / "aotu"
    / "000028.png"
)


# Clearly different products

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
# 10. Run experiments
# ============================================================

compare_patch_features(
    exact_a,
    exact_b,
    "GROUP A - Exact Duplicate"
)

compare_patch_features(
    similar_a,
    similar_b,
    "GROUP B - Same Product, Different Image / Defect"
)

compare_patch_features(
    different_a,
    different_b,
    "GROUP C - Clearly Different Products"
)