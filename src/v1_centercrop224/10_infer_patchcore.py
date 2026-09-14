from pathlib import Path

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

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "3CAD"
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
    / "aluminum_camera_cover_coreset.pt"
)


TEST_IMAGE_PATH = (
    DATA_DIR
    / "Aluminum_Camera_Cover"
    / "test"
    / "aotu"
    / "000028.png"
)


OUTPUT_PATH = (
    REPORT_DIR
    / "patchcore_inference_overlay.png"
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


print(
    "\nCoreset loaded."
)

print(
    "Memory bank shape:",
    tuple(memory_bank.shape)
)


# ============================================================
# 4. Load pretrained ResNet18
# ============================================================

weights = (
    models.ResNet18_Weights.DEFAULT
)

model = models.resnet18(
    weights=weights
)

model = model.to(device)

model.eval()


# ============================================================
# 5. Feature extractor
# ============================================================

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

feature_extractor = (
    feature_extractor.to(device)
)

feature_extractor.eval()


# ============================================================
# 6. Preprocessing
# ============================================================

preprocess = weights.transforms()


# ============================================================
# 7. Extract test patch features
# ============================================================

def extract_test_patches(
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
            feature_extractor(
                tensor
            )
        )


    # [1, 512, 7, 7]

    _, channels, height, width = (
        feature_map.shape
    )


    # [1, 512, 7, 7]
    #
    # ->
    #
    # [1, 7, 7, 512]

    feature_map = (
        feature_map.permute(
            0,
            2,
            3,
            1
        )
    )


    # ->
    #
    # [49, 512]

    patches = (
        feature_map.reshape(
            height
            *
            width,

            channels
        )
    )


    # L2 normalize every patch feature.

    patches = F.normalize(
        patches,
        dim=1
    )


    return (
        patches,
        height,
        width
    )


# ============================================================
# 8. Nearest-neighbour anomaly scoring
# ============================================================

def compute_patch_anomaly_scores(
    test_patches,
    memory_bank
):

    # Both test patches and memory bank
    # are already L2 normalized.
    #
    # Therefore:
    #
    # dot product
    # =
    # cosine similarity


    similarity_matrix = (
        test_patches
        @
        memory_bank.T
    )


    # Shape:
    #
    # [49, 3842]
    #
    # Every test patch is compared
    # with every normal coreset patch.


    # For every test patch:
    #
    # find the most similar
    # normal reference feature.

    nearest_similarity, nearest_indices = (
        similarity_matrix.max(
            dim=1
        )
    )


    # --------------------------------------------------------
    # Convert cosine similarity
    # to Euclidean distance.
    #
    # For L2-normalized vectors:
    #
    # d² = 2 - 2*cosine
    #
    # therefore:
    #
    # d = sqrt(2 - 2*cosine)
    # --------------------------------------------------------

    distances_squared = (
        2.0
        -
        2.0
        *
        nearest_similarity
    )


    # Numerical protection:
    #
    # floating point error could cause
    # something tiny like -0.0000001.

    distances_squared = torch.clamp(
        distances_squared,
        min=0.0
    )


    anomaly_scores = torch.sqrt(
        distances_squared
    )


    return (
        anomaly_scores,
        nearest_similarity,
        nearest_indices
    )


# ============================================================
# 9. Extract test features
# ============================================================

test_patches, h, w = (
    extract_test_patches(
        TEST_IMAGE_PATH
    )
)


print(
    "\nTest patch shape:",
    tuple(test_patches.shape)
)


# ============================================================
# 10. Compute anomaly scores
# ============================================================

(
    anomaly_scores,
    nearest_similarity,
    nearest_indices,

) = compute_patch_anomaly_scores(
    test_patches,
    memory_bank
)


print(
    "\nPatch anomaly statistics:"
)

print(
    f"Minimum distance : "
    f"{anomaly_scores.min().item():.6f}"
)

print(
    f"Average distance : "
    f"{anomaly_scores.mean().item():.6f}"
)

print(
    f"Maximum distance : "
    f"{anomaly_scores.max().item():.6f}"
)


# ============================================================
# 11. Image-level anomaly score
# ============================================================

# Simplest possible image-level score:
#
# take the most anomalous patch.

image_anomaly_score = (
    anomaly_scores.max().item()
)


print(
    f"\nImage anomaly score: "
    f"{image_anomaly_score:.6f}"
)


# ============================================================
# 12. Show most anomalous patches
# ============================================================

sorted_scores, sorted_indices = (
    torch.sort(
        anomaly_scores,
        descending=True
    )
)


print(
    "\nTop 5 anomalous patches:"
)


for rank in range(
    min(
        5,
        len(sorted_scores)
    )
):

    patch_index = (
        sorted_indices[
            rank
        ].item()
    )

    score = (
        sorted_scores[
            rank
        ].item()
    )


    row = (
        patch_index
        //
        w
    )

    col = (
        patch_index
        %
        w
    )


    print(
        f"Patch {patch_index:02d}"
        f" | row={row}"
        f" col={col}"
        f" | distance={score:.6f}"
    )


# ============================================================
# 13. Build 7x7 anomaly map
# ============================================================

anomaly_map = (
    anomaly_scores
    .reshape(
        h,
        w
    )
)


print(
    "\nAnomaly map shape:",
    tuple(anomaly_map.shape)
)


# ============================================================
# 14. Load original image
# ============================================================

original_image = Image.open(
    TEST_IMAGE_PATH
).convert("RGB")


original_array = np.array(
    original_image
)


original_height, original_width = (
    original_array.shape[:2]
)


# ============================================================
# 15. Upsample anomaly map
# ============================================================

anomaly_map_for_resize = (
    anomaly_map
    .unsqueeze(0)
    .unsqueeze(0)
)


upsampled_map = F.interpolate(
    anomaly_map_for_resize,
    size=(
        original_height,
        original_width
    ),
    mode="bilinear",
    align_corners=False,
)


upsampled_map = (
    upsampled_map
    .squeeze()
    .detach()
    .cpu()
    .numpy()
)


# ============================================================
# 16. Normalize ONLY for visualization
# ============================================================
#
# IMPORTANT:
#
# This normalization is just for
# displaying the heatmap nicely.
#
# It is NOT the anomaly score
# used for evaluation.

visual_map = (
    upsampled_map
    -
    upsampled_map.min()
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


# ============================================================
# 17. Save overlay
# ============================================================

plt.figure(
    figsize=(8, 6)
)


plt.imshow(
    original_array
)


plt.imshow(
    visual_map,
    alpha=0.45
)


plt.title(
    f"PatchCore-style Anomaly Map\n"
    f"Image Score = "
    f"{image_anomaly_score:.4f}"
)


plt.axis(
    "off"
)


plt.tight_layout()


plt.savefig(
    OUTPUT_PATH,
    dpi=150,
    bbox_inches="tight"
)


plt.close()


print(
    "\nSaved inference visualization to:"
)

print(
    OUTPUT_PATH
)


print(
    "\nDone."
)