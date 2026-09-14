from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt

from torchvision import models


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "3CAD"

REPORT_DIR = PROJECT_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


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

model = model.to(device)
model.eval()


# ============================================================
# 4. Feature extractor
# ============================================================

# Stop at layer4.
# We do NOT use avgpool or final classifier.

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

    tensor = tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        feature_map = feature_extractor(tensor)

    return feature_map


# ============================================================
# 7. Convert feature map to patch features
# ============================================================

def feature_map_to_patches(feature_map):

    # feature_map:
    # [1, C, H, W]

    feature_map = feature_map.squeeze(0)

    # [C, H, W]
    # ->
    # [H, W, C]

    feature_map = feature_map.permute(
        1,
        2,
        0
    )

    h, w, c = feature_map.shape

    # [H, W, C]
    # ->
    # [H*W, C]

    patches = feature_map.reshape(
        h * w,
        c
    )

    patches = F.normalize(
        patches,
        dim=1
    )

    return patches, h, w


# ============================================================
# 8. Compute anomaly map
# ============================================================

def compute_anomaly_map(
    reference_path: Path,
    test_path: Path
):

    reference_map = extract_feature_map(
        reference_path
    )

    test_map = extract_feature_map(
        test_path
    )


    reference_patches, _, _ = (
        feature_map_to_patches(
            reference_map
        )
    )

    test_patches, h, w = (
        feature_map_to_patches(
            test_map
        )
    )


    # --------------------------------------------------------
    # Similarity matrix
    #
    # test_patches:
    # [49, 512]
    #
    # reference_patches:
    # [49, 512]
    #
    # result:
    # [49, 49]
    #
    # Each test patch is compared with ALL normal patches.
    # --------------------------------------------------------

    similarity_matrix = (
        test_patches
        @
        reference_patches.T
    )


    # For each test patch:
    # find the most similar normal patch.

    best_similarity, _ = (
        similarity_matrix.max(dim=1)
    )


    # --------------------------------------------------------
    # Convert similarity to anomaly score
    #
    # similar:
    # similarity ~ 1
    # anomaly ~ 0
    #
    # different:
    # similarity lower
    # anomaly higher
    # --------------------------------------------------------

    anomaly_scores = (
        1.0 - best_similarity
    )


    # [49]
    # ->
    # [7, 7]

    anomaly_map = anomaly_scores.reshape(
        h,
        w
    )


    return anomaly_map


# ============================================================
# 9. Normalize map for visualization
# ============================================================

def normalize_map(anomaly_map):

    anomaly_map = anomaly_map.detach().cpu()

    minimum = anomaly_map.min()
    maximum = anomaly_map.max()

    normalized = (
        anomaly_map - minimum
    ) / (
        maximum - minimum + 1e-8
    )

    return normalized


# ============================================================
# 10. Upsample anomaly map
# ============================================================

def upsample_anomaly_map(
    anomaly_map,
    output_height,
    output_width
):

    # [H, W]
    # ->
    # [1, 1, H, W]

    anomaly_map = (
        anomaly_map
        .unsqueeze(0)
        .unsqueeze(0)
    )


    upsampled = F.interpolate(
        anomaly_map,
        size=(
            output_height,
            output_width
        ),
        mode="bilinear",
        align_corners=False,
    )


    # [1, 1, H, W]
    # ->
    # [H, W]

    upsampled = (
        upsampled
        .squeeze()
        .numpy()
    )

    return upsampled


# ============================================================
# 11. Images
# ============================================================

reference_image = (
    DATA_DIR
    / "Aluminum_Camera_Cover"
    / "train"
    / "good"
    / "000004.png"
)

test_image = (
    DATA_DIR
    / "Aluminum_Camera_Cover"
    / "test"
    / "aotu"
    / "000028.png"
)


# ============================================================
# 12. Compute anomaly map
# ============================================================

anomaly_map = compute_anomaly_map(
    reference_image,
    test_image
)


print(
    "Raw anomaly map shape:",
    tuple(anomaly_map.shape)
)


print(
    "Raw anomaly min:",
    anomaly_map.min().item()
)

print(
    "Raw anomaly max:",
    anomaly_map.max().item()
)


# ============================================================
# 13. Normalize
# ============================================================

normalized_map = normalize_map(
    anomaly_map
)


print(
    "Normalized anomaly map shape:",
    tuple(normalized_map.shape)
)


# ============================================================
# 14. Load original test image
# ============================================================

original_image = Image.open(
    test_image
).convert("RGB")


original_array = np.array(
    original_image
)


height, width = (
    original_array.shape[:2]
)


# ============================================================
# 15. Upsample to original image size
# ============================================================

upsampled_map = upsample_anomaly_map(
    normalized_map,
    height,
    width
)


# ============================================================
# 16. Plot original image
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    original_array
)

plt.title(
    "Original Test Image"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 17. Plot anomaly heatmap
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    upsampled_map
)

plt.title(
    "Patch-Level Anomaly Heatmap"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 18. Overlay heatmap
# ============================================================

plt.figure(figsize=(8, 6))

plt.imshow(
    original_array
)

plt.imshow(
    upsampled_map,
    alpha=0.45
)

plt.title(
    "Anomaly Heatmap Overlay"
)

plt.axis("off")

plt.tight_layout()

plt.show()


# ============================================================
# 19. Save output
# ============================================================

output_path = (
    REPORT_DIR
    /
    "patch_anomaly_overlay.png"
)


plt.figure(figsize=(8, 6))

plt.imshow(
    original_array
)

plt.imshow(
    upsampled_map,
    alpha=0.45
)

plt.title(
    "Patch-Level Anomaly Heatmap"
)

plt.axis("off")

plt.tight_layout()

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


print(
    "\nSaved visualization to:"
)

print(
    output_path
)