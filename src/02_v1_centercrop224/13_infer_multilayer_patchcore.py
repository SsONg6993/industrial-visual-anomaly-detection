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
    / "aluminum_camera_cover_multilayer_coreset.pt"
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
    / "multilayer_patchcore_overlay.png"
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
    "\nMulti-layer coreset loaded."
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


preprocess = weights.transforms()


# ============================================================
# 5. Multi-layer feature extraction
# ============================================================

def extract_multilayer_features(
    image_tensor
):

    # Stem

    x = model.conv1(
        image_tensor
    )

    x = model.bn1(x)

    x = model.relu(x)

    x = model.maxpool(x)


    # layer1

    x = model.layer1(x)


    # layer2:
    #
    # [B, 128, 28, 28]

    layer2 = model.layer2(
        x
    )


    # layer3:
    #
    # [B, 256, 14, 14]

    layer3 = model.layer3(
        layer2
    )


    # Downsample layer2:
    #
    # 28×28
    # ->
    # 14×14

    layer2_downsampled = (
        F.adaptive_avg_pool2d(
            layer2,
            output_size=(
                14,
                14
            )
        )
    )


    # Concatenate:
    #
    # layer2:
    # [B, 128, 14, 14]
    #
    # layer3:
    # [B, 256, 14, 14]
    #
    # ->
    #
    # [B, 384, 14, 14]

    combined = torch.cat(
        [
            layer2_downsampled,
            layer3,
        ],
        dim=1
    )


    return combined


# ============================================================
# 6. Convert feature map to patch vectors
# ============================================================

def feature_map_to_patches(
    feature_map
):

    # Input:
    #
    # [1, 384, 14, 14]

    _, channels, height, width = (
        feature_map.shape
    )


    # [1, 384, 14, 14]
    #
    # ->
    #
    # [1, 14, 14, 384]

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
    # [196, 384]

    patches = (
        feature_map.reshape(
            height
            *
            width,

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
        height,
        width
    )


# ============================================================
# 7. Extract test patches
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
            extract_multilayer_features(
                tensor
            )
        )


    return (
        feature_map_to_patches(
            feature_map
        )
    )


# ============================================================
# 8. Compute anomaly scores
# ============================================================

def compute_anomaly_scores(
    test_patches,
    memory_bank
):

    # test:
    #
    # [196, 384]
    #
    # memory bank:
    #
    # [15366, 384]
    #
    # result:
    #
    # [196, 15366]

    similarity_matrix = (
        test_patches
        @
        memory_bank.T
    )


    # For each test patch,
    # find its closest normal representative.

    nearest_similarity, nearest_indices = (
        similarity_matrix.max(
            dim=1
        )
    )


    # Convert cosine similarity
    # to Euclidean distance.
    #
    # d² = 2 - 2*cosine

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


    distances = torch.sqrt(
        distance_squared
    )


    return (
        distances,
        nearest_similarity,
        nearest_indices
    )


# ============================================================
# 9. Run feature extraction
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
# 10. Run nearest-neighbour scoring
# ============================================================

(
    anomaly_scores,
    nearest_similarity,
    nearest_indices,

) = compute_anomaly_scores(
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

image_anomaly_score = (
    anomaly_scores.max().item()
)


print(
    f"\nImage anomaly score: "
    f"{image_anomaly_score:.6f}"
)


# ============================================================
# 12. Top anomalous patches
# ============================================================

sorted_scores, sorted_indices = (
    torch.sort(
        anomaly_scores,
        descending=True
    )
)


print(
    "\nTop 10 anomalous patches:"
)


for rank in range(
    min(
        10,
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
        f"Patch {patch_index:03d}"
        f" | row={row:02d}"
        f" col={col:02d}"
        f" | distance={score:.6f}"
    )


# ============================================================
# 13. Build anomaly map
# ============================================================

anomaly_map = (
    anomaly_scores.reshape(
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

anomaly_map_resize = (
    anomaly_map
    .unsqueeze(0)
    .unsqueeze(0)
)


upsampled_map = F.interpolate(
    anomaly_map_resize,
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
    "Multi-layer PatchCore-style Heatmap\n"
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
    "\nSaved visualization to:"
)

print(
    OUTPUT_PATH
)


print(
    "\nDone."
)