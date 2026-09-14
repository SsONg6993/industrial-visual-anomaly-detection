from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image

from torchvision import models
from torchvision.transforms.functional import (
    pil_to_tensor,
    normalize,
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

TEST_IMAGE_PATH = (
    PRODUCT_DIR
    / "test"
    / "aotu"
    / "000028.png"
)

CORESET_PATH = (
    PROJECT_ROOT
    / "models"
    / "aluminum_camera_cover_letterbox256_multilayer_coreset.pt"
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
    "Target size:",
    TARGET_SIZE
)

print(
    "Minimum valid ratio:",
    MIN_VALID_RATIO
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
# 5. Letterbox preprocessing
# ============================================================

def letterbox_image(
    image: Image.Image,
    target_size: int
):

    original_width, original_height = (
        image.size
    )

    scale = min(
        target_size / original_width,
        target_size / original_height
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
            target_size,
            target_size
        ),
        color=padding_color
    )

    left = (
        target_size - new_width
    ) // 2

    top = (
        target_size - new_height
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
            target_size,
            target_size
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
        valid_region,
    )


# ============================================================
# 6. Feature extraction
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
            output_size=(
                layer3.shape[-2],
                layer3.shape[-1],
            )
        )
    )

    combined = torch.cat(
        [
            layer2_downsampled,
            layer3
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

    patch_valid_ratio = (
        F.adaptive_avg_pool2d(
            valid_region,
            output_size=(h, w)
        )
        .reshape(-1)
    )

    valid_patch_mask = (
        patch_valid_ratio
        >=
        MIN_VALID_RATIO
    )

    return (
        valid_patch_mask,
        patch_valid_ratio
    )


# ============================================================
# 9. Load test image
# ============================================================

image = Image.open(
    TEST_IMAGE_PATH
).convert("RGB")


(
    tensor,
    valid_region,

) = letterbox_image(
    image,
    TARGET_SIZE
)


tensor = (
    tensor
    .unsqueeze(0)
    .to(device)
)


# ============================================================
# 10. Inference
# ============================================================

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

    (
        valid_patch_mask,
        patch_valid_ratio,

    ) = valid_region_to_patch_mask(
        valid_region,
        h,
        w
    )


    similarity_matrix = (
        patches
        @
        memory_bank.T
    )


    nearest_similarity = (
        similarity_matrix
        .max(dim=1)
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


# ============================================================
# 11. Ignore padding-heavy patches
# ============================================================

valid_patch_distances = (
    patch_distances[
        valid_patch_mask
    ]
)


image_score = (
    valid_patch_distances
    .max()
    .item()
)


# ============================================================
# 12. Print results
# ============================================================

print(
    "\n"
    +
    "=" * 70
)

print(
    "LETTERBOX256 PATCHCORE INFERENCE"
)

print(
    "=" * 70
)


print(
    "\nTest image:"
)

print(
    TEST_IMAGE_PATH
)


print(
    f"\nFeature map: "
    f"{h}x{w}"
)


print(
    f"Candidate patches: "
    f"{patches.shape[0]}"
)


print(
    f"Valid patches: "
    f"{valid_patch_mask.sum().item()}"
)


print(
    f"\nMinimum distance: "
    f"{valid_patch_distances.min().item():.6f}"
)

print(
    f"Average distance: "
    f"{valid_patch_distances.mean().item():.6f}"
)

print(
    f"Maximum distance: "
    f"{valid_patch_distances.max().item():.6f}"
)


print(
    f"\nIMAGE ANOMALY SCORE: "
    f"{image_score:.6f}"
)


# ============================================================
# 13. Top anomalous patches
# ============================================================

valid_indices = torch.where(
    valid_patch_mask
)[0]


valid_scores = (
    patch_distances[
        valid_patch_mask
    ]
)


top_k = min(
    10,
    len(valid_scores)
)


top_scores, top_positions = (
    torch.topk(
        valid_scores,
        k=top_k
    )
)


print(
    "\nTop anomalous patches:"
)


for rank in range(
    top_k
):

    original_patch_index = int(
        valid_indices[
            top_positions[rank]
        ]
    )

    row = (
        original_patch_index
        //
        w
    )

    col = (
        original_patch_index
        %
        w
    )


    print(
        f"{rank + 1:2d}. "
        f"patch={original_patch_index:3d}"
        f" | row={row:2d}"
        f" | col={col:2d}"
        f" | score="
        f"{top_scores[rank].item():.6f}"
    )