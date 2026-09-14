from pathlib import Path
import time

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = (
    PROJECT_ROOT
    / "data"
    / "3CAD"
    / "Aluminum_Camera_Cover"
    / "train"
    / "good"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_PATH = (
    MODEL_DIR
    / "aluminum_camera_cover_letterbox256_multilayer_memory_bank.pt"
)


# ============================================================
# 2. Configuration
# ============================================================

TARGET_SIZE = 256

MIN_VALID_RATIO = 0.50


# ============================================================
# 3. Device
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
# 4. ResNet18
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
).to(device)

model.eval()


# ============================================================
# 5. Letterbox image
# ============================================================

def letterbox_image(
    image: Image.Image,
    target_size: int = 256
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


    # --------------------------------------------------------
    # Resize real image
    # --------------------------------------------------------

    resized = image.resize(
        (
            new_width,
            new_height
        ),
        resample=Image.Resampling.BILINEAR
    )


    # --------------------------------------------------------
    # Padding colour
    #
    # We use ImageNet mean RGB.
    #
    # After normalization this becomes approximately zero.
    # --------------------------------------------------------

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
        target_size
        -
        new_width
    ) // 2

    top = (
        target_size
        -
        new_height
    ) // 2


    canvas.paste(
        resized,
        (
            left,
            top
        )
    )


    # --------------------------------------------------------
    # Valid-region mask
    #
    # 1 = real image
    # 0 = padding
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # PIL -> tensor
    # --------------------------------------------------------

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


    # Letterbox256:
    #
    # layer2 = 32x32
    # layer3 = 16x16
    #
    # Align layer2 -> 16x16.

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
# 7. Feature map -> patch vectors
# ============================================================

def feature_map_to_patches(
    feature_map
):

    batch_size, channels, h, w = (
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
            batch_size,
            h * w,
            channels
        )
    )


    patches = F.normalize(
        patches,
        p=2,
        dim=2
    )


    return (
        patches,
        h,
        w
    )


# ============================================================
# 8. Valid-region mask -> patch mask
# ============================================================

def valid_region_to_patch_mask(
    valid_region,
    h,
    w
):

    #
    # valid_region:
    # 1 x 256 x 256
    #
    # We average-pool it to feature-map resolution.
    #
    # Result:
    # each value = fraction of real image inside that patch.
    #

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
    )


    patch_valid_ratio = (
        patch_valid_ratio
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
# 9. Collect training images
# ============================================================

image_paths = sorted(
    TRAIN_DIR.glob("*.png")
)

print(
    f"\nTraining normal images: "
    f"{len(image_paths):,}"
)

print(
    f"Target size: "
    f"{TARGET_SIZE}x{TARGET_SIZE}"
)

print(
    f"Minimum valid patch ratio: "
    f"{MIN_VALID_RATIO:.0%}"
)


# ============================================================
# 10. Build memory bank
# ============================================================

all_features = []

total_candidate_patches = 0
total_kept_patches = 0

start_time = time.perf_counter()


for index, image_path in enumerate(
    image_paths,
    start=1
):

    image = Image.open(
        image_path
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


        valid_patch_mask, _ = (
            valid_region_to_patch_mask(
                valid_region,
                h,
                w
            )
        )


        patches = patches.squeeze(0)


        total_candidate_patches += (
            patches.shape[0]
        )


        valid_patches = (
            patches[
                valid_patch_mask
            ]
        )


        total_kept_patches += (
            valid_patches.shape[0]
        )


        all_features.append(
            valid_patches.cpu()
        )


    if (
        index % 50 == 0
        or
        index == len(image_paths)
    ):

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        print(
            f"Processed "
            f"{index:,}/"
            f"{len(image_paths):,}"
            f" | kept patches="
            f"{total_kept_patches:,}"
            f" | elapsed="
            f"{elapsed:.1f}s"
        )


# ============================================================
# 11. Concatenate memory bank
# ============================================================

memory_bank = torch.cat(
    all_features,
    dim=0
)


# ============================================================
# 12. Statistics
# ============================================================

candidate_keep_ratio = (
    total_kept_patches
    /
    total_candidate_patches
)


memory_size_mb = (
    memory_bank.numel()
    *
    memory_bank.element_size()
    /
    (1024 ** 2)
)


print(
    "\n"
    +
    "=" * 70
)

print(
    "LETTERBOX256 MULTILAYER MEMORY BANK"
)

print(
    "=" * 70
)


print(
    f"\nFeature shape: "
    f"{tuple(memory_bank.shape)}"
)

print(
    f"Feature dimension: "
    f"{memory_bank.shape[1]}"
)


print(
    f"\nCandidate patches: "
    f"{total_candidate_patches:,}"
)

print(
    f"Kept valid patches: "
    f"{total_kept_patches:,}"
)

print(
    f"Patch keep ratio: "
    f"{candidate_keep_ratio:.2%}"
)


print(
    f"\nMemory size: "
    f"{memory_size_mb:.2f} MB"
)


# ============================================================
# 13. Save
# ============================================================

checkpoint = {
    "memory_bank":
        memory_bank,

    "target_size":
        TARGET_SIZE,

    "min_valid_ratio":
        MIN_VALID_RATIO,

    "feature_dim":
        memory_bank.shape[1],

    "num_training_images":
        len(image_paths),

    "preprocessing":
        "letterbox256",

    "feature_layers":
        "resnet18_layer2_layer3",
}


torch.save(
    checkpoint,
    OUTPUT_PATH
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)