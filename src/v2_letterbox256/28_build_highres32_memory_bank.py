from pathlib import Path
import time

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
    / "aluminum_camera_cover_letterbox256_highres32_memory_bank.pt"
)


# ============================================================
# 2. Configuration
# ============================================================

TARGET_SIZE = 256

MIN_VALID_RATIO = 0.50

#
# IMPORTANT:
#
# Feature map remains 32x32.
# We only subsample TRAINING memory-bank locations.
#

MEMORY_STRIDE = 2


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

weights = (
    models.ResNet18_Weights.DEFAULT
)

model = models.resnet18(
    weights=weights
).to(device)

model.eval()


# ============================================================
# 5. Letterbox preprocessing
# ============================================================

def letterbox_image(
    image,
    target_size
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


    # 1 = real image
    # 0 = padding

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
        valid_region
    )


# ============================================================
# 6. HIGH-RES feature fusion
# ============================================================

def extract_highres_features(
    tensor
):

    x = model.conv1(tensor)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)

    x = model.layer1(x)

    # 256 input:
    # layer2 = 32x32x128

    layer2 = model.layer2(x)

    # layer3 = 16x16x256

    layer3 = model.layer3(
        layer2
    )


    # ========================================================
    # KEY DIFFERENCE FROM V2a
    #
    # OLD:
    # layer2 32x32 -> downsample 16x16
    #
    # NEW:
    # layer3 16x16 -> upsample 32x32
    # ========================================================

    layer3_upsampled = (
        F.interpolate(
            layer3,
            size=(
                layer2.shape[-2],
                layer2.shape[-1]
            ),
            mode="bilinear",
            align_corners=False,
        )
    )


    combined = torch.cat(
        [
            layer2,
            layer3_upsampled
        ],
        dim=1
    )


    return combined


# ============================================================
# 7. Valid-region ratios at 32x32
# ============================================================

def get_valid_ratio_map(
    valid_region,
    h,
    w
):

    valid_region = (
        valid_region
        .unsqueeze(0)
        .to(device)
    )


    valid_ratio = (
        F.adaptive_avg_pool2d(
            valid_region,
            output_size=(
                h,
                w
            )
        )
    )


    return (
        valid_ratio
        .squeeze(0)
        .squeeze(0)
    )


# ============================================================
# 8. Collect training images
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
    f"Memory spatial stride: "
    f"{MEMORY_STRIDE}"
)


if len(image_paths) == 0:

    raise RuntimeError(
        f"No training images found at:\n"
        f"{TRAIN_DIR}"
    )


# ============================================================
# 9. Build high-resolution memory bank
# ============================================================

all_features = []

total_highres_patches = 0

total_sampled_patches = 0

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
            extract_highres_features(
                tensor
            )
        )


        _, channels, h, w = (
            feature_map.shape
        )


        # Expected:
        # 32 x 32

        total_highres_patches += (
            h * w
        )


        # ----------------------------------------------------
        # Spatial subsampling ONLY for memory bank.
        #
        # 32x32
        # ↓ stride 2
        # 16x16
        # ----------------------------------------------------

        sampled_features = (
            feature_map[
                :,
                :,
                ::MEMORY_STRIDE,
                ::MEMORY_STRIDE
            ]
        )


        sampled_h = (
            sampled_features.shape[-2]
        )

        sampled_w = (
            sampled_features.shape[-1]
        )


        total_sampled_patches += (
            sampled_h
            *
            sampled_w
        )


        # ----------------------------------------------------
        # Valid ratios from full 32x32 map
        # ----------------------------------------------------

        valid_ratio_map = (
            get_valid_ratio_map(
                valid_region,
                h,
                w
            )
        )


        # Match exactly the same sampled positions.

        sampled_valid_ratio = (
            valid_ratio_map[
                ::MEMORY_STRIDE,
                ::MEMORY_STRIDE
            ]
        )


        valid_mask = (
            sampled_valid_ratio.reshape(-1)
            >=
            MIN_VALID_RATIO
        )


        # ----------------------------------------------------
        # Feature map -> vectors
        # ----------------------------------------------------

        patches = (
            sampled_features
            .permute(
                0,
                2,
                3,
                1
            )
            .reshape(
                sampled_h * sampled_w,
                channels
            )
        )


        patches = F.normalize(
            patches,
            p=2,
            dim=1
        )


        valid_patches = (
            patches[
                valid_mask
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
            f" | kept="
            f"{total_kept_patches:,}"
            f" | elapsed="
            f"{elapsed:.1f}s"
        )


# ============================================================
# 10. Build memory bank
# ============================================================

if not all_features:

    raise RuntimeError(
        "No features were collected."
    )


memory_bank = torch.cat(
    all_features,
    dim=0
)


# ============================================================
# 11. Statistics
# ============================================================

memory_size_mb = (
    memory_bank.numel()
    *
    memory_bank.element_size()
    /
    (1024 ** 2)
)


keep_ratio = (
    total_kept_patches
    /
    total_sampled_patches
)


print(
    "\n"
    +
    "=" * 70
)

print(
    "V2B LETTERBOX256 HIGH-RES 32x32 MEMORY BANK"
)

print(
    "=" * 70
)


print(
    f"\nFull feature resolution: "
    f"{h}x{w}"
)

print(
    f"Full patches/image: "
    f"{h * w}"
)


print(
    f"\nMemory sampling stride: "
    f"{MEMORY_STRIDE}"
)

print(
    f"Sampled grid: "
    f"{sampled_h}x{sampled_w}"
)

print(
    f"Sampled patches/image: "
    f"{sampled_h * sampled_w}"
)


print(
    f"\nAll high-res patches: "
    f"{total_highres_patches:,}"
)

print(
    f"Sampled candidate patches: "
    f"{total_sampled_patches:,}"
)

print(
    f"Kept valid patches: "
    f"{total_kept_patches:,}"
)

print(
    f"Valid keep ratio: "
    f"{keep_ratio:.2%}"
)


print(
    f"\nMemory bank shape: "
    f"{tuple(memory_bank.shape)}"
)

print(
    f"Feature dimension: "
    f"{memory_bank.shape[1]}"
)

print(
    f"Memory size: "
    f"{memory_size_mb:.2f} MB"
)


# ============================================================
# 12. Save
# ============================================================

checkpoint = {
    "memory_bank":
        memory_bank,

    "target_size":
        TARGET_SIZE,

    "min_valid_ratio":
        MIN_VALID_RATIO,

    "memory_stride":
        MEMORY_STRIDE,

    "feature_dim":
        memory_bank.shape[1],

    "num_training_images":
        len(image_paths),

    "preprocessing":
        "letterbox256",

    "feature_layers":
        "resnet18_layer2_plus_upsampled_layer3",

    "feature_resolution":
        32,

    "experiment":
        "v2b_highres32",
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