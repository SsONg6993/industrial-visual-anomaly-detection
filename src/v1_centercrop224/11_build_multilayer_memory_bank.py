from pathlib import Path

import torch
import torch.nn.functional as F

from PIL import Image

from torch.utils.data import (
    Dataset,
    DataLoader,
)

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

TRAIN_DIR = (
    DATA_DIR
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


MEMORY_BANK_PATH = (
    MODEL_DIR
    / "aluminum_camera_cover_multilayer_memory_bank.pt"
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
# 3. Load pretrained ResNet18
# ============================================================

weights = models.ResNet18_Weights.DEFAULT

model = models.resnet18(
    weights=weights
)

model = model.to(device)
model.eval()


# ============================================================
# 4. Preprocessing
# ============================================================

preprocess = weights.transforms()


# ============================================================
# 5. Dataset
# ============================================================

class NormalImageDataset(Dataset):

    def __init__(
        self,
        image_dir: Path,
        transform
    ):

        self.image_paths = sorted(
            path
            for path in image_dir.glob("*.png")
            if path.is_file()
        )

        self.transform = transform


    def __len__(self):

        return len(self.image_paths)


    def __getitem__(
        self,
        index
    ):

        image_path = (
            self.image_paths[index]
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        tensor = self.transform(
            image
        )

        return (
            tensor,
            str(image_path)
        )


# ============================================================
# 6. DataLoader
# ============================================================

dataset = NormalImageDataset(
    TRAIN_DIR,
    preprocess
)


print(
    f"Normal training images: "
    f"{len(dataset):,}"
)


BATCH_SIZE = 16


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# 7. Multi-layer feature extraction
# ============================================================

def extract_multilayer_features(
    images
):

    # --------------------------------------------------------
    # ResNet stem
    # --------------------------------------------------------

    x = model.conv1(images)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)


    # --------------------------------------------------------
    # layer1
    #
    # approximately:
    # [B, 64, 56, 56]
    # --------------------------------------------------------

    x = model.layer1(x)


    # --------------------------------------------------------
    # layer2
    #
    # [B, 128, 28, 28]
    # --------------------------------------------------------

    layer2 = model.layer2(x)


    # --------------------------------------------------------
    # layer3
    #
    # [B, 256, 14, 14]
    # --------------------------------------------------------

    layer3 = model.layer3(
        layer2
    )


    # ========================================================
    # Align spatial resolution
    # ========================================================
    #
    # layer2:
    #
    # 28×28×128
    #
    # layer3:
    #
    # 14×14×256
    #
    # We cannot concatenate them directly because
    # H and W are different.
    #
    # Therefore we downsample layer2 to 14×14.
    # ========================================================

    layer2_downsampled = (
        F.adaptive_avg_pool2d(
            layer2,
            output_size=(
                14,
                14
            )
        )
    )


    # --------------------------------------------------------
    # Concatenate along CHANNEL dimension.
    #
    # [B, 128, 14, 14]
    #
    # +
    #
    # [B, 256, 14, 14]
    #
    # =
    #
    # [B, 384, 14, 14]
    # --------------------------------------------------------

    combined = torch.cat(
        [
            layer2_downsampled,
            layer3,
        ],
        dim=1
    )


    return (
        combined,
        layer2,
        layer3
    )


# ============================================================
# 8. Feature map -> patch vectors
# ============================================================

def feature_map_to_patches(
    feature_map
):

    # Input:
    #
    # [B, C, H, W]
    #
    # Expected:
    #
    # [B, 384, 14, 14]

    batch_size = (
        feature_map.shape[0]
    )

    channels = (
        feature_map.shape[1]
    )

    height = (
        feature_map.shape[2]
    )

    width = (
        feature_map.shape[3]
    )


    # --------------------------------------------------------
    # [B, C, H, W]
    #
    # ->
    #
    # [B, H, W, C]
    # --------------------------------------------------------

    feature_map = feature_map.permute(
        0,
        2,
        3,
        1
    )


    # --------------------------------------------------------
    # [B, H, W, C]
    #
    # ->
    #
    # [B*H*W, C]
    # --------------------------------------------------------

    patches = feature_map.reshape(
        batch_size
        *
        height
        *
        width,

        channels
    )


    # --------------------------------------------------------
    # L2 normalize each patch feature
    # --------------------------------------------------------

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
# 9. Build memory bank
# ============================================================

all_patch_features = []


print(
    "\nBuilding multi-layer memory bank..."
)


with torch.no_grad():

    for batch_index, (
        images,
        image_paths
    ) in enumerate(
        dataloader,
        start=1
    ):

        images = images.to(
            device
        )


        (
            combined_features,
            layer2,
            layer3,

        ) = extract_multilayer_features(
            images
        )


        # Print shapes only for first batch.

        if batch_index == 1:

            print(
                "\nFirst batch feature shapes:"
            )

            print(
                "Layer2:",
                tuple(layer2.shape)
            )

            print(
                "Layer3:",
                tuple(layer3.shape)
            )

            print(
                "Combined:",
                tuple(
                    combined_features.shape
                )
            )


        patch_features, h, w = (
            feature_map_to_patches(
                combined_features
            )
        )


        patch_features = (
            patch_features.cpu()
        )


        all_patch_features.append(
            patch_features
        )


        processed = min(
            batch_index
            *
            BATCH_SIZE,

            len(dataset)
        )


        print(
            f"Processed "
            f"{processed:,}/"
            f"{len(dataset):,} images"
        )


# ============================================================
# 10. Combine batches
# ============================================================

memory_bank = torch.cat(
    all_patch_features,
    dim=0
)


print(
    "\nMulti-layer memory bank created."
)


print(
    "Memory bank shape:",
    tuple(memory_bank.shape)
)


# ============================================================
# 11. Validate expected size
# ============================================================

patches_per_image = (
    14
    *
    14
)


expected_patches = (
    len(dataset)
    *
    patches_per_image
)


print(
    f"Expected patches: "
    f"{expected_patches:,}"
)

print(
    f"Actual patches:   "
    f"{memory_bank.shape[0]:,}"
)


# ============================================================
# 12. Memory size
# ============================================================

bytes_used = (
    memory_bank.numel()
    *
    memory_bank.element_size()
)


megabytes_used = (
    bytes_used
    /
    (1024 ** 2)
)


print(
    f"Memory bank size: "
    f"{megabytes_used:.2f} MB"
)


# ============================================================
# 13. Save
# ============================================================

torch.save(
    {
        "product":
            "Aluminum_Camera_Cover",

        "backbone":
            "ResNet18",

        "feature_layers":
            [
                "layer2",
                "layer3",
            ],

        "spatial_resolution":
            (
                h,
                w
            ),

        "num_training_images":
            len(dataset),

        "patches_per_image":
            patches_per_image,

        "feature_dimension":
            memory_bank.shape[1],

        "memory_bank":
            memory_bank,
    },

    MEMORY_BANK_PATH
)


print(
    "\nSaved multi-layer memory bank to:"
)

print(
    MEMORY_BANK_PATH
)


print(
    "\nDone."
)