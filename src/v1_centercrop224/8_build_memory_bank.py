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
# 1. Project paths
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
    / "aluminum_camera_cover_memory_bank.pt"
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

weights = (
    models.ResNet18_Weights.DEFAULT
)

model = models.resnet18(
    weights=weights
)

model = model.to(device)

model.eval()


# ============================================================
# 4. Feature extractor
# ============================================================

# We stop at layer4.
#
# No avgpool.
# No classifier.
#
# Output:
#
# [B, 512, 7, 7]

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
# 5. Preprocessing
# ============================================================

preprocess = weights.transforms()


# ============================================================
# 6. Dataset
# ============================================================

class NormalImageDataset(Dataset):

    def __init__(
        self,
        image_dir: Path,
        transform
    ):

        self.image_paths = sorted(
            [
                path
                for path in image_dir.glob("*.png")
                if path.is_file()
            ]
        )

        self.transform = transform


    def __len__(self):

        return len(
            self.image_paths
        )


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

        image_tensor = (
            self.transform(image)
        )

        return (
            image_tensor,
            str(image_path)
        )


# ============================================================
# 7. Create DataLoader
# ============================================================

dataset = NormalImageDataset(
    TRAIN_DIR,
    preprocess
)


print(
    f"Normal training images: "
    f"{len(dataset):,}"
)


BATCH_SIZE = 32


dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# 8. Feature-map -> patch features
# ============================================================

def feature_map_to_patches(
    feature_map
):

    """
    Input:

        [B, C, H, W]

    Example:

        [32, 512, 7, 7]


    Output:

        [B*H*W, C]

    Example:

        [32*49, 512]
    """

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


    # [B, C, H, W]
    #
    # ->
    #
    # [B, H, W, C]

    feature_map = (
        feature_map.permute(
            0,
            2,
            3,
            1
        )
    )


    # [B, H, W, C]
    #
    # ->
    #
    # [B*H*W, C]

    patches = (
        feature_map.reshape(
            batch_size
            *
            height
            *
            width,

            channels
        )
    )


    # Normalize each patch vector.

    patches = F.normalize(
        patches,
        dim=1
    )


    return patches


# ============================================================
# 9. Extract all normal patch features
# ============================================================

all_patch_features = []


print(
    "\nBuilding normal memory bank..."
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


        # CNN feature extraction

        feature_map = (
            feature_extractor(
                images
            )
        )


        # Convert CNN feature map
        # into local patch vectors.

        patch_features = (
            feature_map_to_patches(
                feature_map
            )
        )


        # Move back to CPU.
        #
        # We don't need to keep the
        # entire memory bank in GPU
        # while building it.

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
            f"{len(dataset):,} "
            f"images"
        )


# ============================================================
# 10. Combine all batches
# ============================================================

memory_bank = torch.cat(
    all_patch_features,
    dim=0
)


print(
    "\nMemory bank created."
)


print(
    "Memory bank shape:",
    tuple(memory_bank.shape)
)


# ============================================================
# 11. Calculate expected patch count
# ============================================================

patches_per_image = 7 * 7

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
# 12. Estimate memory usage
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
# 13. Save memory bank
# ============================================================

torch.save(
    {
        "product":
            "Aluminum_Camera_Cover",

        "backbone":
            "ResNet18",

        "feature_layer":
            "layer4",

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
    "\nSaved memory bank to:"
)

print(
    MEMORY_BANK_PATH
)


print(
    "\nDone."
)