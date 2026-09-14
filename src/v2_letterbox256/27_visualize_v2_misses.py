from pathlib import Path
import csv

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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

GROUND_TRUTH_DIR = (
    PRODUCT_DIR
    / "ground_truth"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

CORESET_PATH = (
    PROJECT_ROOT
    / "models"
    / "aluminum_camera_cover_letterbox256_multilayer_coreset.pt"
)

CROP_AUDIT_PATH = (
    REPORT_DIR
    / "crop_loss_audit.csv"
)

V2_SCORES_PATH = (
    REPORT_DIR
    / "v2_letterbox256_image_scores.csv"
)

OUTPUT_PATH = (
    REPORT_DIR
    / "v2_missed_crop_cases_board.png"
)


# ============================================================
# 2. Config
# ============================================================

V2_THRESHOLD = 0.533522

TOP_N = 12


# ============================================================
# 3. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# 4. Load coreset
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


# ============================================================
# 5. ResNet18
# ============================================================

weights = (
    models.ResNet18_Weights.DEFAULT
)

model = models.resnet18(
    weights=weights
).to(device)

model.eval()


# ============================================================
# 6. Letterbox image
# ============================================================

def letterbox_image(
    image: Image.Image
):

    original_width, original_height = (
        image.size
    )

    scale = min(
        TARGET_SIZE / original_width,
        TARGET_SIZE / original_height
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
            TARGET_SIZE,
            TARGET_SIZE
        ),
        color=padding_color
    )

    left = (
        TARGET_SIZE - new_width
    ) // 2

    top = (
        TARGET_SIZE - new_height
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
            TARGET_SIZE,
            TARGET_SIZE
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
        canvas,
        tensor,
        valid_region,
        scale,
        left,
        top,
        new_width,
        new_height,
    )


# ============================================================
# 7. Letterbox ground-truth mask
# ============================================================

def letterbox_mask(
    mask: Image.Image,
    scale,
    left,
    top,
    new_width,
    new_height,
):

    resized = mask.resize(
        (
            new_width,
            new_height
        ),
        resample=Image.Resampling.NEAREST
    )

    canvas = Image.new(
        "L",
        (
            TARGET_SIZE,
            TARGET_SIZE
        ),
        color=0
    )

    canvas.paste(
        resized,
        (
            left,
            top
        )
    )

    return (
        np.array(canvas) > 127
    ).astype(
        np.uint8
    )


# ============================================================
# 8. Feature extraction
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
# 9. Produce anomaly map
# ============================================================

def infer_anomaly_map(
    image_path
):

    image = Image.open(
        image_path
    ).convert("RGB")

    (
        letterboxed_image,
        tensor,
        valid_region,
        scale,
        left,
        top,
        new_width,
        new_height,

    ) = letterbox_image(
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

        patch_scores = torch.sqrt(
            distance_squared
        )

        anomaly_map = (
            patch_scores
            .reshape(
                1,
                1,
                h,
                w
            )
        )

        anomaly_map = F.interpolate(
            anomaly_map,
            size=(
                TARGET_SIZE,
                TARGET_SIZE
            ),
            mode="bilinear",
            align_corners=False,
        )

    anomaly_map = (
        anomaly_map
        .squeeze()
        .cpu()
        .numpy()
    )

    return (
        image,
        np.array(
            letterboxed_image
        ),
        anomaly_map,
        scale,
        left,
        top,
        new_width,
        new_height,
    )


# ============================================================
# 10. Load V1 fully-lost samples
# ============================================================

fully_lost_samples = set()

with open(
    CROP_AUDIT_PATH,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(
        file
    )

    for row in reader:

        fully_lost = (
            row["fully_lost"]
            .strip()
            .lower()
            in {
                "true",
                "1",
                "yes",
            }
        )

        if fully_lost:

            fully_lost_samples.add(
                (
                    row["defect_type"],
                    row["image_name"],
                )
            )


# ============================================================
# 11. Find V2 misses among those 95
# ============================================================

misses = []

with open(
    V2_SCORES_PATH,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(
        file
    )

    for row in reader:

        defect_type = (
            row["defect_type"]
        )

        image_name = (
            Path(
                row["image_path"]
            ).name
        )

        key = (
            defect_type,
            image_name
        )

        if key not in fully_lost_samples:
            continue

        score = float(
            row["score"]
        )

        if score < V2_THRESHOLD:

            misses.append(
                {
                    "defect_type":
                        defect_type,

                    "image_name":
                        image_name,

                    "score":
                        score,
                }
            )


misses.sort(
    key=lambda x: x["score"]
)

misses = misses[
    :TOP_N
]


print(
    f"Visualizing "
    f"{len(misses)} "
    f"lowest-scoring misses."
)


# ============================================================
# 12. Plot board
# ============================================================

fig, axes = plt.subplots(
    len(misses),
    4,
    figsize=(
        16,
        len(misses) * 4
    )
)


for index, result in enumerate(
    misses
):

    defect_type = (
        result["defect_type"]
    )

    image_path = (
        PRODUCT_DIR
        / "test"
        / defect_type
        / result["image_name"]
    )

    mask_path = (
        GROUND_TRUTH_DIR
        / defect_type
        / result["image_name"]
    )


    (
        original_image,
        letterboxed_array,
        anomaly_map,
        scale,
        left,
        top,
        new_width,
        new_height,

    ) = infer_anomaly_map(
        image_path
    )


    mask = Image.open(
        mask_path
    ).convert("L")


    gt_mask = letterbox_mask(
        mask,
        scale,
        left,
        top,
        new_width,
        new_height,
    )


    # Original

    axes[index, 0].imshow(
        original_image
    )

    axes[index, 0].set_title(
        f"{defect_type}\n"
        f"{result['image_name']}"
    )

    axes[index, 0].axis(
        "off"
    )


    # Letterbox

    axes[index, 1].imshow(
        letterboxed_array
    )

    axes[index, 1].set_title(
        "Letterbox256"
    )

    axes[index, 1].axis(
        "off"
    )


    # Ground truth

    axes[index, 2].imshow(
        gt_mask
    )

    axes[index, 2].set_title(
        "Ground Truth"
    )

    axes[index, 2].axis(
        "off"
    )


    # Heatmap

    axes[index, 3].imshow(
        letterboxed_array
    )

    axes[index, 3].imshow(
        anomaly_map,
        alpha=0.45,
        vmin=0.0,
        vmax=0.8,
    )

    axes[index, 3].set_title(
        f"V2 Heatmap\n"
        f"score="
        f"{result['score']:.4f}"
    )

    axes[index, 3].axis(
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
    "\nSaved:"
)

print(
    OUTPUT_PATH
)