from pathlib import Path
import csv

import numpy as np

from PIL import Image

from torchvision.transforms import (
    Compose,
    Resize,
    CenterCrop,
    InterpolationMode,
)


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PRODUCT_DIR = (
    PROJECT_ROOT
    / "data"
    / "3CAD"
    / "Aluminum_Camera_Cover"
)

TEST_DIR = (
    PRODUCT_DIR
    / "test"
)

GROUND_TRUTH_DIR = (
    PRODUCT_DIR
    / "ground_truth"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_CSV = (
    REPORT_DIR
    / "crop_loss_audit.csv"
)


# ============================================================
# 2. Same spatial transform used by model
# ============================================================

mask_transform = Compose(
    [
        Resize(
            256,
            interpolation=
            InterpolationMode.NEAREST
        ),

        CenterCrop(
            224
        ),
    ]
)


# ============================================================
# 3. Collect results
# ============================================================

results = []


for defect_dir in sorted(
    TEST_DIR.iterdir()
):

    if not defect_dir.is_dir():
        continue

    defect_type = defect_dir.name

    if defect_type == "good":
        continue


    for image_path in sorted(
        defect_dir.glob("*.png")
    ):

        mask_path = (
            GROUND_TRUTH_DIR
            / defect_type
            / image_path.name
        )

        if not mask_path.exists():

            print(
                "Missing mask:",
                mask_path
            )

            continue


        # ----------------------------------------------------
        # Original mask
        # ----------------------------------------------------

        mask = Image.open(
            mask_path
        ).convert("L")


        original_array = np.array(
            mask
        )

        original_binary = (
            original_array > 127
        )

        original_defect_pixels = int(
            original_binary.sum()
        )


        # ----------------------------------------------------
        # After Resize + CenterCrop
        # ----------------------------------------------------

        transformed_mask = (
            mask_transform(
                mask
            )
        )

        transformed_array = np.array(
            transformed_mask
        )

        transformed_binary = (
            transformed_array > 127
        )

        cropped_defect_pixels = int(
            transformed_binary.sum()
        )


        # ----------------------------------------------------
        # Retention
        # ----------------------------------------------------

        if original_defect_pixels > 0:

            retention_ratio = (
                cropped_defect_pixels
                /
                original_defect_pixels
            )

        else:

            retention_ratio = 0.0


        loss_ratio = (
            1.0
            -
            retention_ratio
        )


        fully_lost = (
            original_defect_pixels > 0
            and
            cropped_defect_pixels == 0
        )


        results.append(
            {
                "defect_type":
                    defect_type,

                "image_name":
                    image_path.name,

                "original_width":
                    mask.width,

                "original_height":
                    mask.height,

                "original_defect_pixels":
                    original_defect_pixels,

                "cropped_defect_pixels":
                    cropped_defect_pixels,

                "retention_ratio":
                    retention_ratio,

                "loss_ratio":
                    loss_ratio,

                "fully_lost":
                    fully_lost,
            }
        )


# ============================================================
# 4. Save CSV
# ============================================================

with open(
    OUTPUT_CSV,
    "w",
    newline="",
    encoding="utf-8"
) as file:

    fieldnames = [
        "defect_type",
        "image_name",
        "original_width",
        "original_height",
        "original_defect_pixels",
        "cropped_defect_pixels",
        "retention_ratio",
        "loss_ratio",
        "fully_lost",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in results:

        output_row = row.copy()

        output_row[
            "retention_ratio"
        ] = (
            f"{row['retention_ratio']:.6f}"
        )

        output_row[
            "loss_ratio"
        ] = (
            f"{row['loss_ratio']:.6f}"
        )

        writer.writerow(
            output_row
        )


# ============================================================
# 5. Overall statistics
# ============================================================

total = len(
    results
)

fully_lost_count = sum(
    row["fully_lost"]
    for row in results
)


lost_over_10 = sum(
    row["retention_ratio"] < 0.90
    for row in results
)


lost_over_50 = sum(
    row["retention_ratio"] < 0.50
    for row in results
)


retention_values = np.array(
    [
        row["retention_ratio"]
        for row in results
    ],
    dtype=np.float64
)


print(
    "\n"
    +
    "=" * 70
)

print(
    "CENTER CROP DEFECT RETENTION AUDIT"
)

print(
    "=" * 70
)


print(
    f"\nTotal defect masks: "
    f"{total:,}"
)

print(
    f"Fully lost defects: "
    f"{fully_lost_count:,}"
)

print(
    f"Lost >10% defect area: "
    f"{lost_over_10:,}"
)

print(
    f"Lost >50% defect area: "
    f"{lost_over_50:,}"
)


print(
    f"\nMean retention: "
    f"{retention_values.mean():.4%}"
)

print(
    f"Median retention: "
    f"{np.median(retention_values):.4%}"
)

print(
    f"Minimum retention: "
    f"{retention_values.min():.4%}"
)


# ============================================================
# 6. Fully lost examples
# ============================================================

fully_lost_samples = [
    row
    for row in results
    if row["fully_lost"]
]


print(
    "\nExamples of fully lost defects:"
)


for row in fully_lost_samples[:20]:

    print(
        f"{row['defect_type']:15s}"
        f" | "
        f"{row['image_name']}"
        f" | original pixels="
        f"{row['original_defect_pixels']}"
    )


# ============================================================
# 7. Per defect-type statistics
# ============================================================

defect_types = sorted(
    set(
        row["defect_type"]
        for row in results
    )
)


print(
    "\nPer-defect-type:"
)


for defect_type in defect_types:

    subset = [
        row
        for row in results
        if row["defect_type"]
        ==
        defect_type
    ]


    subset_retention = np.array(
        [
            row["retention_ratio"]
            for row in subset
        ]
    )


    subset_fully_lost = sum(
        row["fully_lost"]
        for row in subset
    )


    print(
        f"{defect_type:15s}"
        f" | n={len(subset):3d}"
        f" | fully_lost="
        f"{subset_fully_lost:3d}"
        f" | mean_retention="
        f"{subset_retention.mean():.2%}"
    )


print(
    "\nSaved report:"
)

print(
    OUTPUT_CSV
)