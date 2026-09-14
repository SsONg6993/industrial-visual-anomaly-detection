from pathlib import Path
import csv

import numpy as np
from PIL import Image


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

TEST_DIR = PRODUCT_DIR / "test"

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
    / "letterbox_retention_audit.csv"
)


# ============================================================
# 2. Config
# ============================================================

TARGET_SIZE = 256


# ============================================================
# 3. Letterbox for mask
# ============================================================

def letterbox_mask(
    mask: Image.Image,
    target_size: int = 256
):

    original_width, original_height = (
        mask.size
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
            target_size,
            target_size
        ),
        color=0
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


    valid_region = np.zeros(
        (
            target_size,
            target_size
        ),
        dtype=np.uint8
    )

    valid_region[
        top:top + new_height,
        left:left + new_width
    ] = 1


    return (
        canvas,
        valid_region,
        scale,
        left,
        top,
        new_width,
        new_height,
    )


# ============================================================
# 4. Audit
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


        mask = Image.open(
            mask_path
        ).convert("L")


        # ---------------------------------------------
        # Original binary mask
        # ---------------------------------------------

        original_array = np.array(
            mask
        )

        original_binary = (
            original_array > 127
        )

        original_pixels = int(
            original_binary.sum()
        )


        # ---------------------------------------------
        # Letterbox
        # ---------------------------------------------

        (
            letterboxed_mask,
            valid_region,
            scale,
            left,
            top,
            new_width,
            new_height,

        ) = letterbox_mask(
            mask,
            TARGET_SIZE
        )


        letterbox_array = np.array(
            letterboxed_mask
        )

        letterbox_binary = (
            letterbox_array > 127
        )

        letterbox_pixels = int(
            letterbox_binary.sum()
        )


        fully_lost = (
            original_pixels > 0
            and
            letterbox_pixels == 0
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
                    original_pixels,

                "letterbox_defect_pixels":
                    letterbox_pixels,

                "scale":
                    scale,

                "new_width":
                    new_width,

                "new_height":
                    new_height,

                "pad_left":
                    left,

                "pad_top":
                    top,

                "fully_lost":
                    fully_lost,
            }
        )


# ============================================================
# 5. Save CSV
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
        "letterbox_defect_pixels",
        "scale",
        "new_width",
        "new_height",
        "pad_left",
        "pad_top",
        "fully_lost",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames
    )

    writer.writeheader()

    for row in results:

        output_row = row.copy()

        output_row["scale"] = (
            f"{row['scale']:.6f}"
        )

        writer.writerow(
            output_row
        )


# ============================================================
# 6. Summary
# ============================================================

total = len(
    results
)

fully_lost_count = sum(
    row["fully_lost"]
    for row in results
)


print(
    "\n"
    +
    "=" * 70
)

print(
    "LETTERBOX DEFECT RETENTION AUDIT"
)

print(
    "=" * 70
)


print(
    f"\nTarget size: "
    f"{TARGET_SIZE}x{TARGET_SIZE}"
)

print(
    f"Total defect masks: "
    f"{total:,}"
)

print(
    f"Fully lost defects: "
    f"{fully_lost_count:,}"
)


# ============================================================
# 7. Per defect type
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

    subset_lost = sum(
        row["fully_lost"]
        for row in subset
    )

    lost_rate = (
        subset_lost
        /
        len(subset)
        if len(subset) > 0
        else 0.0
    )

    print(
        f"{defect_type:15s}"
        f" | n={len(subset):3d}"
        f" | fully_lost="
        f"{subset_lost:3d}"
        f" | lost_rate="
        f"{lost_rate:.2%}"
    )


print(
    "\nSaved report:"
)

print(
    OUTPUT_CSV
)