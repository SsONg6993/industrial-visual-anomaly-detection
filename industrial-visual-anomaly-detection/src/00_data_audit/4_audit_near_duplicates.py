from pathlib import Path
from collections import defaultdict
from PIL import Image
import imagehash
import csv


# ============================================================
# 1. Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "3CAD"

REPORT_DIR = PROJECT_ROOT / "reports"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# 2. Near-duplicate threshold
# ============================================================

# pHash normally gives us a 64-bit perceptual fingerprint.
#
# Hamming distance:
#
# 0     = extremely similar perceptual hash
# 1-2   = very suspicious
# 3-4   = suspicious
#
# IMPORTANT:
# This does NOT mean distance <= 4 must be deleted.
# It only means:
#
# "Please inspect this pair."

PHASH_THRESHOLD = 4


# ============================================================
# 3. Helper functions
# ============================================================

def is_image_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_product_images(
    product_dir: Path,
    split: str
):
    """
    Collect product images from a specific split.

    Example:
        Aluminum_Ipad/train/...
        Aluminum_Ipad/test/...

    Ground-truth masks are NOT included.
    """

    split_dir = product_dir / split

    if not split_dir.exists():
        return []

    return [
        path
        for path in split_dir.rglob("*")
        if is_image_file(path)
    ]


def calculate_phash(image_path: Path):
    """
    Calculate perceptual hash.

    ImageHash.phash internally creates a perceptual
    representation of the image rather than hashing
    the raw file bytes.
    """

    try:

        with Image.open(image_path) as image:

            return imagehash.phash(
                image,
                hash_size=8
            )

    except Exception as error:

        print(
            f"Could not hash: {image_path}"
        )

        print(
            f"Reason: {error}"
        )

        return None


# ============================================================
# 4. Find all product categories
# ============================================================

product_dirs = sorted(
    path
    for path in DATA_DIR.iterdir()
    if path.is_dir()
)


print("=" * 70)
print("3CAD pHash NEAR-DUPLICATE AUDIT")
print("=" * 70)

print(
    f"\nProducts found: "
    f"{len(product_dirs)}"
)

print(
    f"pHash Hamming threshold: "
    f"{PHASH_THRESHOLD}"
)


# ============================================================
# 5. Store suspicious pairs
# ============================================================

suspicious_pairs = []


# ============================================================
# 6. Process one product at a time
# ============================================================

for product_index, product_dir in enumerate(
    product_dirs,
    start=1
):

    product_name = product_dir.name

    print("\n" + "=" * 70)

    print(
        f"[{product_index}/{len(product_dirs)}] "
        f"{product_name}"
    )

    print("=" * 70)


    train_images = get_product_images(
        product_dir,
        "train"
    )

    test_images = get_product_images(
        product_dir,
        "test"
    )


    print(
        f"Train images: "
        f"{len(train_images):,}"
    )

    print(
        f"Test images : "
        f"{len(test_images):,}"
    )


    # ========================================================
    # 7. Calculate train pHashes
    # ========================================================

    print(
        "\nCalculating train pHashes..."
    )

    train_hashes = []

    for index, image_path in enumerate(
        train_images,
        start=1
    ):

        phash = calculate_phash(
            image_path
        )

        if phash is not None:

            train_hashes.append(
                (
                    image_path,
                    phash
                )
            )

        if index % 500 == 0:

            print(
                f"  Train: "
                f"{index:,}/"
                f"{len(train_images):,}"
            )


    # ========================================================
    # 8. Calculate test pHashes
    # ========================================================

    print(
        "Calculating test pHashes..."
    )

    test_hashes = []

    for index, image_path in enumerate(
        test_images,
        start=1
    ):

        phash = calculate_phash(
            image_path
        )

        if phash is not None:

            test_hashes.append(
                (
                    image_path,
                    phash
                )
            )

        if index % 500 == 0:

            print(
                f"  Test : "
                f"{index:,}/"
                f"{len(test_images):,}"
            )


    # ========================================================
    # 9. Compare TRAIN vs TEST
    # ========================================================

    print(
        "\nComparing train ↔ test..."
    )


    product_candidates = 0


    for train_index, (
        train_path,
        train_hash
    ) in enumerate(
        train_hashes,
        start=1
    ):

        for test_path, test_hash in test_hashes:

            # ImageHash subtraction computes
            # Hamming distance.
            distance = (
                train_hash
                -
                test_hash
            )


            if distance <= PHASH_THRESHOLD:

                suspicious_pairs.append(
                    {
                        "product":
                            product_name,

                        "train_image":
                            str(
                                train_path.relative_to(
                                    DATA_DIR
                                )
                            ),

                        "test_image":
                            str(
                                test_path.relative_to(
                                    DATA_DIR
                                )
                            ),

                        "phash_distance":
                            distance,
                    }
                )

                product_candidates += 1


        if train_index % 250 == 0:

            print(
                f"  Compared "
                f"{train_index:,}/"
                f"{len(train_hashes):,} "
                f"train images"
            )


    print(
        f"\nSuspicious pairs found: "
        f"{product_candidates:,}"
    )


# ============================================================
# 10. Sort by similarity
# ============================================================

suspicious_pairs.sort(
    key=lambda item:
    item["phash_distance"]
)


# ============================================================
# 11. Save report to CSV
# ============================================================

report_path = (
    REPORT_DIR
    /
    "near_duplicates_phash.csv"
)


with open(
    report_path,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    fieldnames = [
        "product",
        "train_image",
        "test_image",
        "phash_distance",
    ]


    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames
    )


    writer.writeheader()

    writer.writerows(
        suspicious_pairs
    )


# ============================================================
# 12. Distance distribution
# ============================================================

distance_counts = defaultdict(int)


for pair in suspicious_pairs:

    distance_counts[
        pair["phash_distance"]
    ] += 1


# ============================================================
# 13. Final results
# ============================================================

print("\n" + "=" * 70)
print("NEAR-DUPLICATE RESULTS")
print("=" * 70)


print(
    f"\nTotal suspicious pairs: "
    f"{len(suspicious_pairs):,}"
)


print(
    "\nHamming distance distribution:"
)


for distance in sorted(
    distance_counts
):

    print(
        f"Distance {distance}: "
        f"{distance_counts[distance]:,}"
    )


print(
    "\nMost suspicious examples:"
)


for pair in suspicious_pairs[:20]:

    print(
        "\n"
        f"Product : {pair['product']}\n"
        f"Distance: {pair['phash_distance']}\n"
        f"Train   : {pair['train_image']}\n"
        f"Test    : {pair['test_image']}"
    )


print(
    "\nReport saved to:"
)

print(
    report_path
)


print(
    "\nNear-duplicate audit complete."
)