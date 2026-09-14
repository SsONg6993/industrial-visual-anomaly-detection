from pathlib import Path
from collections import defaultdict
import hashlib


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "3CAD"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# 2. Find product images only
# ============================================================

def find_product_images(folder: Path):
    """
    Find all train/test product images.

    Ground-truth masks are excluded because they are annotations,
    not original product images.
    """

    images = []

    for path in folder.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if "ground_truth" in path.parts:
            continue

        images.append(path)

    return images


# ============================================================
# 3. File hashing
# ============================================================

def calculate_sha256(file_path: Path) -> str:
    """
    Calculate the SHA-256 fingerprint of a file.

    If two files have exactly the same bytes,
    their SHA-256 values should be the same.
    """

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:

        while True:
            chunk = file.read(1024 * 1024)  # Read 1 MB at a time

            if not chunk:
                break

            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 4. Extract metadata from path
# ============================================================

def get_split(path: Path) -> str:

    if "train" in path.parts:
        return "train"

    if "test" in path.parts:
        return "test"

    return "unknown"


def get_product(path: Path) -> str:
    """
    First directory after 3CAD is the product category.
    """

    relative_path = path.relative_to(DATA_DIR)

    return relative_path.parts[0]


# ============================================================
# 5. Start audit
# ============================================================

print("=" * 70)
print("3CAD EXACT DUPLICATE AUDIT")
print("=" * 70)

images = find_product_images(DATA_DIR)

print(f"\nProduct images found: {len(images):,}")


# hash -> list of image paths
hash_to_files = defaultdict(list)


for index, image_path in enumerate(images, start=1):

    file_hash = calculate_sha256(image_path)

    hash_to_files[file_hash].append(image_path)

    if index % 5000 == 0:
        print(
            f"Hashed {index:,}/{len(images):,} images..."
        )


# ============================================================
# 6. Find exact duplicate groups
# ============================================================

duplicate_groups = {
    file_hash: paths
    for file_hash, paths in hash_to_files.items()
    if len(paths) > 1
}


print("\n" + "=" * 70)
print("DUPLICATE RESULTS")
print("=" * 70)

print(
    f"\nExact duplicate groups: "
    f"{len(duplicate_groups):,}"
)


# ============================================================
# 7. Train-test leakage
# ============================================================

train_test_leakage = []

for file_hash, paths in duplicate_groups.items():

    splits = {
        get_split(path)
        for path in paths
    }

    if "train" in splits and "test" in splits:
        train_test_leakage.append(paths)


print(
    f"Train-test leakage groups: "
    f"{len(train_test_leakage):,}"
)


# ============================================================
# 8. Cross-product duplicates
# ============================================================

cross_product_duplicates = []

for file_hash, paths in duplicate_groups.items():

    products = {
        get_product(path)
        for path in paths
    }

    if len(products) > 1:
        cross_product_duplicates.append(paths)


print(
    f"Cross-product duplicate groups: "
    f"{len(cross_product_duplicates):,}"
)


# ============================================================
# 9. Print leakage examples
# ============================================================

if train_test_leakage:

    print("\n" + "=" * 70)
    print("TRAIN-TEST LEAKAGE EXAMPLES")
    print("=" * 70)

    for group_index, group in enumerate(
        train_test_leakage[:10],
        start=1
    ):

        print(f"\nDuplicate group {group_index}")

        for path in group:

            relative_path = path.relative_to(DATA_DIR)

            print(
                f"[{get_split(path):5}] "
                f"{relative_path}"
            )


# ============================================================
# 10. Print ordinary duplicate examples
# ============================================================

if duplicate_groups:

    print("\n" + "=" * 70)
    print("SAMPLE DUPLICATE GROUPS")
    print("=" * 70)

    for group_index, paths in enumerate(
        list(duplicate_groups.values())[:10],
        start=1
    ):

        print(f"\nDuplicate group {group_index}")

        for path in paths:

            relative_path = path.relative_to(DATA_DIR)

            print(relative_path)


# ============================================================
# 11. Final summary
# ============================================================

duplicate_file_count = sum(
    len(paths)
    for paths in duplicate_groups.values()
)

unique_hashes = len(hash_to_files)

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total product images        : {len(images):,}")
print(f"Unique file hashes          : {unique_hashes:,}")
print(f"Exact duplicate groups      : {len(duplicate_groups):,}")
print(f"Files inside duplicate groups: {duplicate_file_count:,}")
print(f"Train-test leakage groups   : {len(train_test_leakage):,}")
print(f"Cross-product duplicate groups: {len(cross_product_duplicates):,}")

print("\nExact duplicate audit complete.")