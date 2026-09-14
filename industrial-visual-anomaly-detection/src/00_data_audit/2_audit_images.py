from pathlib import Path
from collections import Counter, defaultdict
import cv2


# ============================================================
# 1. Project paths
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
# 2. Helper functions
# ============================================================

def is_image_file(path: Path) -> bool:
    """
    Check whether a file has a supported image extension.
    """
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def find_images(folder: Path):
    """
    Recursively find all image files inside a folder.
    """
    if not folder.exists():
        return []

    return [
        file
        for file in folder.rglob("*")
        if is_image_file(file)
    ]


def detect_split(image_path: Path) -> str:
    """
    Identify whether an image belongs to:
    - train
    - test
    - ground_truth
    - unknown
    """
    parts = image_path.parts

    if "train" in parts:
        return "train"

    if "test" in parts:
        return "test"

    if "ground_truth" in parts:
        return "ground_truth"

    return "unknown"


# ============================================================
# 3. Validate dataset path
# ============================================================

print("=" * 70)
print("3CAD IMAGE DATASET AUDIT")
print("=" * 70)

print(f"\nProject root : {PROJECT_ROOT}")
print(f"Dataset path : {DATA_DIR}")
print(f"Dataset exists: {DATA_DIR.exists()}")

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Dataset folder not found:\n{DATA_DIR}"
    )


# ============================================================
# 4. Find every image file
# ============================================================

all_images = find_images(DATA_DIR)

print(f"\nTotal image files found: {len(all_images)}")


# ============================================================
# 5. Counters
# ============================================================

corrupted_files = []

resolution_counts = Counter()
channel_counts = Counter()
extension_counts = Counter()
split_counts = Counter()

product_counts = defaultdict(Counter)


# ============================================================
# 6. Audit each image
# ============================================================

for index, image_path in enumerate(all_images, start=1):

    # --------------------------------------------------------
    # Metadata that does not require decoding the image
    # --------------------------------------------------------

    extension_counts[image_path.suffix.lower()] += 1

    split = detect_split(image_path)

    split_counts[split] += 1

    # First folder after "3CAD" is the product category
    try:
        relative_path = image_path.relative_to(DATA_DIR)
        product_name = relative_path.parts[0]
        product_counts[product_name][split] += 1
    except Exception:
        product_name = "unknown"

    # --------------------------------------------------------
    # Try reading the actual image
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )

    # cv2 returns None if it cannot decode the image
    if image is None:
        corrupted_files.append(image_path)
        continue

    # --------------------------------------------------------
    # Resolution
    # --------------------------------------------------------

    height, width = image.shape[:2]

    resolution_counts[(width, height)] += 1

    # --------------------------------------------------------
    # Number of channels
    # --------------------------------------------------------

    if image.ndim == 2:
        channels = 1
    elif image.ndim == 3:
        channels = image.shape[2]
    else:
        channels = -1

    channel_counts[channels] += 1

    # --------------------------------------------------------
    # Progress reporting
    # --------------------------------------------------------

    if index % 5000 == 0:
        print(
            f"Checked {index:,}/{len(all_images):,} files..."
        )


# ============================================================
# 7. Results
# ============================================================

print("\n" + "=" * 70)
print("AUDIT RESULTS")
print("=" * 70)


# ------------------------------------------------------------
# Corrupted images
# ------------------------------------------------------------

print("\n[1] Corrupted / unreadable images")
print("-" * 70)

print(f"Corrupted images: {len(corrupted_files)}")

if corrupted_files:
    for file in corrupted_files[:20]:
        print(file)

    if len(corrupted_files) > 20:
        print(
            f"... and {len(corrupted_files) - 20} more"
        )


# ------------------------------------------------------------
# Split counts
# ------------------------------------------------------------

print("\n[2] Images by dataset split")
print("-" * 70)

for split, count in split_counts.items():
    print(f"{split:<15}: {count:,}")


# ------------------------------------------------------------
# Product counts
# ------------------------------------------------------------

print("\n[3] Images by product category")
print("-" * 70)

for product_name in sorted(product_counts):
    counts = product_counts[product_name]

    print(f"\n{product_name}")

    for split in ["train", "test", "ground_truth"]:
        print(
            f"  {split:<15}: "
            f"{counts.get(split, 0):,}"
        )


# ------------------------------------------------------------
# File extensions
# ------------------------------------------------------------

print("\n[4] Image file extensions")
print("-" * 70)

for extension, count in extension_counts.most_common():
    print(f"{extension:<10}: {count:,}")


# ------------------------------------------------------------
# Resolutions
# ------------------------------------------------------------

print("\n[5] Most common image resolutions")
print("-" * 70)

print(
    f"Number of unique resolutions: "
    f"{len(resolution_counts)}"
)

for resolution, count in resolution_counts.most_common(20):
    width, height = resolution

    print(
        f"{width} x {height:<8}: "
        f"{count:,}"
    )


# ------------------------------------------------------------
# Channels
# ------------------------------------------------------------

print("\n[6] Image channels")
print("-" * 70)

for channels, count in sorted(channel_counts.items()):
    if channels == 1:
        description = "Grayscale"
    elif channels == 3:
        description = "RGB/BGR"
    elif channels == 4:
        description = "RGBA/BGRA"
    else:
        description = "Other"

    print(
        f"{channels} channel(s) "
        f"({description}): {count:,}"
    )


# ============================================================
# 8. Final summary
# ============================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total files      : {len(all_images):,}")
print(f"Readable images  : {len(all_images) - len(corrupted_files):,}")
print(f"Corrupted images : {len(corrupted_files):,}")
print(f"Unique resolutions: {len(resolution_counts):,}")

print("\nAudit complete.")