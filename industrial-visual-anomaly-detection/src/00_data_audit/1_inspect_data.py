from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "3CAD"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0

    return sum(
        1
        for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


categories = sorted(
    folder
    for folder in DATA_DIR.iterdir()
    if folder.is_dir()
)

print("Number of product categories:", len(categories))

for category in categories:
    print("\n" + "=" * 60)
    print("PRODUCT:", category.name)
    print("=" * 60)

    train_dir = category / "train"
    test_dir = category / "test"
    ground_truth_dir = category / "ground_truth"

    train_good = train_dir / "good"
    test_good = test_dir / "good"

    print("Train good:", count_images(train_good))
    print("Test good :", count_images(test_good))

    defect_total = 0

    print("\nDefect types:")

    for defect_dir in sorted(test_dir.iterdir()):
        if not defect_dir.is_dir():
            continue

        if defect_dir.name == "good":
            continue

        defect_count = count_images(defect_dir)
        defect_total += defect_count

        print(
            f"{defect_dir.name:<20}"
            f"images={defect_count:<5}"
        )

    print("\nTotal test defects:", defect_total)

    print(
        "Ground-truth masks:",
        count_images(ground_truth_dir)
    )