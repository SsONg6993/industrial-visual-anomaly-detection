from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import argparse
import csv
import hashlib
import json
import random


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CSV_FIELDS = [
    "image_path",
    "label",
    "defect_type",
    "mask_path",
    "sha256",
    "source_split",
]


def calculate_sha256(file_path: Path) -> str:
    sha256 = hashlib.sha256()
    with file_path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


def to_project_relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(
        path
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def make_record(
    image_path: Path,
    *,
    project_root: Path,
    source_split: str,
    defect_type: str,
    file_hash: str,
    ground_truth_dir: Path,
) -> dict[str, str | int]:
    label = 0 if defect_type == "good" else 1

    mask_path = ""
    if label == 1:
        candidate = ground_truth_dir / defect_type / image_path.name
        if not candidate.exists():
            raise FileNotFoundError(
                "Ground-truth mask not found for defect image:\n"
                f"  image: {image_path}\n"
                f"  expected mask: {candidate}"
            )
        mask_path = to_project_relative(candidate, project_root)

    return {
        "image_path": to_project_relative(image_path, project_root),
        "label": label,
        "defect_type": defect_type,
        "mask_path": mask_path,
        "sha256": file_hash,
        "source_split": source_split,
    }


def stratified_val_test_split(
    records: list[dict[str, str | int]],
    *,
    validation_fraction: float,
    random_seed: int,
) -> tuple[list[dict[str, str | int]], list[dict[str, str | int]]]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    grouped: dict[str, list[dict[str, str | int]]] = defaultdict(list)
    for record in records:
        grouped[str(record["defect_type"])].append(record)

    rng = random.Random(random_seed)
    validation_records: list[dict[str, str | int]] = []
    test_records: list[dict[str, str | int]] = []

    for defect_type in sorted(grouped):
        group = sorted(grouped[defect_type], key=lambda row: str(row["image_path"]))
        rng.shuffle(group)

        n = len(group)
        if n == 1:
            # Prefer the final untouched test set for a singleton class.
            n_validation = 0
        else:
            n_validation = int(round(n * validation_fraction))
            n_validation = max(1, min(n - 1, n_validation))

        validation_records.extend(group[:n_validation])
        test_records.extend(group[n_validation:])

    validation_records.sort(key=lambda row: str(row["image_path"]))
    test_records.sort(key=lambda row: str(row["image_path"]))

    return validation_records, test_records


def write_manifest(path: Path, records: list[dict[str, str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(records)


def count_by_defect_type(records: list[dict[str, str | int]]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        counts[str(record["defect_type"])] += 1
    return dict(sorted(counts.items()))


def verify_zero_cross_split_overlap(
    train_records: list[dict[str, str | int]],
    validation_records: list[dict[str, str | int]],
    test_records: list[dict[str, str | int]],
) -> dict[str, object]:
    train_hashes = {str(row["sha256"]) for row in train_records}
    validation_hashes = {str(row["sha256"]) for row in validation_records}
    test_hashes = {str(row["sha256"]) for row in test_records}

    overlaps = {
        "train_val": sorted(train_hashes & validation_hashes),
        "train_test": sorted(train_hashes & test_hashes),
        "val_test": sorted(validation_hashes & test_hashes),
    }

    zero_overlap = all(len(values) == 0 for values in overlaps.values())
    if not zero_overlap:
        raise RuntimeError(f"Exact-hash leakage remains across clean splits: {overlaps}")

    return {
        "zero_cross_split_hash_overlap": True,
        "overlap_counts": {name: len(values) for name, values in overlaps.items()},
    }


def build_clean_split(
    *,
    product_dir: Path,
    project_root: Path,
    output_dir: Path,
    validation_fraction: float = 0.50,
    random_seed: int = 42,
) -> dict[str, object]:
    product_dir = product_dir.resolve()
    project_root = project_root.resolve()
    output_dir = output_dir.resolve()

    train_good_dir = product_dir / "train" / "good"
    test_dir = product_dir / "test"
    ground_truth_dir = product_dir / "ground_truth"

    if not train_good_dir.exists():
        raise FileNotFoundError(f"Training directory not found: {train_good_dir}")
    if not test_dir.exists():
        raise FileNotFoundError(f"Test directory not found: {test_dir}")

    train_images = list_images(train_good_dir)
    test_images = list_images(test_dir)

    if not train_images:
        raise RuntimeError(f"No training images found in: {train_good_dir}")
    if not test_images:
        raise RuntimeError(f"No test images found in: {test_dir}")

    # ------------------------------------------------------------------
    # 1. Hash training images.
    #    If train itself contains exact duplicates, keep one canonical
    #    representative per hash because repeated references do not add
    #    new information to the clean benchmark.
    # ------------------------------------------------------------------
    train_hash_to_paths: dict[str, list[Path]] = defaultdict(list)
    for image_path in train_images:
        train_hash_to_paths[calculate_sha256(image_path)].append(image_path)

    train_records: list[dict[str, str | int]] = []
    train_internal_duplicates_removed = 0

    for file_hash in sorted(train_hash_to_paths):
        paths = sorted(train_hash_to_paths[file_hash])
        canonical = paths[0]
        train_internal_duplicates_removed += len(paths) - 1

        train_records.append(
            make_record(
                canonical,
                project_root=project_root,
                source_split="train",
                defect_type="good",
                file_hash=file_hash,
                ground_truth_dir=ground_truth_dir,
            )
        )

    train_hashes = {str(row["sha256"]) for row in train_records}

    # ------------------------------------------------------------------
    # 2. Hash test images.
    #    Any hash already present in train is excluded from clean eval.
    #    For duplicate groups contained only within test, keep one
    #    deterministic canonical path.
    # ------------------------------------------------------------------
    test_hash_to_paths: dict[str, list[Path]] = defaultdict(list)
    for image_path in test_images:
        test_hash_to_paths[calculate_sha256(image_path)].append(image_path)

    leakage_removed = 0
    test_internal_duplicates_removed = 0
    unique_test_records: list[dict[str, str | int]] = []

    leakage_examples: list[dict[str, object]] = []
    duplicate_examples: list[dict[str, object]] = []

    for file_hash in sorted(test_hash_to_paths):
        paths = sorted(test_hash_to_paths[file_hash])

        if file_hash in train_hashes:
            leakage_removed += len(paths)
            if len(leakage_examples) < 10:
                leakage_examples.append(
                    {
                        "sha256": file_hash,
                        "excluded_test_images": [
                            to_project_relative(path, project_root) for path in paths
                        ],
                    }
                )
            continue

        canonical = paths[0]
        test_internal_duplicates_removed += len(paths) - 1

        if len(paths) > 1 and len(duplicate_examples) < 10:
            duplicate_examples.append(
                {
                    "sha256": file_hash,
                    "kept": to_project_relative(canonical, project_root),
                    "excluded_duplicates": [
                        to_project_relative(path, project_root) for path in paths[1:]
                    ],
                }
            )

        # 3CAD test layout: test/<defect_type>/<image>
        defect_type = canonical.parent.name

        unique_test_records.append(
            make_record(
                canonical,
                project_root=project_root,
                source_split="test",
                defect_type=defect_type,
                file_hash=file_hash,
                ground_truth_dir=ground_truth_dir,
            )
        )

    # ------------------------------------------------------------------
    # 3. Split the cleaned, unique test pool into validation + final test.
    # ------------------------------------------------------------------
    validation_records, test_records = stratified_val_test_split(
        unique_test_records,
        validation_fraction=validation_fraction,
        random_seed=random_seed,
    )

    train_records.sort(key=lambda row: str(row["image_path"]))

    verification = verify_zero_cross_split_overlap(
        train_records,
        validation_records,
        test_records,
    )

    # ------------------------------------------------------------------
    # 4. Write manifests and machine-readable summary.
    # ------------------------------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(output_dir / "train_clean.csv", train_records)
    write_manifest(output_dir / "val_clean.csv", validation_records)
    write_manifest(output_dir / "test_clean.csv", test_records)

    summary: dict[str, object] = {
        "product": product_dir.name,
        "random_seed": random_seed,
        "validation_fraction": validation_fraction,
        "raw_counts": {
            "train_good_images": len(train_images),
            "test_images": len(test_images),
        },
        "deduplication": {
            "unique_train_hashes": len(train_records),
            "train_internal_duplicates_removed": train_internal_duplicates_removed,
            "test_images_excluded_due_to_train_test_leakage": leakage_removed,
            "test_internal_duplicates_removed": test_internal_duplicates_removed,
            "unique_clean_test_pool_before_val_test_split": len(unique_test_records),
        },
        "split_counts": {
            "train": len(train_records),
            "validation": len(validation_records),
            "test": len(test_records),
        },
        "split_counts_by_defect_type": {
            "train": count_by_defect_type(train_records),
            "validation": count_by_defect_type(validation_records),
            "test": count_by_defect_type(test_records),
        },
        "verification": verification,
        "examples": {
            "train_test_leakage_removed": leakage_examples,
            "test_internal_duplicates_removed": duplicate_examples,
        },
        "notes": [
            "Raw 3CAD files are never deleted or moved; only manifest membership changes.",
            "Exact duplicate detection uses SHA-256 file hashes.",
            "Near-duplicate/pHash candidates are intentionally not removed automatically.",
            "Training remains one-class: only good training samples are used.",
            "Validation is intended for threshold selection; final test should remain untouched until final evaluation.",
        ],
    }

    summary_path = output_dir / "clean_split_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return summary


def print_summary(summary: dict[str, object], output_dir: Path) -> None:
    split_counts = summary["split_counts"]
    dedup = summary["deduplication"]

    print("=" * 72)
    print("CLEAN LEAKAGE-FREE SPLIT BUILDER")
    print("=" * 72)
    print(f"Product: {summary['product']}")
    print(f"Validation fraction: {summary['validation_fraction']}")
    print(f"Random seed: {summary['random_seed']}")
    print()
    print("Deduplication")
    print(f"  Train internal duplicates removed: {dedup['train_internal_duplicates_removed']}")
    print(f"  Test images removed due to train-test leakage: {dedup['test_images_excluded_due_to_train_test_leakage']}")
    print(f"  Test internal duplicates removed: {dedup['test_internal_duplicates_removed']}")
    print()
    print("Clean split counts")
    print(f"  Train:      {split_counts['train']}")
    print(f"  Validation: {split_counts['validation']}")
    print(f"  Test:       {split_counts['test']}")
    print()
    print("Verification")
    print("  Exact hash overlap across train / val / test: 0")
    print()
    print(f"Manifests written to: {output_dir}")
    print("  - train_clean.csv")
    print("  - val_clean.csv")
    print("  - test_clean.csv")
    print("  - clean_split_summary.json")


def parse_args() -> argparse.Namespace:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]

    default_product_dir = (
        project_root
        / "data"
        / "3CAD"
        / "Aluminum_Camera_Cover"
    )

    default_output_dir = (
        project_root
        / "reports"
        / "clean_evaluation"
    )

    parser = argparse.ArgumentParser(
        description=(
            "Build deterministic, exact-hash-clean train/validation/test manifests "
            "for the Aluminum Camera Cover 3CAD benchmark."
        )
    )
    parser.add_argument(
        "--product-dir",
        type=Path,
        default=default_product_dir,
        help=f"3CAD product directory (default: {default_product_dir})",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=project_root,
        help=f"Project root used for relative manifest paths (default: {project_root})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help=f"Directory for CSV/JSON manifests (default: {default_output_dir})",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.50,
        help="Fraction of each cleaned test defect type assigned to validation (default: 0.50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic stratified validation/test split (default: 42).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    summary = build_clean_split(
        product_dir=args.product_dir,
        project_root=args.project_root,
        output_dir=args.output_dir,
        validation_fraction=args.validation_fraction,
        random_seed=args.seed,
    )

    print_summary(summary, args.output_dir)


if __name__ == "__main__":
    main()
