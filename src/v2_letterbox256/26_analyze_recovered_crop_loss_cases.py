from pathlib import Path
import csv
from collections import defaultdict


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

CROP_AUDIT_PATH = (
    REPORT_DIR
    / "crop_loss_audit.csv"
)

V2_SCORES_PATH = (
    REPORT_DIR
    / "v2_letterbox256_image_scores.csv"
)


# ============================================================
# 2. V2 threshold
# ============================================================

V2_THRESHOLD = 0.533522


# ============================================================
# 3. Load fully-lost V1 samples
# ============================================================

fully_lost_samples = set()

with open(
    CROP_AUDIT_PATH,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        fully_lost_text = (
            row["fully_lost"]
            .strip()
            .lower()
        )

        fully_lost = (
            fully_lost_text
            in {
                "true",
                "1",
                "yes",
            }
        )

        if fully_lost:

            defect_type = (
                row["defect_type"]
            )

            image_name = (
                row["image_name"]
            )

            fully_lost_samples.add(
                (
                    defect_type,
                    image_name,
                )
            )


print(
    "Fully lost V1 samples:",
    len(fully_lost_samples)
)


# ============================================================
# 4. Load V2 scores
# ============================================================

matched_results = []

with open(
    V2_SCORES_PATH,
    "r",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        defect_type = (
            row["defect_type"]
        )

        image_path = Path(
            row["image_path"]
        )

        image_name = (
            image_path.name
        )

        key = (
            defect_type,
            image_name,
        )

        if key not in fully_lost_samples:
            continue

        score = float(
            row["score"]
        )

        predicted_defect = (
            score >= V2_THRESHOLD
        )

        matched_results.append(
            {
                "defect_type":
                    defect_type,

                "image_name":
                    image_name,

                "score":
                    score,

                "predicted_defect":
                    predicted_defect,
            }
        )


# ============================================================
# 5. Overall recovery
# ============================================================

total = len(
    matched_results
)

tp = sum(
    result["predicted_defect"]
    for result in matched_results
)

fn = (
    total
    -
    tp
)

recall = (
    tp / total
    if total > 0
    else 0.0
)


print(
    "\n"
    +
    "=" * 70
)

print(
    "V2 RECOVERY OF V1 FULLY-CROPPED DEFECTS"
)

print(
    "=" * 70
)


print(
    f"\nV2 threshold: "
    f"{V2_THRESHOLD:.6f}"
)

print(
    f"Matched samples: "
    f"{total}"
)

print(
    f"Detected by V2 (TP): "
    f"{tp}"
)

print(
    f"Still missed by V2 (FN): "
    f"{fn}"
)

print(
    f"Recall on these cases: "
    f"{recall:.2%}"
)


# ============================================================
# 6. Per defect type
# ============================================================

stats = defaultdict(
    lambda: {
        "total": 0,
        "tp": 0,
        "fn": 0,
    }
)


for result in matched_results:

    defect_type = (
        result["defect_type"]
    )

    stats[
        defect_type
    ]["total"] += 1

    if result[
        "predicted_defect"
    ]:

        stats[
            defect_type
        ]["tp"] += 1

    else:

        stats[
            defect_type
        ]["fn"] += 1


print(
    "\nPer-defect-type:"
)


for defect_type in sorted(
    stats.keys()
):

    item = stats[
        defect_type
    ]

    type_recall = (
        item["tp"]
        /
        item["total"]
        if item["total"] > 0
        else 0.0
    )

    print(
        f"{defect_type:15s}"
        f" | n={item['total']:3d}"
        f" | TP={item['tp']:3d}"
        f" | FN={item['fn']:3d}"
        f" | recall="
        f"{type_recall:.2%}"
    )


# ============================================================
# 7. Lowest-scoring missed samples
# ============================================================

missed = [
    result
    for result in matched_results
    if not result[
        "predicted_defect"
    ]
]

missed.sort(
    key=lambda x: x["score"]
)


print(
    "\nLowest-scoring V2 misses:"
)


for result in missed[:20]:

    print(
        f"{result['score']:.4f}"
        f" | "
        f"{result['defect_type']}"
        f" | "
        f"{result['image_name']}"
    )