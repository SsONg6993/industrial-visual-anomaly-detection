from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "linkedin"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Experimental Results
# ============================================================

versions = [
    "V1\nCenterCrop224",
    "V2a\nLetterbox256",
    "V2c\nFull32",
]

auroc = [
    0.9210,
    0.8991,
    0.9166,
]

recall = [
    0.9294,
    0.8793,
    0.9304,
]

f1 = [
    0.9104,
    0.8989,
    0.9064,
]

false_positive = [
    121,
    83,
    132,
]

false_negative = [
    76,
    130,
    75,
]


# ============================================================
# 3. Plot 1
#    Model Performance Comparison
# ============================================================

x = np.arange(
    len(versions)
)

width = 0.24


fig, ax = plt.subplots(
    figsize=(10, 6)
)


bars1 = ax.bar(
    x - width,
    auroc,
    width,
    label="AUROC",
)

bars2 = ax.bar(
    x,
    recall,
    width,
    label="Recall",
)

bars3 = ax.bar(
    x + width,
    f1,
    width,
    label="F1 Score",
)


ax.set_title(
    "Industrial Anomaly Detection Performance Comparison",
    fontsize=16,
    fontweight="bold",
)

ax.set_ylabel(
    "Score"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    versions
)

ax.set_ylim(
    0.80,
    0.95
)

ax.legend()


# Add values above bars

for bars in [
    bars1,
    bars2,
    bars3,
]:

    for bar in bars:

        height = (
            bar.get_height()
        )

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,

            height + 0.003,

            f"{height:.4f}",

            ha="center",
            va="bottom",
            fontsize=9,
        )


fig.tight_layout()


output_path = (
    OUTPUT_DIR
    /
    "model_performance_comparison.png"
)


fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


print(
    "Saved:",
    output_path
)


# ============================================================
# 4. Plot 2
#    FP vs FN
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 6)
)


bars1 = ax.bar(
    x - width / 2,
    false_positive,
    width,
    label="False Positive (Overkill)",
)

bars2 = ax.bar(
    x + width / 2,
    false_negative,
    width,
    label="False Negative (Underkill)",
)


ax.set_title(
    "Overkill vs Underkill Trade-off",
    fontsize=16,
    fontweight="bold",
)

ax.set_ylabel(
    "Number of Samples"
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    versions
)

ax.legend()


for bars in [
    bars1,
    bars2,
]:

    for bar in bars:

        height = (
            bar.get_height()
        )

        ax.text(
            bar.get_x()
            +
            bar.get_width() / 2,

            height + 3,

            f"{int(height)}",

            ha="center",
            va="bottom",
            fontsize=10,
        )


ax.text(
    0.5,
    -0.18,
    (
        "False Negative = defective unit classified as good "
        "(Underkill / defect escape)\n"
        "False Positive = good unit classified as defective "
        "(Overkill / unnecessary rejection)"
    ),
    transform=ax.transAxes,
    ha="center",
    fontsize=9,
)


fig.tight_layout()


output_path = (
    OUTPUT_DIR
    /
    "fp_fn_tradeoff.png"
)


fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


print(
    "Saved:",
    output_path
)


# ============================================================
# 5. Plot 3
#    Memory Bank Reduction
# ============================================================

memory_labels = [
    "Full Memory Bank",
    "10% Greedy Coreset",
]

memory_mb = [
    1089.84,
    108.98,
]


fig, ax = plt.subplots(
    figsize=(8, 6)
)


bars = ax.bar(
    memory_labels,
    memory_mb,
)


ax.set_title(
    "Memory Bank Compression",
    fontsize=16,
    fontweight="bold",
)

ax.set_ylabel(
    "Memory Size (MB)"
)


for bar in bars:

    height = (
        bar.get_height()
    )

    ax.text(
        bar.get_x()
        +
        bar.get_width() / 2,

        height + 20,

        f"{height:.2f} MB",

        ha="center",
        va="bottom",
        fontsize=10,
    )


ax.text(
    0.5,
    0.80,
    "90% memory reduction",
    transform=ax.transAxes,
    ha="center",
    fontsize=14,
    fontweight="bold",
)


fig.tight_layout()


output_path = (
    OUTPUT_DIR
    /
    "memory_bank_reduction.png"
)


fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight",
)


plt.close(
    fig
)


print(
    "Saved:",
    output_path
)


# ============================================================
# 6. Print Summary
# ============================================================

print(
    "\n"
    +
    "=" * 65
)

print(
    "LINKEDIN RESULT SUMMARY"
)

print(
    "=" * 65
)


print(
    "\nV1 CenterCrop224"
)

print(
    "AUROC  : 0.9210"
)

print(
    "Recall : 0.9294"
)

print(
    "F1     : 0.9104"
)

print(
    "FP     : 121"
)

print(
    "FN     : 76"
)


print(
    "\nV2a Letterbox256"
)

print(
    "AUROC  : 0.8991"
)

print(
    "Recall : 0.8793"
)

print(
    "F1     : 0.8989"
)

print(
    "FP     : 83"
)

print(
    "FN     : 130"
)


print(
    "\nV2c Full32"
)

print(
    "AUROC  : 0.9166"
)

print(
    "Recall : 0.9304"
)

print(
    "F1     : 0.9064"
)

print(
    "FP     : 132"
)

print(
    "FN     : 75"
)


print(
    "\nMemory Bank:"
)

print(
    "744,000 x 384"
)

print(
    "1089.84 MB"
)


print(
    "\nGreedy 10% Coreset:"
)

print(
    "74,400 x 384"
)

print(
    "108.98 MB"
)

print(
    "Reduction: 90%"
)


print(
    "\nImages saved to:"
)

print(
    OUTPUT_DIR
)