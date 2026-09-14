from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[2]
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

INPUT_PATH = (
    MODEL_DIR
    / "aluminum_camera_cover_letterbox256_full32_memory_bank.pt"
)

OUTPUT_PATH = (
    MODEL_DIR
    / "aluminum_camera_cover_letterbox256_full32_greedy10_coreset.pt"
)


# ============================================================
# 2. Configuration
# ============================================================

CORESET_RATIO = 0.10

PROJECTION_DIM = 64

RANDOM_SEED = 42


# ============================================================
# 3. Load memory bank
# ============================================================

checkpoint = torch.load(
    INPUT_PATH,
    map_location="cpu",
    weights_only=True,
)

memory_bank = (
    checkpoint["memory_bank"]
    .float()
)


print(
    "Loaded memory bank:",
    tuple(memory_bank.shape)
)


num_features = (
    memory_bank.shape[0]
)

feature_dim = (
    memory_bank.shape[1]
)


target_size = max(
    1,
    int(round(
        num_features
        *
        CORESET_RATIO
    ))
)


print(
    f"Coreset ratio: "
    f"{CORESET_RATIO:.0%}"
)

print(
    f"Target coreset size: "
    f"{target_size:,}"
)


# ============================================================
# 4. L2 normalize
# ============================================================

memory_bank = F.normalize(
    memory_bank,
    p=2,
    dim=1
)


# ============================================================
# 5. Random projection
# ============================================================

torch.manual_seed(
    RANDOM_SEED
)


projection_matrix = torch.randn(
    feature_dim,
    PROJECTION_DIM,
    dtype=torch.float32,
)


projection_matrix = (
    projection_matrix
    /
    np.sqrt(
        PROJECTION_DIM
    )
)


print(
    "\nProjecting features..."
)


projection_start = (
    time.perf_counter()
)


projected = (
    memory_bank
    @
    projection_matrix
)


projected = F.normalize(
    projected,
    p=2,
    dim=1
)


projection_time = (
    time.perf_counter()
    -
    projection_start
)


print(
    "Projected shape:",
    tuple(projected.shape)
)

print(
    f"Projection time: "
    f"{projection_time:.2f}s"
)


# ============================================================
# 6. Traditional greedy farthest-first selection
# ============================================================

print(
    "\nStarting traditional greedy selection..."
)

print(
    "WARNING:"
)

print(
    f"This will perform "
    f"{target_size:,} greedy selections "
    f"over {num_features:,} projected features."
)

print()


start_time = (
    time.perf_counter()
)


# ------------------------------------------------------------
# Deterministic starting point
# ------------------------------------------------------------

selected_indices = [
    0
]


first_feature = (
    projected[
        0:1
    ]
)


# ------------------------------------------------------------
# max_similarity[i]
#
# Highest similarity of feature i
# to any currently selected coreset feature.
#
# Low value:
# poorly represented
#
# Therefore:
# next selected point =
# argmin(max_similarity)
# ------------------------------------------------------------

max_similarity = (
    projected
    @
    first_feature.T
).squeeze(1)


#
# Prevent first point being selected again.
#

max_similarity[
    0
] = 1.0


# ============================================================
# 7. Greedy loop
# ============================================================

for step in range(
    1,
    target_size
):

    # --------------------------------------------------------
    # Pick the point that is currently
    # least represented by selected features.
    # --------------------------------------------------------

    next_index = int(
        torch.argmin(
            max_similarity
        )
    )


    selected_indices.append(
        next_index
    )


    # --------------------------------------------------------
    # New selected representative
    # --------------------------------------------------------

    new_feature = (
        projected[
            next_index:
            next_index + 1
        ]
    )


    # --------------------------------------------------------
    # Similarity of every projected feature
    # to the new representative
    # --------------------------------------------------------

    similarity_to_new = (
        projected
        @
        new_feature.T
    ).squeeze(1)


    # --------------------------------------------------------
    # Update coverage:
    #
    # each point keeps its highest similarity
    # to ANY selected representative
    # --------------------------------------------------------

    max_similarity = torch.maximum(
        max_similarity,
        similarity_to_new
    )


    # --------------------------------------------------------
    # Prevent already-selected samples
    # from being selected again.
    # --------------------------------------------------------

    max_similarity[
        selected_indices
    ] = 1.0


    # --------------------------------------------------------
    # Progress logging
    # --------------------------------------------------------

    if (
        (step + 1) % 100 == 0
        or
        step + 1 == target_size
    ):

        elapsed = (
            time.perf_counter()
            -
            start_time
        )


        average_time_per_step = (
            elapsed
            /
            (step + 1)
        )


        remaining_steps = (
            target_size
            -
            (step + 1)
        )


        estimated_remaining = (
            average_time_per_step
            *
            remaining_steps
        )


        print(
            f"Selected "
            f"{step + 1:,}/"
            f"{target_size:,}"
            f" | elapsed="
            f"{elapsed:.1f}s"
            f" | ETA="
            f"{estimated_remaining / 60:.1f} min"
        )


# ============================================================
# 8. Build final coreset
# ============================================================

selected_indices_tensor = (
    torch.tensor(
        selected_indices,
        dtype=torch.long
    )
)


coreset = (
    memory_bank[
        selected_indices_tensor
    ]
)


# ============================================================
# 9. Statistics
# ============================================================

full_size_mb = (
    memory_bank.numel()
    *
    memory_bank.element_size()
    /
    (1024 ** 2)
)


coreset_size_mb = (
    coreset.numel()
    *
    coreset.element_size()
    /
    (1024 ** 2)
)


actual_ratio = (
    coreset.shape[0]
    /
    memory_bank.shape[0]
)


reduction = (
    1.0
    -
    actual_ratio
)


total_time = (
    time.perf_counter()
    -
    start_time
)


print(
    "\n"
    +
    "=" * 70
)

print(
    "V2C FULL32 TRADITIONAL GREEDY 10% CORESET"
)

print(
    "=" * 70
)


print(
    f"\nOriginal memory bank: "
    f"{tuple(memory_bank.shape)}"
)

print(
    f"Coreset: "
    f"{tuple(coreset.shape)}"
)


print(
    f"\nOriginal memory size: "
    f"{full_size_mb:.2f} MB"
)

print(
    f"Coreset size: "
    f"{coreset_size_mb:.2f} MB"
)

print(
    f"Actual coreset ratio: "
    f"{actual_ratio:.2%}"
)

print(
    f"Reduction: "
    f"{reduction:.2%}"
)

print(
    f"Greedy runtime: "
    f"{total_time:.2f}s"
)


# ============================================================
# 10. Save checkpoint
# ============================================================

output_checkpoint = {
    "memory_bank":
        coreset,

    "selected_indices":
        selected_indices_tensor,

    "coreset_ratio":
        CORESET_RATIO,

    "projection_dim":
        PROJECTION_DIM,

    "selection_method":
        "traditional_greedy_farthest_first",

    "target_size":
        checkpoint["target_size"],

    "min_valid_ratio":
        checkpoint["min_valid_ratio"],

    "feature_dim":
        checkpoint["feature_dim"],

    "num_training_images":
        checkpoint["num_training_images"],

    "preprocessing":
        checkpoint["preprocessing"],

    "feature_layers":
        checkpoint["feature_layers"],

    "feature_resolution":
        checkpoint["feature_resolution"],

    "spatial_sampling":
        checkpoint["spatial_sampling"],

    "experiment":
        "v2c_full32_traditional_greedy10",
}


torch.save(
    output_checkpoint,
    OUTPUT_PATH
)


print(
    "\nSaved to:"
)

print(
    OUTPUT_PATH
)