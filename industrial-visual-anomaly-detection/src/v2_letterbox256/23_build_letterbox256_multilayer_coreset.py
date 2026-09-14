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
    / "aluminum_camera_cover_letterbox256_multilayer_memory_bank.pt"
)

OUTPUT_PATH = (
    MODEL_DIR
    / "aluminum_camera_cover_letterbox256_multilayer_coreset.pt"
)


# ============================================================
# 2. Config
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
    int(
        round(
            num_features
            *
            CORESET_RATIO
        )
    )
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
# 4. Normalize
# ============================================================

memory_bank = F.normalize(
    memory_bank,
    p=2,
    dim=1
)


# ============================================================
# 5. Random projection
# ============================================================

#
# We only use the lower-dimensional
# representation to SELECT coreset indices.
#
# Final coreset still uses original
# 384-dimensional features.
#

torch.manual_seed(
    RANDOM_SEED
)

projection_matrix = torch.randn(
    feature_dim,
    PROJECTION_DIM,
    dtype=torch.float32
)

projection_matrix = (
    projection_matrix
    /
    np.sqrt(
        PROJECTION_DIM
    )
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


print(
    "Projected shape:",
    tuple(projected.shape)
)


# ============================================================
# 6. Greedy coverage selection
# ============================================================

start_time = (
    time.perf_counter()
)


#
# Start from one feature.
#
# Deterministic start:
# feature 0.
#

selected_indices = [
    0
]


#
# For every feature:
#
# max_similarity_to_selected
#
# stores the similarity to its
# closest selected representative.
#

first_feature = (
    projected[
        0:1
    ]
)


max_similarity = (
    projected
    @
    first_feature.T
).squeeze(1)


#
# Selected point should not
# be selected again.
#

max_similarity[
    0
] = 1.0


# ============================================================
# 7. Iterative farthest selection
# ============================================================

for step in range(
    1,
    target_size
):

    #
    # Lowest max similarity
    # =
    # least represented feature
    #
    # So select that next.
    #

    next_index = int(
        torch.argmin(
            max_similarity
        )
    )


    selected_indices.append(
        next_index
    )


    new_feature = (
        projected[
            next_index:
            next_index + 1
        ]
    )


    similarity_to_new = (
        projected
        @
        new_feature.T
    ).squeeze(1)


    #
    # Update each feature's similarity
    # to its closest selected representative.
    #

    max_similarity = torch.maximum(
        max_similarity,
        similarity_to_new
    )


    #
    # Prevent reselection.
    #

    max_similarity[
        selected_indices
    ] = 1.0


    if (
        step % 1000 == 0
        or
        step
        ==
        target_size - 1
    ):

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        print(
            f"Selected "
            f"{step + 1:,}/"
            f"{target_size:,}"
            f" | elapsed="
            f"{elapsed:.1f}s"
        )


# ============================================================
# 8. Build coreset
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


reduction = (
    1
    -
    (
        coreset.shape[0]
        /
        memory_bank.shape[0]
    )
)


print(
    "\n"
    +
    "=" * 70
)

print(
    "LETTERBOX256 MULTILAYER CORESET"
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
    f"Reduction: "
    f"{reduction:.2%}"
)


# ============================================================
# 10. Save
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