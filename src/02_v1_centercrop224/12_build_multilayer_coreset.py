from pathlib import Path
import time

import torch
import torch.nn.functional as F


# ============================================================
# 1. Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"

MEMORY_BANK_PATH = (
    MODEL_DIR
    / "aluminum_camera_cover_multilayer_memory_bank.pt"
)

CORESET_PATH = (
    MODEL_DIR
    / "aluminum_camera_cover_multilayer_coreset.pt"
)


# ============================================================
# 2. Configuration
# ============================================================

CORESET_RATIO = 0.10

PROJECTION_DIM = 64

SEED = 42


# ============================================================
# 3. Reproducibility
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# 4. Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("Using device:", device)


# ============================================================
# 5. Load full memory bank
# ============================================================

checkpoint = torch.load(
    MEMORY_BANK_PATH,
    map_location="cpu",
    weights_only=True,
)

memory_bank = checkpoint["memory_bank"].float()

num_features, feature_dim = memory_bank.shape


print("\nMulti-layer memory bank loaded.")

print(
    "Shape:",
    tuple(memory_bank.shape)
)

print(
    f"Normal patch features: "
    f"{num_features:,}"
)

print(
    f"Feature dimension: "
    f"{feature_dim}"
)


# ============================================================
# 6. Target coreset size
# ============================================================

coreset_size = max(
    1,
    round(
        num_features
        *
        CORESET_RATIO
    )
)


print(
    f"\nCoreset ratio: "
    f"{CORESET_RATIO:.0%}"
)

print(
    f"Target coreset size: "
    f"{coreset_size:,}"
)


# ============================================================
# 7. Random projection
# ============================================================

print(
    f"\nProjecting features "
    f"{feature_dim} -> {PROJECTION_DIM}..."
)


features_gpu = memory_bank.to(
    device
)


projection_matrix = torch.randn(
    feature_dim,
    PROJECTION_DIM,
    device=device,
    dtype=features_gpu.dtype,
)


projection_matrix = (
    projection_matrix
    /
    (PROJECTION_DIM ** 0.5)
)


projected = (
    features_gpu
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
# 8. Greedy farthest-point selection
# ============================================================

print(
    "\nStarting greedy coreset selection..."
)

print(
    "Large bank detected — this will perform "
    "many GPU vector comparisons."
)


start_time = time.perf_counter()


# ------------------------------------------------------------
# First representative
# ------------------------------------------------------------

first_index = torch.randint(
    low=0,
    high=num_features,
    size=(1,),
    device=device,
).item()


selected_indices = [
    first_index
]


# ------------------------------------------------------------
# Initial similarity to selected set
# ------------------------------------------------------------

max_similarity_to_selected = (
    projected
    @
    projected[first_index]
)


max_similarity_to_selected[
    first_index
] = float("inf")


# ============================================================
# Greedy loop
# ============================================================

for step in range(
    1,
    coreset_size
):

    # Lowest maximum similarity
    # =
    # currently worst represented point

    next_index = torch.argmin(
        max_similarity_to_selected
    ).item()


    selected_indices.append(
        next_index
    )


    # Compare every feature against
    # newly selected representative.

    new_similarity = (
        projected
        @
        projected[next_index]
    )


    # Keep similarity to nearest selected
    # representative.

    max_similarity_to_selected = (
        torch.maximum(
            max_similarity_to_selected,
            new_similarity
        )
    )


    max_similarity_to_selected[
        next_index
    ] = float("inf")


    if (
        (step + 1) % 500 == 0
        or
        (step + 1) == coreset_size
    ):

        elapsed = (
            time.perf_counter()
            -
            start_time
        )

        print(
            f"Selected "
            f"{step + 1:,}/"
            f"{coreset_size:,} "
            f"| elapsed: "
            f"{elapsed:.1f}s"
        )


# ============================================================
# 9. Selected indices
# ============================================================

selected_indices = torch.tensor(
    selected_indices,
    dtype=torch.long
)


# ============================================================
# 10. Build coreset from ORIGINAL features
# ============================================================

coreset = memory_bank[
    selected_indices
]


print(
    "\nMulti-layer coreset created."
)

print(
    "Coreset shape:",
    tuple(coreset.shape)
)


# ============================================================
# 11. Memory comparison
# ============================================================

full_bytes = (
    memory_bank.numel()
    *
    memory_bank.element_size()
)

coreset_bytes = (
    coreset.numel()
    *
    coreset.element_size()
)


full_mb = (
    full_bytes
    /
    (1024 ** 2)
)

coreset_mb = (
    coreset_bytes
    /
    (1024 ** 2)
)


print("\nMemory comparison:")

print(
    f"Full memory bank: "
    f"{full_mb:.2f} MB"
)

print(
    f"Coreset:          "
    f"{coreset_mb:.2f} MB"
)

print(
    f"Reduction:        "
    f"{100 * (1 - coreset_bytes / full_bytes):.2f}%"
)


# ============================================================
# 12. Save coreset
# ============================================================

torch.save(
    {
        "product":
            checkpoint["product"],

        "backbone":
            checkpoint["backbone"],

        "feature_layers":
            checkpoint["feature_layers"],

        "spatial_resolution":
            checkpoint["spatial_resolution"],

        "num_training_images":
            checkpoint["num_training_images"],

        "original_memory_bank_size":
            num_features,

        "coreset_ratio":
            CORESET_RATIO,

        "projection_dimension":
            PROJECTION_DIM,

        "feature_dimension":
            feature_dim,

        "selected_indices":
            selected_indices,

        "memory_bank":
            coreset,
    },

    CORESET_PATH,
)


# ============================================================
# 13. Final report
# ============================================================

total_time = (
    time.perf_counter()
    -
    start_time
)


print(
    f"\nSelection time: "
    f"{total_time:.2f} seconds"
)

print(
    "\nSaved coreset to:"
)

print(
    CORESET_PATH
)

print("\nDone.")