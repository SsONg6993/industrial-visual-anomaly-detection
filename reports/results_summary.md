# Results Summary

The reported numbers below are **exploratory results on the dataset's existing split**. A data-quality audit identified exact duplicate groups, including train-test overlap, so these results should not be interpreted as a final leakage-free benchmark.

| Version | Main change | AUROC | AP | Precision | Recall | F1 | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V1 | CenterCrop224 + multi-layer features | 0.9210 | 0.9726 | 0.8922 | 0.9294 | 0.9104 | 121 | 76 |
| V2a | Letterbox256 + 16×16 fused features | 0.8991 | 0.9659 | 0.9194 | 0.8793 | 0.8989 | 83 | 130 |
| V2b | High-res 32×32 inference with stride-2 train coverage | 0.4876 | 0.7267 | 0.7453 | 1.0000* | 0.8541 | 368 | 0 |
| V2c | Letterbox256 + full 32×32 train/test coverage | **0.9166** | **0.9716** | 0.8836 | **0.9304** | **0.9064** | 132 | **75** |

\* V2b recall of 1.0 is not a useful operating point because nearly all good samples were classified as defects.

## Key findings

- CenterCrop caused **95 / 1,077** defect masks to become fully empty after preprocessing.
- Letterbox preserved the full field of view and reduced fully lost defect masks to **0** in the corresponding audit.
- V2b exposed a train-test feature-coverage mismatch: training stored only a fixed subset of 32×32 feature positions while inference evaluated the full grid.
- V2c restored full 32×32 coverage during both training-memory construction and inference.
- V2c full memory bank: **744,000 × 384** float32 features, approximately **1.09 GB**.
- A 10% greedy coreset reduced this to **74,400 × 384**, approximately **109 MB**.

## Evaluation caution

For a stricter final benchmark:

1. build a cleaned, leakage-free split;
2. reserve a validation/calibration set for threshold selection;
3. select the operating threshold on validation data;
4. report the final metrics once on an untouched test set.
