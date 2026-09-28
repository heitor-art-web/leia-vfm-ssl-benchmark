# Zero-cost benchmark path

This document records what can be completed before paying for GPU compute.

## Completed without paid compute

- frozen LIDC-IDRI train/validation/test patient manifests;
- nested 1%, 5%, 10% and 25% labelled-patient budgets for seeds 1337, 2026 and 31415;
- exact annotation-count target policy (`0=background`, `1=trusted`, `255=UNKNOWN/ignore`);
- real-data visual QC and XML inventory audit;
- deterministic full-export planner;
- real-data YOLO26 semantic CPU smoke test;
- supervised runner and held-out evaluator;
- pinned `ultralytics==8.4.163`;
- explicit contract check that the pinned Ultralytics semantic loss excludes target value 255;
- metadata-only planning for all 12 supervised seed/budget combinations in GitHub Actions.

## Ultralytics ignore-label contract

The benchmark uses two semantic classes (`0=background`, `1=nodule`), so the pinned semantic criterion uses multiclass cross-entropy. In Ultralytics v8.4.163, `SemanticSegmentationLoss` constructs `CrossEntropyLoss(ignore_index=255)` and also builds a valid-pixel mask using `masks.reshape(-1) != 255` before the semantic loss terms are computed.

The repository checks this contract automatically in `scripts/check_ultralytics_semantic_contract.py`. If the package version or expected behavior changes, the no-cost preflight fails before benchmark training.

Upstream tag used for the audit:

- https://github.com/ultralytics/ultralytics/releases/tag/v8.4.163
- https://github.com/ultralytics/ultralytics/blob/v8.4.163/ultralytics/utils/loss.py

## Free preflight workflow

`.github/workflows/free-preflight.yml` performs two jobs on public GitHub Actions:

1. installs exactly `ultralytics==8.4.163` and verifies the semantic ignore contract;
2. generates metadata-only export plans for all `3 seeds x 4 budgets = 12` supervised runs and verifies:
   - non-empty train plan;
   - exactly 152 frozen validation patients;
   - no patient overlap between train and validation;
   - positive slice counts for every planned scan.

The workflow intentionally does **not** download all CT volumes and does **not** start benchmark training.

## What still requires meaningful compute

The first scientific result remains:

```text
arm      SUP
seed     1337
budget   001pct
model    yolo26n-sem.pt
epochs   100
imgsz    512
batch    8
```

A full run may be performed on any available local NVIDIA GPU or on a genuinely free external GPU allocation. The protocol must not be changed based on which hardware is available. If a free hosted notebook/runtime disconnects, that is an infrastructure failure, not a scientific result.

No paid remote job should be launched merely to complete preflight checks already covered by CI.
