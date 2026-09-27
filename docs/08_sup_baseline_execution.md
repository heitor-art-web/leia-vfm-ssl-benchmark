# Frozen supervised baseline execution

This document freezes the first scientific supervised run before its validation metrics are inspected.

## First run

```text
arm      SUP
seed     1337
budget   1% (001pct)
model    yolo26n-sem.pt
library  ultralytics==8.4.163
input    2.5D z-1 / z / z+1
window   [-1000, 400] HU
imgsz    512
batch    8
workers  8
epochs   100
```

The labelled training cohort is the exact frozen `seed1337/001pct` patient subset already versioned in `splits/lidc/v1/manifest.json`. The complete frozen validation cohort is used for evaluation.

## Checkpoint rule

The v1 supervised baseline uses a **fixed final-epoch checkpoint**.

Held-out validation is disabled during the training epochs. After epoch 100, `last.pt` is evaluated exactly once on the complete frozen validation split. This avoids repeatedly inspecting the held-out cohort and avoids paying the very large cost of evaluating all validation slices after every training epoch.

The same supervised schedule and checkpoint-selection rule must be held constant for the comparable Mean Teacher and Mean Teacher + MedSAM arms unless a deviation is preregistered before seeing their final validation results.

## Metrics

The evaluator treats class `1` as the pulmonary-nodule foreground and excludes target value `255` from all confusion counts.

Primary metrics:

- foreground Dice;
- foreground IoU.

Secondary metrics produced by the first evaluator:

- precision;
- recall / sensitivity;
- patient-level macro summaries;
- case-level macro summaries;
- unknown / ignored pixel fraction;
- inference time;
- deterministic lists of slices with the largest false-negative and false-positive pixel counts for later visual QC.

A patient/case with no positive reference pixels and no positive prediction has undefined foreground Dice/IoU rather than being counted as a perfect score. The evaluator reports how many patient/case metrics are defined.

Lesion-level sensitivity/FROC and HD95 remain planned secondary analyses and will be added without changing the primary Dice/IoU definitions.

## Reproducible command

```bash
python scripts/run_lidc_sup_budget.py \
  --workdir /workspace/leia \
  --results-dir /outputs/sup/seed_1337/001pct \
  --seed 1337 \
  --budget 001pct \
  --epochs 100 \
  --imgsz 512 \
  --batch 8 \
  --eval-batch 16 \
  --workers 8 \
  --device 0
```

The run writes `run.json`, the final checkpoint under `training/`, and `metrics.json` under the chosen results directory.
