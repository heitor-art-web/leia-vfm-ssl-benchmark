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
cls_pw   1.0
```

The labelled training cohort is the exact frozen `seed1337/001pct` patient subset already versioned in `splits/lidc/v1/manifest.json`. The complete frozen validation cohort is used only after training and the train-only sanity gate described below.

## Training-only target profile

Before any validation/test metrics were inspected, the frozen **labelled TRAIN subset only** was profiled from the pinned annotation-count masks. The resulting evidence is versioned in:

```text
experiments/lidc_sup_seed1337_001pct_target_profile.json
```

It contains:

```text
7 patients / 7 scans
2346 slices total
28 slices with trusted foreground          1.19%
74 slices with UNKNOWN but no trusted fg   3.15%
2244 pure-background slices               95.65%
0 all-ignore slices

trusted voxels    1,014 / 614,989,824   ~1.65e-6
UNKNOWN voxels   11,586 / 614,989,824   ~1.88e-5
```

Only three of the seven labelled patients contain any `>=3`-vote trusted foreground under the frozen target policy. This is a property of the already-frozen target-blind 1% patient subset, not a post-hoc resampling choice.

The profile deliberately does not open validation or test masks.

## Foreground imbalance policy

Ultralytics semantic segmentation supports training-pixel class weights through `cls_pw`. The v1 supervised baseline freezes:

```text
cls_pw = 1.0
```

At this setting the pinned semantic trainer applies its full ENet-style inverse-log pixel-frequency weighting computed **only from the labelled TRAIN masks**. The Dice term remains part of the pinned semantic loss.

The v1 baseline does **not** introduce a custom positive-slice sampler. Every exported labelled training slice remains eligible in the standard shuffled training loader. Slice oversampling, if studied later, is an explicit ablation rather than an undocumented baseline change.

## 2.5D-safe augmentation policy

The three input channels are adjacent CT slices, not natural-image RGB channels. Natural-image color transforms and cross-image mosaics would therefore alter the meaning of the 2.5D representation.

The first baseline freezes the following augmentation policy:

```text
hsv_h       0.0
hsv_s       0.0
hsv_v       0.0
bgr         0.0
mosaic      0.0
mixup       0.0
cutmix      0.0
copy_paste  0.0

rotation    +/- 5 degrees
translate   0.05
scale       +/- 0.10
horizontal flip probability 0.5
vertical flip               0.0
shear                       0.0
perspective                 0.0
```

Spatial transforms are applied jointly to the three CT channels and the semantic target. The pinned augmentation contract is executable: semantic affine transforms use nearest-neighbor interpolation, newly introduced border support is set to `255` rather than false background, and horizontal flips transform the semantic target exactly.

This exact labelled-stream augmentation policy must also be used by the comparable MT and MT+MedSAM arms.

## Verified ignore-label contract

The benchmark relies on target value `255` being excluded from the training loss. Version-pinned executable contract tests now check `ultralytics==8.4.163` directly before benchmark training.

For the pinned semantic loss:

- the dataset has two explicit classes (`0=background`, `1=pulmonary_nodule`);
- multi-class cross-entropy is configured with `ignore_index=255`;
- Dice filters the same invalid pixels;
- changing logits only at an ignored pixel leaves the loss unchanged;
- a synthetic all-ignore batch remains finite.

For the pinned semantic spatial transforms:

- affine mask interpolation is nearest-neighbor;
- affine border pixels are `255` (UNKNOWN/ignore), not background;
- trusted and UNKNOWN class ids remain discrete;
- horizontal flips transform image and target in the same spatial direction.

This closes the earlier implementation uncertainty around whether `255` behaves as the benchmark's intended UNKNOWN/ignore label in both loss and the frozen spatial augmentation path.

## Checkpoint rule and strict held-out isolation

The v1 supervised baseline uses a **fixed final-epoch checkpoint** (`last.pt`).

An important pinned-library detail is handled explicitly: in Ultralytics `8.4.163`, the trainer validates on the final epoch even when `val=False`, and then executes `final_eval()`. Therefore merely setting `val=False` is **not sufficient** to keep held-out validation unseen during training.

The benchmark runner writes a separate trainer-only dataset YAML in which:

```text
train = labelled TRAIN images
val   = labelled TRAIN images
```

The real frozen validation split remains present in the exported benchmark dataset for the external evaluator, but its path is never supplied to the Ultralytics trainer. Consequently, any mandatory internal final-epoch/final evaluation can inspect **TRAIN only**.

After epoch 100, `last.pt` is selected without consulting held-out validation performance. Internal `best.pt` is irrelevant to the registered benchmark and is never used for the scientific comparison.

The same supervised schedule and checkpoint-selection rule must be held constant for the comparable Mean Teacher and Mean Teacher + MedSAM arms unless a deviation is preregistered before seeing their final validation results.

## Train-only degeneration gate

Before opening the held-out validation metrics, the final checkpoint is evaluated once on the labelled TRAIN subset by the benchmark's external evaluator.

The run aborts without evaluating the real frozen validation split if either is true:

```text
predicted foreground pixels on TRAIN == 0
true-positive overlap on TRAIN == 0
```

This is not a model-selection criterion and does not set a performance threshold. It is only a software/scientific sanity gate to prevent a completely collapsed model from triggering inspection of held-out results. The gate uses no validation or test labels.

Only after this gate passes does the external evaluator open the frozen validation masks and write `metrics.json`.

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

## Run provenance

`run.json` records:

- Git revision;
- exact split-manifest SHA-256;
- exported dataset provenance SHA-256;
- trainer-only YAML SHA-256;
- Python/platform information;
- Ultralytics/PyTorch/NumPy/NiBabel versions;
- CUDA/cuDNN visibility and GPU device names;
- frozen training/augmentation policy;
- final checkpoint and metric-file SHA-256 hashes.

For `SUP / seed1337 / 001pct`, the runner refuses accidental deviations from the frozen schedule unless `--allow-protocol-override` is supplied explicitly. An override is not the registered baseline and must be documented separately.

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
  --cls-pw 1.0 \
  --degrees 5 \
  --translate 0.05 \
  --scale 0.10 \
  --fliplr 0.5 \
  --device 0
```

The run writes `run.json`, the final checkpoint under `training/`, `train_metrics.json` for the train-only sanity gate, and `metrics.json` for the first external frozen-validation evaluation.

## Zero-cost preflight

The public repository includes `.github/workflows/sup-1pct-free-preflight.yml`. It uses standard GitHub-hosted runners on this public repository and performs three non-scientific checks:

1. verifies the exact pinned Ultralytics `255` loss and spatial-augmentation contracts;
2. regenerates the TRAIN-only target profile without opening validation/test masks;
3. exports the complete seven-patient labelled TRAIN subset and runs one CPU epoch at `512` resolution to catch memory, data-loader, loss and checkpoint failures.

The preflight exports **no held-out data**. Because Ultralytics still performs mandatory final internal validation, the preflight YAML aliases `val` to TRAIN. Any internal validation therefore remains TRAIN-only.

The one-epoch CPU output is explicitly integration/resource evidence and must never be reported as benchmark performance.
