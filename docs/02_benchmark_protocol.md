# Benchmark protocol

## Dataset

Primary dataset: **LIDC-IDRI thoracic CT**.

Phase-1 target: pulmonary nodule semantic segmentation from expert contour annotations.

Before freezing any split, run `scripts/audit_lidc_annotations.py` to inspect per-patient nodule-cluster counts and reader-agreement density.

## Split discipline

**Never split by slice.** Every slice from one patient must remain in the same partition.

The split policy must be frozen and versioned only after inspecting the annotation audit. The initial split should preserve patient-level independence and should avoid accidental concentration of high-agreement nodule cases in one partition.

For each annotation budget, select labelled patients **only from the frozen train pool**. The remaining train patients are treated as unlabelled.

Exact patient IDs must be stored in versioned split files.

## Annotation budgets

Core budgets:

- 1%
- 5%
- 10%
- 25%

Optional ceiling:

- 100% supervised

Budget counts are computed from the final frozen training-patient pool, not from slices.

At very small budgets, especially 1%, variance may be high. If the audit shows heterogeneous nodule burden, use more seeds or a pre-declared constrained sampling rule rather than post-hoc cherry-picking.

## Input preprocessing

### DICOM → HU

Read DICOM slices in pylidc's scan order and explicitly apply `RescaleSlope` and `RescaleIntercept`.

### 2.5D representation

For central axial slice `z`:

```text
channel 0 = z-1
channel 1 = z
channel 2 = z+1
```

Replicate edge slices.

### Intensity window

The converter exposes configurable lower/upper HU limits.

The initial engineering default is `[-1000, 400] HU`. This is a tunable preprocessing parameter, not a clinical truth. Freeze it before benchmark runs and do not tune it on test performance.

## Human target construction

Use `pylidc` to cluster annotations referring to the same physical nodule.

Use `pylidc.utils.consensus(..., ret_masks=True)` only to align reader masks into a common bounding box. The benchmark then applies its own conservative target policy.

Default policy:

- cluster has fewer than 3 reader annotations → annotation union becomes `UNKNOWN=255`;
- eligible cluster with `N` masks → positive threshold `ceil(0.5 × N)`;
- pixels meeting threshold → `nodule=1`;
- pixels in the reader union but below threshold → `UNKNOWN=255`;
- other pixels → `background=0`.

When cluster targets overlap, precedence is:

```text
trusted positive > UNKNOWN > background
```

This avoids erasing strong positive evidence with an overlapping ambiguous target.

The target policy itself must be sensitivity-tested later.

## Dataset export

YOLO26 semantic layout:

```text
dataset/
├── images/{train,val,test}/
└── masks/{train,val,test}/
```

Each image is a 2.5D RGB PNG and each mask is a single-channel PNG with values `{0,1,255}`.

Run `scripts/validate_lidc_yolo26.py` before training. Validation checks:

- image/mask pairing;
- size consistency;
- allowed mask values;
- patient leakage across splits.

## Supervised specialist baseline (SUP)

YOLO26 semantic segmentation trained only on labelled patients.

No unlabelled image may contribute a supervised target.

## SSL baseline (MT)

EMA teacher + student using the same YOLO26 semantic architecture.

Conceptually:

`θ_teacher ← α θ_teacher + (1-α) θ_student`

Student objective:

`L = L_sup + λ(t) L_unsup`

Unsupervised loss is confidence-gated and must ignore pixels marked `UNKNOWN`.

## VFM-assisted SSL (MT+MedSAM)

Keep specialist architecture, train/val/test split, augmentations and optimization policy identical to MT.

For an unlabelled image:

1. EMA teacher predicts nodule probability/mask;
2. trusted candidate foreground yields a prompt (initially a bounding box);
3. frozen MedSAM predicts a prompted mask;
4. teacher/MedSAM agreement and confidence determine trusted pseudo-label pixels;
5. rejected/disputed pixels remain `UNKNOWN`, never forced to background;
6. student learns from the trusted pseudo-label target.

The first implementation should remain one-way:

`teacher → prompt → MedSAM refinement → student`

Bidirectional cross-prompting is a later ablation, not part of v1.

## Evaluation

### Segmentation

Primary:

- nodule Dice on trusted reference pixels;
- IoU/Jaccard;
- patient-level or lesion-level aggregation with confidence intervals.

Secondary where defined:

- HD95;
- predicted volume / reference volume error.

### Lesion-aware detection

Because pulmonary nodules are sparse, a segmentation model should also be evaluated after connected-component extraction:

- lesion sensitivity;
- false positives per scan;
- FROC-style operating points in a later stage.

Do not use pixel accuracy as a primary metric because background dominates.

### Pseudo-label diagnostics

On validation labels only:

- teacher pseudo-label precision/recall/Dice;
- MedSAM-refined pseudo-label precision/recall/Dice;
- accepted-pixel coverage;
- teacher↔MedSAM agreement;
- quality vs reader-agreement level.

## Statistics

At minimum:

- 3 fixed random seeds;
- paired comparison on identical test patients;
- mean ± SD across seeds;
- patient-level bootstrap 95% CIs for key contrasts.

Increase seeds for 1% / 5% budgets if variance is high.

Do not report only the best seed.

## Leakage controls

- patient-level splitting;
- test set frozen before model/hyperparameter selection;
- label-budget subsets drawn only from train;
- no test-derived preprocessing statistics;
- no human GT generated from model output;
- no ground-truth box used to prompt MedSAM on unlabelled/test data;
- record potential foundation-model pretraining overlap as a contamination risk;
- preserve specialist and VFM outputs separately for audit.

## Stop/go criteria

A null result is acceptable.

If `MT+MedSAM` does not improve pseudo-label quality or downstream segmentation beyond uncertainty, do not add architectural complexity merely to force a gain.

Continue method development if VFM guidance improves pseudo-label quality, low-label performance, calibration/coverage, or performance specifically in high-disagreement annotation regions.
