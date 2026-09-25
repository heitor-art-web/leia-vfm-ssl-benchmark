# LIDC-IDRI → YOLO26-sem dataset pipeline

## Scope

The first pulmonary benchmark task is **semantic segmentation of the contourable LIDC pulmonary nodules in thoracic CT**.

For phase 1, the positive target is the LIDC nodule category with contour annotations (nodules ≥3 mm, represented by `pylidc.Annotation`). LIDC's smaller-nodule and non-nodule mark types are not treated as positive targets here. Consequently, `background=0` means background relative to this benchmark target; it must not be interpreted as “no pulmonary abnormality.”

This phase is deliberately narrower than diagnosis or malignancy prediction. It asks whether a specialist segmentation model can learn the task efficiently from limited / incomplete expert annotations and whether VFM guidance adds measurable value.

## Source data

Use the official **LIDC-IDRI** collection from The Cancer Imaging Archive (TCIA).

Important properties for this benchmark:

- thoracic CT;
- four experienced thoracic radiologists participated in a two-phase annotation protocol;
- no forced consensus was required;
- annotations for nodules ≥3 mm contain contours suitable for segmentation;
- TCIA recommends using `pylidc` or the standardized DICOM representation rather than writing a new XML parser unless necessary;
- the collection contains more CT series than patients, so filenames must identify the series as well as the patient.

The repository does not redistribute DICOM images or annotation files.

## Unit of splitting

**Patient, never slice or series.**

All CT series and all slices belonging to one patient must remain in exactly one of train / validation / test.

The pipeline therefore uses the patient ID for leakage control but adds a stable hash of `SeriesInstanceUID` to each scan filename to prevent collisions for patients with multiple CT series.

Before freezing the partition:

```bash
python scripts/audit_lidc_annotations.py \
  --output data/lidc_annotation_audit.csv
```

This writes both scan-level and patient-level audit tables. The v1 split planner stratifies coarsely by whether a patient has at least one cluster supported by ≥3 readers; more granular stratification should only be introduced if the real audit distribution supports it.

```bash
python scripts/plan_lidc_splits.py \
  data/lidc_annotation_audit_patients.csv \
  --output-dir data/splits/lidc_v1
```

Annotation-budget subsets (1%, 5%, 10%, 25%) are selected only from the frozen training-patient pool. The budget generator uses deterministic prefix sampling so budgets are nested within each seed:

```bash
python scripts/make_lidc_label_budgets.py \
  data/splits/lidc_v1/train_patients.txt \
  --output-dir data/splits/lidc_v1/budgets
```

## 2.5D image representation

YOLO26 semantic segmentation is 2D. The first benchmark therefore uses a reproducible 2.5D representation:

- R channel: slice `z-1`
- G channel: slice `z`
- B channel: slice `z+1`
- edge slices replicate the nearest available slice

The CT intensity window is configurable. The initial engineering default is `[-1000, 400] HU`; it is **not** treated as a clinically privileged choice and must be frozen before benchmark runs.

Because the three channels are adjacent CT slices rather than colour channels, colour-space augmentation is disabled. Mosaic, MixUp and Copy-Paste are also disabled in the initial protocol because they can create anatomically implausible CT composites.

## Conservative target construction

Ultralytics semantic masks support pixel value `255` as an ignore label. We use that to preserve uncertainty instead of silently converting disagreement to background.

For a clustered contourable nodule:

1. `pylidc.utils.consensus(..., ret_masks=True)` is used only to align the reader masks into a common bounding box.
2. If fewer than `min_cluster_readers` readers annotated the nodule cluster, all pixels in the annotation union become `255 / UNKNOWN`.
3. Otherwise, pixels with at least `ceil(N * positive_fraction)` positive reader masks are label `1 / nodule`.
4. Pixels inside the union but below that threshold become `255 / UNKNOWN`.
5. Pixels outside these contourable-nodule annotation regions remain `0 / background` for the phase-1 target.

Initial defaults:

```text
min_cluster_readers = 3
positive_fraction   = 0.5
```

These are benchmark parameters, not medical truth. Sensitivity analyses should test alternative policies.

When cluster regions overlap, target precedence is:

```text
trusted positive > UNKNOWN > background
```

so a strongly supported positive region cannot be erased by an overlapping ambiguous cluster.

## Why UNKNOWN matters

A region annotated by one reader and rejected or omitted by others is not equivalent to known background. Treating it as negative would inject a supervision assumption into the benchmark.

The semantic target therefore has:

```text
0   background relative to the phase-1 contourable-nodule target
1   trusted contourable nodule
255 unresolved / insufficient annotation evidence
```

## Output layout

```text
data/lidc_yolo26/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
├── masks/
│   ├── train/
│   ├── val/
│   └── test/
├── manifest_train.csv
├── manifest_val.csv
└── manifest_test.csv
```

Image and mask stems are identical. A stem contains the patient ID, a stable series-derived scan key and the axial slice index.

## First smoke test

Do not convert the full collection immediately. Start with a known patient:

```bash
python scripts/prepare_lidc_yolo26.py \
  --output data/lidc_yolo26_smoke \
  --patient-id LIDC-IDRI-0078 \
  --split train \
  --signal-only
```

The converter refuses an unscoped export and refuses overwrites unless explicitly requested.

Check:

1. image and mask sizes match;
2. all mask values are in `{0, 1, 255}`;
3. positive masks visually overlap contourable nodules;
4. UNKNOWN pixels appear only around disputed/low-reader annotation regions;
5. the stable scan identifier prevents collisions if a patient has multiple CT series;
6. no patient appears in multiple splits.

Then run:

```bash
python scripts/validate_lidc_yolo26.py data/lidc_yolo26_smoke --splits train
```

Only after visual + automated QA should the pipeline be scaled to the development cohort and then the full benchmark.

## Known limitations

- `pylidc` is mature but old; keep its use isolated to dataset preparation.
- The current pipeline treats clusters with low reader count conservatively as UNKNOWN instead of positives or negatives.
- The current export is 2.5D, not volumetric 3D.
- Benchmark conversion preserves all axial slices by default. `--signal-only` exists only for development/smoke tests; training-time sampling can then be controlled without discarding source data.
- The phase-1 target does not model LIDC's small-nodule or non-nodule mark types.
- Malignancy ratings and diagnosis labels are intentionally out of scope for phase 1.
