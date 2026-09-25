# LIDC-IDRI → YOLO26-sem dataset pipeline

## Scope

The first pulmonary benchmark task is **semantic segmentation of pulmonary nodules in thoracic CT**.

This phase is deliberately narrower than diagnosis or malignancy prediction. It asks whether a specialist segmentation model can learn the task efficiently from limited / incomplete expert annotations and whether VFM guidance adds measurable value.

## Source data

Use the official **LIDC-IDRI** collection from The Cancer Imaging Archive (TCIA).

Important properties for this benchmark:

- thoracic CT;
- four experienced thoracic radiologists participated in a two-phase annotation protocol;
- no forced consensus was required;
- annotations for nodules ≥3 mm contain contours suitable for segmentation;
- TCIA recommends using `pylidc` or the standardized DICOM representation rather than writing a new XML parser unless necessary.

The repository does not redistribute DICOM images or annotation files.

## Unit of splitting

**Patient, never slice.**

All slices from one patient must remain in exactly one of train / validation / test.

Annotation-budget subsets (1%, 5%, 10%, 25%) are selected from the frozen training-patient pool only.

## 2.5D image representation

YOLO26 semantic segmentation is 2D. The first benchmark therefore uses a reproducible 2.5D representation:

- R channel: slice `z-1`
- G channel: slice `z`
- B channel: slice `z+1`
- edge slices replicate the nearest available slice

The CT intensity window is configurable. The initial engineering default is `[-1000, 400] HU`; it is **not** treated as a clinically privileged choice and must be frozen before benchmark runs.

## Conservative target construction

Ultralytics semantic masks support pixel value `255` as an ignore label. We use that to preserve uncertainty instead of silently converting disagreement to background.

For a clustered nodule:

1. `pylidc.utils.consensus(..., ret_masks=True)` is used only to align the reader masks into a common bounding box.
2. If fewer than `min_cluster_readers` readers annotated the nodule cluster, all pixels in the annotation union become `255 / UNKNOWN`.
3. Otherwise, pixels with at least `ceil(N * positive_fraction)` positive reader masks are label `1 / nodule`.
4. Pixels inside the union but below that threshold become `255 / UNKNOWN`.
5. Pixels outside all annotated nodule regions remain `0 / background`.

Initial defaults:

```text
min_cluster_readers = 3
positive_fraction   = 0.5
```

These are benchmark parameters, not medical truth. Sensitivity analyses should test alternative policies.

## Why UNKNOWN matters

A region annotated by one reader and rejected or omitted by others is not equivalent to known background. Treating it as negative would inject a supervision assumption into the benchmark.

The semantic target therefore has:

```text
0   known/background for the benchmark target
1   trusted nodule
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

Image and mask stems are identical.

## First smoke test

Do not convert the full collection immediately. Start with a known patient:

```bash
python scripts/prepare_lidc_yolo26.py \
  --output data/lidc_yolo26_smoke \
  --patient-id LIDC-IDRI-0078 \
  --split train \
  --signal-only
```

Check:

1. image and mask sizes match;
2. all mask values are in `{0, 1, 255}`;
3. positive masks visually overlap nodules;
4. UNKNOWN pixels appear only around disputed/low-reader annotation regions;
5. no patient appears in multiple splits.

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
- Malignancy ratings and diagnosis labels are intentionally out of scope for phase 1.
