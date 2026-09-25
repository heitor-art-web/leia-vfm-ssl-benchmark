# VFM + SSL Pulmonary Nodule Segmentation Benchmark

Independent, public research scaffold for a narrow question:

> **Does classical semi-supervised learning still add measurable value when a medical vision foundation model is available, especially under very low and incomplete annotation budgets?**

The first application domain is **pulmonary nodule segmentation in thoracic CT**.

## Minimal benchmark

- **Primary dataset:** LIDC-IDRI
- **Task:** semantic segmentation of the contourable LIDC pulmonary nodules (the ≥3 mm nodule annotations represented by `pylidc.Annotation`) in phase 1
- **Specialist:** YOLO26 semantic segmentation, using a 2.5D CT representation
- **SSL method:** Mean Teacher / EMA teacher around the same specialist
- **Foundation model:** frozen MedSAM used as a co-teacher / pseudo-label refiner
- **Label budgets:** 1%, 5%, 10%, 25% of the frozen training-patient pool
- **Optional ceiling:** 100% supervised
- **Primary metrics:** Dice / IoU on trusted nodule pixels, plus lesion-aware evaluation
- **Secondary metrics:** HD95 where defined, pseudo-label acceptance coverage, pseudo-label quality, compute cost
- **Repetitions:** 3 fixed seeds minimum; stronger repeated/CV analysis later

The phase-1 target is deliberately narrow. LIDC marks nodules <3 mm and non-nodules differently from contourable nodules ≥3 mm; those other mark types are **not positive targets in this benchmark**. `background=0` therefore means background relative to this phase-1 segmentation target, not “absence of all pulmonary pathology.”

## Why LIDC-IDRI?

LIDC-IDRI is a public thoracic CT collection with expert pulmonary nodule annotations from multiple experienced radiologists. The annotation protocol did not force consensus, which makes the dataset useful not only for low-label SSL but also for studying **annotation disagreement and UNKNOWN regions** rather than pretending every non-agreed pixel is background.

The repository does **not** redistribute DICOM images or annotation files.

## Experimental conditions

For each annotation fraction `f ∈ {0.01, 0.05, 0.10, 0.25}`:

1. `SUP`: YOLO26-sem trained only on labelled patients.
2. `MT`: same specialist + Mean Teacher, using the remaining training patients as unlabelled.
3. `MT+MedSAM`: identical MT setup, with frozen MedSAM used to validate/refine teacher pseudo-labels.

Main contrasts:

- `SSL gain(f) = MT(f) - SUP(f)`
- `VFM gain(f) = MT+MedSAM(f) - MT(f)`

MedSAM is not allowed to rewrite human ground truth.

A separate validation-only diagnostic measures whether MedSAM improves teacher-derived pseudo-labels when prompted by boxes generated from the specialist prediction. This diagnostic is not counted as a separate training condition.

## Annotation states

The phase-1 semantic target is deliberately conservative:

```text
0   = background relative to the contourable-nodule target
1   = trusted pulmonary nodule
255 = UNKNOWN / disputed annotation evidence
```

Ultralytics semantic segmentation ignores mask value `255` during loss/metric computation, allowing disagreement to remain unresolved instead of being converted to a negative label.

## 2.5D input

YOLO26-sem is a 2D semantic model. The initial CT representation uses:

```text
R = slice z-1
G = slice z
B = slice z+1
```

Edge slices replicate the nearest available slice.

The CT window is configurable and must be frozen before benchmark runs.

## Repository map

```text
configs/                 experiment / dataset configuration
docs/                    scope, literature, protocol and data pipeline
literature/              living bibliography + search log
src/leia_benchmark/      benchmark utilities
scripts/                 audit, conversion, validation and experiment tools
experiments/             frozen run matrix
results/                 result templates (no fabricated numbers)
tests/                   unit tests
```

## LIDC dataset preparation

Install the optional preparation dependencies:

```bash
pip install -r requirements-lidc.txt
```

Configure `pylidc` to point at the downloaded LIDC-IDRI DICOM root.

Audit reader-agreement density before freezing splits:

```bash
python scripts/audit_lidc_annotations.py \
  --output data/lidc_annotation_audit.csv
```

Plan a patient-level split only after reviewing that audit:

```bash
python scripts/plan_lidc_splits.py \
  data/lidc_annotation_audit_patients.csv \
  --output-dir data/splits/lidc_v1
```

Then generate deterministic nested 1/5/10/25% label budgets from the frozen train pool:

```bash
python scripts/make_lidc_label_budgets.py \
  data/splits/lidc_v1/train_patients.txt \
  --output-dir data/splits/lidc_v1/budgets
```

Run a one-patient smoke conversion first:

```bash
python scripts/prepare_lidc_yolo26.py \
  --output data/lidc_yolo26_smoke \
  --patient-id LIDC-IDRI-0078 \
  --split train \
  --signal-only
```

Validate the exported semantic dataset:

```bash
python scripts/validate_lidc_yolo26.py \
  data/lidc_yolo26_smoke \
  --splits train
```

See `docs/06_lidc_pipeline.md`.

## Current status

**Phase 0 — dataset/protocol engineering.**

Implemented on the LIDC pipeline branch:

- HU conversion from ordered DICOM slices;
- 2.5D slice construction;
- configurable CT windowing;
- reader-vote target construction;
- explicit `UNKNOWN=255`;
- positive-over-unknown merge precedence;
- multi-scan-safe sample identifiers;
- patient-level annotation audit and split planning;
- nested annotation-budget generation;
- YOLO26 semantic dataset config;
- export manifest;
- image/mask validator;
- patient-leakage validator;
- reference-compatible MedSAM prompt/refinement wrapper;
- buffer-safe Mean Teacher EMA implementation;
- unit-test CI.

No performance claims are made yet.

## Reproducibility rules

- Split at **patient level**, never slice level.
- Freeze test patients before hyperparameter selection.
- Freeze train/val/test before selecting 1/5/10/25% labelled subsets.
- Do not use model predictions as human ground truth.
- `unlabelled` means **unknown label**, not background / negative.
- Preserve radiologist disagreement as an auditable state.
- Use the same frozen splits for all methods.
- Report all seeds; do not select only the best run.
- Tune pseudo-label thresholds on validation only.
- Keep specialist and VFM predictions separately auditable.
- Do not claim clinical diagnosis or malignancy prediction from a segmentation-only phase.

## Project status and affiliation

This is an independent benchmark/reproduction project. It is not an official repository of INESC TEC, NIAR, LEIA, TCIA, LIDC-IDRI, Ultralytics or MedSAM, and it does not claim affiliation with those projects. External names are used only for scientific attribution.
