# VFM + SSL Medical Segmentation Benchmark

Independent, public research scaffold to test a narrow question:

> **Does classical semi-supervised learning still add measurable value when a medical vision foundation model is available, especially under very low annotation budgets and incomplete expert annotations?**

The benchmark is intentionally falsifiable. The pulmonary track uses one public CT dataset, one specialist segmentation model, one SSL mechanism, one frozen medical foundation model and fixed patient-level annotation budgets.

## Primary pulmonary benchmark

- **Dataset:** LIDC-IDRI chest CT
- **Clinical task:** semantic segmentation of volumetrically annotated pulmonary nodules >= 3 mm
- **Specialist:** YOLO26 semantic segmentation (`YOLO26-sem`)
- **SSL method:** Mean Teacher using an EMA YOLO26-sem teacher and YOLO26-sem student
- **Foundation model:** frozen MedSAM used as a co-teacher / pseudo-label refiner
- **Label budgets:** 1%, 5%, 10%, 25% of the frozen training-patient pool
- **Optional ceiling:** 100% supervised
- **Incomplete annotation signal:** radiologist contour disagreement represented explicitly with ignore/unknown pixels
- **Primary metrics:** Dice and IoU for the nodule class
- **Secondary metrics:** HD95, sensitivity, precision, confidence/coverage, compute cost and annotation-efficiency curves

ACDC remains useful as a small secondary cardiac benchmark, but it is no longer the primary clinical track.

## Core experimental arms

For each annotation fraction `f ∈ {0.01, 0.05, 0.10, 0.25}`:

1. `SUP`: YOLO26-sem trained only on labelled patients.
2. `MT`: the same YOLO26-sem architecture under Mean Teacher, using the remaining training patients as unlabelled data.
3. `MT+MedSAM`: the same Mean Teacher setup, with frozen MedSAM used to refine or validate teacher-generated pseudo-labels.

This separates:

- gain from **unlabelled data**: `MT - SUP`;
- additional gain from the **foundation-model prior**: `MT+MedSAM - MT`.

A standalone `MedSAM` baseline is scientifically useful, but its prompt source must be defined without using ground-truth masks or boxes. That arm remains deliberately unresolved until a leakage-free prompt protocol is frozen.

## LIDC-IDRI annotation policy

The first implementation uses volumetric contours for nodules >= 3 mm and preserves reader disagreement:

```text
0   = task background
1   = trusted nodule pixel (>= 3 of 4 possible reader votes)
255 = ambiguous / unknown (1-2 reader votes)
```

The `255` value is excluded from Ultralytics semantic-segmentation loss and metrics. This prevents disputed pixels from being silently converted into background.

The 3-reader rule is a benchmark policy, not a medical definition of truth, and should later be ablated.

See `docs/06_lidc_phase0.md` for the complete dataset contract and limitations.

## CT representation

The Phase 0 export is 2.5D:

```text
channel 0 = slice z-1
channel 1 = slice z
channel 2 = slice z+1
```

DICOM `RescaleSlope` and `RescaleIntercept` are applied before deterministic HU windowing. The initial window is `[-1000, 400] HU`.

## Repository map

```text
configs/                 experiment and dataset configuration
docs/                    scope, literature, protocols and dataset contracts
literature/              living bibliography + search log
src/leia_benchmark/      benchmark code and data primitives
scripts/                 export, validation and split utilities
experiments/             frozen run matrix
results/                 result templates (no fabricated numbers)
tests/                   unit tests
```

## Current status

**Phase 0 — LIDC-IDRI dataset engineering and protocol freeze.** No performance claims are made yet.

Implemented on the development branch:

- reader-vote consensus primitives;
- explicit `0 / 1 / 255` semantic targets;
- DICOM rescale to Hounsfield units;
- deterministic 2.5D CT export;
- Ultralytics semantic dataset layout;
- export manifest;
- integrity checks for patient leakage, image/mask pairing and mask values.

The repository does not include LIDC-IDRI DICOM files, MedSAM weights or YOLO weights. Obtain external data and weights from their official sources and respect their licenses and terms.

## Reproducibility rules

- Split at **patient level**, never slice level.
- Freeze test patients before any tuning.
- Never create human ground truth from model predictions.
- `unlabelled` means **unknown label**, not background / negative.
- Keep reader disagreement auditable instead of forcing consensus everywhere.
- Reuse the same frozen patient splits across all benchmark arms.
- Report all seeds; never select only the best run.
- Tune pseudo-label thresholds on validation data only.
- Keep specialist, EMA teacher and foundation-model predictions separately auditable.
- Do not use ground-truth-derived prompts when evaluating a supposedly unlabelled MedSAM path.

## Project status and affiliation

This is an independent benchmark/reproduction project. It is not an official repository of INESC TEC, NIAR, LEIA, LIDC-IDRI, Ultralytics or MedSAM, and it does not claim affiliation with those projects. Names of external datasets, methods and projects are used only for scientific attribution.
