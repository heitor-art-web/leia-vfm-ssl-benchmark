# LIDC-IDRI experiment protocol

## Research question

Does classical semi-supervised learning still add measurable value when a frozen medical vision foundation model is available, especially under very low annotation budgets?

## Primary task

Semantic segmentation of volumetrically annotated pulmonary nodules >= 3 mm on chest CT.

## Experimental arms

For each patient-level label fraction in `{1%, 5%, 10%, 25%}`:

### SUP

YOLO26-sem trained only on the labelled subset.

### MT

Mean Teacher with:

- YOLO26-sem student;
- EMA copy of the same model as teacher;
- labelled supervised loss on the labelled stream;
- consistency / pseudo-label loss on the unlabelled stream.

### MT + MedSAM

Identical Mean Teacher setup, except that the EMA teacher prediction is additionally used to generate prompts for frozen MedSAM. MedSAM output is used only as a second source of evidence for pseudo-label acceptance/refinement.

The intended first policy is conservative:

1. EMA teacher predicts a nodule region.
2. The teacher prediction generates a prompt without using ground truth.
3. Frozen MedSAM returns a refined mask.
4. High-confidence agreement becomes a pseudo-label.
5. Disagreement remains unknown / ignored rather than being forced into foreground or background.

## Why MedSAM is not the sole teacher

A promptable model needs a prompt source. Ground-truth-derived boxes or points would leak annotation information into the unlabelled arm. Using the specialist teacher prediction as the prompt source keeps the unlabelled path annotation-free.

## Standalone VFM arm

A standalone MedSAM baseline is desirable but not yet frozen because its prompt protocol must be fair and leakage-free. It must not use boxes, points or masks derived from held-out ground truth.

Possible future prompt sources include a frozen external detector or a specialist prediction, but those choices introduce additional variables and must be documented explicitly.

## Label budgets and splits

- Split train / validation / test by patient.
- Freeze validation and test patients before any hyperparameter tuning.
- Apply 1%, 5%, 10% and 25% only inside the frozen training pool.
- The remaining training patients form the unlabelled pool for MT and MT+MedSAM.
- Use exactly the same labelled patient IDs across methods for a given fraction and seed.

## Seeds

Initial benchmark seeds:

```text
1337
2026
31415
```

At 1% and 5%, more seeds may be added because patient selection variance is expected to be large. Additional seeds must be declared before looking at final test performance.

## Ground-truth policy

The v1 target is restricted to LIDC-IDRI volumetric contours for nodules >= 3 mm.

Reader votes are encoded as:

```text
0   background for the current task
1   trusted nodule (>= 3 of 4 possible reader votes)
255 ambiguous / unknown (1-2 reader votes)
```

The consensus threshold is fixed before model comparison. Alternative thresholds are ablations, not tuning knobs chosen from test results.

## Input representation

Initial representation is 2.5D:

```text
R = z-1
G = z
B = z+1
```

DICOM modality rescale is applied before a fixed CT window of `[-1000, 400] HU`.

## Fairness constraints

Across SUP, MT and MT+MedSAM, keep constant whenever technically possible:

- YOLO26-sem model size;
- train / val / test patients;
- labelled patient subset;
- image resolution;
- CT window;
- augmentations applied to the labelled stream;
- optimizer family;
- supervised training schedule;
- validation metric implementation;
- checkpoint-selection rule.

Any unavoidable difference must be recorded in the run metadata.

## Primary comparisons

```text
Delta_SSL = metric(MT) - metric(SUP)
Delta_VFM = metric(MT+MedSAM) - metric(MT)
```

The benchmark must report negative, zero and positive deltas without changing the protocol after seeing the result.

## Metrics

Primary:

- Dice for the nodule class;
- IoU for the nodule class.

Secondary:

- HD95 for cases with valid foreground in reference and prediction;
- sensitivity / recall;
- precision;
- pseudo-label coverage;
- pseudo-label agreement rate;
- fraction of pixels kept as unknown;
- training time and inference time;
- GPU memory use where available.

Patient-level aggregation must be reported in addition to any slice-level summaries so scans with many slices do not dominate the benchmark.

## Quality gates before training

Do not start benchmark training until all are true:

1. train / val / test patient sets are disjoint;
2. exported image and mask dimensions match;
3. masks contain only `{0, 1, 255}`;
4. CT intensity rescale is verified on real DICOM examples;
5. visual overlays are reviewed for several 4-reader, 3-reader, 2-reader and 1-reader nodules;
6. empty CT slices are not silently removed from the official benchmark;
7. all frozen patient lists and manifests are versioned;
8. no ground-truth information enters the unlabelled prompt path.

## Current implementation boundary

Phase 0 implements data conversion, consensus handling, validation and the supervised YOLO26-sem entry point. Mean Teacher and MedSAM co-teacher integration should only be implemented after a real LIDC export passes all dataset quality gates.
