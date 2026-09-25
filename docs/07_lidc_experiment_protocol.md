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

## Frozen patient split policy

The benchmark unit is the **patient**, never an individual CT slice.

The v1 held-out split is fixed before model training as:

```text
70% train
15% validation
15% test
split seed = 20260925
```

The train/validation/test assignment is stratified only by whether the patient has any foreground in the mirror's documented default nodule reference. This prevents a pathological held-out prevalence imbalance while keeping every scan from a patient in exactly one split.

Validation and test patients are frozen before hyperparameter tuning. The exact patient manifests are generated from the pinned MedOtter metadata revision and versioned with the benchmark.

## Label budgets inside the training pool

Label fractions `{1%, 5%, 10%, 25%}` are applied **only inside the frozen training pool**.

Within each seed, labelled patient subsets are generated from a deterministic target-blind ordering of the training patient IDs. The subset selector does **not** inspect nodule presence, malignancy score, mask size or any other target-derived property. This avoids making the low-label regimes artificially easier through label-aware patient selection.

The subsets are nested within a seed:

```text
1% subset ⊂ 5% subset ⊂ 10% subset ⊂ 25% subset
```

The remaining training patients form the unlabelled pool for MT and MT+MedSAM. SUP sees only the labelled subset.

For a given fraction and seed, SUP, MT and MT+MedSAM must use exactly the same labelled patient IDs.

## Seeds

Initial label-budget seeds:

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

DICOM modality rescale is applied before a fixed CT window of `[-1000, 400] HU`. For the public NIfTI development mirror, image geometry and mask alignment are checked explicitly before rendering or export.

## YOLO26-sem interface

The specialist baseline uses Ultralytics YOLO26 semantic segmentation, initially `yolo26n-sem.pt`. The dataset representation is single-channel class-index PNG masks with:

```text
0   background
1   pulmonary nodule
255 ignore / unknown
```

The benchmark keeps semantic and instance segmentation conceptually separate: `YOLO26-sem` is used for the primary semantic task; instance masks are used only where exact nodule identity is required for QC.

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
4. CT intensity rescale is verified on real examples;
5. visual overlays are reviewed for representative trusted and ambiguous annotations;
6. empty CT slices are not silently removed from the official benchmark;
7. all frozen patient lists and manifests are versioned;
8. no ground-truth information enters the unlabelled prompt path.

### Visual-QC status

The trusted-reference gate is accepted for five exact-instance cases: one no-volumetric-nodule control plus four radiologist high-suspicion nodules spanning approximately 7-45 mm. Exact instance isolation is used so an unrelated larger nodule in the same scan cannot replace the metadata-selected lesion in the preview.

The QC cohort has now been expanded to seven patient-unique cases with two deterministic ambiguity examples:

```text
one-reader contour -> UNKNOWN / ignore
 two-reader contour -> UNKNOWN / ignore
```

Each ambiguity scan contains exactly one volumetrically annotated nodule and zero default-reference nodules. The rendered slices therefore exercise the scan-wide annotation-count path without contamination by a second volumetric nodule. Automated geometry, metadata and target checks are green. The two new amber UNKNOWN overlays still require final visual confirmation before training starts.

## Current implementation boundary

Phase 0 now implements data conversion, consensus handling, real-data visual QC, deterministic patient-level split generation, frozen exact patient manifests, nested label-budget manifests, validation and the supervised YOLO26-sem entry point. Mean Teacher and MedSAM co-teacher integration remains deferred until the final ambiguity visual-QC gate is accepted.
