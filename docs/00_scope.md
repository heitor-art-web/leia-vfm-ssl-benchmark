# Scope and falsifiable question

The goal is **not** to invent a new method on day one. The goal is to establish an auditable baseline that can answer a narrow question before adding complexity.

## Clinical task in phase 1

**Pulmonary nodule semantic segmentation in thoracic CT (LIDC-IDRI).**

This phase does not claim cancer diagnosis, malignancy prediction or clinical decision support. It tests the learning methodology on a clinically relevant segmentation problem with expensive and partially disagreeing expert annotations.

## Core question

Given the same CT scans and the same small human-labelled patient subset:

1. How much does a modern specialist learn from labelled data alone?
2. How much does classical SSL gain from the unlabelled remainder?
3. Does a frozen medical foundation model add anything beyond that SSL gain?
4. At what annotation fraction does the VFM contribution become negligible?
5. Does VFM guidance help specifically in regions/cases with annotation disagreement?

## Main training conditions

Let `L_f` be the patient subset labelled at fraction `f` and `U_f` the remaining frozen training patients.

### SUP

`YOLO26-sem(L_f)`

### MT

`MeanTeacher(YOLO26-sem, L_f + U_f)`

### MT + MedSAM

`MeanTeacher(YOLO26-sem, L_f + U_f) + frozen MedSAM co-teacher`

MedSAM can only provide a second opinion / refinement signal on pseudo-labels for unlabelled images. It is not allowed to overwrite human ground truth.

## VFM diagnostic

On labelled validation data only:

1. specialist teacher produces a candidate nodule mask;
2. candidate foreground is converted to a prompt/box;
3. MedSAM predicts a refined mask;
4. teacher mask and MedSAM-refined mask are both scored against the held-out human target.

This isolates whether the foundation model is improving pseudo-label quality before attributing any downstream training gain to it.

## Evidence

For each fraction and seed, compare paired results on the same frozen test patients.

Primary contrasts:

`Δ_SSL(f) = Metric(MT, f) - Metric(SUP, f)`

`Δ_VFM(f) = Metric(MT+MedSAM, f) - Metric(MT, f)`

The interesting result is the **shape of these deltas across annotation budgets**, not a single best score.

The null result is valid: if MedSAM does not improve a well-controlled MT baseline, the benchmark should say so.
