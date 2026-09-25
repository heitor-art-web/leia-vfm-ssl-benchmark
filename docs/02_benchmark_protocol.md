# Benchmark protocol

## Dataset

ACDC public training database: 100 patients with expert segmentations and five diagnostic groups.

### Split discipline

**Never split by slice.** Every slice/frame from a patient must remain in the same partition.

Recommended first-stage split:

- 20 patients frozen test set, stratified by diagnostic group;
- 10 patients validation set, stratified where possible;
- 70 patients train pool.

For each seed and label fraction, select labelled patients **only from the train pool**. The remaining train patients are treated as unlabelled.

A stronger second stage is 5-fold patient-level cross-validation.

## Annotation budgets

If train pool = 70 patients:

- 1% → 1 labelled patient (round up)
- 5% → 4 labelled patients
- 10% → 7 labelled patients
- 25% → 18 labelled patients

Store exact patient IDs in versioned JSON split files. Percentage labels in papers are not sufficiently reproducible without IDs.

## Preprocessing

- Load native NIfTI volumes.
- Use ED/ES frames with ground truth.
- Resample/resize to a fixed in-plane size only if required by the specialist.
- Normalise per image or per volume with a frozen policy.
- For MedSAM, convert the single-channel MRI slice to 3-channel input without changing semantic content.
- Do not use test-set statistics.

## Supervised specialist baseline (SUP)

2D U-Net trained on labelled slices from labelled patients.

Loss:

`L_sup = CE + DiceLoss`

## SSL baseline (MT)

EMA teacher with independent perturbations.

`θ_teacher ← α θ_teacher + (1-α) θ_student`

Student loss:

`L = L_sup + λ(t) L_consistency`

where consistency is applied only to unlabelled samples and can be confidence-gated.

## VFM-assisted SSL (MT+MedSAM)

Keep the specialist and Mean Teacher machinery identical to MT.

For an unlabelled slice:

1. teacher predicts probability map;
2. high-confidence foreground prediction yields a candidate mask;
3. derive a bounding box from the candidate foreground;
4. frozen MedSAM predicts a mask from image + candidate box;
5. use only regions where specialist teacher and MedSAM sufficiently agree, or use MedSAM mask as a refinement target with a confidence weight;
6. rejected/ambiguous pixels remain **unknown**, not background.

This isolates the contribution of the foundation-model prior without changing the specialist architecture.

## Metrics

Primary:

- Dice per RV / myocardium / LV;
- macro Dice;
- HD95 macro.

Secondary:

- ASD;
- pseudo-label acceptance coverage;
- pseudo-label precision on validation labels;
- calibration / confidence histograms;
- wall-clock and GPU memory.

## Statistics

At minimum:

- 3 fixed random seeds;
- paired comparison on identical test patients;
- report mean ± SD across seeds;
- bootstrap 95% CIs across test patients for key contrasts.

Do not report only the best seed.

## Leakage controls

- Patient-level splitting.
- Test set frozen before hyperparameter selection.
- If a foundation model may have seen ACDC during pretraining/fine-tuning, record that as a contamination risk.
- Run a sensitivity analysis with a general-domain VFM later if contamination becomes central to the theory.
- Never derive human GT from model output.

## Stop/go criteria

### Stop after phase 1 if

`MT+MedSAM` does not improve over `MT` outside uncertainty across all four label budgets.

That is already a useful result.

### Continue if

The gain is concentrated in 1–5% labels or if pseudo-label quality improves even when final Dice does not. Then investigate confidence calibration, uncertainty gating, active learning or incomplete/partial labels.
