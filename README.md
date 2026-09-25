# VFM + SSL Medical Segmentation Benchmark

Independent, public research scaffold to test a narrow question:

> **Does classical semi-supervised learning still add measurable value when a medical vision foundation model is available, especially under very low annotation budgets?**

This repository is intentionally small and falsifiable. It starts with one dataset, one foundation model, one supervised specialist baseline, one SSL method, and four annotation budgets.

## Minimal benchmark

- **Dataset:** ACDC cardiac MRI segmentation (100 labelled training patients in the public challenge release)
- **Foundation model:** MedSAM (frozen generalist)
- **Specialist baseline:** 2D U-Net, supervised only
- **SSL method:** Mean Teacher (same U-Net architecture)
- **VFM-assisted SSL:** Mean Teacher + frozen MedSAM pseudo-label refinement/agreement
- **Label budgets:** 1%, 5%, 10%, 25% of the patient-level training pool
- **Optional ceiling:** 100% supervised
- **Primary metrics:** Dice (LV / MYO / RV and macro), HD95
- **Secondary metrics:** ASD, calibration/confidence coverage, compute time, label-efficiency curve
- **Repetitions:** 3 seeds minimum; 5-fold CV as the stronger second stage

## Why ACDC?

1. Public, small enough to run locally, but medically meaningful.
2. Patient-level ground truth avoids needing a private clinical dataset.
3. It is a long-standing semi-supervised medical segmentation benchmark.
4. It is already used by recent SAM/VFM + SSL work, which makes comparison with the literature possible.
5. Cardiac MRI gives a clean bridge between classical medical imaging and current foundation-model work.

## Experimental conditions

For each annotation fraction `f ∈ {0.01, 0.05, 0.10, 0.25}`:

1. `SUP`: U-Net trained only on labelled patients.
2. `MT`: Mean Teacher with the same labelled patients + remaining train patients as unlabelled.
3. `MT+MedSAM`: identical Mean Teacher setup, with frozen MedSAM used only to refine/validate pseudo-labels from the unlabelled stream.

This lets us separate:

- gain from **unlabelled data** (`MT - SUP`), and
- gain from **foundation-model prior** (`MT+MedSAM - MT`).

The same patient split, augmentation policy, optimizer family, validation set and test set must be reused across conditions.

## Scientific hypothesis

The working hypothesis is deliberately testable:

> The marginal value of VFM guidance should be largest in the 1–5% label regime, shrink as label coverage increases, and may disappear when task-specific supervision is already sufficient.

The opposite result is equally useful: if `MT+MedSAM` does not consistently beat `MT`, that is evidence that a foundation model is not automatically useful just because it exists.

## Repository map

```text
configs/                 experiment configuration
docs/                    scope, literature map, protocol and open questions
literature/              living bibliography + search log
src/leia_benchmark/      original benchmark scaffold
scripts/                 split generation / experiment utilities
experiments/             frozen run matrix
results/                 result templates (no fabricated numbers)
tests/                   basic unit tests
```

## Current status

**Phase 0 — research + protocol + scaffold.** No performance claims are made yet.

The repository deliberately does not include the ACDC dataset or MedSAM weights. Obtain those from their official sources and respect their licenses/terms.

## Reproducibility rules

- Split at **patient level**, never slice level.
- Freeze test patients before any tuning.
- Do not use model predictions as human ground truth.
- `unlabelled` means **unknown label**, not background / negative.
- Use the same frozen splits for all methods.
- Report all seeds; do not select only the best run.
- Any pseudo-label acceptance threshold must be tuned on validation only.
- Keep foundation-model and specialist predictions separately auditable.

See `docs/02_benchmark_protocol.md` for the full protocol.

## Project status and affiliation

This is an independent benchmark/reproduction project. It is not an official repository of INESC TEC, NIAR, LEIA, ACDC or MedSAM, and it does not claim affiliation with those projects. Names of external datasets, methods and projects are used only for scientific attribution.
