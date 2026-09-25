# Open questions after the minimal benchmark

Only open these branches after the 36-run core matrix and pseudo-label diagnostic are stable.

## Consensus / incomplete-annotation policy

LIDC-IDRI already contains multi-reader disagreement. Test whether conclusions change under alternative pre-declared policies:

- minimum reader count 2 vs 3 vs 4;
- consensus fraction 0.5 vs 0.75;
- disagreement as UNKNOWN vs confidence-weighted soft targets;
- reader-count-aware pseudo-label thresholds.

## Active learning

Use teacher uncertainty and teacher↔VFM disagreement to choose the next patient/slice for annotation. Compare against random annotation acquisition.

## Foundation-model choice

Replace MedSAM with SAM2 / MedSAM2 / SAM-Med3D or a general-domain VFM to test whether medical pretraining is the key variable.

## Specialist choice

Replicate the central result with a second specialist architecture (e.g. U-Net-family or a 3D model) to test whether conclusions are YOLO26-specific.

## 2D vs 2.5D vs 3D

The first benchmark uses 2.5D because YOLO26-sem is 2D. Later compare against a volumetric specialist or VFM while preserving the patient split and target policy.

## Lesion-aware evaluation

Extend semantic masks to connected-component / nodule-level evaluation:

- lesion matching;
- sensitivity;
- false positives per scan;
- FROC curves.

## Nodule characterization

Only after segmentation is stable, investigate morphology or malignancy-related prediction. Keep this as a separate clinical task with its own labels, splits and claims.

## Domain shift

Evaluate transfer to another public lung CT cohort where licensing and annotation compatibility permit.

## Annotation economics

Convert annotation budgets into approximate radiologist time/cost and report performance per unit of expert annotation.
