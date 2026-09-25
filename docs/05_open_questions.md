# Open questions after the minimal benchmark

Only open these branches after the 36-run core matrix is complete.

## Incomplete annotations instead of fully missing labels

Move from patient-level labelled/unlabelled to:

- partially labelled slices;
- partial masks / scribbles;
- missing structures (e.g. LV known, myocardium unknown);
- positive / negative / unknown pixel states.

## Active learning

Use uncertainty and VFM disagreement to choose the next patient/slice for annotation. Compare random annotation acquisition against uncertainty/disagreement sampling.

## Foundation-model choice

Replace MedSAM with SAM2 / MedSAM2 / SAM-Med3D or a general-domain VFM to test whether medical pretraining is the key variable.

## Pseudo-label quality

Measure pseudo-label precision/recall directly on held-out labelled validation data, not only final segmentation Dice.

## Domain shift

Train/adapt on ACDC and test transfer on another cardiac MRI dataset where licensing permits.

## Annotation economics

Convert fractions into approximate expert annotation time and report performance per hour of human labelling.
