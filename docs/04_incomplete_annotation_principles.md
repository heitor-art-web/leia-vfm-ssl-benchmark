# Incomplete annotation principles

This benchmark treats missing supervision as **unknown**, not as a negative label.

That distinction matters in both fully unlabeled and partially annotated medical images. A structure that is not annotated may be absent, may be present but unreviewed, or may fall outside the annotation protocol. Collapsing all missing labels into background silently converts uncertainty into false supervision.

## Core states

For each target region or training signal, preserve three conceptual states when the dataset permits it:

- `POSITIVE`: explicitly annotated target;
- `NEGATIVE`: explicitly reviewed absence/background;
- `UNKNOWN`: not annotated or not reviewed.

`UNKNOWN` pixels or structures should be masked out of supervised losses unless a pseudo-labeling rule accepts them under a documented confidence criterion.

## Why this belongs in the benchmark

The first benchmark phase uses patient-level labeled/unlabeled splits. A second phase should simulate incomplete annotation inside otherwise labeled cases, for example:

1. **class-missing masks** — one cardiac structure omitted from a subset of patients;
2. **partial masks** — selected slices or regions withheld;
3. **sparse supervision** — only a subset of slices annotated;
4. **mixed supervision** — dense masks for some cases and weaker prompts/boxes for others.

The benchmark should never evaluate a method by pretending hidden ground truth is genuine background. Hidden labels remain available only for evaluation.

## Experimental extension

For a masking operator `M`, construct observed supervision

`Y_obs = M(Y_full)`

while keeping `Y_full` sealed for evaluation. Training receives `Y_obs` plus an explicit validity mask `V`.

The supervised loss is then applied only where `V = 1`:

`L_sup = sum(V * loss(pred, Y_obs)) / sum(V)`

This makes the incomplete-annotation assumption auditable and avoids conflating missing labels with negatives.
