# Experiment matrix

Core training matrix = **3 conditions × 4 annotation budgets × 3 seeds = 36 runs**.

Optional 100% supervised ceiling adds 3 runs.

| Condition | Labels | Unlabelled pool | MedSAM | Purpose |
|---|---:|---|---|---|
| SUP | 1/5/10/25% | ignored | no | YOLO26 specialist label-efficiency baseline |
| MT | 1/5/10/25% | yes | no | classical Mean Teacher gain |
| MT+MedSAM | 1/5/10/25% | yes | frozen co-teacher | incremental VFM gain |
| SUP-100 | 100% | none | no | approximate specialist ceiling |

## Validation-only VFM diagnostic

Not counted as an additional training arm:

| Diagnostic | Input prompt | Ground-truth access at inference? | Purpose |
|---|---|---|---|
| Teacher mask | none | no | pseudo-label baseline |
| Teacher → MedSAM | teacher-derived box | no | measure VFM refinement effect |

This diagnostic is run on labelled validation cases so both pseudo-label candidates can be scored against hidden-from-model human targets.

## Derived quantities

- `SSL gain = MT - SUP`
- `VFM gain = MT+MedSAM - MT`
- `Total low-label gain = MT+MedSAM - SUP`
- `Label efficiency`: performance as a function of labelled patients
- `VFM pseudo-label delta = quality(Teacher→MedSAM) - quality(Teacher)`

## Interpretation

Do not pre-commit to a positive VFM result.

Scientifically useful patterns include:

- positive VFM gain concentrated at 1–5% labels;
- no VFM gain once the specialist teacher is strong;
- pseudo-label quality improves but downstream training does not;
- VFM gains are concentrated in high-reader-disagreement regions;
- VFM refinement increases precision while reducing pseudo-label coverage.

The benchmark should report whichever pattern the data support.
