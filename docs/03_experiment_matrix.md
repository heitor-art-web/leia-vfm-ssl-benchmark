# Experiment matrix

Core matrix = **3 conditions × 4 annotation budgets × 3 seeds = 36 runs**.

Optional 100% supervised ceiling adds 3 runs.

| Condition | Labels | Unlabelled pool | MedSAM | Purpose |
|---|---:|---|---|---|
| SUP | 1/5/10/25% | ignored | no | specialist label-efficiency baseline |
| MT | 1/5/10/25% | yes | no | classical SSL gain |
| MT+MedSAM | 1/5/10/25% | yes | frozen | incremental VFM gain |
| SUP-100 | 100% | none | no | approximate task ceiling |

## Derived quantities

- `SSL gain = MT - SUP`
- `VFM gain = MT+MedSAM - MT`
- `Total low-label gain = MT+MedSAM - SUP`
- `Label efficiency`: smallest label fraction that reaches a chosen percentage of SUP-100 performance.

## Strong result pattern

A convincing result is not necessarily “highest Dice”. One informative pattern would be:

- large positive VFM gain at 1%;
- smaller positive gain at 5%;
- near-zero gain at 10–25%;
- pseudo-label acceptance precision increasing as specialist quality improves.

That would empirically support the idea that foundation priors are most valuable when human supervision is extremely scarce.
