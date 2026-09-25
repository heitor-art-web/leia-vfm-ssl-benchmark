# Results

Do not enter performance values until they come from the frozen evaluation protocol.

## Main segmentation table

| Labels | SUP Dice | MT Dice | MT+MedSAM Dice | SSL gain | VFM gain |
|---:|---:|---:|---:|---:|---:|
| 1% | TBD | TBD | TBD | TBD | TBD |
| 5% | TBD | TBD | TBD | TBD | TBD |
| 10% | TBD | TBD | TBD | TBD | TBD |
| 25% | TBD | TBD | TBD | TBD | TBD |

Report each entry as mean ± SD across the frozen seeds, with a patient-level bootstrap 95% CI for the key paired contrasts.

## Lesion-aware table

Semantic Dice alone is insufficient for sparse pulmonary nodules. After connected-component matching is frozen, report:

| Labels | Condition | Lesion sensitivity | False positives / scan | IoU | HD95 |
|---:|---|---:|---:|---:|---:|
| 1% | SUP | TBD | TBD | TBD | TBD |
| 1% | MT | TBD | TBD | TBD | TBD |
| 1% | MT+MedSAM | TBD | TBD | TBD | TBD |
| 5% | SUP | TBD | TBD | TBD | TBD |
| 5% | MT | TBD | TBD | TBD | TBD |
| 5% | MT+MedSAM | TBD | TBD | TBD | TBD |
| 10% | SUP | TBD | TBD | TBD | TBD |
| 10% | MT | TBD | TBD | TBD | TBD |
| 10% | MT+MedSAM | TBD | TBD | TBD | TBD |
| 25% | SUP | TBD | TBD | TBD | TBD |
| 25% | MT | TBD | TBD | TBD | TBD |
| 25% | MT+MedSAM | TBD | TBD | TBD | TBD |

## Pseudo-label diagnostic

Run this on the labelled validation subset only. Prompts for MedSAM must come from the EMA teacher prediction, never from the reference mask.

| Labels | Teacher pseudo Dice | Teacher→MedSAM Dice | Δ pseudo Dice | Precision Δ | Recall Δ | Accepted coverage |
|---:|---:|---:|---:|---:|---:|---:|
| 1% | TBD | TBD | TBD | TBD | TBD | TBD |
| 5% | TBD | TBD | TBD | TBD | TBD | TBD |
| 10% | TBD | TBD | TBD | TBD | TBD | TBD |
| 25% | TBD | TBD | TBD | TBD | TBD | TBD |

## Reader-disagreement analysis

Report performance separately for targets derived from different annotation-support levels where sample size permits.

| Reader support | SUP | MT | MT+MedSAM | Notes |
|---|---:|---:|---:|---|
| 4 readers | TBD | TBD | TBD | highest support |
| 3 readers | TBD | TBD | TBD | high support |
| 1–2 readers / UNKNOWN boundary | diagnostic only | diagnostic only | diagnostic only | not converted to hard negative |

## Required reporting

- exact patient split IDs and label-budget subset IDs;
- exact model checkpoint (`yolo26s-sem.pt` for the core matrix unless protocol is amended before runs);
- exact Ultralytics version;
- mean ± SD across all declared seeds;
- patient-level bootstrap CI for key paired contrasts;
- nodule Dice and IoU;
- lesion sensitivity and false positives / scan once matching policy is frozen;
- HD95 only where the metric is defined;
- pseudo-label precision / recall / Dice and accepted coverage;
- reader-support / disagreement analysis;
- GPU, wall-clock and peak-memory budget;
- failures and excluded scans, with reasons.

Do not report pixel accuracy as the main result: background dominates this task.
Do not select the best seed for headline reporting.
Do not claim malignancy prediction or clinical diagnosis from these segmentation results.
