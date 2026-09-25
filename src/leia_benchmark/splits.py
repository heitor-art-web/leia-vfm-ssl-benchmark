from __future__ import annotations

import math
import random
from dataclasses import dataclass, asdict
from typing import Iterable


@dataclass(frozen=True)
class LabelSplit:
    seed: int
    fraction: float
    labelled: list[str]
    unlabelled: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def make_label_split(patient_ids: Iterable[str], fraction: float, seed: int) -> LabelSplit:
    """Split a frozen train-patient pool into labelled and unlabelled subsets.

    The unit is always a patient, never a slice. At least one patient is labelled.
    """
    ids = sorted(set(patient_ids))
    if not ids:
        raise ValueError("patient_ids is empty")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")

    rng = random.Random(seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_labelled = max(1, math.ceil(len(ids) * fraction))
    labelled = sorted(shuffled[:n_labelled])
    unlabelled = sorted(shuffled[n_labelled:])
    return LabelSplit(seed=seed, fraction=fraction, labelled=labelled, unlabelled=unlabelled)
