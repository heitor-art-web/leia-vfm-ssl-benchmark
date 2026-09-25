from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from leia_benchmark.data.acdc import discover_pairs, parse_patient_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='data/ACDC')
    args = ap.parse_args()

    pairs = discover_pairs(args.root)
    patients = sorted({p.patient_id for p in pairs})
    groups = Counter()
    for pid in patients:
        group = parse_patient_group(Path(args.root) / pid)
        groups[group or 'UNKNOWN'] += 1

    print(f'patients with GT frames: {len(patients)}')
    print(f'image/GT frame pairs: {len(pairs)}')
    print('groups:', dict(groups))


if __name__ == '__main__':
    main()
