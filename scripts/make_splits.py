from __future__ import annotations

import argparse
import json
from pathlib import Path

from leia_benchmark.splits import make_label_split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--patients', required=True, help='Text file with one frozen train-patient ID per line')
    ap.add_argument('--out', required=True)
    ap.add_argument('--fractions', nargs='+', type=float, default=[0.01, 0.05, 0.10, 0.25])
    ap.add_argument('--seeds', nargs='+', type=int, default=[1337, 2026, 31415])
    args = ap.parse_args()

    ids = [x.strip() for x in Path(args.patients).read_text().splitlines() if x.strip()]
    payload = []
    for seed in args.seeds:
        for frac in args.fractions:
            payload.append(make_label_split(ids, frac, seed).to_dict())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f'wrote {len(payload)} splits to {out}')


if __name__ == '__main__':
    main()
