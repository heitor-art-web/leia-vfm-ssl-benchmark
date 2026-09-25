# Research showcase UI

A small TypeScript/React presentation layer for the pulmonary LIDC-IDRI benchmark.

The web app is deliberately separated from the scientific Python code. Page content and benchmark state are stored in JSON under `src/data/`; React components render those records without inventing benchmark metrics.

## Run locally

```bash
cd web
npm install
npm run dev
```

Build:

```bash
npm run build
```

## Pages

- Overview — research question, methods, annotation policy and pipeline
- Cases — seven patient-unique visual-QC cases
- Viewer — real QC contact sheets when assets are synced
- Benchmark — frozen split and label-budget protocol
- Results — result registry; scores remain blank until full benchmark runs complete
- About — scope, provenance and clinical boundaries

## Real QC assets

Generate/download the seven-case QC bundle first, then from the repository root:

```bash
python scripts/sync_showcase_qc_assets.py \
  --qc-root /path/to/lidc_qc7_magenta \
  --web-public web/public/lidc
```

The sync script converts each real `contact_sheet.png` to a compact WebP file suitable for the showcase.
