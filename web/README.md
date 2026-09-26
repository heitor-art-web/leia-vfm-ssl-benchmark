# Research showcase UI

Static TypeScript/React/Vite presentation layer for the pulmonary LIDC-IDRI benchmark.

The web app is deliberately separated from the scientific Python code. Page content and benchmark state are stored in JSON under `src/data/`; React components render those records without inventing benchmark metrics.

The showcase also keeps a strict distinction between **benchmark-construction evidence** and **model results**:

- the seven bundled CT contact sheets are real visual-QC evidence;
- green means trusted foreground;
- magenta means `UNKNOWN / ignore`;
- Dice/IoU stay blank until frozen full benchmark runs complete;
- nothing in this UI is a cancer diagnosis or a medical-device output.

## Run locally

```bash
cd web
npm install
npm run dev
```

Production validation/build:

```bash
npm run build
```

The production build first checks that all seven real QC assets referenced by `src/data/cases.json` exist and that `public/lidc/provenance.json` is present.

## Pages

- Overview — research question, methods, annotation policy and pipeline
- Cases — seven patient-unique visual-QC cases
- Viewer — real trusted/control/UNKNOWN QC contact sheets
- Benchmark — frozen split and label-budget protocol
- Results — result registry; scores remain blank until full benchmark runs complete
- About — scope, provenance and clinical boundaries

## Real QC assets

The committed showcase assets live under:

```text
public/lidc/<case_id>/contact_sheet.webp
```

They are regenerated from the pinned seven-case LIDC development cohort by `.github/workflows/sync-web-qc-assets.yml`. The workflow downloads the pinned source, reruns the same geometry/target checks used by the QC renderer, converts the contact sheets to compact WebP files and commits them back to the development branch.

`public/lidc/provenance.json` records the source metadata and target/visual policy used for these images.

## Vercel

When importing the repository into Vercel, set **Root Directory** to `web`.

Vercel should detect Vite automatically. If entered manually:

```text
Build Command: npm run build
Output Directory: dist
```

No runtime backend or environment variables are required for the current showcase. Navigation uses URL hashes, so static hosting does not need SPA rewrite rules.
