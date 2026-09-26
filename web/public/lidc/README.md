# LIDC showcase assets

This directory contains the real seven-case visual-QC evidence used by the research showcase. No fabricated medical images are used here.

Layout:

```text
public/lidc/<case_id>/contact_sheet.webp
public/lidc/provenance.json
```

The assets are generated reproducibly by `.github/workflows/sync-web-qc-assets.yml` from the pinned LIDC development cohort and the repository's QC renderer.

Visual convention:

```text
green   trusted foreground
magenta UNKNOWN / ignore
```

These sheets document benchmark construction and annotation handling. They are not model predictions and must not be presented as cancer diagnoses.
