# LIDC showcase assets

This directory is intentionally populated from the real seven-case QC export rather than with fabricated demo images.

Expected layout:

```text
public/lidc/<case_id>/contact_sheet.webp
```

From the repository root, after generating or downloading the seven-case QC directory, run:

```bash
python scripts/sync_showcase_qc_assets.py \
  --qc-root /path/to/lidc_qc7_magenta \
  --web-public web/public/lidc
```

The UI has a clear fallback state when an asset has not yet been synced.
