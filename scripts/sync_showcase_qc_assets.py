from __future__ import annotations

import argparse
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy real seven-case LIDC QC contact sheets into the TypeScript showcase."
    )
    parser.add_argument("--qc-root", type=Path, required=True)
    parser.add_argument("--web-public", type=Path, default=Path("web/public/lidc"))
    parser.add_argument("--quality", type=int, default=82)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from PIL import Image
    except ImportError as exc:
        raise SystemExit("Pillow is required: pip install Pillow") from exc

    if not args.qc_root.is_dir():
        raise SystemExit(f"QC root does not exist: {args.qc_root}")

    case_dirs = sorted(path for path in args.qc_root.iterdir() if path.is_dir())
    if not case_dirs:
        raise SystemExit(f"No case directories found under {args.qc_root}")

    args.web_public.mkdir(parents=True, exist_ok=True)
    written = 0
    for case_dir in case_dirs:
        source = case_dir / "contact_sheet.png"
        if not source.exists():
            continue
        destination_dir = args.web_public / case_dir.name
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "contact_sheet.webp"

        image = Image.open(source).convert("RGB")
        if image.width > 1200:
            height = round(image.height * 1200 / image.width)
            image = image.resize((1200, height), Image.Resampling.LANCZOS)
        image.save(destination, "WEBP", quality=args.quality, method=6)
        written += 1
        print(f"wrote {destination}")

    if written == 0:
        raise SystemExit("No contact_sheet.png files were found")

    # Do not leave generated platform metadata behind if the source bundle contains it.
    for name in (".DS_Store", "Thumbs.db"):
        stale = args.web_public / name
        if stale.exists():
            stale.unlink()

    print(f"synced {written} QC cases into {args.web_public}")


if __name__ == "__main__":
    main()
