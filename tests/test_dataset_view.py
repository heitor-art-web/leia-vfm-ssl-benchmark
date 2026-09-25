import csv
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_lidc_dataset_view.py"


def _write_manifest(root: Path, split: str, rows: list[tuple[str, str]]) -> None:
    path = root / f"manifest_{split}.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["patient_id", "image", "mask"])
        writer.writeheader()
        for patient_id, stem in rows:
            image = root / "images" / split / f"{stem}.png"
            mask = root / "masks" / split / f"{stem}.png"
            image.parent.mkdir(parents=True, exist_ok=True)
            mask.parent.mkdir(parents=True, exist_ok=True)
            image.touch()
            mask.touch()
            writer.writerow(
                {
                    "patient_id": patient_id,
                    "image": str(image.relative_to(root)),
                    "mask": str(mask.relative_to(root)),
                }
            )


def test_dataset_view_hides_unlabelled_train_masks_from_supervised_yaml(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_manifest(
        dataset,
        "train",
        [
            ("LIDC-IDRI-0001", "LIDC-IDRI-0001_scan_z0001"),
            ("LIDC-IDRI-0002", "LIDC-IDRI-0002_scan_z0001"),
        ],
    )
    _write_manifest(dataset, "val", [("LIDC-IDRI-0003", "LIDC-IDRI-0003_scan_z0001")])
    _write_manifest(dataset, "test", [("LIDC-IDRI-0004", "LIDC-IDRI-0004_scan_z0001")])

    labelled = tmp_path / "labelled.txt"
    labelled.write_text("LIDC-IDRI-0001\n", encoding="utf-8")
    view = tmp_path / "view"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(dataset),
            str(labelled),
            "--output-dir",
            str(view),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    train_labelled = (view / "train_labelled.txt").read_text(encoding="utf-8")
    train_unlabelled = (view / "train_unlabelled.txt").read_text(encoding="utf-8")
    assert "LIDC-IDRI-0001" in train_labelled
    assert "LIDC-IDRI-0002" not in train_labelled
    assert "LIDC-IDRI-0002" in train_unlabelled
    assert "LIDC-IDRI-0001" not in train_unlabelled

    data = yaml.safe_load((view / "supervised.yaml").read_text(encoding="utf-8"))
    assert Path(data["train"]).resolve() == (view / "train_labelled.txt").resolve()
    assert data["masks_dir"] == "masks"
    assert data["names"] == {0: "background", 1: "pulmonary_nodule"}


def test_dataset_view_rejects_labelled_patient_outside_train(tmp_path):
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    _write_manifest(dataset, "train", [("LIDC-IDRI-0001", "LIDC-IDRI-0001_scan_z0001")])
    _write_manifest(dataset, "val", [("LIDC-IDRI-0002", "LIDC-IDRI-0002_scan_z0001")])
    _write_manifest(dataset, "test", [("LIDC-IDRI-0003", "LIDC-IDRI-0003_scan_z0001")])

    labelled = tmp_path / "labelled.txt"
    labelled.write_text("LIDC-IDRI-9999\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(dataset),
            str(labelled),
            "--output-dir",
            str(tmp_path / "view"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "absent from train manifest" in (result.stderr + result.stdout)
