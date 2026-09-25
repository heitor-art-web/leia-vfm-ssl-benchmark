import csv
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "lidc_benchmark.yaml"
RUN_MATRIX_PATH = ROOT / "experiments" / "run_matrix.csv"


def _config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _runs() -> list[dict[str, str]]:
    with RUN_MATRIX_PATH.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_core_contract_is_lidc_yolo26_mean_teacher_medsam():
    cfg = _config()
    assert cfg["dataset"]["name"] == "LIDC-IDRI"
    assert cfg["dataset"]["split_unit"] == "patient"
    assert cfg["dataset"]["mask_values"] == {
        "background": 0,
        "nodule": 1,
        "ignore_unknown": 255,
    }
    assert cfg["model"]["specialist_family"] == "yolo26-sem"
    assert cfg["model"]["benchmark_checkpoint"] == "yolo26s-sem.pt"
    assert cfg["ssl"]["method"] == "mean_teacher"
    assert cfg["foundation"]["model"] == "MedSAM"
    assert cfg["foundation"]["prompt_source"] == "ema_teacher_prediction"
    assert cfg["foundation"]["ground_truth_prompt_for_unlabelled_or_test"] is False


def test_annotation_budgets_and_seeds_are_frozen():
    cfg = _config()
    assert cfg["benchmark"]["label_fractions"] == [0.01, 0.05, 0.10, 0.25]
    assert cfg["benchmark"]["seeds"] == [1337, 2026, 31415]
    assert cfg["benchmark"]["conditions"] == ["SUP", "MT", "MT_MEDSAM"]


def test_ct_augmentation_does_not_mix_adjacent_slice_channels_or_patients():
    aug = _config()["training"]["augmentation"]
    assert aug["hsv_h"] == 0.0
    assert aug["hsv_s"] == 0.0
    assert aug["hsv_v"] == 0.0
    assert aug["mosaic"] == 0.0
    assert aug["mixup"] == 0.0
    assert aug["copy_paste"] == 0.0


def test_run_matrix_matches_core_config():
    cfg = _config()
    runs = _runs()

    core = [row for row in runs if row["status"] == "planned"]
    optional = [row for row in runs if row["status"] == "optional"]

    assert len(core) == 36
    assert len(optional) == 3
    assert {row["dataset"] for row in runs} == {"LIDC-IDRI"}
    assert {row["specialist"] for row in runs} == {cfg["model"]["benchmark_checkpoint"]}

    expected = {
        (condition, f"{fraction:g}", str(seed))
        for condition in cfg["benchmark"]["conditions"]
        for fraction in cfg["benchmark"]["label_fractions"]
        for seed in cfg["benchmark"]["seeds"]
    }
    observed = {
        (row["condition"], f"{float(row['label_fraction']):g}", row["seed"])
        for row in core
    }
    assert observed == expected


def test_medsam_component_prompt_policy_is_explicit():
    foundation = _config()["foundation"]
    assert foundation["initial_prompt_type"] == "bounding_box"
    assert int(foundation["min_teacher_component_pixels"]) >= 1
    assert int(foundation["box_padding_pixels"]) >= 0
    assert foundation["rejected_pixels"] == 255
