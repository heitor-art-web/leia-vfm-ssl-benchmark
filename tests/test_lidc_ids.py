import pytest

from leia_benchmark.data.lidc import patient_id_from_path, stable_scan_key


def test_stable_scan_key_keeps_patient_and_is_uid_specific():
    a = stable_scan_key("LIDC-IDRI-0007", "1.2.3.4")
    b = stable_scan_key("LIDC-IDRI-0007", "1.2.3.5")
    assert a.startswith("LIDC-IDRI-0007_")
    assert b.startswith("LIDC-IDRI-0007_")
    assert a != b
    assert patient_id_from_path(a) == "LIDC-IDRI-0007"


def test_stable_scan_key_is_deterministic():
    first = stable_scan_key("LIDC-IDRI-0101", "1.2.840.113619.2.55")
    second = stable_scan_key("path/LIDC-IDRI-0101", "1.2.840.113619.2.55")
    assert first == second


def test_stable_scan_key_rejects_empty_uid():
    with pytest.raises(ValueError):
        stable_scan_key("LIDC-IDRI-0001", "")
