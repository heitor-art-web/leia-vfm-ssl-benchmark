.PHONY: test

test:
	PYTHONPATH=src pytest -q

check-data:
	PYTHONPATH=src python scripts/check_dataset.py --root data/ACDC
