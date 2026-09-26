from __future__ import annotations

import argparse
import csv
import io
from pathlib import Path
import statistics
import xml.etree.ElementTree as ET
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cross-check a cohort against the canonical LIDC XML annotations. "
            "The audit intentionally reports raw per-reading-session evidence and "
            "does not invent cross-reader nodule clusters from XML alone."
        )
    )
    parser.add_argument("--xml-zip", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str | None:
    for child in list(element):
        if _local(child.tag) == name:
            text = (child.text or "").strip()
            return text or None
    return None


def _desc_text(element: ET.Element, name: str) -> str | None:
    for child in element.iter():
        if _local(child.tag) == name:
            text = (child.text or "").strip()
            if text:
                return text
    return None


def _series_uid(root: ET.Element) -> str | None:
    return _desc_text(root, "SeriesInstanceUid")


def _inventory(root: ET.Element) -> dict[str, object]:
    sessions = [element for element in root.iter() if _local(element.tag) == "readingSession"]
    volumetric = 0
    point_marks = 0
    non_nodules = 0
    malignancy_values: list[int] = []
    per_session: list[str] = []

    for index, session in enumerate(sessions, start=1):
        session_volumetric = 0
        session_points = 0
        session_non = 0
        session_malignancy: list[int] = []

        for child in list(session):
            name = _local(child.tag)
            if name == "nonNodule":
                non_nodules += 1
                session_non += 1
                continue
            if name != "unblindedReadNodule":
                continue

            characteristics = next(
                (item for item in list(child) if _local(item.tag) == "characteristics"),
                None,
            )
            if characteristics is None:
                point_marks += 1
                session_points += 1
                continue

            volumetric += 1
            session_volumetric += 1
            malignancy = _child_text(characteristics, "malignancy")
            if malignancy is not None:
                value = int(malignancy)
                malignancy_values.append(value)
                session_malignancy.append(value)

        per_session.append(
            f"r{index}:vol={session_volumetric},point={session_points},"
            f"non={session_non},mal={';'.join(map(str, session_malignancy)) or '-'}"
        )

    return {
        "reading_sessions": len(sessions),
        "volumetric_reader_annotations": volumetric,
        "point_nodule_marks": point_marks,
        "non_nodule_marks": non_nodules,
        "all_malignancy_values": ";".join(map(str, malignancy_values)),
        "all_malignancy_median": (
            "" if not malignancy_values else float(statistics.median(malignancy_values))
        ),
        "per_session_inventory": " | ".join(per_session),
    }


def main() -> None:
    args = parse_args()
    if not args.xml_zip.exists():
        raise SystemExit(f"XML ZIP not found: {args.xml_zip}")
    if not args.cohort.exists():
        raise SystemExit(f"cohort CSV not found: {args.cohort}")

    with args.cohort.open(newline="", encoding="utf-8") as handle:
        cohort = list(csv.DictReader(handle))
    if not cohort:
        raise SystemExit("cohort CSV is empty")

    wanted = {row["series_uid"]: row for row in cohort}
    found: dict[str, tuple[str, dict[str, object]]] = {}

    with zipfile.ZipFile(args.xml_zip) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".xml"):
                continue
            try:
                root = ET.parse(io.BytesIO(archive.read(name))).getroot()
            except ET.ParseError as exc:
                raise RuntimeError(f"invalid XML in {name}") from exc
            uid = _series_uid(root)
            if uid in wanted:
                if uid in found:
                    raise RuntimeError(f"duplicate XML series UID in archive: {uid}")
                found[uid] = (name, _inventory(root))

    rows: list[dict[str, object]] = []
    for cohort_row in cohort:
        uid = cohort_row["series_uid"]
        item = found.get(uid)
        row: dict[str, object] = {
            "patient_id": cohort_row.get("patient_id", ""),
            "case_id": cohort_row.get("case_id", ""),
            "role": cohort_row.get("role", ""),
            "series_uid": uid,
            "xml_found": item is not None,
            "xml_path": "" if item is None else item[0],
        }
        if item is not None:
            row.update(item[1])
        rows.append(row)

    missing = [row["series_uid"] for row in rows if not row["xml_found"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Audited {len(rows)} cohort series; matched {len(rows) - len(missing)} XML files.")
    if missing:
        raise RuntimeError(f"missing XML for cohort series: {missing}")
    print(
        "Raw XML inventory written without cross-reader clustering. "
        "Use individual contours/DICOM geometry for benchmark target construction."
    )


if __name__ == "__main__":
    main()
