from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
from Semi_ATE import STDF


def read_stdf_v4(payload: bytes) -> pd.DataFrame:
    rows = []
    with tempfile.NamedTemporaryFile(suffix=".stdf") as handle:
        handle.write(payload); handle.flush()
        part_number = 0
        for record in STDF.records_from_file(handle.name):
            record_id = record.id
            if record_id == "PIR":
                part_number += 1
            if record_id != "PTR":
                continue
            rows.append({
                "part_id": f"STDF-{max(part_number, 1):06d}", "lot_id": "STDF-UPLOAD",
                "param": record.get_value("TEST_TXT") or f"test_{record.get_value('TEST_NUM')}",
                "value_0h": float(record.get_value("RESULT")), "value_24h": float(record.get_value("RESULT")),
            })
    if not rows:
        raise ValueError("No PTR parametric records were found in this STDF v4 file")
    return pd.DataFrame(rows)
