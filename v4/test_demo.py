from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from core import analyze, audit_pdf, demo_lot
from stdf_io import read_stdf_v4


def main() -> None:
    frame = demo_lot()
    decisions, audit = analyze(frame)
    assert len(decisions) == 240
    assert audit["decision_hour"] == 24
    assert audit["n_flagged"] > 0
    assert audit_pdf(audit).startswith(b"%PDF")
    from Semi_ATE import STDF
    records = [STDF.FAR("V4", "<").__repr__(), STDF.PIR("V4", "<").__repr__()]
    ptr = STDF.PTR("V4", "<")
    ptr.set_value("TEST_NUM", 1001); ptr.set_value("RESULT", 1.25); ptr.set_value("TEST_TXT", "leakage_uA")
    records.append(ptr.__repr__())
    prr = STDF.PRR("V4", "<"); prr.set_value("NUM_TEST", 1); prr.set_value("PART_ID", "P001")
    records.append(prr.__repr__())
    parsed = read_stdf_v4(b"".join(records))
    assert parsed.iloc[0].param == "leakage_uA"
    result = {"status": "passed", "n_parts": audit["n_parts"], "n_flagged": audit["n_flagged"], "input_sha256": audit["input_sha256"], "stdf_roundtrip_rows": len(parsed)}
    (ROOT / "out" / "v4" / "demo_test.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
