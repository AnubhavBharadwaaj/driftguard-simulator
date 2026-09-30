from __future__ import annotations

import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas


MODEL_VERSION = "sahodrift-v4-demo-0.1"
REQUIRED = {"part_id", "lot_id", "param", "value_0h", "value_24h"}


def demo_lot(seed: int = 26170) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(240):
        defect = index in {7, 41, 67, 119, 188, 223}
        for param, centre, sigma in [("leakage_uA", 2.0, 0.11), ("iddq_uA", 17.0, 0.65), ("delay_ns", 8.0, 0.25)]:
            value0 = rng.normal(centre, sigma)
            drift = rng.normal(0.02 * centre, 0.015 * centre)
            if defect:
                drift += centre * (0.35 + 0.15 * rng.random())
            rows.append({
                "part_id": f"DEMO-{index:03d}", "lot_id": "DEMO-LOT-01", "param": param,
                "value_0h": value0, "value_24h": value0 + drift,
                "value_96h": value0 + 2.2 * drift, "value_168h": value0 + 3.1 * drift,
            })
    return pd.DataFrame(rows)


def validate_long(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))
    result = frame.copy()
    for column in ["value_0h", "value_24h", "value_96h", "value_168h"]:
        if column in result:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if result[["value_0h", "value_24h"]].isna().any().any():
        raise ValueError("0 h and 24 h values must be numeric and nonempty")
    return result


def input_hash(frame: pd.DataFrame) -> str:
    payload = frame.sort_values(["lot_id", "part_id", "param"]).to_csv(index=False).encode()
    return hashlib.sha256(payload).hexdigest()


def analyze(frame: pd.DataFrame, far_budget_pct: float = 0.5, tier: str = "REJECT") -> tuple[pd.DataFrame, dict]:
    frame = validate_long(frame)
    parts = []
    for (lot, param), local in frame.groupby(["lot_id", "param"], sort=False):
        median0 = float(local.value_0h.median())
        median24 = float(local.value_24h.median())
        mad0 = max(float((local.value_0h - median0).abs().median()), 1e-9)
        mad24 = max(float((local.value_24h - median24).abs().median()), 1e-9)
        local = local.copy()
        local["z_level"] = (local.value_24h - median24).abs() / (1.4826 * mad24)
        local["z_drift"] = ((local.value_24h - local.value_0h) - (median24 - median0)).abs() / (1.4826 * max(mad0, mad24))
        local["source_score"] = local[["z_level", "z_drift"]].max(axis=1)
        parts.append(local)
    scored = pd.concat(parts, ignore_index=True)
    wide = scored.pivot_table(index=["part_id", "lot_id"], columns="param", values="source_score", aggfunc="max")
    aggregate = wide.max(axis=1).rename("score").reset_index()
    rank = aggregate.groupby("lot_id").score.rank(method="average", ascending=False)
    count = aggregate.groupby("lot_id").part_id.transform("count")
    aggregate["conformal_p"] = rank / (count + 1.0)
    threshold = far_budget_pct / 100.0 if tier == "REJECT" else 0.20
    aggregate["flagged"] = aggregate.conformal_p <= threshold
    aggregate["evidence_source"] = aggregate.apply(
        lambda row: scored[scored.part_id.eq(row.part_id)].sort_values("source_score", ascending=False).iloc[0]["param"], axis=1
    )
    aggregate["counterfactual_score"] = aggregate.score.where(~aggregate.flagged, aggregate.score * 0.8)
    audit = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "model_version": MODEL_VERSION,
        "input_sha256": input_hash(frame), "calibration_set_sha256": "demo-rank-calibration-not-production",
        "tier": tier, "false_alarm_budget_pct": far_budget_pct, "threshold": threshold,
        "decision_hour": 24, "n_parts": int(len(aggregate)), "n_flagged": int(aggregate.flagged.sum()),
        "decisions": aggregate.to_dict(orient="records"),
        "limitations": "Inspector prototype. It demonstrates the audit contract and reason cards; production scores require the frozen v4 calibration artefact.",
    }
    return aggregate, audit


def audit_pdf(audit: dict) -> bytes:
    output = io.BytesIO()
    pdf = canvas.Canvas(output, pagesize=A4)
    y = A4[1] - 45
    for line in [
        "Sahodrift inspection audit", f"Model: {audit['model_version']}", f"Created: {audit['created_utc']}",
        f"Input SHA256: {audit['input_sha256']}", f"Tier: {audit['tier']}",
        f"Decision hour: {audit['decision_hour']}", f"Parts: {audit['n_parts']}", f"Flagged: {audit['n_flagged']}",
    ]:
        pdf.drawString(45, y, line[:110]); y -= 18
    y -= 8
    for row in audit["decisions"]:
        pdf.drawString(45, y, f"{row['part_id']}  p={row['conformal_p']:.4g}  flagged={row['flagged']}  source={row['evidence_source']}")
        y -= 14
        if y < 50:
            pdf.showPage(); y = A4[1] - 45
    pdf.save()
    return output.getvalue()


def append_override(path: Path, part_id: str, engineer: str, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "part_id": part_id, "engineer": engineer, "reason": reason}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")
