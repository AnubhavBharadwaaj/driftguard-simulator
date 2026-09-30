from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from core import analyze, append_override, audit_pdf, demo_lot
from stdf_io import read_stdf_v4


ROOT = Path(__file__).resolve().parents[1]
st.set_page_config(page_title="Sahodrift Inspector", layout="wide")
st.title("Sahodrift Inspector")
st.caption("Auditable 24 hour burn in screening prototype")

source = st.radio("Input", ["Built in simulated lot", "CSV", "STDF v4"], horizontal=True)
if source == "Built in simulated lot":
    raw = demo_lot()
else:
    upload = st.file_uploader("Upload inspection data", type=["csv"] if source == "CSV" else ["stdf", "std"])
    if upload is None:
        st.stop()
    raw = pd.read_csv(upload) if source == "CSV" else read_stdf_v4(upload.getvalue())

tier = st.selectbox("Operating tier", ["REJECT", "REVIEW"])
budget = st.slider("False alarm budget for REJECT", 0.1, 2.0, 0.5, 0.1, format="%.1f%%", disabled=tier != "REJECT")
decisions, audit = analyze(raw, budget, tier)
st.metric("Flagged parts", f"{audit['n_flagged']} / {audit['n_parts']}")
st.info("REJECT controls the healthy part budget. REVIEW prioritises sensitivity and therefore has a larger measured workload.")

lot = st.selectbox("Lot", sorted(raw.lot_id.unique()))
plot = raw[raw.lot_id.eq(lot)].melt(["part_id", "lot_id", "param"], [column for column in ["value_0h", "value_24h", "value_96h", "value_168h"] if column in raw], var_name="checkpoint", value_name="value")
st.line_chart(plot, x="checkpoint", y="value", color="part_id")
st.dataframe(decisions, hide_index=True)

flagged = decisions[decisions.flagged]
if len(flagged):
    chosen = st.selectbox("Reason card", flagged.part_id.tolist())
    row = flagged[flagged.part_id.eq(chosen)].iloc[0]
    local = raw[raw.part_id.eq(chosen)].copy()
    st.subheader(chosen)
    st.write({
        "conformal p value": round(float(row.conformal_p), 6), "budget": audit["threshold"],
        "strongest evidence": row.evidence_source, "counterfactual": f"not flagged once rank p exceeds {audit['threshold']:.4g}",
        "168 h forecast": "shown when value_168h or the production forecaster is available",
        "three similar past parts": "requires the production retrieval index",
    })
    st.dataframe(local, hide_index=True)
    with st.form("override"):
        engineer = st.text_input("Engineer")
        reason = st.text_input("Override reason")
        if st.form_submit_button("Append override"):
            append_override(ROOT / "out" / "v4" / "demo_overrides.jsonl", chosen, engineer, reason)
            st.success("Override appended. Existing records were not changed.")

audit_json = json.dumps(audit, indent=2).encode()
col1, col2 = st.columns(2)
col1.download_button("Download audit JSON", audit_json, "sahodrift_audit.json", "application/json")
col2.download_button("Download audit PDF", audit_pdf(audit), "sahodrift_audit.pdf", "application/pdf")

