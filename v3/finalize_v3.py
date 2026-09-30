from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from run_v3 import ROOT, OUT, best_detector, impute, generate_evaluation, split_calibration_lots, write_json


BUNDLE = ROOT.parent / "Rival_Trace" / "codex_v3_work" / "codex_v3"


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def make_cards() -> list[dict]:
    cfg = yaml.safe_load((ROOT / "config_v2.yaml").read_text())
    seed = int(cfg["simulation"]["seeds"][0])
    sets = generate_evaluation(cfg, seed)
    _, calibration, test, _ = impute(
        sets["train"],
        [sets["train"], sets["calibration"], sets["test_inprior"], sets["test_shifted"]],
        cfg,
    )
    c1, c2 = split_calibration_lots(calibration, seed)
    bundle = best_detector(c1, c2, test, cfg, seed)
    selected = "twice_mean"
    pvalues = np.asarray(bundle["pvalues"][selected])
    sources = np.asarray(bundle["source_pvalues"])

    defect_type = test.defect_type.fillna("healthy").astype(str).to_numpy()
    is_defect = test.is_defect.to_numpy(int) == 1
    chosen: list[tuple[str, int]] = []
    for label in ["D1_high_in_limit", "D2_fast_drifter", "D4_multivariate"]:
        idx = np.flatnonzero(defect_type == label)
        if len(idx):
            prefix = "strong" if label == "D4_multivariate" else "representative"
            chosen.append((f"{prefix}_{label.split('_', 1)[0]}", int(idx[np.argmin(pvalues[idx])])))
    false_alarm = np.flatnonzero((~is_defect) & (pvalues <= 0.005))
    if len(false_alarm):
        chosen.append(("false_alarm", int(false_alarm[np.argmin(pvalues[false_alarm])])))
    missed = np.flatnonzero(is_defect & (pvalues > 0.005))
    if len(missed):
        chosen.append(("missed_defect", int(missed[np.argmax(pvalues[missed])])))

    cards = []
    for card_type, idx in chosen[:5]:
        row = test.iloc[idx]
        flagged = bool(pvalues[idx] <= 0.005)
        cards.append(
            {
                "product": "Sahodrift",
                "card_type": card_type,
                "seed": seed,
                "part_id": row.part_id,
                "lot_id": row.lot_id,
                "true_label": defect_type[idx],
                "decision_hour": 24,
                "selected_combiner": selected,
                "conformal_p_value": float(pvalues[idx]),
                "reject_threshold": 0.005,
                "status": "REJECT" if flagged else "CONTINUE",
                "evidence": {
                    "selected_component_fisher_p": float(sources[idx, 0]),
                    "conditional_residual_p": float(sources[idx, 1]),
                },
                "reason": "combined evidence crossed the frozen threshold" if flagged else "combined evidence remained above the frozen threshold",
                "failure_case": card_type in {"false_alarm", "missed_defect"},
            }
        )
    return cards


def make_literature_status() -> list[dict]:
    rows = []
    with (BUNDLE / "papers_v3.tsv").open(newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            rows.append(
                {
                    "priority": row["priority"],
                    "work_package": row["wp"],
                    "reference": row["ref"],
                    "source_url": row["access_url"],
                    "requested_extraction": row["what_to_extract"],
                    "found": None,
                    "status": "not_read_in_simulation_phase",
                    "key_numbers": [],
                    "limitations": "No paper metric is used as an experimental result in v3.",
                    "difference_from_sahodrift": None,
                }
            )
    return rows


def update_results(commit: str) -> None:
    path = OUT / "driftguard_results_v3.json"
    payload = json.loads(path.read_text())
    payload["meta"]["git_commit"] = commit
    payload["meta"]["venvs"] = {"simulation": ".venv, Python 3.11"}
    status = payload["meta"]["wp_status"]
    status.update(
        {
            "WP9": {"status": "skipped", "reason": "No WP9 action is defined in the supplied phase plan."},
            "WP10": {"status": "skipped", "reason": "No WP10 action is defined in the supplied phase plan."},
            "WP11": {"status": "skipped", "reason": "No WP11 action is defined in the supplied phase plan."},
            "WP12": {"status": "skipped", "reason": "No WP12 action is defined in the supplied phase plan."},
            "WP13": {"status": "skipped", "reason": "No WP13 action is defined in the supplied phase plan."},
            "WP14": {"status": "skipped", "reason": "No WP14 action is defined in the supplied phase plan."},
            "WP15": {"status": "partial", "reason": "CORA extracted MAT files and gEOL are complete; Bosch numeric archives remain incomplete."},
            "WP16": {"status": "skipped", "reason": "The literature queue is preserved in literature_v3.json, but full paper extraction is outside the completed simulation phase."},
        }
    )
    write_json(path, payload)


def main() -> None:
    pd.read_csv(OUT / "operating_curves.csv").to_csv(OUT / "curves_v3.csv", index=False)
    write_json(OUT / "cards_v3.json", {"cards": make_cards()})
    write_json(ROOT / "v3" / "literature_v3.json", {"papers": make_literature_status()})
    update_results(git_head())


if __name__ == "__main__":
    main()
