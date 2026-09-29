from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from conformal import fisher_p, pvalues_from_calibration
from driftpfn.model import DriftPFN
from driftpfn.train_eval import TYPES, predict_lots
from module_a_v2 import combined_pvalues, component_scores, conditional_residual_scores
from simulate_v2 import generate_evaluation, impute


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"


def best_p_getter(cal, cfg):
    cal_components, _, _ = component_scores(cal, cfg)
    cal_conditional = conditional_residual_scores(cal)
    healthy = cal.is_defect.to_numpy() == 0
    cal_combo = combined_pvalues(cal, cal_components, cal_components)
    cal_conditional_p = pvalues_from_calibration(cal_conditional[healthy], cal_conditional)
    raw_cal = fisher_p(np.column_stack([cal_combo["fisher"], cal_conditional_p]))

    def get_p(frame):
        components, _, _ = component_scores(frame, cfg)
        combo = combined_pvalues(cal, cal_components, components)
        conditional = conditional_residual_scores(frame)
        conditional_p = pvalues_from_calibration(cal_conditional[healthy], conditional)
        raw_test = fisher_p(np.column_stack([combo["fisher"], conditional_p]))
        return pvalues_from_calibration(
            -np.log(np.clip(raw_cal[healthy], 1e-300, 1.0)),
            -np.log(np.clip(raw_test, 1e-300, 1.0)),
        )

    return get_p


def counterfactual(test, row_index: int, parameter: str, get_p) -> tuple[float | None, str]:
    median = float(test.loc[test.lot_id == test.loc[row_index, "lot_id"], f"{parameter}_24h"].median())
    lo, hi = max(1e-9, 0.1 * median), 6.0 * median
    grid = np.geomspace(lo, hi, 65)
    pvalues = []
    for value in grid:
        changed = test.copy()
        changed.loc[row_index, f"{parameter}_24h"] = value
        pvalues.append(float(get_p(changed)[row_index]))
    safe = np.asarray(pvalues) > 0.005
    if not safe.any():
        return None, "No unflagged value found in the search range"
    j = int(np.flatnonzero(safe)[-1])
    if j == len(grid) - 1:
        return float(grid[j]), "Unflagged through the search ceiling of six times the lot median"
    left, right = float(grid[j]), float(grid[j + 1])
    # Refine only when the adjacent point is flagged.  The detector is not
    # guaranteed globally monotone, so the grid establishes the local branch.
    for _ in range(24):
        mid = (left + right) / 2
        changed = test.copy()
        changed.loc[row_index, f"{parameter}_24h"] = mid
        if float(get_p(changed)[row_index]) > 0.005:
            left = mid
        else:
            right = mid
    return left, "Local upper boundary found by grid search followed by bisection"


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config_v2.yaml").read_text())
    sets = generate_evaluation(cfg, 0)
    train, cal, test, _ = impute(
        sets["train"],
        [sets["train"], sets["calibration"], sets["test_inprior"], sets["test_shifted"]],
        cfg,
    )
    get_p = best_p_getter(cal, cfg)
    payload = json.loads((OUT / "cards.json").read_text())

    type_prob = None
    checkpoint = OUT / "driftpfn.pt"
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = DriftPFN()
        model.load_state_dict(saved["model"])
        scores, quantiles, type_prob = predict_lots(model, test, "cpu")

    reverse_types = {v: k for k, v in TYPES.items()}
    for card in payload["cards"]:
        if card.get("status") != "available":
            continue
        match = np.flatnonzero(test.part_id.to_numpy() == card["part_id"])
        if not len(match):
            card["counterfactual_note"] = "Part identifier was not found in regenerated seed zero test data"
            continue
        idx = int(match[0])
        if "counterfactual_note" not in card:
            value, note = counterfactual(test, idx, card["parameter"], get_p)
            card["counterfactual_max_24h_not_flagged"] = None if value is None else float(f"{value:.4g}")
            card["counterfactual_note"] = note
        if type_prob is not None:
            pred = int(np.argmax(type_prob[idx]))
            card["predicted_defect_type"] = reverse_types[pred]
            card["predicted_type_probability"] = float(f"{type_prob[idx, pred]:.4g}")
    (OUT / "cards.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
