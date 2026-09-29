from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

from conformal import pvalues_from_calibration
from driftpfn.model import DriftPFN
from driftpfn.train_eval import predict_lots
from simulate_v2 import generate_evaluation, impute


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"
CFG = yaml.safe_load((ROOT / "config_v2.yaml").read_text())


def ms(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return {
        "mean": None if not len(a) else float(np.mean(a)),
        "sd": None if not len(a) else (0.0 if len(a) == 1 else float(np.std(a, ddof=1))),
    }


def load_model(path: Path, device: str):
    saved = torch.load(path, map_location="cpu", weights_only=False)
    model = DriftPFN().to(device)
    model.load_state_dict(saved["model"])
    model.eval()
    return model, int(saved.get("lots_seen", 0))


def evaluate_variant(name: str, checkpoint: Path, *, use_slot: bool, no_context: bool, device: str):
    model, lots_seen = load_model(checkpoint, device)
    recalls, maes, d4 = [], [], []
    for seed in CFG["simulation"]["seeds"]:
        sets = generate_evaluation(CFG, int(seed))
        train, cal, test, shifted = impute(
            sets["train"],
            [sets["train"], sets["calibration"], sets["test_inprior"], sets["test_shifted"]],
            CFG,
        )
        cal_score, _, _ = predict_lots(model, cal, device, use_slot=use_slot, no_context=no_context)
        score, quantile, _ = predict_lots(model, test, device, use_slot=use_slot, no_context=no_context)
        healthy_cal = cal.is_defect.to_numpy() == 0
        pvalue = pvalues_from_calibration(cal_score[healthy_cal], score)
        y = test.is_defect.to_numpy()
        flags = pvalue <= 0.005
        recalls.append(100 * np.mean(flags[y == 1]))
        dtype = test.defect_type.to_numpy()
        d4.append(100 * np.mean(flags[dtype == "D4_multivariate"]) if np.any(dtype == "D4_multivariate") else np.nan)
        prediction = test["leakage_uA_24h"].to_numpy() * np.exp(quantile[:, 0, 1])
        maes.append(np.mean(np.abs(test["leakage_uA_168h"].to_numpy() - prediction)))
    return {
        "name": name,
        "checkpoint_lots_seen": lots_seen,
        "recall_pct_at_0p5_far": ms(recalls),
        "recall_D4_pct": ms(d4),
        "mae_leakage_uA": ms(maes),
    }


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    final = OUT / "driftpfn.pt"
    half = OUT / "driftpfn_half.pt"
    rows = [
        evaluate_variant("no_lot_context", final, use_slot=True, no_context=True, device=device),
        evaluate_variant("no_slot_features", final, use_slot=False, no_context=False, device=device),
    ]
    if half.exists():
        rows.append(evaluate_variant("half_training_budget", half, use_slot=True, no_context=False, device=device))
    else:
        rows.append({
            "name": "half_training_budget",
            "status": "skipped",
            "reason": "The 15 minute checkpoint was not produced before training stopped.",
            "recall_pct_at_0p5_far": {"mean": None, "sd": None},
            "recall_D4_pct": {"mean": None, "sd": None},
            "mae_leakage_uA": {"mean": None, "sd": None},
        })
    (OUT / "driftpfn_ablation.json").write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
