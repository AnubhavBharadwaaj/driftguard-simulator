from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "v4"
DATA = ROOT / "data_ext" / "rival_bench"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "v3"))
sys.path.insert(0, str(ROOT / "v4"))

from conformal import evaluate
from v3.run_v3 import mergers, rival_baselines, smooth_conformal_p
from run_phase_a import cp_interval, source_pvalues, split_three


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def fill_common(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy().reset_index(drop=True)
    result["tester_id"] = result.get("tester_id", "unknown")
    result["slot_row"] = result.get("slot_row", 0)
    result["slot_col"] = result.get("slot_col", 0)
    result["defect_type"] = result.get("defect_type", np.where(result.is_defect == 1, "external_anomaly", "healthy"))
    return result


def aabhas_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    base = DATA / "Aabhas"
    measurements = pd.read_csv(base / "data" / "burnin_measurements.csv.gz")
    truth = pd.read_csv(base / "data" / "ground_truth.csv")
    chosen = measurements[measurements.checkpoint_h.isin([0.0, 24.0, 168.0])].copy()
    wide = chosen.pivot(index=["component_id", "lot_id", "board_id", "socket_row", "socket_col"], columns="checkpoint_h", values=["leakage_na", "iddq_ua", "prop_delay_ns"])
    wide.columns = [f"{name}_{int(hour)}h" for name, hour in wide.columns]
    wide = wide.reset_index().rename(columns={"component_id": "part_id", "board_id": "tester_id", "socket_row": "slot_row", "socket_col": "slot_col"})
    for hour in [0, 24, 168]:
        wide[f"leakage_uA_{hour}h"] = wide[f"leakage_na_{hour}h"] / 1000.0
        wide[f"iddq_uA_{hour}h"] = wide[f"iddq_ua_{hour}h"]
        wide[f"delay_ns_{hour}h"] = wide[f"prop_delay_ns_{hour}h"]
    merged = wide.merge(truth[["component_id", "defect_type", "is_defective", "is_trap"]].rename(columns={"component_id": "part_id"}), on="part_id", how="left")
    merged["is_defect"] = (merged.is_defective.astype(bool) & ~merged.is_trap.astype(bool)).astype(int)
    merged.loc[merged.is_defect == 0, "defect_type"] = "healthy"
    lot_number = merged.lot_id.str.extract(r"(\d+)")[0].astype(int)
    train = fill_common(merged[lot_number < 144])
    calibration = fill_common(merged[(lot_number >= 144) & (lot_number < 192)])
    test = fill_common(merged[(lot_number >= 192) & (lot_number <= 239)])
    meta = {"trap_classes_counted_good": ["Va", "VI", "VII", "all rows where is_trap is true"], "source_split": "train LOT000 to LOT143, calibration LOT144 to LOT191, test LOT192 to LOT239"}
    return train, calibration, test, meta


def burntrace_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    base = DATA / "BurnTrace" / "incoming" / "chaitanya" / "riddhi_inner" / "02_CALIBRATE" / "ModuleA_Calibration.csv"
    raw = pd.read_csv(base)
    mapped = pd.DataFrame({
        "part_id": raw.component_id,
        "lot_id": raw.lot_id,
        "leakage_uA_0h": raw.Input_Leakage_Current_0h,
        "leakage_uA_24h": raw.Input_Leakage_Current_24h,
        "leakage_uA_168h": raw.Input_Leakage_Current_168h,
        "iddq_uA_0h": raw.IDDQ_0h,
        "iddq_uA_24h": raw.IDDQ_24h,
        "iddq_uA_168h": raw.IDDQ_168h,
        "delay_ns_0h": raw.Propagation_Delay_0h,
        "delay_ns_24h": raw.Propagation_Delay_24h,
        "delay_ns_168h": raw.Propagation_Delay_168h,
        "is_defect": raw.is_anomalous.astype(int),
    })
    mapped["defect_type"] = np.where(mapped.is_defect == 1, "external_anomaly", "healthy")
    mapped = fill_common(mapped)
    lots = np.array(sorted(mapped.lot_id.unique()))
    rng = np.random.default_rng(26170)
    rng.shuffle(lots)
    groups = np.array_split(lots, [max(1, int(0.6 * len(lots))), max(2, int(0.8 * len(lots)))])
    train = mapped[mapped.lot_id.isin(groups[0])].reset_index(drop=True)
    calibration = mapped[mapped.lot_id.isin(groups[1])].reset_index(drop=True)
    test = mapped[mapped.lot_id.isin(groups[2])].reset_index(drop=True)
    meta = {"source_split": "lot grouped 60/20/20 inside the published labelled calibration table", "split_differs_from_their_published_holdout": True, "reason": "The designated public holdout has no labels"}
    return train, calibration, test, meta


def at_hour(frame: pd.DataFrame, hour: int) -> pd.DataFrame:
    result = frame.copy()
    if hour == 168:
        for parameter in ["leakage_uA", "iddq_uA", "delay_ns"]:
            result[f"{parameter}_24h"] = result[f"{parameter}_168h"]
    return result


def metric(dataset: str, detector: str, hour: int, frame: pd.DataFrame, flags: np.ndarray, source_repo: str, commit: str, split_note: str) -> dict:
    y = frame.is_defect.to_numpy(int)
    good = y == 0
    defect = y == 1
    ev = evaluate(y, flags, frame.lot_id, frame.defect_type)
    tp = int(np.sum(flags & defect))
    fp = int(np.sum(flags & good))
    return {
        "dataset": dataset,
        "detector": detector,
        "source_repo": source_repo,
        "commit": commit,
        "split_note": split_note,
        "decision_hour": hour,
        "target_far_pct": 0.5,
        "achieved_far_pct": ev["achieved_far_pct"],
        "recall_pct": ev["recall_pct"],
        "n_pos": int(defect.sum()),
        "n_good": int(good.sum()),
        "recall_cp95": cp_interval(tp, int(defect.sum())),
        "far_cp95": cp_interval(fp, int(good.sum())),
    }


def evaluate_dataset(name: str, loader, repo_url: str, published: list[dict], cfg: dict, run_rivals: bool = True) -> dict:
    train, calibration, test, meta = loader()
    measurement_columns = [
        f"{parameter}_{hour}h"
        for parameter in ["leakage_uA", "iddq_uA", "delay_ns"]
        for hour in [0, 24, 168]
    ]
    imputation = {}
    for column in measurement_columns:
        value = float(np.nanmedian(train[column].to_numpy(float)))
        imputation[column] = value
        for frame in [train, calibration, test]:
            frame[column] = frame[column].fillna(value)
    meta["missing_value_policy"] = "training period median per parameter and checkpoint"
    commit = subprocess.check_output(["git", "-C", str(DATA / name), "rev-parse", "HEAD"], text=True).strip()
    rows = []
    rival_rows = []
    for hour in [24, 168]:
        tr, cal, te = at_hour(train, hour), at_hour(calibration, hour), at_hour(test, hour)
        c1, _, c2b = split_three(cal, 26170 + hour)
        rng = np.random.default_rng(26170 + hour)
        c2_sources, _ = source_pvalues(c1, c2b, cfg, rng)
        test_sources, _ = source_pvalues(c1, te, cfg, rng)
        ref_merged = mergers(c2_sources)["twice_mean"]
        test_merged = mergers(test_sources)["twice_mean"]
        good = c2b.is_defect.to_numpy() == 0
        pvalue = smooth_conformal_p(
            -np.log(np.clip(ref_merged[good], 1e-300, 1.0)),
            -np.log(np.clip(test_merged, 1e-300, 1.0)),
            rng,
        )
        rows.append(metric(name, "sahodrift_twice_mean", hour, te, pvalue <= 0.005, repo_url, commit, meta["source_split"]))
        if not run_rivals:
            continue
        rival_train = tr
        baseline_train_note = "full training split"
        if len(tr) > 3000:
            rng_train = np.random.default_rng(26170 + hour)
            defects = np.flatnonzero(tr.is_defect.to_numpy() == 1)
            good_train = np.flatnonzero(tr.is_defect.to_numpy() == 0)
            chosen_good = rng_train.choice(good_train, size=max(1, 3000 - len(defects)), replace=False)
            rival_train = tr.iloc[np.r_[defects, chosen_good]].reset_index(drop=True)
            baseline_train_note = "3,000 part computational cap with every training defect retained after larger caps exceeded the time budget"
        for row in rival_baselines(rival_train, cal, {"test": te}, 26170 + hour):
            if row["variant"] != "recalibrated_0p5":
                continue
            n_pos = int(te.is_defect.sum())
            n_good = int((te.is_defect == 0).sum())
            tp = int(round(row["recall_pct"] * n_pos / 100))
            fp = int(round(row["achieved_far_pct"] * n_good / 100))
            flags_note = {
                "dataset": name,
                "detector": row["id"],
                "source_repo": repo_url,
                "commit": commit,
                "split_note": meta["source_split"] + "; rival baseline fit uses " + baseline_train_note,
                "decision_hour": hour,
                "target_far_pct": 0.5,
                "achieved_far_pct": row["achieved_far_pct"],
                "recall_pct": row["recall_pct"],
                "n_pos": n_pos,
                "n_good": n_good,
                "recall_cp95": cp_interval(tp, n_pos),
                "far_cp95": cp_interval(fp, n_good),
                "ci_note": "Counts reconstructed from the helper's unrounded percentages and exact denominators.",
            }
            rival_rows.append(flags_note)
    return {
        "dataset": name,
        "meta": meta,
        "source_repo": repo_url,
        "commit": commit,
        "sahodrift": rows,
        "rival_baselines": rival_rows,
        "rival_baselines_status": (
            "completed" if run_rivals else
            "skipped_compute_cap: the five reference implementations require repeated tree ensembles on 72,000 training parts; Sahodrift and the repository's published comparator are reported without changing either method"
        ),
        "their_published": published,
    }


def main() -> None:
    cfg = yaml.safe_load((ROOT / "config_v2.yaml").read_text())
    results = [
        evaluate_dataset(
            "Aabhas",
            aabhas_frames,
            "https://github.com/Mihir1107/Aabhas",
            [
                {"decision_hour": 24, "recall_pct": 29.7, "achieved_far_pct": 0.54, "file": "results/safety_slopes_union.csv"},
                {"decision_hour": 168, "recall_pct": 92.7, "yield_loss_pct": 7.0, "file": "modulea/evaluation.py:169", "caveat": "threshold taken from test good parts"},
            ],
            cfg,
            run_rivals=False,
        ),
        evaluate_dataset(
            "BurnTrace",
            burntrace_frames,
            "https://github.com/chaitanychoudhari3107-dotcom/SIH26170-Improved",
            [
                {"decision_hour": 24, "recall_pct": 21.1, "achieved_far_pct": 2.55, "file": "results/09_holdout_metrics.csv"},
                {"decision_hour": 168, "recall_pct": 72.2, "achieved_far_pct": 1.04, "file": "results/09_holdout_metrics.csv"},
            ],
            cfg,
            run_rivals=True,
        ),
    ]
    write_json(OUT / "rival_benchmark_v4.json", {"benchmarks": results})
    print(json.dumps({row["dataset"]: row["sahodrift"] for row in results}, indent=2))


if __name__ == "__main__":
    main()
