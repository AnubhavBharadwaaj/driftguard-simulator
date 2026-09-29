from __future__ import annotations

import numpy as np
import pandas as pd


PARAMS = ["leakage_uA", "iddq_uA", "delay_ns"]


def _corr_matrix(rho: float) -> np.ndarray:
    return np.full((3, 3), rho) + np.eye(3) * (1.0 - rho)


def simulate_dataset(cfg: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    s = cfg["simulation"]
    rng = np.random.default_rng(seed)
    corr = _corr_matrix(float(s["parameter_correlation"]))
    chol = np.linalg.cholesky(corr)
    limits = {p: float(s["parameters"][p]["limit"]) for p in PARAMS}
    tester_offsets = rng.uniform(
        -float(s["tester_offset_max"]), float(s["tester_offset_max"]),
        size=(int(s["n_testers"]), len(PARAMS)),
    )
    rows: list[dict] = []
    lot_meta: list[dict] = []
    part_counter = 0
    for lot_index in range(int(s["n_lots"])):
        lot_id = f"L{lot_index:02d}"
        family = int(rng.integers(0, int(s["n_families"])))
        lot_size = int(rng.choice(s["lot_sizes"]))
        rate = float(rng.uniform(*s["defect_rate_range"]))
        n_defects = max(1, int(round(lot_size * rate)))
        defect_positions = set(rng.choice(lot_size, size=n_defects, replace=False).tolist())
        lot_log_shift = chol @ rng.normal(0, float(s["lot_median_log_sigma"]), 3)
        medians = np.array([
            float(s["parameters"][p]["family_medians"][family]) for p in PARAMS
        ]) * np.exp(lot_log_shift)
        medians = np.minimum(medians, np.array([limits[p] for p in PARAMS]) * 0.72)
        lot_drift = max(0.01, rng.normal(float(s["healthy_drift_mean"]), float(s["lot_drift_sd"])))
        lot_meta.append({"lot_id": lot_id, "device_family": family, "lot_size": lot_size, "defect_rate": rate})
        for j in range(lot_size):
            tester = int(rng.integers(0, int(s["n_testers"])))
            slot = int(rng.integers(1, int(s["n_board_slots"]) + 1))
            part_id = f"P{part_counter:06d}"
            part_counter += 1
            z = chol @ rng.normal(size=3)
            x0 = medians * np.exp(z * float(s["part_log_sigma"]))
            n_exp = rng.uniform(*s["healthy_n_range"], size=3)
            a = np.maximum(0.005, rng.normal(lot_drift, float(s["healthy_drift_sd"]), 3))
            defect = j in defect_positions
            defect_type = "healthy"
            defect_param = None
            late_amp = np.zeros(3)
            if defect:
                types = list(s["defect_type_probs"].keys())
                probs = np.array(list(s["defect_type_probs"].values()), dtype=float)
                probs /= probs.sum()
                defect_type = str(rng.choice(types, p=probs))
                k = int(rng.integers(0, 3))
                defect_param = PARAMS[k]
                if defect_type == "D1_high_in_limit":
                    mult = float(rng.uniform(*s["d1_level_multiple_range"]))
                    x0[k] = min(medians[k] * mult, limits[PARAMS[k]] * float(s["d1_limit_fraction_cap"]))
                    a[k] = float(rng.uniform(0.015, 0.06))
                elif defect_type == "D2_fast_drifter":
                    a[k] = float(rng.uniform(*s["d2_drift_range"]))
                    desired = float(rng.uniform(0.82, 1.18)) * limits[PARAMS[k]]
                    a[k] = min(float(s["d2_drift_range"][1]), max(a[k], desired / max(x0[k], 1e-9) - 1.0))
                    n_exp[k] = float(rng.uniform(0.55, 0.95))
                elif defect_type == "D3_late_accelerator":
                    late_amp[k] = float(rng.uniform(*s["d3_late_amplitude_range"]))
                    n_exp[k] = float(rng.uniform(0.20, 0.40))
                elif defect_type == "D4_multivariate":
                    signs = np.array([1.0, -1.0, 1.0])
                    rng.shuffle(signs)
                    x0 = medians * np.exp(signs * float(s["d4_sigma_shift"]) * float(s["part_log_sigma"]))
                    x0 = np.minimum(x0, np.array([limits[p] for p in PARAMS]) * 0.88)
            base = {
                "part_id": part_id, "lot_id": lot_id, "device_family": family,
                "tester_id": tester, "board_slot_id": slot, "temperature_c": float(s["temperature_c"]),
                "is_defect": int(defect), "defect_type": defect_type, "defect_parameter": defect_param or "",
            }
            for pi, p in enumerate(PARAMS):
                base[f"true_{p}_x0"] = float(x0[pi])
                for t in s["checkpoints_h"]:
                    frac = float(t) / 168.0
                    true_val = x0[pi] * (1.0 + a[pi] * (frac ** n_exp[pi] if t > 0 else 0.0))
                    if t > 96:
                        true_val *= 1.0 + late_amp[pi] * ((float(t) - 96.0) / 72.0) ** 2
                    noise_sd = float(rng.uniform(*s["measurement_noise_range"]))
                    observed = true_val * (1.0 + tester_offsets[tester, pi]) * rng.normal(1.0, noise_sd)
                    if rng.random() < float(s["missing_rate"]):
                        observed = np.nan
                    base[f"{p}_{int(t)}h"] = float(observed) if np.isfinite(observed) else np.nan
            rows.append(base)
    df = pd.DataFrame(rows)
    lot_ids = np.array(sorted(df["lot_id"].unique()))
    shuffled = rng.permutation(lot_ids)
    ntr = int(s["split_lots"]["train"])
    nca = int(s["split_lots"]["calibration"])
    split_map = {lot: "train" for lot in shuffled[:ntr]}
    split_map.update({lot: "calibration" for lot in shuffled[ntr:ntr + nca]})
    split_map.update({lot: "test" for lot in shuffled[ntr + nca:]})
    df["split"] = df["lot_id"].map(split_map)
    return df, {"lots": lot_meta, "split_map": split_map, "tester_offsets": tester_offsets.tolist()}


def impute_by_family(train: pd.DataFrame, frames: list[pd.DataFrame], columns: list[str]) -> list[pd.DataFrame]:
    global_med = train[columns].median(numeric_only=True)
    family_med = train.groupby("device_family")[columns].median()
    out = []
    for frame in frames:
        f = frame.copy()
        for c in columns:
            f[c] = f[c].fillna(f["device_family"].map(family_med[c])).fillna(global_med[c])
        out.append(f)
    return out
