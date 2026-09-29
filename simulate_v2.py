from __future__ import annotations

import numpy as np
import pandas as pd

PARAMS = ["leakage_uA", "iddq_uA", "delay_ns"]


def _corr(rho: float) -> np.ndarray:
    return np.full((3, 3), rho) + np.eye(3) * (1.0 - rho)


def sample_prior(rng: np.random.Generator, cfg: dict) -> dict:
    p = cfg["simulation"]["broad_prior"]
    return {
        "lot_size": int(rng.integers(int(p["lot_size_range"][0]), int(p["lot_size_range"][1]) + 1)),
        "lot_median_log_sigma": float(rng.uniform(*p["lot_median_log_sigma"])),
        "part_log_sigma": float(rng.uniform(*p["part_log_sigma"])),
        "correlation": float(rng.uniform(*p["correlation"])),
        "healthy_drift_mean": float(rng.uniform(*p["healthy_drift_mean"])),
        "n_range": [float(p["exponent_n"][0]), float(p["exponent_n"][1])],
        "noise_range": [float(rng.uniform(*p["noise"])), float(rng.uniform(*p["noise"]))],
        "tester_offset_max": float(rng.uniform(*p["tester_offset_max"])),
        "defect_rate": float(rng.uniform(*p["defect_rate"])),
        "defect_amplitude_scale": float(rng.uniform(0.55, 1.30)),
        "law": "power",
    }


def _arrhenius_factor(temp_c: np.ndarray, cfg: dict) -> np.ndarray:
    s = cfg["simulation"]; ea = float(s["activation_energy_ev"]); kb = float(s["boltzmann_ev_per_k"])
    tref = float(s["temperature_c"]) + 273.15; t = temp_c + 273.15
    return np.exp(-ea / kb * (1.0 / t - 1.0 / tref))


def sample_lot(rng: np.random.Generator, prior: dict, cfg: dict, lot_id: str, family: int | None = None) -> pd.DataFrame:
    s = cfg["simulation"]; family = int(rng.integers(0, 2)) if family is None else int(family)
    n = int(prior["lot_size"]); rho = float(prior["correlation"]); chol = np.linalg.cholesky(_corr(rho))
    limits = np.array([float(s["parameters"][p]["limit"]) for p in PARAMS])
    base_med = np.array([float(s["parameters"][p]["family_medians"][family]) for p in PARAMS])
    med = base_med * np.exp(chol @ rng.normal(0, float(prior["lot_median_log_sigma"]), 3))
    med = np.minimum(med, limits * 0.72)
    theta = float(rng.uniform(0, 2 * np.pi)); magnitude = float(rng.uniform(0, float(s["slot_gradient_max_c"])))
    tester_offsets = rng.uniform(-float(prior["tester_offset_max"]), float(prior["tester_offset_max"]), size=(int(s["n_testers"]), 3))
    rate = float(prior["defect_rate"]); nd = int(round(n * rate)); positions = set(rng.choice(n, nd, replace=False).tolist()) if nd else set()
    rows = []
    for j in range(n):
        slot = int(rng.integers(0, 16)); row = slot // 4; col = slot % 4
        xr = (row - 1.5) / 1.5; xc = (col - 1.5) / 1.5
        delta_t = magnitude * (np.cos(theta) * xr + np.sin(theta) * xc) / np.sqrt(2)
        temp = float(s["temperature_c"]) + delta_t; arr = float(_arrhenius_factor(np.array([temp]), cfg)[0])
        tester = int(rng.integers(0, int(s["n_testers"])))
        z = chol @ rng.normal(size=3); x0 = med * np.exp(z * float(prior["part_log_sigma"]))
        n_exp = rng.uniform(*prior["n_range"], size=3)
        lot_drift = max(0.003, rng.normal(float(prior["healthy_drift_mean"]), 0.25 * float(prior["healthy_drift_mean"])))
        a = np.maximum(0.002, rng.normal(lot_drift, 0.25 * lot_drift, 3)) * arr
        dtype = "healthy"; dparam = ""; late = np.zeros(3)
        if j in positions:
            names = list(s["defect_type_probs"]); probs = np.array(list(s["defect_type_probs"].values()), float); probs /= probs.sum()
            dtype = str(rng.choice(names, p=probs)); k = int(rng.integers(0, 3)); dparam = PARAMS[k]; scale = float(prior["defect_amplitude_scale"])
            if dtype == "D1_high_in_limit":
                mult = 1.0 + (float(rng.uniform(*s["d1_multiple_range"])) - 1.0) * scale
                x0[k] = min(med[k] * mult, 0.97 * limits[k]); a[k] = rng.uniform(0.015, 0.06) * arr
            elif dtype == "D2_fast_drifter":
                a[k] = rng.uniform(*s["d2_drift_range"]) * scale * arr; n_exp[k] = rng.uniform(0.55, 0.95)
            elif dtype == "D3_late_accelerator":
                late[k] = rng.uniform(*s["d3_late_range"]) * scale * arr; n_exp[k] = rng.uniform(0.20, 0.40)
            elif dtype == "D4_multivariate":
                signs = np.array([1.0, -1.0, 1.0]); rng.shuffle(signs)
                x0 = med * np.exp(signs * float(s["d4_sigma_shift"]) * float(prior["part_log_sigma"]) * scale)
                x0 = np.minimum(x0, limits * 0.88)
        rec = {"part_id": f"{lot_id}_P{j:03d}", "lot_id": lot_id, "device_family": family, "tester_id": tester,
               "board_slot_id": slot + 1, "slot_row": row, "slot_col": col, "temperature_c": temp,
               "is_defect": int(dtype != "healthy"), "defect_type": dtype, "defect_parameter": dparam}
        tau = float(prior.get("tau_h", 36.0))
        for pi, p in enumerate(PARAMS):
            for t in s["checkpoints_h"]:
                if prior.get("law", "power") == "logtime":
                    shape = np.log1p(float(t) / tau) / np.log1p(168.0 / tau) if t else 0.0
                else:
                    shape = (float(t) / 168.0) ** n_exp[pi] if t else 0.0
                true = x0[pi] * (1.0 + a[pi] * shape)
                if t > 96: true *= 1.0 + late[pi] * ((float(t) - 96.0) / 72.0) ** 2
                nl, nh = sorted(prior["noise_range"]); noise = float(rng.uniform(nl, nh))
                obs = true * (1.0 + tester_offsets[tester, pi]) * rng.normal(1.0, noise)
                if rng.random() < float(s["missing_rate"]): obs = np.nan
                rec[f"{p}_{int(t)}h"] = obs
        rows.append(rec)
    return pd.DataFrame(rows)


def _inprior(cfg: dict, rng: np.random.Generator) -> dict:
    x = cfg["simulation"]["inprior"]
    nr = list(x["noise_range"])
    return {"lot_size": int(rng.choice(cfg["simulation"]["lot_sizes"])), "lot_median_log_sigma": float(x["lot_median_log_sigma"]),
            "part_log_sigma": float(x["part_log_sigma"]), "correlation": float(x["correlation"]),
            "healthy_drift_mean": float(x["healthy_drift_mean"]), "n_range": list(x["healthy_n_range"]),
            "noise_range": nr, "tester_offset_max": float(x["tester_offset_max"]),
            "defect_rate": float(rng.uniform(*x["defect_rate_range"])), "defect_amplitude_scale": 1.0, "law": "power"}


def generate_evaluation(cfg: dict, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed); s = cfg["simulation"]
    lots = [sample_lot(rng, _inprior(cfg, rng), cfg, f"S{seed}_L{i:02d}") for i in range(int(s["n_lots"]))]
    order = rng.permutation(len(lots)); ntr = int(s["split_lots"]["train"]); ncal = int(s["split_lots"]["calibration"])
    def join(indices, split):
        f = pd.concat([lots[i] for i in indices], ignore_index=True); f["split"] = split; f["test_set"] = "inprior"; return f
    train = join(order[:ntr], "train"); cal = join(order[ntr:ntr+ncal], "calibration"); test = join(order[ntr+ncal:], "test")
    shifted = []
    sh = s["shifted"]
    for i in range(int(s["shifted_test_lots"])):
        pr = _inprior(cfg, rng); pr.update({"correlation": float(sh["correlation"]), "noise_range": list(sh["noise_range"]),
            "tester_offset_max": float(sh["tester_offset_max"]), "defect_amplitude_scale": float(sh["defect_amplitude_scale"]),
            "law": "logtime", "tau_h": float(rng.uniform(*sh["tau_h_range"]))})
        shifted.append(sample_lot(rng, pr, cfg, f"S{seed}_X{i:02d}"))
    shifted = pd.concat(shifted, ignore_index=True); shifted["split"] = "test"; shifted["test_set"] = "shifted"
    return {"train": train, "calibration": cal, "test_inprior": test, "test_shifted": shifted}


def impute(train: pd.DataFrame, frames: list[pd.DataFrame], cfg: dict) -> list[pd.DataFrame]:
    cols = [f"{p}_{t}h" for p in PARAMS for t in cfg["simulation"]["checkpoints_h"]]
    gm = train[cols].median(); fm = train.groupby("device_family")[cols].median()
    out=[]
    for frame in frames:
        f=frame.copy().reset_index(drop=True)
        for c in cols: f[c]=f[c].fillna(f["device_family"].map(fm[c])).fillna(gm[c])
        out.append(f)
    return out
