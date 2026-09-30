from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import curve_fit


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Dataset" / "doi-10"
OUT = ROOT / "out" / "v3"


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def cycle_trace(path: Path) -> dict:
    dut = loadmat(path, squeeze_me=True, struct_as_record=False)[path.stem]
    time = np.asarray(dut.Electric.E_time, dtype=float).ravel()
    vce = np.asarray(dut.Electric.Vce, dtype=float).ravel()
    current = np.asarray(dut.Electric.Ic, dtype=float).ravel()
    cycle = np.floor(time / 8.0).astype(np.int64)
    valid = np.isfinite(vce) & np.isfinite(current) & (current > 30.0) & (vce > 1.0) & (vce < 3.5)
    count = np.bincount(cycle[valid])
    total = np.bincount(cycle[valid], weights=vce[valid])
    keep = count >= 50
    y = total[keep] / count[keep]
    x = np.flatnonzero(keep).astype(float)
    return {
        "device": path.stem,
        "x": x,
        "y": y,
        "n_raw_electrical_samples": int(len(vce)),
        "n_cycles": int(len(y)),
        "preconditioned": path.stem == "DUT2",
    }


def tail_median(y: np.ndarray) -> float:
    return float(np.median(y[-max(10, len(y) // 100) :]))


def early_median(y: np.ndarray) -> float:
    return float(np.median(y[-max(10, len(y) // 20) :]))


def evaluate() -> dict:
    traces = [cycle_trace(path) for path in sorted(DATA.glob("DUT*.mat"))]
    if len(traces) != 3:
        raise RuntimeError(f"Expected three CORA DUT files, found {len(traces)}")

    rows = []
    fractions = [0.15, 0.25, 0.50]
    for held in traces:
        peers = [peer for peer in traces if peer["device"] != held["device"]]
        target = tail_median(held["y"])
        for fraction in fractions:
            n_early = max(20, int(np.floor(fraction * len(held["y"]))))
            xh = held["x"][:n_early]
            yh = held["y"][:n_early]
            xnorm = (xh - xh[0]) / max(1.0, held["x"][-1] - xh[0])
            early = early_median(yh)
            predictions = {"persistence": early}

            linear = np.polyfit(xnorm, yh, 1)
            predictions["linear"] = float(np.polyval(linear, 1.0))

            ratios = []
            for peer in peers:
                n_peer = max(20, int(np.floor(fraction * len(peer["y"]))))
                ratios.append(tail_median(peer["y"]) / max(early_median(peer["y"][:n_peer]), 1e-12))
            predictions["leave_one_device_out_ratio"] = early * float(np.median(ratios))

            try:
                def power_path(t, a, b):
                    return yh[0] + a * np.power(np.maximum(t, 0), b)

                guess = [max(1e-5, yh[-1] - yh[0]), 1.0]
                fitted, _ = curve_fit(
                    power_path,
                    xnorm,
                    yh,
                    p0=guess,
                    bounds=([0.0, 0.1], [5.0, 5.0]),
                    maxfev=20000,
                )
                predictions["power_law"] = float(power_path(np.array([1.0]), *fitted)[0])
            except Exception:
                predictions["power_law"] = None

            for method, prediction in predictions.items():
                rows.append(
                    {
                        "device": held["device"],
                        "preconditioned": held["preconditioned"],
                        "early_fraction": fraction,
                        "method": method,
                        "target_final_vce_on_v": target,
                        "prediction_v": prediction,
                        "absolute_error_v": None if prediction is None else abs(prediction - target),
                        "absolute_percentage_error": None if prediction is None else 100.0 * abs(prediction - target) / abs(target),
                    }
                )

    summary = []
    for fraction in fractions:
        for method in sorted({row["method"] for row in rows}):
            values = [row for row in rows if row["early_fraction"] == fraction and row["method"] == method and row["prediction_v"] is not None]
            errors = np.array([row["absolute_error_v"] for row in values])
            percentages = np.array([row["absolute_percentage_error"] for row in values])
            summary.append(
                {
                    "early_fraction": fraction,
                    "method": method,
                    "mae_v": float(np.mean(errors)),
                    "median_absolute_error_v": float(np.median(errors)),
                    "mape_pct": float(np.mean(percentages)),
                    "n_devices": len(values),
                }
            )

    best_15 = min((row for row in summary if row["early_fraction"] == 0.15), key=lambda row: row["mae_v"])
    return {
        "dataset": "CORA.RDR Si IGBT accelerated power cycling",
        "doi": "10.34810/data3206",
        "licence": "CC BY 4.0",
        "n_devices": 3,
        "n_lots": 3,
        "lots_definition": "one continuously aged physical device per file",
        "protocol": "On-state Vce is averaged per eight-second cycle after rejecting off-state and artifact samples. The first 15, 25 or 50 percent of each device trajectory predicts the median final one percent. Ratio priors use the other two devices only.",
        "trace_inventory": [
            {
                "device": trace["device"],
                "n_raw_electrical_samples": trace["n_raw_electrical_samples"],
                "n_cycles": trace["n_cycles"],
                "preconditioned": trace["preconditioned"],
            }
            for trace in traces
        ],
        "metrics": summary,
        "per_device": rows,
        "best_at_15_percent": best_15,
        "caveat": "Accelerated power cycling to failure, not component burn-in. DUT2 was preconditioned by cutting one of six bond wires. Three devices are too few for a production performance claim.",
        "evidence_label": "PUBLIC DATASET SANITY CHECK",
        "rivals_using_it": 0,
    }


def integrate(cora: dict) -> None:
    unused_path = OUT / "realdata_unused_v3.json"
    unused = json.loads(unused_path.read_text())
    unused["datasets"] = [row for row in unused.get("datasets", []) if not row.get("dataset", "").startswith("CORA")]
    unused["datasets"].append(cora)
    unused["unavailable"] = [row for row in unused.get("unavailable", []) if not row.get("dataset", "").startswith("CORA")]
    write_json(unused_path, unused)

    result_path = OUT / "driftguard_results_v3.json"
    results = json.loads(result_path.read_text())
    results["realdata_unused"] = [row for row in results.get("realdata_unused", []) if not row.get("dataset", "").startswith("CORA")]
    results["realdata_unused"].append(cora)
    results["headline"]["cora_best_at_15_percent"] = cora["best_at_15_percent"]
    results["meta"]["wp_status"]["WP5b"] = {
        "status": "partial",
        "reason": "Aalborg gEOL and CORA IGBT completed. SPRC and the remaining optional datasets were not evaluated in this simulation phase.",
    }
    results["meta"]["wp_status"]["WP5"] = {
        "status": "partial",
        "reason": "Aalborg gEOL and CORA IGBT completed; Bosch numeric archives remain incomplete and the other planned real datasets were not evaluated.",
    }
    results["meta"]["wp_status"]["WP15"] = {
        "status": "partial",
        "reason": "CORA extracted MAT files and gEOL are complete; Bosch numeric archives remain incomplete.",
    }
    losses = results["integrity"]["where_new_methods_lose"]
    retained_losses = [
        text
        for text in losses
        if not text.startswith("CORA and Bosch")
        and text != "Bosch cannot be evaluated because its local numeric archives are incomplete"
    ]
    results["integrity"]["where_new_methods_lose"] = retained_losses + [
        "Bosch cannot be evaluated because its local numeric archives are incomplete",
        "CORA has only three devices and DUT2 is deliberately preconditioned",
    ]
    note = "CORA uses three devices, including one deliberately preconditioned device, so its forecast errors are descriptive only"
    if note not in results["integrity"]["notes"]:
        results["integrity"]["notes"].append(note)
    write_json(result_path, results)


def main() -> None:
    cora = evaluate()
    write_json(OUT / "cora_igbt_v3.json", cora)
    integrate(cora)
    print(json.dumps(cora["best_at_15_percent"], indent=2))


if __name__ == "__main__":
    main()
