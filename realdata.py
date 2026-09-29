from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.optimize import least_squares


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "igbt_inner"
OUT = ROOT / "out" / "realdata_result.json"


def trajectory(device: int) -> tuple[np.ndarray, np.ndarray]:
    rows: list[tuple[float, float]] = []
    for path in DATA.rglob(f"Device{device}*  1.mat"):
        if "check" in path.name.lower():
            continue
        measurement = loadmat(path, squeeze_me=True, struct_as_record=False)["measurement"]
        for transient in np.atleast_1d(measurement.transient):
            td = transient.timeDomain
            voltage = np.asarray(td.collectorEmitterVoltage, dtype=float)
            current = np.asarray(td.collectorEmitterCurrentSignal, dtype=float)
            good = np.isfinite(voltage) & np.isfinite(current)
            if good.sum() < 50:
                continue
            voltage, current = voltage[good], current[good]
            on = current >= np.quantile(current, 0.75)
            rows.append((float(transient.timeEpoch), float(np.median(voltage[on]))))
    arr = np.asarray(rows, dtype=float)
    arr = arr[np.argsort(arr[:, 0])]
    # Duplicate timestamps can occur across acquisition files.  Median aggregation
    # avoids giving a repeated capture extra influence.
    times = np.unique(arr[:, 0])
    values = np.asarray([np.median(arr[arr[:, 0] == t, 1]) for t in times])
    times = (times - times[0]) / max(times[-1] - times[0], 1.0)
    return times, values


def fit_power(t: np.ndarray, y: np.ndarray, exponent: float | None = None) -> tuple[float, float, float]:
    y0 = float(np.median(y[: max(2, len(y) // 5)]))
    if exponent is None:
        def residual(theta: np.ndarray) -> np.ndarray:
            a, n = theta
            return y0 + a * np.maximum(t, 1e-9) ** n - y
        fit = least_squares(residual, x0=np.array([max(y[-1] - y0, 0.05), 0.6]), bounds=([-20.0, 0.05], [20.0, 3.0]))
        a, exponent = map(float, fit.x)
    else:
        z = np.maximum(t, 1e-9) ** exponent
        a = float(np.dot(z, y - y0) / max(np.dot(z, z), 1e-12))
    return y0, a, float(exponent)


def main() -> None:
    try:
        devices = {d: trajectory(d) for d in range(2, 6)}
        errors = {"persistence": [], "linear": [], "power_law": [], "eb_forecaster": []}
        detail = []
        for held, (t, y) in devices.items():
            n_early = max(5, int(np.ceil(0.15 * len(t))))
            te, ye = t[:n_early], y[:n_early]
            actual = float(y[-1])
            persistence = float(ye[-1])
            linear_coef = np.polyfit(te, ye, 1)
            linear = float(np.polyval(linear_coef, 1.0))
            y0, a, n_local = fit_power(te, ye)
            power = float(y0 + a)
            train_n = []
            for other, (to, yo) in devices.items():
                if other == held:
                    continue
                train_n.append(fit_power(to, yo)[2])
            prior_n = float(np.median(train_n))
            # Empirical Bayes style shrinkage: a noisy early exponent receives one
            # part weight and the leave-one-device-out population receives three.
            n_eb = float((n_local + len(train_n) * prior_n) / (1 + len(train_n)))
            eb_y0, eb_a, _ = fit_power(te, ye, n_eb)
            eb = float(eb_y0 + eb_a)
            predictions = {"persistence": persistence, "linear": linear, "power_law": power, "eb_forecaster": eb}
            for name, pred in predictions.items():
                errors[name].append(abs(pred - actual))
            detail.append({
                "device": held,
                "n_points": int(len(t)),
                "observed_first_15pct_points": int(n_early),
                "actual_final_V": round(actual, 6),
                "predictions_V": {k: round(v, 6) for k, v in predictions.items()},
                "local_exponent": round(n_local, 6),
                "leave_one_device_out_prior_exponent": round(prior_n, 6),
            })
        result = {
            "status": "partial",
            "reason": "IGBT pipeline check completed on four devices. The separate NASA MOSFET archive was reachable but is 7.85 GB, beyond the remaining task budget, so it was not downloaded or scored.",
            "datasets": [{
                "name": "NASA PCoE IGBT Accelerated Aging",
                "n_devices": 4,
                "signal": "Median collector emitter voltage during the highest current quartile of each switching transient",
                "task": "Use the first 15 percent of each device trajectory to predict its final observed value; leave one device out when estimating the EB exponent prior.",
                "mae": {k: round(float(np.mean(v)), 6) for k, v in errors.items()},
                "mae_unit": "V",
                "device_results": detail,
                "caveat": "A four device ageing set is only an end to end pipeline sanity check, not evidence of factory generalization. Device durations and acquisition counts differ."
            }, {
                "name": "NASA PCoE MOSFET Thermal Overstress Aging",
                "n_devices": 0,
                "signal": "not evaluated",
                "mae": {},
                "status": "skipped",
                "reason": "The official zip responded successfully but reports a content length of 7,849,865,909 bytes."
            }]
        }
    except Exception as exc:
        result = {"status": "skipped", "reason": f"{type(exc).__name__}: {exc}", "datasets": []}
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
