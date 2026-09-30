from __future__ import annotations

import json
import hashlib
import re
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT.parent / "Dataset" / "Bosch_Production_Line_Performance"
OUT = ROOT / "out" / "v3"
CACHE = DATA / ".sahodrift_cache"
CHUNK = 5000
RNG_SEED = 26170


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def cp_interval(successes: int, total: int) -> list[float]:
    if total == 0:
        return [None, None]
    low = 0.0 if successes == 0 else beta.ppf(0.025, successes, total - successes + 1)
    high = 1.0 if successes == total else beta.ppf(0.975, successes + 1, total - successes)
    return [100.0 * float(low), 100.0 * float(high)]


def date_features() -> dict[str, np.ndarray]:
    cache = CACHE / "date_features.npz"
    if cache.exists():
        return dict(np.load(cache))
    CACHE.mkdir(parents=True, exist_ok=True)
    archive = DATA / "train_date.csv.zip"
    with zipfile.ZipFile(archive) as zf, zf.open("train_date.csv") as stream:
        header = pd.read_csv(stream, nrows=0).columns.tolist()
    columns = header[1:]
    stations = np.array([int(re.search(r"_S(\d+)_", name).group(1)) for name in columns], dtype=np.int16)
    unique_stations = np.unique(stations)
    station_columns = [np.flatnonzero(stations == station) for station in unique_stations]
    output = {key: [] for key in ["id", "start", "end", "duration", "n_stations", "first_station", "last_station", "path_code"]}
    with zipfile.ZipFile(archive) as zf, zf.open("train_date.csv") as stream:
        reader = pd.read_csv(stream, chunksize=CHUNK, dtype=np.float32)
        for chunk_no, frame in enumerate(reader):
            ids = frame.iloc[:, 0].to_numpy(np.int64)
            values = frame.iloc[:, 1:].to_numpy(np.float32)
            finite = np.isfinite(values)
            with np.errstate(all="ignore"):
                start = np.nanmin(values, axis=1)
                end = np.nanmax(values, axis=1)
            active = np.column_stack([finite[:, idx].any(axis=1) for idx in station_columns])
            n_stations = active.sum(axis=1).astype(np.int16)
            first_idx = np.argmax(active, axis=1)
            last_idx = active.shape[1] - 1 - np.argmax(active[:, ::-1], axis=1)
            no_station = n_stations == 0
            first_station = unique_stations[first_idx].astype(np.int16)
            last_station = unique_stations[last_idx].astype(np.int16)
            first_station[no_station] = -1
            last_station[no_station] = -1
            path = np.zeros(len(frame), dtype=np.uint64)
            for position, station in enumerate(unique_stations):
                path |= active[:, position].astype(np.uint64) << np.uint64(station)
            output["id"].append(ids)
            output["start"].append(start)
            output["end"].append(end)
            output["duration"].append(end - start)
            output["n_stations"].append(n_stations)
            output["first_station"].append(first_station)
            output["last_station"].append(last_station)
            output["path_code"].append(path)
            if chunk_no % 25 == 0:
                print(f"date rows {sum(map(len, output['id'])):,}", flush=True)
    result = {key: np.concatenate(value) for key, value in output.items()}
    np.savez_compressed(cache, **result)
    return result


def numeric_first_pass(n_rows: int, train_mask: np.ndarray) -> tuple[dict[str, np.ndarray], list[str]]:
    cache = CACHE / "numeric_compact.npz"
    selected_file = CACHE / "selected_columns.json"
    signature_file = CACHE / "numeric_split_signature.txt"
    signature = hashlib.sha256(train_mask.tobytes()).hexdigest()
    if cache.exists() and selected_file.exists() and signature_file.exists() and signature_file.read_text().strip() == signature:
        return dict(np.load(cache)), json.loads(selected_file.read_text())

    path = DATA / "train_numeric.csv"
    header = pd.read_csv(path, nrows=0).columns.tolist()
    feature_names = header[1:-1]
    p = len(feature_names)
    sums = {"pos": np.zeros(p), "neg": np.zeros(p), "pos2": np.zeros(p), "neg2": np.zeros(p)}
    counts = {"pos": np.zeros(p, dtype=np.int64), "neg": np.zeros(p, dtype=np.int64)}
    compact = {key: np.empty(n_rows, dtype=dtype) for key, dtype in {
        "id": np.int64, "label": np.int8, "observed": np.int16, "mean": np.float32,
        "std": np.float32, "minimum": np.float32, "maximum": np.float32,
    }.items()}
    offset = 0
    for chunk_no, frame in enumerate(pd.read_csv(path, chunksize=CHUNK, dtype=np.float32)):
        length = len(frame)
        stop = offset + length
        values = frame.iloc[:, 1:-1].to_numpy(np.float32)
        labels = frame.iloc[:, -1].to_numpy(np.int8)
        compact["id"][offset:stop] = frame.iloc[:, 0].to_numpy(np.int64)
        compact["label"][offset:stop] = labels
        compact["observed"][offset:stop] = np.sum(np.isfinite(values), axis=1)
        with np.errstate(all="ignore"):
            compact["mean"][offset:stop] = np.nanmean(values, axis=1)
            compact["std"][offset:stop] = np.nanstd(values, axis=1)
            compact["minimum"][offset:stop] = np.nanmin(values, axis=1)
            compact["maximum"][offset:stop] = np.nanmax(values, axis=1)
        local_train = train_mask[offset:stop]
        for label_name, label_value in [("pos", 1), ("neg", 0)]:
            local = values[local_train & (labels == label_value)]
            finite = np.isfinite(local)
            clean = np.where(finite, local, 0.0).astype(np.float64)
            sums[label_name] += clean.sum(axis=0)
            sums[label_name + "2"] += np.square(clean).sum(axis=0)
            counts[label_name] += finite.sum(axis=0)
        offset = stop
        if chunk_no % 25 == 0:
            print(f"numeric pass 1 rows {offset:,}", flush=True)
    if offset != n_rows:
        raise RuntimeError(f"Date and numeric row counts differ: {n_rows} versus {offset}")
    mean_pos = sums["pos"] / np.maximum(counts["pos"], 1)
    mean_neg = sums["neg"] / np.maximum(counts["neg"], 1)
    var_neg = sums["neg2"] / np.maximum(counts["neg"], 1) - np.square(mean_neg)
    effect = np.abs(mean_pos - mean_neg) / np.sqrt(np.maximum(var_neg, 1e-8))
    eligible = (counts["pos"] >= 30) & (counts["neg"] >= 3000) & np.isfinite(effect)
    chosen = np.flatnonzero(eligible)
    chosen = chosen[np.argsort(effect[chosen])[-48:]][::-1]
    selected = [feature_names[index] for index in chosen]
    np.savez_compressed(cache, **compact)
    selected_file.write_text(json.dumps(selected, indent=2) + "\n")
    signature_file.write_text(signature + "\n")
    return compact, selected


def selected_numeric(selected: list[str], n_rows: int) -> np.ndarray:
    cache = CACHE / "selected_values.npy"
    metadata = CACHE / "selected_values_columns.json"
    if cache.exists() and metadata.exists() and json.loads(metadata.read_text()) == selected:
        values = np.load(cache)
        if values.shape == (n_rows, len(selected)):
            return values
    usecols = ["Id", *selected, "Response"]
    output = np.empty((n_rows, len(selected)), dtype=np.float32)
    offset = 0
    for chunk_no, frame in enumerate(pd.read_csv(DATA / "train_numeric.csv", usecols=usecols, chunksize=20000, dtype=np.float32)):
        stop = offset + len(frame)
        output[offset:stop] = frame[selected].to_numpy(np.float32)
        offset = stop
        if chunk_no % 20 == 0:
            print(f"numeric pass 2 rows {offset:,}", flush=True)
    np.save(cache, output)
    metadata.write_text(json.dumps(selected, indent=2) + "\n")
    return output


def result_row(tier: str, target: str, threshold: float, scores: np.ndarray, labels: np.ndarray) -> dict:
    flags = scores >= threshold
    good = labels == 0
    defect = labels == 1
    fp = int(np.sum(flags & good))
    tp = int(np.sum(flags & defect))
    return {
        "tier": tier,
        "target": target,
        "threshold": float(threshold),
        "achieved_far_pct": 100.0 * fp / int(good.sum()),
        "recall_pct": 100.0 * tp / int(defect.sum()),
        "false_alarms_per_1000_good": 1000.0 * fp / int(good.sum()),
        "escapes_per_1000_defects": 1000.0 * (1.0 - tp / int(defect.sum())),
        "n_good": int(good.sum()),
        "n_defects": int(defect.sum()),
        "far_cp95": cp_interval(fp, int(good.sum())),
        "recall_cp95": cp_interval(tp, int(defect.sum())),
    }


def evaluate() -> dict:
    start_clock = time.perf_counter()
    dates = date_features()
    split_time = dates["start"].astype(float).copy()
    timestamp_valid = np.isfinite(split_time)
    missing_timestamps = int(np.sum(~timestamp_valid))
    if missing_timestamps:
        order_by_id = np.argsort(dates["id"], kind="stable")
        ordered_id = dates["id"][order_by_id]
        ordered_time = split_time[order_by_id]
        valid = np.isfinite(ordered_time)
        ordered_time[~valid] = np.interp(ordered_id[~valid], ordered_id[valid], ordered_time[valid])
        split_time[order_by_id] = ordered_time
    order = np.argsort(split_time, kind="stable")
    n = len(order)
    train_idx = order[: int(0.60 * n)]
    cal_idx = order[int(0.60 * n) : int(0.80 * n)]
    test_idx = order[int(0.80 * n) :]
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    compact, selected = numeric_first_pass(n, train_mask)
    if not np.array_equal(compact["id"], dates["id"]):
        raise RuntimeError("Numeric and date Id order differs")
    chosen = selected_numeric(selected, n)

    base = np.column_stack([
        dates["start"], dates["duration"], dates["n_stations"], dates["first_station"], dates["last_station"],
        compact["observed"], compact["mean"], compact["std"], compact["minimum"], compact["maximum"], chosen,
    ]).astype(np.float32)
    median = np.nanmedian(base[train_idx], axis=0)
    missing = ~np.isfinite(base)
    base[missing] = np.take(median, np.where(missing)[1])
    labels = compact["label"].astype(np.int8)

    rng = np.random.default_rng(RNG_SEED)
    positive_train = train_idx[labels[train_idx] == 1]
    negative_train = train_idx[labels[train_idx] == 0]
    keep_negative = rng.choice(negative_train, size=min(250000, len(negative_train)), replace=False)
    fit_idx = np.concatenate([positive_train, keep_negative])
    rng.shuffle(fit_idx)
    model = HistGradientBoostingClassifier(
        learning_rate=0.08,
        max_iter=180,
        max_leaf_nodes=31,
        min_samples_leaf=40,
        l2_regularization=1.0,
        class_weight="balanced",
        random_state=RNG_SEED,
    ).fit(base[fit_idx], labels[fit_idx])
    cal_score = model.predict_proba(base[cal_idx])[:, 1]
    test_score = model.predict_proba(base[test_idx])[:, 1]
    cal_labels = labels[cal_idx]
    test_labels = labels[test_idx]

    operating = []
    good_cal = cal_score[cal_labels == 0]
    for far in [0.5, 1.0, 5.0]:
        threshold = np.quantile(good_cal, 1.0 - far / 100.0, method="higher")
        operating.append(result_row("REJECT", f"FAR {far}%", threshold, test_score, test_labels))
    defect_cal = cal_score[cal_labels == 1]
    for recall in [80, 90, 95]:
        threshold = np.quantile(defect_cal, 1.0 - recall / 100.0, method="lower")
        operating.append(result_row("REVIEW", f"recall {recall}%", threshold, test_score, test_labels))

    path = dates["path_code"][test_idx]
    time_window = np.floor(split_time[test_idx] / 10.0).astype(np.int64)
    lots = np.array([f"{window}:{int(code)}" for window, code in zip(time_window, path)])
    return {
        "dataset": "Bosch Production Line Performance",
        "source": "Kaggle Bosch Production Line Performance competition",
        "licence": "Competition data. Use and redistribution are subject to the Kaggle competition rules.",
        "n_parts": int(n),
        "n_failures": int(labels.sum()),
        "failure_prevalence_pct": 100.0 * float(labels.mean()),
        "split": {
            "type": "chronological 60/20/20 by earliest station timestamp",
            "missing_timestamps_interpolated_from_ordered_part_id": missing_timestamps,
            "train_parts": int(len(train_idx)),
            "calibration_parts": int(len(cal_idx)),
            "test_parts": int(len(test_idx)),
            "test_failures": int(test_labels.sum()),
        },
        "lots_definition": "ten unit production time window crossed with exact station path",
        "n_test_lots": int(len(np.unique(lots))),
        "model": "HistGradientBoosting on row statistics and 48 measurements selected using training data only",
        "selected_measurements": selected,
        "average_precision": float(average_precision_score(test_labels, test_score)),
        "operating_points": operating,
        "runtime_s": time.perf_counter() - start_clock,
        "caveat": "Real rare production failures, but not burn-in trajectories. Labels identify line failures and do not identify semiconductor failure mechanisms. The competition data cannot be redistributed.",
        "evidence_label": "PUBLIC DATASET SANITY CHECK",
        "rivals_using_it": 0,
    }


def integrate(bosch: dict) -> None:
    unused_path = OUT / "realdata_unused_v3.json"
    unused = json.loads(unused_path.read_text())
    unused["datasets"] = [row for row in unused.get("datasets", []) if not row.get("dataset", "").startswith("Bosch")]
    unused["datasets"].append(bosch)
    unused["unavailable"] = [row for row in unused.get("unavailable", []) if not row.get("dataset", "").startswith("Bosch")]
    write_json(unused_path, unused)

    result_path = OUT / "driftguard_results_v3.json"
    results = json.loads(result_path.read_text())
    results["realdata_unused"] = [row for row in results.get("realdata_unused", []) if not row.get("dataset", "").startswith("Bosch")]
    results["realdata_unused"].append(bosch)
    reject = next(row for row in bosch["operating_points"] if row["tier"] == "REJECT" and row["target"] == "FAR 0.5%")
    results["headline"]["bosch_reject_at_0p5_far"] = reject
    results["meta"]["wp_status"]["WP5"] = {"status": "partial", "reason": "Aalborg gEOL, CORA IGBT and Bosch completed; the other planned real datasets were not evaluated."}
    results["meta"]["wp_status"]["WP5b"] = {"status": "partial", "reason": "Aalborg gEOL, CORA IGBT and Bosch completed. SPRC and the remaining optional datasets were not evaluated."}
    results["meta"]["wp_status"]["WP15"] = {"status": "done", "reason": "The user supplied complete gEOL, CORA and Bosch training data required for this phase."}
    losses = [
        text
        for text in results["integrity"]["where_new_methods_lose"]
        if not text.startswith("Bosch cannot")
        and text != "Bosch recall is low at the controlled 0.5 percent false alarm setting"
    ]
    results["integrity"]["where_new_methods_lose"] = losses + [
        "Bosch recall is low at the controlled 0.5 percent false alarm setting"
    ]
    note = "Bosch is a production line failure benchmark, not a burn-in trajectory dataset"
    if note not in results["integrity"]["notes"]:
        results["integrity"]["notes"].append(note)
    write_json(result_path, results)


def main() -> None:
    result = evaluate()
    write_json(OUT / "bosch_production_v3.json", result)
    integrate(result)
    print(json.dumps({"average_precision": result["average_precision"], "operating_points": result["operating_points"]}, indent=2))


if __name__ == "__main__":
    main()
