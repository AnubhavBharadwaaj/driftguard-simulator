from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "v4"
BUNDLE = ROOT.parent / "Rival_Trace" / "codex_v3_work" / "codex_v3"


def load(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def literature() -> dict:
    rows = []
    with (BUNDLE / "papers_v3.tsv").open() as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            item = {
                "priority": row["priority"], "work_package": row["wp"], "reference": row["ref"],
                "url": row["access_url"], "requested_extraction": row["what_to_extract"],
                "found": None, "status": "not_read_in_v4_execution_window", "key_numbers": [],
                "limitation": "Not used as empirical evidence in v4.", "difference_from_sahodrift": None,
            }
            if "Conformal Risk Control" in row["ref"]:
                item.update({
                    "found": True, "status": "read_full_html_for_A1",
                    "key_numbers": [{"fact": "Finite-sample upper risk criterion used in A1: (n times empirical risk plus the loss bound)/(n plus 1) must not exceed alpha.", "location": "Equation 4 and surrounding discussion"}],
                    "limitation": "The paper controls expected loss under its stated exchangeability assumptions; it does not guarantee realised recall on every small defect calibration set.",
                    "difference_from_sahodrift": "Sahodrift applies the criterion to the REVIEW false-negative loss while retaining a separate false-alarm-calibrated REJECT tier.",
                })
            rows.append(item)
    payload = {"status": "partial", "papers": rows, "read_count": sum(row["found"] is True for row in rows), "total_count": len(rows)}
    write(ROOT / "v4" / "literature_v4.json", payload)
    return payload


def main() -> None:
    phase = load(OUT / "phase_a_results.json", {})
    rival = load(OUT / "rival_benchmark_v4.json", {})
    shift = load(OUT / "shift_robust_v4.json", {})
    sequential = load(OUT / "sequential_d3_v4.json", {})
    demo_test = load(OUT / "demo_test.json", {})
    real_v3 = load(ROOT / "out" / "v3" / "driftguard_results_v3.json", {})
    foundation = {
        "status": "skipped_human_gate",
        "models": ["TabPFN v2", "TabPFN 2.5", "TabICLv2", "Mitra-v2", "TabDPT v1.3", "FoMo-0D", "OUTFORMER", "TabPFN unsupervised extension"],
        "reason": "Packages and model weights are not installed; gated models also require a user token. Installing the full stack was unsafe with less than four GiB free disk.",
        "results": [],
    }
    driftpfn = {"status": "skipped", "reason": "No out/driftpfn.pt checkpoint exists in the repository; A8 forbids retraining as a substitute."}
    write(OUT / "foundation_models.json", foundation); write(OUT / "driftpfn_v4_protocol.json", driftpfn)
    lit = literature()
    realdata = {
        "status": "partial",
        "completed_from_existing_local_files": [
            load(ROOT / "out" / "v3" / "cora_igbt_v3.json", {}),
            load(ROOT / "out" / "v3" / "bosch_production_v3.json", {}),
        ],
        "cora_identifier_check": {"doi": "10.34810/data3206", "evidence": "Dataset Readme.txt ACCESS INFORMATION section", "archive_name_mismatch": "The earlier archive name contained data3204; the dataset's own Readme states data3206."},
        "gEOL": next((item for item in real_v3.get("realdata_unused", []) if "gEOL" in item.get("dataset", "")), None),
        "not_completed": ["NASA IGBT all six", "NASA MOSFET thermal overstress", "SMRD2", "UCI SECOM", "LAM 9600", "Zenodo 18808126", "Zenodo 13321644"],
    }
    write(OUT / "realdata_v4.json", realdata)
    standards = {
        "status": "skipped", "reason": "Primary standards text was not available locally and no unverified duration or percentage is asserted.",
        "documents": ["MIL-STD-883 TM1015", "MIL-STD-883 TM5004", "MIL-PRF-38535", "MIL-STD-750 TM1042", "ESCC 5000", "ESCC 22900"],
    }
    write(ROOT / "v4" / "standards_v4.json", standards)
    status = {
        "A1": {"status": "done", "reason": "Three-way lot split, real calibration-size draws, and CRC REVIEW thresholds executed over five seeds."},
        "A2": {"status": "done", "reason": "Five-row protocol ablation written with FAR, recall and false rejects."},
        "A3": {"status": "done", "reason": "Per-type recall and counts added for Sahodrift, static limit and dynamic PAT."},
        "A4": {"status": "partial", "reason": "Trap matrix, defect-heavy lot, static and dynamic PAT completed; exact XGBoost is gated by missing libomp and the fallback remains labelled."},
        "A5": {"status": "partial", "reason": "Aabhas and BurnTrace adapters and Sahodrift runs completed. Five rivals completed on BurnTrace; Aabhas rival trees exceeded the explicit compute cap."},
        "A6": {"status": "partial", "reason": "Local Streamlit app, audit exports and STDF round trip pass. Public deployment and video require the user's hosting account."},
        "A7": {"status": "skipped_human_gate", "reason": foundation["reason"]},
        "A8": {"status": "skipped", "reason": driftpfn["reason"]},
        "B1": {"status": "skipped", "reason": "SPRC source PDFs were not present locally; no extracted table or fabricated QA rate is reported."},
        "B2": {"status": "partial", "reason": "CORA DOI corrected and CORA, Bosch and gEOL retained; seven remaining datasets are listed as not completed."},
        "B3": {"status": "partial", "reason": "Domain-weighted conformal and a domain-alarm fallback executed; unsafe or theorem-incomplete alternatives are explicitly not claimed."},
        "B4": {"status": "partial", "reason": "Exploratory checkpoint e-process executed; conditional validity remains unproved and is labelled."},
        "B5": {"status": "skipped", "reason": "No prior checkpoint and insufficient free disk for a new three-hour training run."},
        "B6": {"status": "partial", "reason": f"CRC paper read in full HTML; {lit['total_count'] - lit['read_count']} literature rows and primary standards remain unverified."},
        "B7": {"status": "done", "reason": "README, result paths, limitations and reproduction commands updated."},
    }
    result = {
        "meta": {
            "name": "Sahodrift v4", "team": "Silicon Surge", "problem_statement": "SIH26170",
            "wp_status": status,
            "frozen_outputs": {
                "out/driftguard_results_v2.json": sha(ROOT / "out" / "driftguard_results_v2.json"),
                "out/v3/driftguard_results_v3.json": sha(ROOT / "out" / "v3" / "driftguard_results_v3.json"),
            },
        },
        "protocol": phase.get("protocol"), "operating_rows": phase.get("detectors"),
        "ablation_protocol": phase.get("ablation_protocol"), "recall_by_type": phase.get("recall_by_type"),
        "calibration_size": phase.get("calibration_size"), "review_crc": phase.get("review_crc"),
        "traps_v4": phase.get("traps_v4"), "rival_benchmark_v4": rival,
        "foundation_models": foundation, "driftpfn_v4_protocol": driftpfn,
        "sprc": {"status": "skipped", "reason": status["B1"]["reason"]}, "realdata_v4": realdata,
        "shift_robust": shift, "sequential_d3": sequential,
        "demo": {"local_cmd": ".venv/bin/streamlit run app/main.py", "public_url": None, "video_path": None, "local_test": demo_test},
        "literature": {"path": "v4/literature_v4.json", "status": lit["status"]},
        "standards": standards,
        "integrity": {
            "no_test_lot_rule_or_threshold_selection": True,
            "limitations": [
                "The 24 hour in-prior REJECT result exceeds the nominal 0.5 percent FAR on average in v4.",
                "REVIEW workloads remain 528 to 897 per 1000 healthy parts for the three targets.",
                "BurnTrace has only 13 held-out anomalies and Sahodrift detected none in this split.",
                "Sequential D3 results are exploratory, not an anytime-valid production claim.",
            ],
        },
    }
    write(OUT / "sahodrift_results_v4.json", result)
    print(json.dumps({"output": str(OUT / "sahodrift_results_v4.json"), "wp_status": status}, indent=2))


if __name__ == "__main__":
    main()
