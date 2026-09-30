from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "v5"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    result_path = OUT / "sahodrift_results_v5.json"
    rival_path = OUT / "rival_benchmark_v5.json"
    v4_path = ROOT / "out" / "v4" / "sahodrift_results_v4.json"
    result = json.loads(result_path.read_text())
    rival = json.loads(rival_path.read_text())
    rival["demo"] = {"public_url": "https://sahodrift-app.vercel.app"}
    rival_path.write_text(json.dumps(rival, indent=2, allow_nan=False) + "\n")
    v4 = json.loads(v4_path.read_text())
    main_row = next(x for x in result["operating_aggregates"] if x["detector"] == "v5_cross_conformal" and x["test_set"] == "inprior" and x["decision_hour"] == 24)
    result["ablation_protocol"] = {
        "v4_rows_frozen_for_comparison": v4["ablation_protocol"],
        "v5_fair_selection": result["protocol"]["selection_summary"],
        "v5_cross_conformal": main_row,
    }
    result["rival_benchmark_v5"] = {"file": "out/v5/rival_benchmark_v5.json", "sha256": sha256(rival_path), "benchmarks": rival["benchmarks"]}
    result["integrity"]["limitations"].extend([
        "At 168 hours the nominal in-prior mean FAR is 0.614 percent and three seeds exceed 0.75 percent; the conservative alpha/2 row is the defensible sensitivity result.",
        "BurnTrace has only 144 healthy calibration parts, so min p = 1/145 exceeds alpha 0.005; its nominal 0.5 percent rows are marked infeasible rather than treated as zero-recall failures.",
        "R3 was not rerun because the full supervised fallback exceeded the 30-minute cap in prior profiling; no capped approximation was substituted.",
        "The R4 robust covariance subcomponent uses a disclosed deterministic 12,000-row numerical cap; R1, R2 and R5 use the full training split.",
        "No supported browser screen recorder or installed Playwright browser was available. Deployment was checked by production build, HTTP title response and the 18-test suite.",
    ])
    result["integrity"]["frozen_input_hashes"] = {
        "v2": "a69e4834451f23cd21bbd4473d25847c409dd2a47a7b419789c587fe8231e823",
        "v3": "b51c8007b6e42aa34f8b6eddddf060442d1e9892096583ae874cab9d39dcdd89",
        "v4": "0be4ea37a8959d490f14db15ae8091a7e264499456cb632f8fb2539c3d66e29d",
    }
    result_path.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"v5_result_sha256": sha256(result_path), "rival_sha256": sha256(rival_path)}, indent=2))


if __name__ == "__main__": main()
