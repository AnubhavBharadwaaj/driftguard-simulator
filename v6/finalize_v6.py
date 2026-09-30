from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "v6"


def sha(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""): h.update(b)
    return h.hexdigest()


def main():
    main_path=OUT/"sahodrift_results_v6.json"; rival_path=OUT/"simulator_rivals_v5_rules.json"
    data=json.loads(main_path.read_text())
    data["demo"]={
        "public_url":"https://sahodrift-app.vercel.app",
        "deployment_id":"dpl_B2aDKbSZrHGZX2Qb6d9wRXvEJ7MJ",
        "tests_passed":18,
        "production_build_passed":True,
        "guided_demo_parts":1632,
        "official_example":{"part_id":"LOT-A01-P045","disposition":"REJECT","reason_card_verified":True},
        "performance_view_verified":True,
        "binary_round_trip":{"files":64,"ptr_records":19584,"both_byte_orders_covered_by_test_suite":True},
        "walkthrough_release_url":"https://github.com/AnubhavBharadwaaj/driftguard-simulator/releases/download/v6-results/Sahodrift_Inspector_demo.mp4"
    }
    data["integrity"]["limitations"]=[
        "The within-lot MCD candidate failed the preregistered simulator recall gate and was not evaluated on Aabhas.",
        "Aabhas R4 exceeds Sahodrift at 24 hours and R1 lot MAD exceeds Sahodrift at 168 hours; the class overlap table identifies where.",
        "Fresh seed 5 has 1.427 percent FAR at 24 hours. The preregistered aggregate rule still passes because mean FAR is below 0.5 percent and exactly one seed exceeds 0.75 percent.",
        "BurnTrace has only thirteen held-out defects. Exact confidence intervals are wide and rankings are not stable evidence of general superiority.",
        "Sharing alpha does not guarantee identical achieved FAR; all tables retain achieved FAR beside recall.",
        "R3 is the preregistered sklearn GradientBoostingClassifier fallback, not XGBoost."
    ]
    data["integrity"]["frozen_hashes"]={
        "out_v5_sahodrift":sha(ROOT/"out/v5/sahodrift_results_v5.json"),
        "out_v5_rivals":sha(ROOT/"out/v5/rival_benchmark_v5.json"),
        "preregistration":sha(ROOT/"v6/PREREGISTRATION.md")
    }
    main_path.write_text(json.dumps(data,indent=2,allow_nan=False)+"\n")
    rivals=json.loads(rival_path.read_text()); rivals["demo"]=data["demo"]
    rival_path.write_text(json.dumps(rivals,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"sahodrift_results_v6_sha256":sha(main_path),"simulator_rivals_sha256":sha(rival_path)},indent=2))


if __name__=="__main__": main()
