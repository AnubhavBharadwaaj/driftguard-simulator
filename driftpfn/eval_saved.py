from __future__ import annotations

import json
from pathlib import Path

import torch

from driftpfn.model import DriftPFN
from driftpfn.train_eval import eval_model


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    saved = torch.load(OUT / "driftpfn.pt", map_location="cpu", weights_only=False)
    model = DriftPFN().to(device)
    model.load_state_dict(saved["model"])
    result = eval_model(model, int(saved["params"]), int(saved["lots_seen"]), 1800.0, device)
    (OUT / "driftpfn_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
