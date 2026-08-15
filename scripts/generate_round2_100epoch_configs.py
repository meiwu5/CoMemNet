#!/usr/bin/env python3
"""Generate 100-epoch matched configs for round-2 runtime comparisons."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=4) + "\n", encoding="utf-8")
    print(path.relative_to(ROOT))


def patch_common(d: dict, dataset: str, logname: str) -> dict:
    d = dict(d)
    d["logname"] = logname
    d["epoch"] = 100
    d["data_process"] = 0
    d["use_contrastive_loss"] = False
    d["evaluate_continual"] = True
    d["raw_data_path"] = f"data/{dataset}/finaldata/"
    d["save_data_path"] = f"data/{dataset}/FastData/"
    d["graph_path"] = f"data/{dataset}/graph/"
    d["model_path"] = f"res/reviewer/{dataset}"
    return d


def generate_dataset(dataset: str, comemnet_src: str, retrained_src: str, prefix: str = "") -> None:
    base = ROOT / "config" / "reviewer" / dataset

    d = json.loads((base / comemnet_src).read_text())
    d = patch_common(d, dataset, f"{prefix}comemnet_100epoch")
    d.update({
        "strategy": "incremental",
        "init": True,
        "replay": True,
        "use_target_branch": True,
    })
    write(base / f"{prefix}comemnet_100epoch.json", d)

    d = json.loads((base / retrained_src).read_text())
    d = patch_common(d, dataset, f"{prefix}current_retrained_100epoch")
    d.update({
        "strategy": "retrained",
        "init": False,
        "replay": False,
    })
    write(base / f"{prefix}current_retrained_100epoch.json", d)


def main() -> None:
    generate_dataset(
        "PEMSD3-stream",
        "sampler_feature.json",
        "forgetting_current_retrained.json",
        "round2_",
    )
    for dataset in ["PEMSD4-scale25", "PEMSD4-scale50", "PEMSD4-scale75", "PEMSD4-large"]:
        generate_dataset(
            dataset,
            "round2_scaling_comemnet.json",
            "round2_scaling_current_retrained.json",
            "round2_scaling_",
        )


if __name__ == "__main__":
    main()
