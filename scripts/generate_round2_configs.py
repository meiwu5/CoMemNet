#!/usr/bin/env python3
"""Generate configs for second-round Reviewer 2 experiments."""
from __future__ import annotations

import copy
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=4) + "\n", encoding="utf-8")


def years(dataset: str) -> tuple[int, int]:
    files = sorted((ROOT / "data" / dataset / "finaldata").glob("*.npz"))
    ys = [int(p.stem) for p in files if p.stem.isdigit()]
    if not ys:
        raise FileNotFoundError(f"No finaldata years for {dataset}")
    return min(ys), max(ys)


def reviewer_base(base: dict, dataset: str, logname: str) -> dict:
    cfg = copy.deepcopy(base)
    begin, end = years(dataset)
    cfg.update({
        "dataset": dataset,
        "begin_year": begin,
        "end_year": end,
        "logname": logname,
        "strategy": "incremental",
        "init": True,
        "increase": True,
        "replay": True,
        "replay_strategy": "feature",
        "replay_ratio": 0.05,
        "is_TMRB": True,
        "is_update": True,
        "select_k": True,
        "use_target_branch": True,
        "sampler_branch": "target",
        "use_contrastive_loss": False,
        "evaluate_continual": True,
        "continual_memory_protocols": ["task_context", "current_state"],
        "data_process": 0,
        "raw_data_path": f"data/{dataset}/finaldata/",
        "save_data_path": f"data/{dataset}/FastData/",
        "graph_path": f"data/{dataset}/graph/",
        "model_path": f"res/reviewer/{dataset}",
    })
    return cfg


def write_scaling_configs(base_dataset: str, datasets: list[str]) -> None:
    base = json.loads((ROOT / "config" / base_dataset / "model.json").read_text())
    for dataset in datasets:
        out = ROOT / "config" / "reviewer" / dataset
        cfg = reviewer_base(base, dataset, "round2_scaling_comemnet")
        write_json(out / "round2_scaling_comemnet.json", cfg)

        cfg = reviewer_base(base, dataset, "round2_scaling_current_retrained")
        cfg["strategy"] = "retrained"
        cfg["init"] = False
        cfg["replay"] = False
        write_json(out / "round2_scaling_current_retrained.json", cfg)

        cfg = reviewer_base(base, dataset, "round2_scaling_all_history_retrained")
        cfg["strategy"] = "all_history_retrained"
        cfg["init"] = False
        cfg["replay"] = False
        write_json(out / "round2_scaling_all_history_retrained.json", cfg)


def main() -> None:
    specs = {
        "PEMSD4-large": ["PEMSD4-scale25", "PEMSD4-scale50", "PEMSD4-scale75", "PEMSD4-large"],
        "PEMSD3-stream": ["PEMSD3-scale50", "PEMSD3-stream"],
    }
    generated: list[str] = []
    for base_dataset, datasets in specs.items():
        write_scaling_configs(base_dataset, datasets)
        generated.extend(datasets)
    print("generated round-2 configs for", ", ".join(generated))


if __name__ == "__main__":
    main()
