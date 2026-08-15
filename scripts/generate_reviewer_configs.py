import copy
import json
from pathlib import Path


DATASETS = {
    "PEMSD3-stream": Path("config/PEMSD3-stream/model.json"),
    "PEMSD4-large": Path("config/PEMSD4-large/model.json"),
    "PEMSD8-mini": Path("config/PEMSD8-mini/model.json"),
}

SAMPLER_STRATEGIES = [
    "feature",
    "random",
    "recency",
    "high_error",
    "feature_l2",
    "feature_kl",
    "feature_js",
    "feature_mmd",
    "raw_l2",
]

MOMENTUM_VALUES = [0.9, 0.95, 0.99, 0.995]


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
        f.write("\n")


def reviewer_base(base, logname):
    cfg = copy.deepcopy(base)
    cfg["logname"] = logname
    cfg["strategy"] = "incremental"
    cfg["init"] = True
    cfg["increase"] = True
    cfg["replay"] = True
    cfg["is_TMRB"] = True
    cfg["is_update"] = True
    cfg["select_k"] = True
    cfg["use_target_branch"] = True
    cfg["sampler_branch"] = "target"
    cfg["save_sampler_debug"] = False
    cfg["save_sampler_plot"] = False
    cfg["replay_window_steps"] = 288 * 7
    cfg["recency_steps"] = 288
    cfg["high_error_windows"] = 256
    cfg["evaluate_continual"] = True
    # Controlled reviewer comparisons use MAE only.
    cfg["use_contrastive_loss"] = False
    cfg["contrastive_weight"] = 0.01
    cfg["contrastive_temperature"] = 0.2
    cfg["contrastive_max_nodes"] = 128
    cfg["model_path"] = str(Path("res/reviewer") / cfg["dataset"])
    # Reviewer runs reuse the frozen chronological split artifacts. Rebuilding
    # multi-GB FastData for every seed is both wasteful and a reproducibility risk.
    cfg["data_process"] = 0
    return cfg


def main():
    for dataset, base_path in DATASETS.items():
        with base_path.open("r", encoding="utf-8") as f:
            base = json.load(f)

        out_dir = Path("config/reviewer") / dataset

        for strategy in SAMPLER_STRATEGIES:
            cfg = reviewer_base(base, "sampler_{}".format(strategy))
            cfg["replay_strategy"] = strategy
            cfg["save_sampler_debug"] = True
            write_json(out_dir / "sampler_{}.json".format(strategy), cfg)

        cfg = reviewer_base(base, "target_no_target")
        cfg["use_target_branch"] = False
        cfg["sampler_branch"] = "online"
        cfg["use_contrastive_loss"] = False
        write_json(out_dir / "target_no_target.json", cfg)

        cfg = reviewer_base(base, "target_online_sampler")
        cfg["sampler_branch"] = "online"
        write_json(out_dir / "target_online_sampler.json", cfg)

        cfg = reviewer_base(base, "loss_mae_only")
        cfg["use_contrastive_loss"] = False
        write_json(out_dir / "loss_mae_only.json", cfg)

        cfg = reviewer_base(base, "no_replay")
        cfg["replay"] = False
        write_json(out_dir / "no_replay.json", cfg)

        cfg = reviewer_base(base, "static")
        cfg["strategy"] = "static"
        cfg["replay"] = False
        cfg["use_contrastive_loss"] = False
        write_json(out_dir / "static.json", cfg)

        cfg = reviewer_base(base, "retrained")
        cfg["strategy"] = "retrained"
        cfg["replay"] = False
        cfg["use_contrastive_loss"] = False
        write_json(out_dir / "retrained.json", cfg)

        # Reviewer-grouped forgetting audit. These configurations intentionally
        # use both task-context and current-state historical evaluation.
        cfg = reviewer_base(base, "forgetting_full")
        cfg["continual_memory_protocols"] = ["task_context", "current_state"]
        write_json(out_dir / "forgetting_full.json", cfg)

        cfg = reviewer_base(base, "forgetting_no_replay")
        cfg["replay"] = False
        cfg["continual_memory_protocols"] = ["task_context", "current_state"]
        write_json(out_dir / "forgetting_no_replay.json", cfg)

        cfg = reviewer_base(base, "forgetting_no_tmrb")
        cfg["is_TMRB"] = False
        cfg["continual_memory_protocols"] = ["task_context", "current_state"]
        write_json(out_dir / "forgetting_no_tmrb.json", cfg)

        cfg = reviewer_base(base, "forgetting_static")
        cfg["strategy"] = "static"
        cfg["replay"] = False
        cfg["continual_memory_protocols"] = ["task_context", "current_state"]
        write_json(out_dir / "forgetting_static.json", cfg)

        cfg = reviewer_base(base, "forgetting_current_retrained")
        cfg["strategy"] = "retrained"
        cfg["init"] = False
        cfg["replay"] = False
        cfg["continual_memory_protocols"] = ["task_context", "current_state"]
        write_json(out_dir / "forgetting_current_retrained.json", cfg)

        cfg = reviewer_base(base, "all_history_retrained")
        cfg["strategy"] = "all_history_retrained"
        cfg["init"] = False
        cfg["replay"] = False
        cfg["continual_memory_protocols"] = ["task_context", "current_state"]
        write_json(out_dir / "all_history_retrained.json", cfg)

        for weight in [0.01, 0.05, 0.1, 0.2]:
            cfg = reviewer_base(base, "contrastive_weight_{}".format(str(weight).replace(".", "_")))
            cfg["use_contrastive_loss"] = True
            cfg["contrastive_weight"] = weight
            write_json(out_dir / "contrastive_weight_{}.json".format(str(weight).replace(".", "_")), cfg)

        cfg = reviewer_base(base, "graph_selected_nodes_only")
        cfg["num_hops"] = 0
        cfg["topology_assisted_update"] = False
        write_json(out_dir / "graph_selected_nodes_only.json", cfg)

        for hops in [1, 2, 3]:
            cfg = reviewer_base(base, "graph_{}_hop".format(hops))
            cfg["num_hops"] = hops
            cfg["topology_assisted_update"] = True
            write_json(out_dir / "graph_{}_hop.json".format(hops), cfg)

        for momentum in MOMENTUM_VALUES:
            cfg = reviewer_base(base, "momentum_{}".format(str(momentum).replace(".", "_")))
            cfg["momentum"] = momentum
            write_json(out_dir / "momentum_{}.json".format(str(momentum).replace(".", "_")), cfg)

        for ratio in [0.01, 0.02, 0.05, 0.10]:
            tag = str(ratio).replace(".", "_")
            cfg = reviewer_base(base, "budget_{}".format(tag))
            cfg["replay_ratio"] = ratio
            cfg["save_sampler_debug"] = True
            cfg["continual_memory_protocols"] = ["task_context", "current_state"]
            write_json(out_dir / "budget_{}.json".format(tag), cfg)

        for name, overrides in {
            "fwt_full": {},
            "fwt_no_replay": {"replay": False},
            "fwt_static": {"strategy": "static", "replay": False},
        }.items():
            cfg = reviewer_base(base, name)
            cfg.update(overrides)
            cfg["evaluate_fwt"] = True
            cfg["fwt_reference_seed"] = 0
            cfg["continual_memory_protocols"] = ["task_context", "current_state"]
            write_json(out_dir / (name + ".json"), cfg)


if __name__ == "__main__":
    main()
