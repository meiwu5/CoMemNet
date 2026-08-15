import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="res/reviewer")
    parser.add_argument("--output", default="res/reviewer/aggregate.json")
    args = parser.parse_args()

    grouped = defaultdict(list)
    for path in Path(args.root).glob("**/metrics/summary.json"):
        with path.open(encoding="utf-8") as handle:
            record = json.load(handle)
        key = (record["dataset"], record["variant"], record.get("config_hash", "legacy"))
        grouped[key].append(record)

    aggregate = []
    for (dataset, variant, config_hash), records in sorted(grouped.items()):
        aip = [r["continual_summary"].get("aip_mae") for r in records
               if r.get("continual_summary") and "aip_mae" in r["continual_summary"]]
        forgetting = [r["continual_summary"].get("average_forgetting_mae") for r in records
                      if r.get("continual_summary") and "average_forgetting_mae" in r["continual_summary"]]
        bwt = [r["continual_summary"].get("bwt_negative_mae") for r in records
               if r.get("continual_summary") and "bwt_negative_mae" in r["continual_summary"]]
        protocol_metrics = {}
        protocol_names = sorted({name for r in records for name in r.get("continual_summaries", {})})
        for protocol in protocol_names:
            summaries = [r.get("continual_summaries", {}).get(protocol, {}) for r in records]
            protocol_metrics[protocol] = {}
            for metric in ("aip_mae", "current_task_mae", "final_seen_task_mae",
                           "bwt_negative_mae", "average_forgetting_mae",
                           "average_relative_forgetting"):
                values = [row[metric] for row in summaries if row.get(metric) is not None]
                protocol_metrics[protocol][metric + "_mean"] = float(np.mean(values)) if values else None
                protocol_metrics[protocol][metric + "_std"] = (
                    float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None)
        aggregate.append({
            "dataset": dataset,
            "variant": variant,
            "config_hash": config_hash,
            "seeds": sorted(r["seed"] for r in records),
            "runs": len(records),
            "aip_mae_mean": float(np.mean(aip)) if aip else None,
            "aip_mae_std": float(np.std(aip, ddof=1)) if len(aip) > 1 else 0.0 if aip else None,
            "forgetting_mae_mean": float(np.mean(forgetting)) if forgetting else None,
            "forgetting_mae_std": float(np.std(forgetting, ddof=1)) if len(forgetting) > 1 else 0.0 if forgetting else None,
            "bwt_negative_mae_mean": float(np.mean(bwt)) if bwt else None,
            "bwt_negative_mae_std": float(np.std(bwt, ddof=1)) if len(bwt) > 1 else 0.0 if bwt else None,
            "protocol_metrics": protocol_metrics,
        })

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(aggregate, handle, indent=2)
        handle.write("\n")
    print("aggregated {} variants into {}".format(len(aggregate), output))


if __name__ == "__main__":
    main()
