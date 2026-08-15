#!/usr/bin/env python3
"""Aggregate only final CoMemNet main runs with matching configs."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
ROW=re.compile(r"^(3|6|12)\s+(mae|rmse|mape)\s+.*average:\s*(-?\d+(?:\.\d+)?)$")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--root",default="res/reviewer"); p.add_argument("--seeds",default="0 1 2"); p.add_argument("--datasets",default="PEMSD3-stream PEMSD4-large PEMSD8-mini"); p.add_argument("--output",default="res/reviewer/final_main_multiseed.json"); a=p.parse_args()
 seeds={int(x) for x in a.seeds.split()}; out={}
 for ds in a.datasets.split():
  cfg=Path("config/reviewer")/ds/"sampler_feature.json"; expected=hashlib.sha256(cfg.read_bytes()).hexdigest(); runs={}
  for summary in (Path(a.root)/ds).glob("sampler_feature*/metrics/summary.json"):
   d=json.loads(summary.read_text()); seed=int(d.get("seed",-1))
   if seed not in seeds or d.get("variant")!="sampler_feature" or d.get("config_hash")!=expected: continue
   log=next(summary.parent.parent.glob("*.log"),None)
   if not log: continue
   metrics={}
   for line in log.read_text(errors="replace").splitlines():
    m=ROW.match(line.split(" - ",1)[-1].strip())
    if m: metrics[f"{m.group(1)}_{m.group(2)}"]=float(m.group(3))
   if len(metrics)==9: runs[seed]={"metrics":metrics,"continual":d.get("continual_summary",{}),"run":str(summary.parent.parent)}
  missing=sorted(seeds-set(runs))
  if missing: raise SystemExit(f"{ds}: missing completed seeds {missing}")
  keys=sorted(next(iter(runs.values()))["metrics"]); agg={}
  for k in keys:
   values=[runs[s]["metrics"][k] for s in sorted(seeds)]
   agg[k]={"values":values,"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)) if len(values)>1 else 0.0}
  out[ds]={"seeds":sorted(seeds),"runs":runs,"aggregate":agg}
 path=Path(a.output); path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(out,indent=2)+"\n"); print(f"wrote {path}")
if __name__=="__main__": main()
