#!/usr/bin/env python3
import argparse,json,re
from pathlib import Path
ROW=re.compile(r"^(3|6|12|Avg)\s+(MAE|RMSE|MAPE)\s+(.+?)\s*$")
def main():
    p=argparse.ArgumentParser()
    for name in ("dataset","config","log","output"): p.add_argument("--"+name,required=True)
    p.add_argument("--seed",required=True,type=int); a=p.parse_args(); cfg=json.loads(Path(a.config).read_text()); metrics={}
    for line in Path(a.log).read_text(errors="replace").splitlines():
        match=ROW.match(line.split(" - ",1)[-1].strip())
        if not match: continue
        h,m,tail=match.groups(); vals=[float(x) for x in re.findall(r"-?\d+(?:\.\d+)?",tail)]; expected=cfg["end_year"]-cfg["begin_year"]+2
        if len(vals)>=expected: metrics.setdefault(h,{})[m]={"per_period":vals[:expected-1],"mean":vals[expected-1]}
    missing=[f"{h}/{m}" for h in ("3","6","12") for m in ("MAE","RMSE","MAPE") if m not in metrics.get(h,{})]
    if missing: raise SystemExit("incomplete EAC log; missing "+", ".join(missing))
    payload={"baseline":"EAC","dataset":a.dataset,"seed":a.seed,"protocol":cfg["adapter_protocol"],"official_commit":cfg["official_commit"],"config":a.config,"log":a.log,"metrics":metrics}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(payload,indent=2)+"\n"); print(f"wrote {out}")
if __name__=="__main__": main()
