#!/usr/bin/env python3
"""Generate EAC configs and zero-copy links to CoMemNet frozen splits."""
import json, os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
EAC_ROOT=ROOT/"baseline/EAC-official"; LINK_ROOT=ROOT/"baseline/EAC-data"; CONFIG_ROOT=ROOT/"config/baseline/eac"
DATASETS={"PEMSD3-stream":(2011,2017),"PEMSD4-large":(2009,2015),"PEMSD8-mini":(2012,2018)}
def replace_link(link,target):
    link.parent.mkdir(parents=True,exist_ok=True)
    if link.is_symlink() and link.resolve()==target.resolve(): return
    if link.exists() or link.is_symlink(): raise RuntimeError(f"refusing to replace non-matching path: {link}")
    link.symlink_to(os.path.relpath(target.resolve(),start=link.parent.resolve()))
def main():
    official=json.loads((EAC_ROOT/"conf/PEMS/eac.json").read_text()); CONFIG_ROOT.mkdir(parents=True,exist_ok=True)
    for dataset,(begin,end) in DATASETS.items():
        source,linked=ROOT/"data"/dataset,LINK_ROOT/dataset
        for year in range(begin,end+1):
            replace_link(linked/"FastData"/f"{year}.npz",source/"FastData"/f"{year}_30day.npz")
            replace_link(linked/"RawData"/f"{year}.npz",source/"finaldata"/f"{year}.npz")
            replace_link(linked/"graph"/f"{year}_adj.npz",source/"graph"/f"{year}_adj.npz")
        cfg=dict(official); cfg.update({"begin_year":begin,"end_year":end,"gpuid":0,"train":1,"auto_test":1,"data_process":0,"raw_data_path":str(linked/"RawData")+"/","save_data_path":str(linked/"FastData")+"/","graph_path":str(linked/"graph")+"/","model_path":str(ROOT/"res/baseline/EAC"/dataset)+"/","logname":"eac","method":"EAC","adapter_protocol":"same frozen CoMemNet split; traffic channel only","official_commit":"0a99297e01e484d56b2dfc845eacbbcf733efd1b"})
        (CONFIG_ROOT/f"{dataset}.json").write_text(json.dumps(cfg,indent=2)+"\n"); print(f"prepared EAC config: {dataset}")
if __name__=="__main__": main()
