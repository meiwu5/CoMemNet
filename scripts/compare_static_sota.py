#!/usr/bin/env python3
"""Compare a per-period static baseline with final CoMemNet results/resources."""
import argparse, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--baseline',choices=('STAEformer','STID'),required=True)
 p.add_argument('--dataset',default='PEMSD3-stream');p.add_argument('--seed',type=int,default=0)
 p.add_argument('--output');a=p.parse_args()
 slug=a.baseline.lower(); bp=ROOT/'res'/'baseline'/a.baseline/a.dataset/f'{slug}-retrained-{a.seed}'/'metrics'/'summary.json'
 if not bp.exists(): raise SystemExit(f'missing baseline summary: {bp}')
 base=json.loads(bp.read_text());multi=json.loads((ROOT/'res'/'reviewer'/'final_main_multiseed.json').read_text())[a.dataset]
 run=Path(multi['runs'][str(a.seed)]['run']);cp=ROOT/run/'metrics'/'summary.json';co=json.loads(cp.read_text());eff=co['efficiency_by_period']
 co_metrics=multi['runs'][str(a.seed)]['metrics'];periods=base['periods']
 table={}
 for h in ('3','6','12'):
  table[h]={}
  for m in ('mae','rmse','mape'):
   cv=float(co_metrics[f'{h}_{m}']);bv=float(base['metrics'][h][m.upper()]['mean'])
   table[h][m.upper()]={'CoMemNet':cv,a.baseline:bv,'CoMemNet_relative_change_percent':(cv-bv)/bv*100}
 co_time=sum(float(x['total_time']) for x in eff.values());base_time=sum(float(x['train_seconds']) for x in periods)
 co_peak=max(float(x['peak_memory_mb']) for x in eff.values());base_peak=max(float(x['peak_vram_mb']) for x in periods)
 co_params=max(int(x['total_params']) for x in eff.values());base_params=max(int(x['parameters']) for x in periods)
 payload={'dataset':a.dataset,'seed':a.seed,'baseline':a.baseline,'metric_note':'negative relative change means CoMemNet has lower error',
  'metrics':table,'resources':{
   'cumulative_train_seconds':{'CoMemNet':co_time,a.baseline:base_time,'baseline_over_comemnet':base_time/co_time},
   'peak_vram_mb':{'CoMemNet':co_peak,a.baseline:base_peak,'baseline_over_comemnet':base_peak/co_peak},
   'parameters':{'CoMemNet':co_params,a.baseline:base_params,'baseline_over_comemnet':base_params/co_params}},
  'sources':{'CoMemNet':str(cp.relative_to(ROOT)),'baseline':str(bp.relative_to(ROOT))}}
 out=Path(a.output) if a.output else ROOT/'res'/'baseline'/a.baseline/a.dataset/f'comparison_with_comemnet_seed{a.seed}.json'
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(payload,indent=2)+'\n')
 print(json.dumps(payload,indent=2));print('saved:',out)
if __name__=='__main__':main()
