#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import torch

def latest(root,variant):
    hits=sorted(Path(root).glob(variant+'*/metrics/summary.json'))
    if not hits: raise FileNotFoundError(variant)
    return hits[-1]
def nbytes_tensor_dict(d): return sum(v.numel()*v.element_size() for v in d.values() if torch.is_tensor(v))
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--root',default='res/reviewer/PEMSD3-stream'); ap.add_argument('--output',required=True); a=ap.parse_args()
    variants=['forgetting_full','all_history_retrained','forgetting_current_retrained']; rows={}
    for variant in variants:
        summary_path=latest(a.root,variant); summary=json.loads(summary_path.read_text()); run=summary_path.parent.parent
        config_path=Path('config/reviewer/PEMSD3-stream')/(variant+'.json'); cfg=json.loads(config_path.read_text())
        years=range(cfg['begin_year'],cfg['end_year']+1); final=cfg['end_year']; checkpoints=list((run/str(final)).glob('*.pkl')); ckpt=min(checkpoints,key=lambda p:float(p.stem)); state=torch.load(ckpt,map_location='cpu')
        sd=state['model_state_dict']; categories={'target':{},'tmrb':{},'predictor_shared':{}}
        for k,v in sd.items():
            key='target' if k.startswith(('target_backbone.','target_projection.')) else 'tmrb' if k.startswith('TMRB.') else 'predictor_shared'
            categories[key][k]=v
        raw_files=[Path(cfg['raw_data_path'])/f'{y}.npz' for y in years]
        fast_files=[Path(cfg['save_data_path'])/f'{y}_30day.npz' for y in years]
        graph_files=[Path(cfg['graph_path'])/f'{y}_adj.npz' for y in years]
        if variant=='forgetting_full': history_files=raw_files[:-1]
        elif variant=='all_history_retrained': history_files=fast_files
        else: history_files=[]
        efficiency=summary['efficiency_by_period']
        rows[variant]={
          'run_root':str(run),'cumulative_train_seconds':sum(x['total_time'] for x in efficiency.values()),
          'peak_vram_mb':max(x['peak_memory_mb'] for x in efficiency.values()),
          'final_checkpoint_bytes':ckpt.stat().st_size,'model_state_bytes':nbytes_tensor_dict(sd),
          'predictor_shared_bytes':nbytes_tensor_dict(categories['predictor_shared']),
          'target_branch_bytes':nbytes_tensor_dict(categories['target']),'tmrb_parameter_bytes':nbytes_tensor_dict(categories['tmrb']),
          'tmrb_state_bytes':nbytes_tensor_dict(state.get('hidden_states_per_year',{})),
          'historical_training_access_bytes':sum(p.stat().st_size for p in history_files if p.exists()),
          'historical_training_files':[str(p) for p in history_files],
          'graph_metadata_bytes':sum(p.stat().st_size for p in graph_files if p.exists()),
          'accounting_note':'unique on-disk files required/accessed; not cumulative repeated I/O'}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(rows,indent=2)); print('saved',out)
if __name__=='__main__':main()
