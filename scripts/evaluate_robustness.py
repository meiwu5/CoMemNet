#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np,torch
sys.path.append('src')
from model.model import Basic_Model


def mae(y,p,mask_nodes=None):
    if mask_nodes is not None: y,p=y[:,:,mask_nodes,:],p[:,:,mask_nodes,:]
    valid=y!=0
    return float(np.abs(y-p)[valid].mean())

def metrics(y,p,masked=None):
    out={}
    observed=None if masked is None else np.setdiff1d(np.arange(y.shape[2]),masked)
    for h in (3,6,12):
        out[str(h)]={'mae_all':mae(y[:,:h],p[:,:h]),
                     'mae_observed':mae(y[:,:h],p[:,:h],observed) if observed is not None else mae(y[:,:h],p[:,:h])}
    return out

def best_checkpoint(root,year):
    files=list((Path(root)/str(year)).glob('*.pkl'))
    if not files: raise FileNotFoundError(f'no checkpoint in {root}/{year}')
    return min(files,key=lambda p:float(p.stem))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--run-root',required=True); ap.add_argument('--config',required=True)
    ap.add_argument('--output',required=True); ap.add_argument('--year',type=int,default=2017); ap.add_argument('--seed',type=int,default=0)
    a=ap.parse_args(); cfg=json.load(open(a.config)); cfg.update(device=torch.device('cuda:0'),year=a.year)
    args=SimpleNamespace(**cfg); model=Basic_Model(args).to(args.device)
    ckpt=best_checkpoint(a.run_root,a.year); state=torch.load(ckpt,map_location=args.device)
    model.load_state_dict(state['model_state_dict'],strict=False); model.hidden_states_per_year={int(k):v.to(args.device) for k,v in state.get('hidden_states_per_year',{}).items()}; model.eval()
    data=np.load(Path(args.save_data_path)/f'{a.year}_30day.npz'); xs,ys=data['test_x'],data['test_y']; rng=np.random.default_rng(a.seed)
    node_count=xs.shape[2]; masks={r:np.sort(rng.choice(node_count,max(1,round(node_count*r)),replace=False)) for r in (.1,.2)}
    conditions=[('clean',None,None),('missing_10',masks[.1],None),('missing_20',masks[.2],None),('noise_10',None,.1)]
    results={}; context=max(model.hidden_states_per_year) if model.hidden_states_per_year else a.year-1
    for name,masked,noise_ratio in conditions:
        preds=[]; truths=[]
        for start in range(0,len(xs),args.batch_size):
            x=xs[start:start+args.batch_size].copy(); y=ys[start:start+args.batch_size]
            if masked is not None: x[:,:,masked,0]=0.0
            if noise_ratio is not None:
                sigma=float(xs[:,:,:,0].std())*noise_ratio
                x[:,:,:,0]+=rng.normal(0,sigma,size=x[:,:,:,0].shape).astype(x.dtype)
            with torch.no_grad(): pred=model({'x':torch.as_tensor(x,dtype=torch.float32,device=args.device)},a.year,memory_context_year=context)
            preds.append(pred.cpu().numpy()); truths.append(y)
        results[name]=metrics(np.concatenate(truths),np.concatenate(preds),masked)
        if masked is not None: results[name]['masked_nodes']=masked.tolist()
    clean=results['clean']['12']['mae_all']
    for name,row in results.items(): row['relative_degradation_12']=row['12']['mae_all']/clean-1
    output={'variant':cfg.get('logname'),'run_root':a.run_root,'checkpoint':str(ckpt),'year':a.year,'seed':a.seed,
            'protocol':'fixed checkpoint; perturb test inputs only; common seeded masks','results':results}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(output,indent=2)); print('saved',out)
if __name__=='__main__':main()
