#!/usr/bin/env python3
"""Per-period retraining adapter for static baselines.

STID/STAEformer use the checked-out official implementations.  DLinear and
PatchTST are lightweight in-repo adapters used only for the round-2 static
retraining stress test under the same frozen CoMemNet splits.
"""
from __future__ import annotations
import argparse, importlib, json, math, random, sys, time, types
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROOT=Path(__file__).resolve().parents[1]
DATASETS={"PEMSD3-stream":(2011,2017),"PEMSD4-large":(2009,2015),"PEMSD8-mini":(2012,2018),"PEMS08":(2016,2016)}
COMMITS={"staeformer":"fc49d39b2f1a8e3cf37b6289d7240680e1690f3f","stid":"e8b313bc591bdd0101a1619962c9b503e75127c0"}

class Split(Dataset):
 def __init__(self,x,y): self.x=x;self.y=y
 def __len__(self): return len(self.x)
 def __getitem__(self,i): return torch.from_numpy(self.x[i]).float(),torch.from_numpy(self.y[i]).float()

class DLinear(nn.Module):
 def __init__(self,input_len=12,output_len=12):
  super().__init__();self.linear=nn.Linear(input_len,output_len)
 def forward(self,x):
  z=x[...,0].permute(0,2,1);y=self.linear(z).permute(0,2,1).unsqueeze(-1);return y

class PatchTSTLite(nn.Module):
 def __init__(self,input_len=12,output_len=12,d_model=64,nhead=4,layers=2,patch_len=3,stride=3,dropout=.1):
  super().__init__();self.patch_len=patch_len;self.stride=stride;self.output_len=output_len
  patch_count=max(1,(input_len-patch_len)//stride+1)
  self.patch=nn.Linear(patch_len,d_model);self.pos=nn.Parameter(torch.zeros(1,patch_count,d_model))
  enc_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=d_model*2,dropout=dropout,batch_first=True,activation='gelu')
  self.encoder=nn.TransformerEncoder(enc_layer,num_layers=layers);self.head=nn.Linear(d_model*patch_count,output_len)
 def forward(self,x):
  # x: B,T,N,C -> (B*N),P,L, shared across nodes.
  z=x[...,0].permute(0,2,1).contiguous();b,n,t=z.shape;z=z.view(b*n,t)
  patches=z.unfold(dimension=1,size=self.patch_len,step=self.stride)
  h=self.patch(patches)+self.pos[:,:patches.size(1)]
  h=self.encoder(h).reshape(b*n,-1)
  y=self.head(h).view(b,n,self.output_len).permute(0,2,1).unsqueeze(-1)
  return y


class ITransformerLite(nn.Module):
 def __init__(self,input_len=12,output_len=12,d_model=64,nhead=4,layers=2,dropout=.1):
  super().__init__();self.input_len=input_len;self.output_len=output_len
  self.value_embedding=nn.Linear(input_len,d_model)
  enc_layer=nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,dim_feedforward=d_model*2,dropout=dropout,batch_first=True,activation='gelu')
  self.encoder=nn.TransformerEncoder(enc_layer,num_layers=layers)
  self.head=nn.Linear(d_model,output_len)
 def forward(self,x):
  # iTransformer-style inverted tokens: x B,T,N,C -> node tokens B,N,T.
  z=x[...,0].permute(0,2,1).contiguous()
  h=self.value_embedding(z)
  h=self.encoder(h)
  y=self.head(h).permute(0,2,1).unsqueeze(-1)
  return y

def seed_all(s):
 random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)

def infer_years(dataset):
 if dataset in DATASETS: return DATASETS[dataset]
 root=ROOT/'data'/dataset/'FastData'
 years=sorted(int(p.name.split('_')[0]) for p in root.glob('*_30day.npz') if p.name.split('_')[0].isdigit())
 if not years: raise FileNotFoundError(f'unknown dataset {dataset}; no FastData files under {root}')
 return min(years),max(years)

def write_json(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2)+"\n");q.replace(p)

def load_data(ds,year):
 p=ROOT/'data'/ds/'FastData'/f'{year}_30day.npz'
 with np.load(p) as z: d={k:z[k] for k in ('train_x','train_y','val_x','val_y','test_x','test_y')}
 # Recover traffic values from CoMemNet's train-only MinMax scaler.
 with np.load(ROOT/'data'/ds/'finaldata'/f'{year}.npz') as z:
  raw=z['x'][:len(d['train_x'])+11];lo=float(raw.min());hi=float(raw.max())
 def restore(a): return ((a+1)/2)*(hi-lo)+lo
 train_raw=restore(d['train_x'][...,0]);mean=float(train_raw.mean());std=float(train_raw.std()) or 1.0
 for part in ('train','val','test'):
  x=d[part+'_x'].astype(np.float32);x[...,0]=(restore(x[...,0])-mean)/std
  # CoMemNet day feature is integer 0..6; both official models expect that,
  # except STID internally multiplies a normalized day value by seven.
  d[part+'_x']=x;d[part+'_y']=((d[part+'_y'].astype(np.float32)-mean)/std)
 return p,d,mean,std

def year_restore_stats(ds,year):
 with np.load(ROOT/'data'/ds/'FastData'/f'{year}_30day.npz') as f:
  train_len=len(f['train_x'])
 with np.load(ROOT/'data'/ds/'finaldata'/f'{year}.npz') as z:
  raw=z['x'][:train_len+11];lo=float(raw.min());hi=float(raw.max())
 return lo,hi

def restore_with_stats(a,lo,hi): return ((a+1)/2)*(hi-lo)+lo

def pad_nodes(a,node_count):
 if a.shape[2]==node_count: return a
 if a.shape[2]>node_count: return a[:,:,:node_count,:]
 pad=list(a.shape);pad[2]=node_count-a.shape[2]
 return np.concatenate([a,np.zeros(pad,dtype=a.dtype)],axis=2)

def load_train_split_current_scale(ds,year,current_mean,current_std,current_nodes):
 p=ROOT/'data'/ds/'FastData'/f'{year}_30day.npz'
 with np.load(p) as z:
  x=z['train_x'].astype(np.float32);y=z['train_y'].astype(np.float32)
 lo,hi=year_restore_stats(ds,year)
 x=pad_nodes(x,current_nodes);y=pad_nodes(y,current_nodes)
 x[...,0]=(restore_with_stats(x[...,0],lo,hi)-current_mean)/current_std
 y=(restore_with_stats(y,lo,hi)-current_mean)/current_std
 return x,y

class ChainSplit(Dataset):
 def __init__(self,parts):
  self.parts=parts;self.bounds=[];total=0
  for x,y in parts:
   total+=len(x);self.bounds.append(total)
 def __len__(self): return self.bounds[-1] if self.bounds else 0
 def __getitem__(self,i):
  prev=0
  for (x,y),bound in zip(self.parts,self.bounds):
   if i<bound:
    j=i-prev;return torch.from_numpy(x[j]).float(),torch.from_numpy(y[j]).float()
   prev=bound
  raise IndexError(i)

def build_model(name,nodes):
 if name=='dlinear': return DLinear(input_len=12,output_len=12)
 if name=='patchtst': return PatchTSTLite(input_len=12,output_len=12)
 if name=='itransformer': return ITransformerLite(input_len=12,output_len=12)
 if name=='staeformer':
  sys.path.insert(0,str(ROOT/'baseline'/'STAEformer-official'))
  if 'torchinfo' not in sys.modules:
   try: import torchinfo  # noqa: F401
   except ImportError:
    shim=types.ModuleType('torchinfo');shim.summary=lambda *a,**k: None;sys.modules['torchinfo']=shim
  M=importlib.import_module('model.STAEformer').STAEformer
  return M(num_nodes=nodes,in_steps=12,out_steps=12,steps_per_day=288,input_dim=3,output_dim=1,
   input_embedding_dim=24,tod_embedding_dim=24,dow_embedding_dim=24,spatial_embedding_dim=0,
   adaptive_embedding_dim=80,feed_forward_dim=256,num_heads=4,num_layers=3,dropout=.1)
 sys.path.insert(0,str(ROOT/'baseline'/'STID-official'))
 M=importlib.import_module('stid.arch.stid_arch').STID
 return M(num_nodes=nodes,input_len=12,input_dim=1,embed_dim=32,output_len=12,num_layer=3,
  if_node=True,node_dim=32,if_T_i_D=True,if_D_i_W=True,temp_dim_tid=32,temp_dim_diw=32,
  time_of_day_size=288,day_of_week_size=7)

def forward(model,name,x):
 if name=='stid':
  x=x.clone();x[...,2]/=7.0
  return model(x,None,0,0,model.training).permute(0,1,2,3)
 return model(x)

def metrics(y,p):
 out={}
 for h in (3,6,12):
  a,b=y[:,:h],p[:,:h];m=a!=0;e=b-a
  nz=m&(np.abs(a)>1e-5)
  out[str(h)]={'MAE':float(np.abs(e)[m].mean()),'RMSE':float(np.sqrt((e*e)[m].mean())),
               'MAPE':float(np.abs(e[nz]/a[nz]).mean()*100)}
 return out

def run(args,ds,year,device,root,begin_year):
 out=root/str(year)/'metrics.json'
 if args.resume and out.exists(): print('[resume]',ds,year,args.model);return json.loads(out.read_text())
 split,d,mean,std=load_data(ds,year);nodes=d['train_x'].shape[2]
 if args.model=='staeformer' and nodes>args.max_stae_nodes and not args.allow_large:
  raise RuntimeError(f'STAEformer O(N^2) guard: {ds}/{year} has {nodes} nodes; use --allow-large and a smaller batch')
 if args.protocol=='all_history':
  parts=[load_train_split_current_scale(ds,y,mean,std,nodes) for y in range(begin_year,year+1)]
  train_ds=ChainSplit(parts);protocol='all-history static retraining; frozen CoMemNet split; historical train splits padded to current nodes'
 else:
  train_ds=Split(d['train_x'],d['train_y']);protocol='per-period static retraining; frozen CoMemNet split'
 loaders={'train':DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,num_workers=0),
          'val':DataLoader(Split(d['val_x'],d['val_y']),batch_size=args.batch_size,shuffle=False,num_workers=0),
          'test':DataLoader(Split(d['test_x'],d['test_y']),batch_size=args.batch_size,shuffle=False,num_workers=0)}
 seed_all(args.seed);model=build_model(args.model,nodes).to(device)
 opt=torch.optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
 sch=torch.optim.lr_scheduler.MultiStepLR(opt,[15,30,50],.1)
 ck=root/str(year)/'best.pt';ck.parent.mkdir(parents=True,exist_ok=True)
 best=math.inf;stale=0;seconds=0.;peak=0.
 for ep in range(args.epochs):
  model.train();tic=time.time();total=n=0
  for x,y in loaders['train']:
   x,y=x.to(device),y.to(device);opt.zero_grad(set_to_none=True);pred=forward(model,args.model,x)
   loss=F.huber_loss(pred,y);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);opt.step();total+=float(loss);n+=1
  seconds+=time.time()-tic;model.eval();vals=[]
  with torch.no_grad():
   for x,y in loaders['val']:
    pred=forward(model,args.model,x.to(device));vals.append(float(torch.abs(pred-y.to(device)).mean()))
  val=float(np.mean(vals));sch.step()
  if device.type=='cuda':peak=max(peak,torch.cuda.max_memory_allocated(device)/1024**2)
  print(f'[{args.model}] {ds}/{year} epoch={ep:03d} train={total/max(n,1):.5f} val={val*std:.4f}',flush=True)
  if val<best:best=val;stale=0;torch.save(model.state_dict(),ck)
  else:
   stale+=1
   if stale>=args.patience:break
 model.load_state_dict(torch.load(ck,map_location=device));model.eval();ys=[];ps=[]
 with torch.no_grad():
  for x,y in loaders['test']:
   ps.append((forward(model,args.model,x.to(device)).cpu().numpy()*std+mean));ys.append(y.numpy()*std+mean)
 baseline_name={'staeformer':'STAEformer','stid':'STID','dlinear':'DLinear','patchtst':'PatchTST-Lite','itransformer':'iTransformer-Lite'}[args.model]
 payload={'baseline':baseline_name,'official_commit':COMMITS.get(args.model,'in-repo-adapter'),
  'protocol':protocol,'dataset':ds,'year':year,'seed':args.seed,'nodes':nodes,'static_protocol':args.protocol,
  'split':str(split.relative_to(ROOT)),'parameters':sum(p.numel() for p in model.parameters()),
  'train_seconds':seconds,'peak_vram_mb':peak,'best_val_mae':best*std,'metrics':metrics(np.concatenate(ys),np.concatenate(ps))}
 write_json(out,payload);return payload

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=('staeformer','stid','dlinear','patchtst','itransformer'),required=True);p.add_argument('--dataset',required=True)
 p.add_argument('--seed',type=int,default=0);p.add_argument('--gpu',type=int,default=0);p.add_argument('--years',nargs='*',type=int)
 p.add_argument('--epochs',type=int,default=100);p.add_argument('--batch-size',type=int,default=16);p.add_argument('--patience',type=int,default=20)
 p.add_argument('--lr',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=.0005)
 p.add_argument('--protocol',choices=('current','all_history'),default='current')
 p.add_argument('--run-tag',default='',help='optional suffix inserted into output directory, e.g. bs128')
 p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--max-stae-nodes',type=int,default=1200);p.add_argument('--allow-large',action='store_true');a=p.parse_args()
 begin,end=infer_years(a.dataset);years=a.years or list(range(begin,end+1));device=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() and a.gpu>=0 else 'cpu')
 family={'staeformer':'STAEformer','stid':'STID','dlinear':'DLinear','patchtst':'PatchTST-Lite','itransformer':'iTransformer-Lite'}[a.model]
 suffix='retrained' if a.protocol=='current' else 'all-history-retrained'
 tag=f'-{a.run_tag}' if a.run_tag else ''
 root=ROOT/'res'/'baseline'/family/a.dataset/f'{a.model}-{suffix}{tag}-{a.seed}'
 periods=[run(a,a.dataset,y,device,root,begin) for y in years]
 summary={'baseline':periods[0]['baseline'],'official_commit':COMMITS.get(a.model,'in-repo-adapter'),'dataset':a.dataset,'seed':a.seed,'protocol':periods[0]['protocol'],'static_protocol':a.protocol,'periods':periods,'metrics':{},'efficiency':{'cumulative_train_seconds':float(np.sum([x['train_seconds'] for x in periods])),'peak_vram_mb':float(np.max([x['peak_vram_mb'] for x in periods])),'parameters':int(periods[-1]['parameters'])}}
 for h in ('3','6','12'):
  summary['metrics'][h]={m:{'per_period':[x['metrics'][h][m] for x in periods],'mean':float(np.mean([x['metrics'][h][m] for x in periods]))} for m in ('MAE','RMSE','MAPE')}
 write_json(root/'metrics'/'summary.json',summary);print(root/'metrics'/'summary.json')
if __name__=='__main__':main()
