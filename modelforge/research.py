from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import time
import torch
import torch.nn as nn
from .runtime import output_tensor

@dataclass
class ProbeSpec:
    path:str
    kind:str='activation'
    full_tensor:bool=False
    top_k:int=12

@dataclass
class InterventionSpec:
    path:str
    kind:str
    value:float|None=None
    min_value:float|None=None
    max_value:float|None=None
    indices:list[int]|None=None


def _summary(t:torch.Tensor,top_k:int=12)->dict[str,Any]:
    td=t.detach().float().cpu()
    if td.numel()==0:return {'shape':list(td.shape),'numel':0}
    r={'shape':list(td.shape),'dtype':str(t.dtype).replace('torch.',''),'numel':td.numel(),'mean':float(td.mean()),'std':float(td.std(unbiased=False)),'min':float(td.min()),'max':float(td.max()),'mean_abs':float(td.abs().mean()),'rms':float(torch.sqrt((td*td).mean())),'l2':float(torch.linalg.vector_norm(td)),'positive_fraction':float((td>0).float().mean()),'zero_fraction':float((td==0).float().mean())}
    if td.ndim>=2:
        z=td[0]; score=z.abs() if z.ndim==1 else z.abs().reshape(z.shape[0],-1).mean(1); k=min(top_k,score.numel()); vals,idx=torch.topk(score,k); r['top_units']=[{'index':int(i),'score':float(v)} for v,i in zip(vals,idx)]
    return r


def _apply_output_intervention(out:Any,spec:InterventionSpec)->Any:
    if isinstance(out,tuple) and out and isinstance(out[0],torch.Tensor): return (_apply_output_intervention(out[0],spec),*out[1:])
    if isinstance(out,list) and out and isinstance(out[0],torch.Tensor): return [_apply_output_intervention(out[0],spec),*out[1:]]
    if not isinstance(out,torch.Tensor): return out
    x=out
    if spec.kind=='ablate':
        if spec.indices and x.ndim>=2:
            y=x.clone(); idx=torch.tensor(spec.indices,dtype=torch.long,device=x.device); idx=idx[idx<x.shape[1]]
            if idx.numel(): y.index_fill_(1,idx,0)
            return y
        return torch.zeros_like(x)
    if spec.kind=='scale': return x*float(1.0 if spec.value is None else spec.value)
    if spec.kind=='noise': return x+torch.randn_like(x)*float(0.01 if spec.value is None else spec.value)
    if spec.kind=='clamp': return torch.clamp(x,-float('inf') if spec.min_value is None else spec.min_value,float('inf') if spec.max_value is None else spec.max_value)
    if spec.kind=='detach': return x.detach()
    if spec.kind=='mask':
        if not spec.indices: raise ValueError('mask intervention requires at least one index')
        if x.ndim<2: raise ValueError('mask intervention requires tensor with a unit/channel axis')
        y=x.clone(); idx=torch.tensor(spec.indices,dtype=torch.long,device=x.device); idx=idx[idx<x.shape[1]]
        if idx.numel(): y.index_fill_(1,idx,0)
        return y
    return x


def parameter_inventory(model:nn.Module,query:str='')->list[dict[str,Any]]:
    q=query.lower().strip(); rows=[]
    for name,p in model.named_parameters():
        if q and q not in name.lower(): continue
        d=p.detach().float().cpu(); rows.append({'name':name,'shape':list(p.shape),'numel':p.numel(),'trainable':bool(p.requires_grad),'dtype':str(p.dtype).replace('torch.',''),'mean':float(d.mean()) if d.numel() else 0.0,'std':float(d.std(unbiased=False)) if d.numel() else 0.0,'l2':float(torch.linalg.vector_norm(d)) if d.numel() else 0.0,'sparsity':float((d==0).float().mean()) if d.numel() else 0.0})
    return rows


def profile_model_call(model:nn.Module,args:tuple,kwargs:dict)->dict[str,Any]:
    times={}; starts={}; handles=[]
    for name,mod in model.named_modules():
        if not name or any(True for _ in mod.children()): continue
        handles.append(mod.register_forward_pre_hook(lambda m,i,n=name: starts.__setitem__(n,time.perf_counter())))
        handles.append(mod.register_forward_hook(lambda m,i,o,n=name: times.__setitem__(n,(time.perf_counter()-starts.get(n,time.perf_counter()))*1000.0)))
    t0=time.perf_counter()
    with torch.inference_mode(): y=model(*args,**kwargs)
    total=(time.perf_counter()-t0)*1000.0
    for h in handles:h.remove()
    ranked=sorted(({'path':k,'latency_ms':v} for k,v in times.items()),key=lambda z:z['latency_ms'],reverse=True)
    yt=output_tensor(y)
    return {'total_ms':total,'modules':ranked,'output':_summary(yt) if yt is not None else {'type':type(y).__name__}}


def profile_model(model:nn.Module,x:torch.Tensor)->dict[str,Any]: return profile_model_call(model,(x,),{})


def run_research_call(model:nn.Module,args:tuple,kwargs:dict,probes:list[ProbeSpec]|None=None,interventions:list[InterventionSpec]|None=None,capture_gradients:bool=False,target_index:int|None=None)->dict[str,Any]:
    probes=probes or []; interventions=interventions or []; mods=dict(model.named_modules()); probe_map={p.path:p for p in probes}; int_map={}
    for i in interventions:int_map.setdefault(i.path,[]).append(i)
    captures={}; handles=[]; timings={}; starts={}
    for path,mod in mods.items():
        if not path or (path not in probe_map and path not in int_map):continue
        def pre(module,inp,n=path): starts[n]=time.perf_counter()
        def post(module,inp,out,n=path):
            timings[n]=(time.perf_counter()-starts.get(n,time.perf_counter()))*1000.0; actual=out
            for spec in int_map.get(n,[]): actual=_apply_output_intervention(actual,spec)
            p=probe_map.get(n); t=output_tensor(actual)
            if p and t is not None:
                captures.setdefault(n,{})['activation']=_summary(t,p.top_k); captures[n]['latency_ms']=timings[n]
                if p.full_tensor and t.numel()<=4096: captures[n]['tensor']=t.detach().float().cpu().flatten().tolist()
                if capture_gradients and t.requires_grad: t.retain_grad(); captures[n]['_tensor_ref']=t
            return actual
        handles.extend([mod.register_forward_pre_hook(pre),mod.register_forward_hook(post)])
    freeze_restore=[]
    for spec in interventions:
        if spec.kind=='freeze':
            if spec.path not in mods: raise ValueError(f'Unknown module path {spec.path!r}')
            for p in mods[spec.path].parameters(recurse=True): freeze_restore.append((p,p.requires_grad)); p.requires_grad_(False)
    was_training=model.training; model.eval()
    if capture_gradients:
        # Gradient mode is meaningful for tensor inputs; other inputs can still work if model internally creates grad tensors.
        grad_args=list(args)
        if grad_args and isinstance(grad_args[0],torch.Tensor): grad_args[0]=grad_args[0].detach().requires_grad_(True)
        y=model(*tuple(grad_args),**kwargs); yt=output_tensor(y)
        if yt is None: raise ValueError('Gradient capture requires a tensor-like model output')
        objective=yt[:,target_index].sum() if target_index is not None and yt.ndim==2 and 0<=target_index<yt.shape[1] else yt.float().pow(2).mean(); objective.backward()
        for cap in captures.values():
            ref=cap.pop('_tensor_ref',None)
            if ref is not None and ref.grad is not None: cap['gradient']=_summary(ref.grad)
    else:
        with torch.inference_mode(): y=model(*args,**kwargs)
    for p,old in freeze_restore:p.requires_grad_(old)
    for h in handles:h.remove()
    if was_training:model.train()
    yt=output_tensor(y); out=_summary(yt) if yt is not None else {'type':type(y).__name__}
    if yt is not None and yt.ndim==2:
        logits=yt[0].detach().float().cpu(); probs=torch.softmax(logits,-1); vals,idx=torch.topk(probs,min(5,probs.numel())); out['top']=[{'index':int(i),'probability':float(v),'logit':float(logits[int(i)])} for v,i in zip(vals,idx)]
    return {'output':out,'captures':captures,'interventions':[asdict(i) for i in interventions],'probes':[asdict(p) for p in probes]}


def run_research(model:nn.Module,x:torch.Tensor,probes=None,interventions=None,capture_gradients=False,target_index=None): return run_research_call(model,(x,),{},probes,interventions,capture_gradients,target_index)
