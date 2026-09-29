from __future__ import annotations
from typing import Any, Callable
import torch
import torch.nn as nn
from .runtime import output_tensor


def capture_module_output(model:nn.Module,path:str,args:tuple,kwargs:dict)->Any:
    model.eval()
    mods=dict(model.named_modules())
    if path not in mods: raise ValueError(f'Unknown module path {path!r}')
    box={}
    def hook(mod,inp,out): box['out']=out.detach().clone() if isinstance(out,torch.Tensor) else out
    h=mods[path].register_forward_hook(hook)
    try:
        with torch.inference_mode(): model(*args,**kwargs)
    finally: h.remove()
    if 'out' not in box: raise RuntimeError('Module did not produce a capturable output')
    return box['out']


def activation_patch(model:nn.Module,clean_args:tuple,clean_kwargs:dict,corrupt_args:tuple,corrupt_kwargs:dict,path:str,index:int|None=None)->dict:
    model.eval()
    clean=capture_module_output(model,path,clean_args,clean_kwargs)
    mods=dict(model.named_modules()); target=mods[path]
    def patch(mod,inp,out):
        if not isinstance(out,torch.Tensor) or not isinstance(clean,torch.Tensor): return out
        if index is None: return clean.to(out.device,dtype=out.dtype)
        y=out.clone(); axis=1 if y.ndim>=2 else 0
        if index < y.shape[axis]:
            sl=[slice(None)]*y.ndim; sl[axis]=index; y[tuple(sl)]=clean.to(y.device,dtype=y.dtype)[tuple(sl)]
        return y
    with torch.inference_mode(): base=model(*corrupt_args,**corrupt_kwargs)
    h=target.register_forward_hook(patch)
    try:
        with torch.inference_mode(): patched=model(*corrupt_args,**corrupt_kwargs)
    finally:h.remove()
    bt=output_tensor(base); pt=output_tensor(patched)
    result={'path':path,'index':index}
    if bt is not None and pt is not None:
        b=bt.detach().float(); p=pt.detach().float(); result.update({'l2_delta':float(torch.linalg.vector_norm(p-b)),'mean_abs_delta':float((p-b).abs().mean())})
        if b.ndim==2:
            pb=torch.softmax(b,dim=-1); pp=torch.softmax(p,dim=-1); result['kl_patched_vs_base']=float((pp*(pp.clamp_min(1e-9).log()-pb.clamp_min(1e-9).log())).sum(-1).mean())
    return result
