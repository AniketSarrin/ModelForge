from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import hashlib, json, platform
import torch
import torch.nn as nn
from .research import ProbeSpec, InterventionSpec, run_research_call, parameter_inventory, profile_model_call


def model_hash(model: nn.Module) -> str:
    h=hashlib.sha256()
    for name,p in model.state_dict().items():
        h.update(name.encode()); h.update(str(tuple(p.shape)).encode()); h.update(p.detach().cpu().numpy().tobytes()[:4096])
    return h.hexdigest()[:16]

@dataclass
class OpenModel:
    model: nn.Module
    name: str='model'

    def modules(self):
        return [{"path":n,"type":m.__class__.__name__,"params":sum(p.numel() for p in m.parameters(recurse=False)),"children":len(list(m.children()))} for n,m in self.model.named_modules() if n]

    def parameters(self, query:str=''):
        return parameter_inventory(self.model,query)

    def run(self,*args,probes=None,interventions=None,capture_gradients=False,target_index=None,**kwargs):
        ps=[p if isinstance(p,ProbeSpec) else ProbeSpec(**p) for p in (probes or [])]
        ins=[i if isinstance(i,InterventionSpec) else InterventionSpec(**i) for i in (interventions or [])]
        result=run_research_call(self.model,args,kwargs,ps,ins,capture_gradients,target_index)
        result['reproducibility']={
            'model_hash':model_hash(self.model),'torch':torch.__version__,'python':platform.python_version()
        }
        return result

    def profile(self,*args,**kwargs):
        return profile_model_call(self.model,args,kwargs)


def open(model:nn.Module,name:str='model')->OpenModel:
    if not isinstance(model,nn.Module): raise TypeError('mf.open(model) expects torch.nn.Module')
    return OpenModel(model,name)
