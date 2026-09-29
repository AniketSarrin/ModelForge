from __future__ import annotations
from pathlib import Path
from uuid import uuid4
import os, torch
from torch import nn
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from .models import SUPPORTED, load_model
from .store import ROOT, register_import, get_import, list_imports

IMPORT_DIR = ROOT / 'imports'
IMPORT_DIR.mkdir(parents=True, exist_ok=True)


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ('state_dict','model_state_dict','model','module'):
            value=obj.get(key)
            if isinstance(value,dict) and value and all(isinstance(k,str) for k in value):
                obj=value; break
    if not isinstance(obj,dict) or not obj: raise ValueError('File does not contain a recognizable state_dict')
    out={}
    for k,v in obj.items():
        if not isinstance(k,str) or not isinstance(v,torch.Tensor): continue
        nk=k
        for prefix in ('module.','model.','net.'):
            if nk.startswith(prefix): nk=nk[len(prefix):]
        out[nk]=v
    if not out: raise ValueError('No tensor weights were found in this checkpoint')
    return out


def _load_checkpoint(path:Path):
    if path.name.lower().endswith('.safetensors'): return safe_load_file(str(path),device='cpu')
    return torch.load(path,map_location='cpu',weights_only=True)


def detect_architecture(state_dict:dict[str,torch.Tensor])->str:
    keys=set(state_dict); scores=[]
    for arch in SUPPORTED:
        candidate=load_model(arch); cstate=candidate.state_dict(); ckeys=set(cstate)
        matching=sum(1 for k in keys&ckeys if tuple(state_dict[k].shape)==tuple(cstate[k].shape))
        wrong=sum(1 for k in keys&ckeys if tuple(state_dict[k].shape)!=tuple(cstate[k].shape))
        key_cov=matching/max(1,len(ckeys)); tensor_cov=matching/max(1,len(keys)); score=matching-3*wrong
        scores.append((key_cov,tensor_cov,score,arch))
    key_cov,tensor_cov,_,arch=max(scores)
    if key_cov<0.40 or tensor_cov<0.30:
        raise ValueError('Could not identify this checkpoint as one of the currently supported torchvision architectures: '+', '.join(SUPPORTED))
    return arch


def _compatible_state(model:nn.Module,state:dict[str,torch.Tensor]):
    target=model.state_dict(); kept={}; skipped=[]
    for k,v in state.items():
        if k in target and tuple(v.shape)==tuple(target[k].shape): kept[k]=v
        else: skipped.append(k)
    return kept,skipped


def register_weight_file(path:Path,original_name:str)->dict:
    state=_extract_state_dict(_load_checkpoint(path)); arch=detect_architecture(state); model=load_model(arch)
    compatible,skipped=_compatible_state(model,state)
    missing,unexpected=model.load_state_dict(compatible,strict=False)
    ref=uuid4().hex[:12]; suffix='.safetensors' if original_name.lower().endswith('.safetensors') else '.pth'; final=IMPORT_DIR/f'{ref}{suffix}'
    if suffix=='.safetensors': safe_save_file(state,str(final))
    else: torch.save(state,final)
    meta={"missing":list(missing),"unexpected":list(unexpected),"skipped_shape_mismatch":skipped}
    register_import(ref,original_name,arch,str(final),meta)
    return {"ref":ref,"model":f'import:{ref}',"architecture":arch,"name":original_name,**meta}


def resolve_model(name:str)->nn.Module:
    if not name.startswith('import:'): return load_model(name)
    ref=name.split(':',1)[1]; meta=get_import(ref)
    if not meta: raise ValueError('Imported model was not found in the persistent ModelForge store')
    model=load_model(meta['arch']); state=_extract_state_dict(_load_checkpoint(Path(meta['path']))); compatible,_=_compatible_state(model,state); model.load_state_dict(compatible,strict=False); return model


def imported_metadata(name:str)->dict|None:
    if not name.startswith('import:'): return None
    return get_import(name.split(':',1)[1])


def imported_catalog()->list[dict]: return list_imports()
