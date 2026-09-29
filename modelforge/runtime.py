from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import io, torch
from PIL import Image
from torchvision import transforms

@dataclass
class PreparedInput:
    args: tuple
    kwargs: dict
    consumed: list[str]
    unused: list[str]
    preview: dict


def prepare_vision_input(image_bytes:bytes|None,image_size:int=96,text:str='',audio_present:bool=False)->PreparedInput:
    unused=[]
    if text.strip(): unused.append('text')
    if audio_present: unused.append('audio')
    if image_bytes:
        im=Image.open(io.BytesIO(image_bytes)).convert('RGB')
        prep=transforms.Compose([transforms.Resize((image_size,image_size)),transforms.ToTensor(),transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
        x=prep(im).unsqueeze(0); consumed=['image']; preview={'kind':'image','size':list(im.size)}
    else:
        x=torch.zeros(1,3,image_size,image_size); consumed=['synthetic_image']; preview={'kind':'synthetic_image'}
    return PreparedInput((x,),{},consumed,unused,preview)


def output_tensor(output:Any)->torch.Tensor|None:
    if isinstance(output,torch.Tensor): return output
    if hasattr(output,'logits') and isinstance(output.logits,torch.Tensor): return output.logits
    if isinstance(output,(tuple,list)) and output and isinstance(output[0],torch.Tensor): return output[0]
    if isinstance(output,dict):
        for k in ('logits','output','last_hidden_state'):
            if isinstance(output.get(k),torch.Tensor): return output[k]
    return None
