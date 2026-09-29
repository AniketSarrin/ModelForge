from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch

SUPPORTED_TEXT_MODELS={
    'gpt2':'gpt2',
    'gpt2-medium':'gpt2-medium',
    'pythia-70m':'EleutherAI/pythia-70m',
    'pythia-160m':'EleutherAI/pythia-160m',
}

@dataclass
class HFHandle:
    model: Any
    tokenizer: Any
    model_id: str

    def prepare(self,text:str):
        if not text: raise ValueError('Text input is required')
        batch=self.tokenizer(text,return_tensors='pt')
        device=next(self.model.parameters()).device
        return (),{k:v.to(device) for k,v in batch.items()}

    def decode_top(self,output,top_k:int=5):
        logits=output.logits if hasattr(output,'logits') else output[0]
        last=logits[0,-1].detach().float().cpu(); probs=torch.softmax(last,-1); vals,idx=torch.topk(probs,min(top_k,last.numel()))
        return [{'token_id':int(i),'token':self.tokenizer.decode([int(i)]),'probability':float(v),'logit':float(last[int(i)])} for v,i in zip(vals,idx)]


def load_text_model(name:str,device:str='cpu',dtype:str='float32')->HFHandle:
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError as exc:
        raise RuntimeError('Hugging Face support requires: pip install "modelforge[transformers]"') from exc
    model_id=SUPPORTED_TEXT_MODELS.get(name,name)
    tok=AutoTokenizer.from_pretrained(model_id)
    torch_dtype={'float16':torch.float16,'bfloat16':torch.bfloat16,'float32':torch.float32}.get(dtype,torch.float32)
    model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=torch_dtype).to(device).eval()
    return HFHandle(model,tok,model_id)
