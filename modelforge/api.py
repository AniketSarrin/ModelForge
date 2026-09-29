from __future__ import annotations
from pathlib import Path
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Request, Response, Depends
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
from PIL import Image
import io
import time
from torchvision import transforms, models as tv_models

from .ir import trace_named_modules, hierarchy
from .models import module_shapes, SUPPORTED, SOURCE_CUTS, TARGET_ENTRIES
from .imports import resolve_model, register_weight_file, imported_metadata, imported_catalog
from .stitch import build_hybrid, plan_adapter
from .train import synthetic_train
from . import __version__
from .research import ProbeSpec, InterventionSpec, run_research, run_research_call, parameter_inventory, profile_model, profile_model_call
from .store import create_run, list_runs, get_run, create_experiment, attach_run, compare_runs, list_experiments
from .plugins import catalog as plugin_catalog
from .exporter import export_onnx, python_source, export_python_bundle
from .runtime import prepare_vision_input
from .codegen import run_to_python
from .cloud import cloud_config
from .auth import signup as auth_signup, login as auth_login, logout as auth_logout, user_for_token
from .datasets import save_upload as dataset_save_upload, register_uploaded_path, list_datasets as dataset_list, preview as dataset_preview, stats as dataset_stats, chart as dataset_chart, filter_rows as dataset_filter_rows, clean as dataset_clean, delete as dataset_delete, quota_info

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
app = FastAPI(title="ModelForge Research Workbench", version=__version__)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
BEGINNER = ROOT / "beginner"
if BEGINNER.exists(): app.mount("/beginner", StaticFiles(directory=BEGINNER, html=True), name="beginner")

class ModelRequest(BaseModel):
    model: str
    image_size: int = Field(96, ge=32, le=512)
    root: str = ""

class UnitsRequest(BaseModel):
    model: str
    path: str
    offset: int = Field(0, ge=0)
    count: int = Field(64, ge=1, le=256)

class StitchRequest(BaseModel):
    source: str
    source_cut: str
    target: str
    target_entry: str
    image_size: int = Field(96, ge=32, le=512)

class TrainRequest(StitchRequest):
    steps: int = Field(2, ge=1, le=20)
    batch_size: int = Field(1, ge=1, le=8)

@app.get("/")
def home(): return FileResponse(STATIC / "index.html")

@app.get("/api/health")
def health(): return {"ok": True, "version": __version__, "torch": torch.__version__}

@app.get("/api/cloud/config")
def cloud_status(): return cloud_config()



class AuthRequest(BaseModel):
    email: str
    password: str

class FilterRequest(BaseModel):
    column: str
    op: str
    value: str = ""
    limit: int = Field(100, ge=1, le=500)

class CleanRequest(BaseModel):
    name: str | None = None
    drop_duplicates: bool = False
    drop_missing: bool = False
    fill_missing: dict[str, str | float | int] = {}
    drop_columns: list[str] = []
    rename_columns: dict[str, str] = {}

def _current_user(request: Request):
    token = request.cookies.get("mf_session")
    user = user_for_token(token)
    if not user:
        raise HTTPException(401, "Sign in to use datasets")
    return user

@app.post("/api/auth/signup")
def signup_endpoint(req: AuthRequest, response: Response):
    try:
        auth_signup(req.email, req.password)
        user, token = auth_login(req.email, req.password)
        response.set_cookie("mf_session", token, httponly=True, samesite="lax", max_age=7*24*3600)
        return {"user": user, "quota": quota_info(user["id"])}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/auth/login")
def login_endpoint(req: AuthRequest, response: Response):
    try:
        user, token = auth_login(req.email, req.password)
        response.set_cookie("mf_session", token, httponly=True, samesite="lax", max_age=7*24*3600)
        return {"user": user, "quota": quota_info(user["id"])}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/auth/logout")
def logout_endpoint(request: Request, response: Response):
    auth_logout(request.cookies.get("mf_session")); response.delete_cookie("mf_session"); return {"ok": True}

@app.get("/api/auth/me")
def me_endpoint(user=Depends(_current_user)):
    return {"user": user, "quota": quota_info(user["id"])}

@app.get("/api/datasets")
def datasets_endpoint(user=Depends(_current_user)):
    return {"datasets": dataset_list(user["id"]), "quota": quota_info(user["id"])}

@app.post("/api/datasets/upload")
async def dataset_upload_endpoint(file: UploadFile = File(...), user=Depends(_current_user)):
    tmp = Path(tempfile.mkstemp(suffix=Path(file.filename or "dataset.csv").suffix)[1])
    size = 0
    try:
        remaining = quota_info(user["id"])["remaining_bytes"]
        with tmp.open("wb") as out:
            while True:
                chunk = await file.read(8 * 1024 * 1024)
                if not chunk: break
                size += len(chunk)
                if size > remaining:
                    raise HTTPException(413, "This upload would exceed your 1 GB dataset quota")
                out.write(chunk)
        ds = register_uploaded_path(user["id"], file.filename or "dataset.csv", tmp, size)
        return {"dataset": ds, "quota": quota_info(user["id"])}
    except HTTPException:
        tmp.unlink(missing_ok=True); raise
    except Exception as e:
        tmp.unlink(missing_ok=True); raise HTTPException(400, str(e))

@app.get("/api/datasets/{dataset_id}/preview")
def dataset_preview_endpoint(dataset_id: str, limit: int = 50, user=Depends(_current_user)):
    try: return dataset_preview(user["id"], dataset_id, limit)
    except KeyError: raise HTTPException(404, "Dataset not found")
    except Exception as e: raise HTTPException(400, str(e))

@app.get("/api/datasets/{dataset_id}/stats")
def dataset_stats_endpoint(dataset_id: str, user=Depends(_current_user)):
    try: return dataset_stats(user["id"], dataset_id)
    except KeyError: raise HTTPException(404, "Dataset not found")
    except Exception as e: raise HTTPException(400, str(e))

@app.get("/api/datasets/{dataset_id}/chart")
def dataset_chart_endpoint(dataset_id: str, column: str, bins: int = 20, user=Depends(_current_user)):
    try: return dataset_chart(user["id"], dataset_id, column, bins)
    except KeyError: raise HTTPException(404, "Dataset not found")
    except Exception as e: raise HTTPException(400, str(e))

@app.post("/api/datasets/{dataset_id}/filter")
def dataset_filter_endpoint(dataset_id: str, req: FilterRequest, user=Depends(_current_user)):
    try: return dataset_filter_rows(user["id"], dataset_id, req.column, req.op, req.value, req.limit)
    except KeyError: raise HTTPException(404, "Dataset not found")
    except Exception as e: raise HTTPException(400, str(e))

@app.post("/api/datasets/{dataset_id}/clean")
def dataset_clean_endpoint(dataset_id: str, req: CleanRequest, user=Depends(_current_user)):
    try:
        ops={"drop_duplicates":req.drop_duplicates,"drop_missing":req.drop_missing,"fill_missing":req.fill_missing,"drop_columns":req.drop_columns,"rename_columns":req.rename_columns}
        return {"dataset": dataset_clean(user["id"], dataset_id, ops, req.name), "quota": quota_info(user["id"])}
    except KeyError: raise HTTPException(404, "Dataset not found")
    except Exception as e: raise HTTPException(400, str(e))

@app.delete("/api/datasets/{dataset_id}")
def dataset_delete_endpoint(dataset_id: str, user=Depends(_current_user)):
    try:
        dataset_delete(user["id"], dataset_id); return {"ok":True,"quota":quota_info(user["id"])}
    except KeyError: raise HTTPException(404, "Dataset not found")

@app.get("/api/catalog")
def catalog():
    return {
        "models": SUPPORTED,
        "source_cuts": SOURCE_CUTS,
        "target_entries": TARGET_ENTRIES,
        "default_image_size": 96,
        "import_formats": [".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".pth.tar", ".pt.tar"],
    }

@app.post("/api/import")
async def import_weights(file: UploadFile = File(...)):
    lower = (file.filename or "weights.pth").lower()
    valid = any(lower.endswith(ext) for ext in (".pt", ".pth", ".ckpt", ".bin", ".safetensors", ".pth.tar", ".pt.tar"))
    if not valid:
        raise HTTPException(400, "Use a .pt, .pth, .ckpt, .bin, .safetensors, .pth.tar, or .pt.tar checkpoint file")
    tmp = Path(tempfile.mkstemp(suffix=Path(lower).suffix or '.pth')[1])
    try:
        data = await file.read()
        if len(data) > 1024 * 1024 * 1024:
            raise HTTPException(413, "Checkpoint exceeds the 1 GB MVP upload limit")
        tmp.write_bytes(data)
        return register_weight_file(tmp, file.filename or tmp.name)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, str(e))
    finally:
        tmp.unlink(missing_ok=True)

@app.post("/api/model")
def inspect_model(req: ModelRequest):
    try:
        model = resolve_model(req.model)
        sample = torch.zeros(1, 3, req.image_size, req.image_size)
        return {
            "model": req.model,
            "semantic_shapes": module_shapes(model, req.image_size),
            "nodes": [n.to_dict() for n in trace_named_modules(model, sample)],
            "parameters": sum(p.numel() for p in model.parameters()),
            "imported": imported_metadata(req.model),
        }
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/hierarchy")
def inspect_hierarchy(req: ModelRequest):
    try:
        model = resolve_model(req.model)
        sample = torch.zeros(1,3,req.image_size,req.image_size)
        data = hierarchy(model,sample,req.root)
        data["model"] = req.model
        data["imported"] = imported_metadata(req.model)
        return data
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/units")
def inspect_units(req: UnitsRequest):
    try:
        model = resolve_model(req.model)
        mods = dict(model.named_modules())
        if req.path not in mods:
            raise ValueError(f"Unknown module path '{req.path}'")
        mod = mods[req.path]
        if isinstance(mod, nn.Conv2d):
            total = mod.out_channels; kind = "channel"; w = mod.weight.detach().cpu()
        elif isinstance(mod, nn.Linear):
            total = mod.out_features; kind = "neuron"; w = mod.weight.detach().cpu()
        else:
            raise ValueError("Unit-level expansion is currently available for Conv2d channels and Linear neurons")
        start = min(req.offset,total); end = min(total,start+req.count); units=[]
        for i in range(start,end):
            wi = w[i]
            units.append({"index":i,"name":f"{kind} {i}","shape":list(wi.shape),"mean":float(wi.mean()),"std":float(wi.std(unbiased=False)),"min":float(wi.min()),"max":float(wi.max()),"l2":float(torch.linalg.vector_norm(wi))})
        return {"path":req.path,"module_type":mod.__class__.__name__,"kind":kind,"total":total,"offset":start,"units":units}
    except Exception as e:
        raise HTTPException(400, str(e))



@app.post("/api/activate")
async def activate_model(
    model: str = Form(...),
    text: str = Form(""),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
    image_size: int = Form(96),
):
    """Run an input through a supported model and summarize activation intensity.

    The current built-in models are vision classifiers, so image input is consumed.
    Text/audio are accepted by the multimodal workbench and reported as unused
    modalities until a model with those input paths is loaded.
    """
    try:
        m = resolve_model(model)
        m.eval()
        unused = []
        if text.strip(): unused.append("text")
        if audio is not None and audio.filename: unused.append("audio")
        if image is not None:
            raw = await image.read()
            im = Image.open(io.BytesIO(raw)).convert("RGB")
            prep = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
            x = prep(im).unsqueeze(0)
            consumed = ["image"]
        else:
            # Still permit a dry activation run so the visualizer can be tested.
            x = torch.zeros(1,3,image_size,image_size)
            consumed = ["synthetic_image"]

        stats = {}
        hooks = []
        for name, mod in m.named_modules():
            if not name or any(True for _ in mod.children()):
                continue
            def hook(module, inp, out, n=name):
                t = out[0] if isinstance(out,(tuple,list)) and out and isinstance(out[0],torch.Tensor) else out
                if not isinstance(t, torch.Tensor): return
                td = t.detach().float().cpu()
                mean_abs = float(td.abs().mean())
                rms = float(torch.sqrt((td*td).mean()))
                positive = float((td > 0).float().mean())
                max_abs = float(td.abs().max())
                score = rms
                top_units = []
                if td.ndim >= 2:
                    z = td[0]
                    if z.ndim == 1:
                        us = z.abs()
                    else:
                        us = z.abs().reshape(z.shape[0], -1).mean(dim=1)
                    k = min(12, us.numel())
                    vals, idx = torch.topk(us, k)
                    top_units = [{"index": int(i), "score": float(v)} for v,i in zip(vals,idx)]
                stats[n] = {
                    "score": score, "mean_abs": mean_abs, "rms": rms,
                    "positive_fraction": positive, "max_abs": max_abs,
                    "shape": list(td.shape), "top_units": top_units,
                    "op_type": module.__class__.__name__,
                }
            hooks.append(mod.register_forward_hook(hook))
        run_started = time.perf_counter()
        with torch.inference_mode():
            y = m(x)
        total_ms = (time.perf_counter() - run_started) * 1000.0
        for h in hooks: h.remove()
        mx = max([v["score"] for v in stats.values()] + [1e-9])
        for v in stats.values(): v["normalized"] = v["score"] / mx
        output = {"shape": list(y.shape)} if isinstance(y, torch.Tensor) else {"type": type(y).__name__}
        if isinstance(y, torch.Tensor):
            yd = y.detach().float().cpu()
            output["dtype"] = str(y.dtype).replace("torch.", "")
            output["summary"] = {
                "min": float(yd.min()), "max": float(yd.max()),
                "mean": float(yd.mean()), "std": float(yd.std(unbiased=False)),
                "l2": float(torch.linalg.vector_norm(yd)),
            }
            flat = yd.flatten()
            output["preview"] = [float(v) for v in flat[:min(16, flat.numel())]]
        if isinstance(y, torch.Tensor) and y.ndim == 2:
            logits = y[0].float().cpu()
            probs = torch.softmax(logits, dim=0)
            vals, idx = torch.topk(probs, min(5, probs.numel()))
            top_rows = [{"index": int(i), "probability": float(v), "logit": float(logits[int(i)])} for v,i in zip(vals,idx)]
            # Add standard ImageNet labels where torchvision exposes the mapping.
            # For imported/fine-tuned checkpoints this mapping is only a reference;
            # custom class semantics must come from the user's dataset.
            try:
                meta = imported_metadata(model)
                architecture = (meta or {}).get("arch") if meta else model
                weights_enum = tv_models.get_model_weights(architecture)
                categories = weights_enum.DEFAULT.meta.get("categories", [])
                if len(categories) == logits.numel():
                    for row in top_rows:
                        row["label"] = categories[row["index"]]
                    output["label_space"] = "ImageNet-1K reference labels"
            except Exception:
                pass
            output["top"] = top_rows
            lvals, lidx = torch.topk(logits, min(5, logits.numel()))
            output["top_logits"] = [{"index": int(i), "logit": float(v)} for v,i in zip(lvals,lidx)]
            output["task"] = "classification"
        return {
            "model": model, "consumed_modalities": consumed, "unused_modalities": unused,
            "model_state": "uploaded checkpoint" if model.startswith("import:") else "built-in untrained architecture",
            "runtime_ms": total_ms,
            "output": output, "activations": stats,
            "note": "Activation intensity is a numeric summary of tensor responses, not proof that a neuron semantically caused the output.",
        }
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/plan")
def plan(req: StitchRequest):
    try:
        return plan_adapter(resolve_model(req.source),req.source_cut,resolve_model(req.target),req.target_entry,req.image_size).to_dict()
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/train")
def train(req: TrainRequest):
    try:
        hybrid, ap = build_hybrid(resolve_model(req.source),req.source_cut,resolve_model(req.target),req.target_entry,req.image_size)
        result = synthetic_train(hybrid, steps=req.steps, batch_size=req.batch_size, image_size=req.image_size)
        return {"plan":ap.to_dict(),"training":result.to_dict()}
    except Exception as e:
        raise HTTPException(400, str(e))


class ProbeModel(BaseModel):
    path: str
    kind: str = "activation"
    full_tensor: bool = False
    top_k: int = Field(12, ge=1, le=128)

class InterventionModel(BaseModel):
    path: str
    kind: str
    value: float | None = None
    min_value: float | None = None
    max_value: float | None = None
    indices: list[int] | None = None

class ResearchRunRequest(BaseModel):
    model: str
    name: str = ""
    image_size: int = Field(96, ge=32, le=512)
    probes: list[ProbeModel] = []
    interventions: list[InterventionModel] = []
    capture_gradients: bool = False
    target_index: int | None = None
    seed: int = 0

class CompareRequest(BaseModel):
    run_a: str
    run_b: str

class ExperimentRequest(BaseModel):
    name: str
    baseline_run: str | None = None
    parent: str | None = None
    graph_diff: dict = {}

class AttachRunRequest(BaseModel):
    experiment_id: str
    run_id: str

class SweepRequest(BaseModel):
    model: str
    path: str
    kind: str = "scale"
    values: list[float]
    image_size: int = Field(96, ge=32, le=512)

@app.get("/api/plugins")
def plugins():
    return {"plugins": plugin_catalog()}

@app.get("/api/parameters/{model_name:path}")
def parameters(model_name: str, q: str = ""):
    try:
        model = resolve_model(model_name)
        rows = parameter_inventory(model, q)
        return {"model": model_name, "count": len(rows), "parameters": rows[:5000]}
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/research/run")
def research_run(req: ResearchRunRequest):
    try:
        torch.manual_seed(req.seed)
        model = resolve_model(req.model)
        x = torch.randn(1,3,req.image_size,req.image_size)
        result = run_research(
            model, x,
            probes=[ProbeSpec(**p.model_dump()) for p in req.probes],
            interventions=[InterventionSpec(**i.model_dump()) for i in req.interventions],
            capture_gradients=req.capture_gradients, target_index=req.target_index,
        )
        row = create_run(req.model_dump(), result)
        return row
    except Exception as e:
        raise HTTPException(400, str(e))

@app.post("/api/profile")
def profile(req: ModelRequest):
    try:
        model = resolve_model(req.model)
        x = torch.randn(1,3,req.image_size,req.image_size)
        return profile_model(model, x)
    except Exception as e:
        raise HTTPException(400, str(e))

@app.get("/api/runs")
def runs():
    return {"runs": list_runs()}

@app.get("/api/runs/{run_id}")
def run_detail(run_id: str):
    try: return get_run(run_id)
    except KeyError: raise HTTPException(404, "Unknown run")

@app.post("/api/runs/compare")
def compare(req: CompareRequest):
    try: return compare_runs(get_run(req.run_a), get_run(req.run_b))
    except KeyError: raise HTTPException(404, "Unknown run")

@app.post("/api/experiments")
def experiment(req: ExperimentRequest):
    return create_experiment(req.name, req.baseline_run, req.parent, req.graph_diff)

@app.get("/api/experiments")
def experiments():
    return {"experiments": list_experiments()}

@app.post("/api/experiments/attach")
def experiment_attach(req: AttachRunRequest):
    try: return attach_run(req.experiment_id, req.run_id)
    except KeyError: raise HTTPException(404, "Unknown experiment or run")

@app.post("/api/sweep")
def sweep(req: SweepRequest):
    if not req.values or len(req.values) > 32:
        raise HTTPException(400, "Provide 1-32 sweep values")
    # Every sweep value sees the exact same input. This avoids confounding the parameter effect with input randomness.
    torch.manual_seed(0)
    x=torch.randn(1,3,req.image_size,req.image_size)
    rows=[]
    for v in req.values:
        model=resolve_model(req.model)
        torch.manual_seed(0)
        result=run_research(model,x.clone(),probes=[ProbeSpec(path=req.path)],interventions=[InterventionSpec(path=req.path,kind=req.kind,value=v)])
        rows.append({"value":v,"output":result["output"],"capture":result["captures"].get(req.path)})
    return {"model":req.model,"path":req.path,"kind":req.kind,"input_seed":0,"results":rows}

@app.get("/api/export/python/{model_name:path}")
def export_python(model_name: str):
    try:
        model=resolve_model(model_name)
        meta=imported_metadata(model_name)
        architecture=(meta or {}).get("arch") if meta else model_name
        content=python_source(model_name, architecture, 224)
        return {"filename":"model.py","architecture":architecture,"content":content,"complete":True}
    except Exception as e:
        raise HTTPException(400,str(e))

@app.get("/api/export/python-bundle/{model_name:path}")
def export_python_bundle_api(model_name: str):
    try:
        model=resolve_model(model_name)
        meta=imported_metadata(model_name)
        architecture=(meta or {}).get("arch") if meta else model_name
        path=export_python_bundle(model, model_name, architecture, 224)
        return FileResponse(path,filename=path.name,media_type="application/zip")
    except Exception as e:
        raise HTTPException(400,str(e))

@app.post("/api/export/onnx")
def export_onnx_api(req: ModelRequest):
    try:
        path=export_onnx(resolve_model(req.model),req.image_size,"modelforge_export")
        return FileResponse(path,filename=path.name,media_type="application/octet-stream")
    except Exception as e:
        raise HTTPException(400,str(e))

@app.get('/api/imports')
def imports_catalog_api():
    return {'imports': imported_catalog()}

@app.post('/api/research/run-input')
async def research_run_input(
    model: str = Form(...),
    name: str = Form(''),
    probes_json: str = Form('[]'),
    interventions_json: str = Form('[]'),
    capture_gradients: bool = Form(False),
    target_index: int | None = Form(None),
    seed: int = Form(0),
    image_size: int = Form(96),
    text: str = Form(''),
    image: UploadFile | None = File(None),
    audio: UploadFile | None = File(None),
):
    """Run probes/interventions against the exact user input supplied to the workbench."""
    import json, platform
    try:
        torch.manual_seed(seed)
        raw = await image.read() if image is not None else None
        prepared = prepare_vision_input(raw, image_size=image_size, text=text, audio_present=bool(audio and audio.filename))
        m = resolve_model(model)
        probes = [ProbeSpec(**x) for x in json.loads(probes_json or '[]')]
        interventions = [InterventionSpec(**x) for x in json.loads(interventions_json or '[]')]
        result = run_research_call(m, prepared.args, prepared.kwargs, probes, interventions, capture_gradients, target_index)
        result['input']={'consumed_modalities':prepared.consumed,'unused_modalities':prepared.unused,'preview':prepared.preview}
        payload={'model':model,'name':name,'image_size':image_size,'probes':[p.__dict__ for p in probes],'interventions':[i.__dict__ for i in interventions],'capture_gradients':capture_gradients,'target_index':target_index,'seed':seed,'input':result['input']}
        result['reproducibility']={'seed':seed,'torch':torch.__version__,'python':platform.python_version()}
        return create_run(payload,result)
    except Exception as e:
        raise HTTPException(400,str(e))

@app.get('/api/runs/{run_id}/code')
def run_code(run_id:str, flavor:str='pytorch'):
    try:
        row=get_run(run_id); content=run_to_python(row,flavor); return {'run_id':run_id,'flavor':flavor,'filename':f'run_{run_id}.py','content':content}
    except KeyError: raise HTTPException(404,'Unknown run')
    except Exception as e: raise HTTPException(400,str(e))

class HFRunRequest(BaseModel):
    model: str = 'gpt2'
    text: str
    probes: list[ProbeModel] = []
    interventions: list[InterventionModel] = []
    capture_gradients: bool = False
    seed: int = 0
    device: str = 'cpu'

@app.get('/api/hf/catalog')
def hf_catalog():
    from .hf import SUPPORTED_TEXT_MODELS
    return {'models':SUPPORTED_TEXT_MODELS,'optional_dependency':'pip install "modelforge[transformers]"'}

@app.post('/api/hf/run')
def hf_run(req:HFRunRequest):
    from .hf import load_text_model
    try:
        torch.manual_seed(req.seed)
        handle=load_text_model(req.model,device=req.device)
        args,kwargs=handle.prepare(req.text)
        result=run_research_call(handle.model,args,kwargs,[ProbeSpec(**p.model_dump()) for p in req.probes],[InterventionSpec(**i.model_dump()) for i in req.interventions],req.capture_gradients,None)
        with torch.inference_mode(): raw=handle.model(*args,**kwargs)
        result['output']['top_tokens']=handle.decode_top(raw)
        payload={'model':f'hf:{handle.model_id}','name':'text research run','text':req.text,'probes':[p.model_dump() for p in req.probes],'interventions':[i.model_dump() for i in req.interventions],'capture_gradients':req.capture_gradients,'seed':req.seed}
        return create_run(payload,result)
    except Exception as e:
        raise HTTPException(400,str(e))

@app.post('/api/patch/image')
async def patch_image(
    model: str = Form(...),
    path: str = Form(...),
    index: int | None = Form(None),
    image_size: int = Form(96),
    clean: UploadFile = File(...),
    corrupted: UploadFile = File(...),
):
    """Patch an internal activation from a clean image into a corrupted-image run."""
    from .patching import activation_patch
    try:
        clean_p=prepare_vision_input(await clean.read(),image_size=image_size)
        corrupt_p=prepare_vision_input(await corrupted.read(),image_size=image_size)
        return activation_patch(resolve_model(model),clean_p.args,clean_p.kwargs,corrupt_p.args,corrupt_p.kwargs,path,index)
    except Exception as e: raise HTTPException(400,str(e))

class HFDatasetRequest(BaseModel):
    model: str = 'gpt2'
    prompts: list[str]
    probe_path: str | None = None
    intervention: InterventionModel | None = None
    seed: int = 0
    device: str = 'cpu'

@app.post('/api/hf/dataset-run')
def hf_dataset_run(req:HFDatasetRequest):
    from .hf import load_text_model
    try:
        if not req.prompts or len(req.prompts)>256: raise ValueError('Provide 1-256 prompts')
        torch.manual_seed(req.seed); handle=load_text_model(req.model,device=req.device); rows=[]
        for i,prompt in enumerate(req.prompts):
            args,kwargs=handle.prepare(prompt)
            probes=[ProbeSpec(path=req.probe_path)] if req.probe_path else []
            ints=[InterventionSpec(**req.intervention.model_dump())] if req.intervention else []
            result=run_research_call(handle.model,args,kwargs,probes,ints,False,None)
            with torch.inference_mode(): raw=handle.model(*args,**kwargs)
            score=None
            if req.probe_path and req.probe_path in result['captures']:
                score=result['captures'][req.probe_path]['activation'].get('rms')
            rows.append({'index':i,'prompt':prompt,'probe_score':score,'top_tokens':handle.decode_top(raw,5)})
        ranked=sorted(rows,key=lambda r:float('-inf') if r['probe_score'] is None else r['probe_score'],reverse=True)
        return {'model':handle.model_id,'count':len(rows),'probe_path':req.probe_path,'top_activating_examples':ranked[:20],'rows':rows}
    except Exception as e: raise HTTPException(400,str(e))
