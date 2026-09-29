from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Any
import torch
import torch.nn as nn


@dataclass
class TensorSpec:
    shape: list[int]
    dtype: str


@dataclass
class IRNode:
    id: str
    op_type: str
    params: int
    input: TensorSpec | None = None
    output: TensorSpec | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _spec(x: Any) -> TensorSpec | None:
    if isinstance(x, torch.Tensor):
        return TensorSpec(shape=list(x.shape), dtype=str(x.dtype).replace("torch.", ""))
    if isinstance(x, (tuple, list)) and x and isinstance(x[0], torch.Tensor):
        return TensorSpec(shape=list(x[0].shape), dtype=str(x[0].dtype).replace("torch.", ""))
    return None


def trace_named_modules(model: nn.Module, sample: torch.Tensor, leaves_only: bool = True) -> list[IRNode]:
    """Trace modules into a minimal compiler-friendly IR.

    When leaves_only=False, parent modules are traced too, which powers the
    Vivado-like hierarchical browser in the UI.
    """
    nodes: dict[str, IRNode] = {}
    hooks = []

    for name, module in model.named_modules():
        if not name:
            continue
        if leaves_only and any(True for _ in module.children()):
            continue
        nodes[name] = IRNode(
            id=name,
            op_type=module.__class__.__name__,
            params=sum(p.numel() for p in module.parameters(recurse=False)),
        )

        def hook(mod, inp, out, node_name=name):
            node = nodes[node_name]
            node.input = _spec(inp)
            node.output = _spec(out)

        hooks.append(module.register_forward_hook(hook))

    was_training = model.training
    model.eval()
    with torch.inference_mode():
        model(sample)
    if was_training:
        model.train()
    for h in hooks:
        h.remove()
    return list(nodes.values())



def module_details(module: nn.Module) -> dict[str, Any]:
    """Return literal PyTorch configuration for paper-style inspection.

    These are model attributes, not invented visualization semantics.
    """
    d: dict[str, Any] = {}
    if isinstance(module, nn.Conv2d):
        d = {
            "in_channels": module.in_channels, "out_channels": module.out_channels,
            "kernel_size": list(module.kernel_size), "stride": list(module.stride),
            "padding": list(module.padding), "dilation": list(module.dilation),
            "groups": module.groups, "bias": module.bias is not None,
        }
    elif isinstance(module, nn.Linear):
        d = {"in_features": module.in_features, "out_features": module.out_features, "bias": module.bias is not None}
    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        d = {"num_features": module.num_features, "eps": module.eps, "momentum": module.momentum, "affine": module.affine}
    elif isinstance(module, nn.LayerNorm):
        ns = module.normalized_shape
        d = {"normalized_shape": list(ns) if isinstance(ns, (tuple,list)) else [ns], "eps": module.eps, "elementwise_affine": module.elementwise_affine}
    elif isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d, nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
        def li(v): return list(v) if isinstance(v, tuple) else v
        d = {"kernel_size": li(module.kernel_size), "stride": li(module.stride), "padding": li(module.padding)}
    elif isinstance(module, nn.AdaptiveAvgPool2d):
        v = module.output_size
        d = {"output_size": list(v) if isinstance(v, tuple) else v}
    elif isinstance(module, nn.Dropout):
        d = {"p": module.p, "inplace": module.inplace}
    elif isinstance(module, nn.ReLU):
        d = {"inplace": module.inplace}
    elif isinstance(module, nn.Embedding):
        d = {"num_embeddings": module.num_embeddings, "embedding_dim": module.embedding_dim, "padding_idx": module.padding_idx}
    elif isinstance(module, nn.MultiheadAttention):
        d = {"embed_dim": module.embed_dim, "num_heads": module.num_heads, "dropout": module.dropout, "batch_first": module.batch_first}
    # Torchvision residual containers expose these attributes even though their
    # classes are defined outside this module.
    cname = module.__class__.__name__
    if cname in {"BasicBlock", "Bottleneck"}:
        d["stride"] = list(module.stride) if isinstance(getattr(module, "stride", 1), tuple) else getattr(module, "stride", 1)
        d["residual"] = True
    if isinstance(module, nn.Sequential):
        children = list(module.children())
        if children:
            types = [c.__class__.__name__ for c in children]
            d["length"] = len(children)
            if len(set(types)) == 1:
                d["repeated_type"] = types[0]
    return d


def paper_label(module: nn.Module, name: str = "") -> str:
    """Compact notation commonly used in architecture figures."""
    d = module_details(module)
    if isinstance(module, nn.Conv2d):
        k = module.kernel_size[0] if module.kernel_size[0] == module.kernel_size[1] else f"{module.kernel_size[0]}×{module.kernel_size[1]}"
        if isinstance(k, int): k = f"{k}×{k}"
        s = module.stride[0] if module.stride[0] == module.stride[1] else f"{module.stride[0]}×{module.stride[1]}"
        suffix = f" /{s}" if s != 1 else ""
        return f"Conv {k}, {module.out_channels}{suffix}"
    if isinstance(module, nn.Linear): return f"FC {module.in_features}→{module.out_features}"
    if isinstance(module, nn.BatchNorm2d): return f"BN {module.num_features}"
    if isinstance(module, nn.LayerNorm): return f"LayerNorm {tuple(module.normalized_shape)}"
    if isinstance(module, nn.MaxPool2d):
        k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
        s = module.stride if isinstance(module.stride, int) else module.stride[0]
        return f"MaxPool {k}×{k} /{s}"
    if isinstance(module, nn.AvgPool2d):
        k = module.kernel_size if isinstance(module.kernel_size, int) else module.kernel_size[0]
        s = module.stride if isinstance(module.stride, int) else module.stride[0]
        return f"AvgPool {k}×{k} /{s}"
    if isinstance(module, nn.AdaptiveAvgPool2d): return f"AdaptiveAvgPool {module.output_size}"
    if isinstance(module, nn.ReLU): return "ReLU"
    if isinstance(module, nn.GELU): return "GELU"
    if isinstance(module, nn.Flatten): return "Flatten"
    if isinstance(module, nn.Dropout): return f"Dropout p={module.p:g}"
    if isinstance(module, nn.Embedding): return f"Embedding {module.num_embeddings}×{module.embedding_dim}"
    if isinstance(module, nn.MultiheadAttention): return f"MHA {module.embed_dim}, {module.num_heads} heads"
    cname = module.__class__.__name__
    if cname == "BasicBlock": return "BasicBlock"
    if cname == "Bottleneck": return "Bottleneck"
    if isinstance(module, nn.Sequential) and d.get("repeated_type"):
        return f"{d['repeated_type']} ×{d['length']}"
    return cname

def hierarchy(model: nn.Module, sample: torch.Tensor, root: str = "") -> dict[str, Any]:
    """Return one level of the real module hierarchy with runtime tensor shapes."""
    traced = {n.id: n for n in trace_named_modules(model, sample, leaves_only=False)}
    modules = dict(model.named_modules())
    if root and root not in modules:
        raise ValueError(f"Unknown module path '{root}'")
    container = modules[root] if root else model
    children = []
    for child_name, child in container.named_children():
        full = f"{root}.{child_name}" if root else child_name
        node = traced.get(full)
        descendants = list(child.named_children())
        children.append({
            "id": full,
            "name": child_name,
            "op_type": child.__class__.__name__,
            "params_self": sum(p.numel() for p in child.parameters(recurse=False)),
            "params_total": sum(p.numel() for p in child.parameters()),
            "has_children": bool(descendants),
            "child_count": len(descendants),
            "input": asdict(node.input) if node and node.input else None,
            "output": asdict(node.output) if node and node.output else None,
            "details": module_details(child),
            "paper_label": paper_label(child, child_name),
        })
    root_node = traced.get(root) if root else None
    return {
        "root": root,
        "root_type": container.__class__.__name__,
        "root_params": sum(p.numel() for p in container.parameters()),
        "root_input": asdict(root_node.input) if root_node and root_node.input else None,
        "root_output": asdict(root_node.output) if root_node and root_node.output else None,
        "root_details": module_details(container),
        "root_paper_label": paper_label(container, root or "model"),
        "children": children,
    }
