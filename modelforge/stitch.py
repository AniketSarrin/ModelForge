from __future__ import annotations
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
import torch.nn.functional as F
from .models import Prefix, Suffix, module_shapes


@dataclass
class AdapterPlan:
    kind: str
    source_shape: list[int]
    target_shape: list[int]
    source_channels: int
    target_channels: int
    target_hw: tuple[int, int]
    params: int

    def to_dict(self):
        d = asdict(self)
        d["target_hw"] = list(self.target_hw)
        return d


class ConvSpatialAdapter(nn.Module):
    """Simple learned channel projection + deterministic spatial resize."""
    def __init__(self, in_channels: int, out_channels: int, target_hw: tuple[int, int]):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
        self.target_hw = target_hw

    def forward(self, x):
        if tuple(x.shape[-2:]) != self.target_hw:
            x = F.interpolate(x, size=self.target_hw, mode="bilinear", align_corners=False)
        return self.act(self.norm(self.proj(x)))


class HybridModel(nn.Module):
    def __init__(self, prefix: nn.Module, adapter: nn.Module, suffix: nn.Module):
        super().__init__()
        self.prefix = prefix
        self.adapter = adapter
        self.suffix = suffix

    def forward(self, x):
        return self.suffix(self.adapter(self.prefix(x)))

    def freeze_parents(self):
        for p in self.prefix.parameters(): p.requires_grad = False
        for p in self.suffix.parameters(): p.requires_grad = False
        for p in self.adapter.parameters(): p.requires_grad = True
        return self


def plan_adapter(source_model, source_cut: str, target_model, target_entry: str, image_size: int = 224) -> AdapterPlan:
    src = module_shapes(source_model, image_size)[source_cut]["output"]
    tgt = module_shapes(target_model, image_size)[target_entry]["input"]
    if len(src) != 4 or len(tgt) != 4:
        raise ValueError("MVP currently supports 4-D CNN feature-map stitching only.")
    in_c, out_c = src[1], tgt[1]
    hw = (tgt[2], tgt[3])
    # conv1x1 weights + bias + batchnorm affine params
    params = in_c * out_c + out_c + 2 * out_c
    return AdapterPlan(
        kind="conv1x1+batchnorm+gelu+resize",
        source_shape=src,
        target_shape=tgt,
        source_channels=in_c,
        target_channels=out_c,
        target_hw=hw,
        params=params,
    )


def build_hybrid(source_model, source_cut: str, target_model, target_entry: str, image_size: int = 224) -> tuple[HybridModel, AdapterPlan]:
    plan = plan_adapter(source_model, source_cut, target_model, target_entry, image_size)
    prefix = Prefix(source_model, source_cut)
    suffix = Suffix(target_model, target_entry)
    adapter = ConvSpatialAdapter(plan.source_channels, plan.target_channels, plan.target_hw)
    return HybridModel(prefix, adapter, suffix).freeze_parents(), plan
