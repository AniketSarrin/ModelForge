from __future__ import annotations
from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision import models

# Broad inspection support. Stitching remains ResNet-oriented in this MVP.
SUPPORTED = (
    "resnet18", "resnet34", "resnet50",
    "densenet121", "mobilenet_v3_small", "mobilenet_v3_large",
    "efficientnet_b0", "vgg11", "alexnet",
)
SOURCE_CUTS = ("stem", "layer1", "layer2", "layer3", "layer4")
TARGET_ENTRIES = ("layer1", "layer2", "layer3", "layer4", "classifier")
RESNET_FAMILY = {"resnet18", "resnet34", "resnet50"}


def load_model(name: str, pretrained: bool = False) -> nn.Module:
    if name not in SUPPORTED:
        raise ValueError(f"Unsupported model '{name}'. Supported: {', '.join(SUPPORTED)}")
    fn = getattr(models, name)
    return fn(weights="DEFAULT" if pretrained else None)


def stem(model: nn.Module) -> nn.Sequential:
    required = ["conv1", "bn1", "relu", "maxpool"]
    if not all(hasattr(model, x) for x in required):
        raise ValueError("The current stitching MVP supports ResNet-family models for prefix/suffix slicing")
    return nn.Sequential(OrderedDict([
        ("conv1", model.conv1),
        ("bn1", model.bn1),
        ("relu", model.relu),
        ("maxpool", model.maxpool),
    ]))


class Prefix(nn.Module):
    def __init__(self, model: nn.Module, cut: str):
        super().__init__()
        self.stem = stem(model)
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.cut = cut

    def forward(self, x):
        x = self.stem(x)
        if self.cut == "stem": return x
        x = self.layer1(x)
        if self.cut == "layer1": return x
        x = self.layer2(x)
        if self.cut == "layer2": return x
        x = self.layer3(x)
        if self.cut == "layer3": return x
        x = self.layer4(x)
        return x


class Suffix(nn.Module):
    def __init__(self, model: nn.Module, entry: str):
        super().__init__()
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
        self.fc = model.fc
        self.entry = entry

    def forward(self, x):
        if self.entry == "layer1": x = self.layer1(x)
        if self.entry in ("layer1", "layer2"): x = self.layer2(x)
        if self.entry in ("layer1", "layer2", "layer3"): x = self.layer3(x)
        if self.entry in ("layer1", "layer2", "layer3", "layer4"): x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def module_shapes(model: nn.Module, image_size: int = 224) -> dict[str, dict[str, list[int]]]:
    info: dict[str, dict[str, list[int]]] = {}
    handles = []
    sem = {}
    for key in ("maxpool", "layer1", "layer2", "layer3", "layer4", "fc"):
        if hasattr(model, key):
            sem[{"maxpool": "stem", "fc": "classifier"}.get(key, key)] = getattr(model, key)
    if not sem:
        return info
    for name, module in sem.items():
        def hook(mod, inp, out, n=name):
            x_in = inp[0] if isinstance(inp, tuple) else inp
            info[n] = {
                "input": list(x_in.shape) if isinstance(x_in, torch.Tensor) else [],
                "output": list(out.shape) if isinstance(out, torch.Tensor) else [],
            }
        handles.append(module.register_forward_hook(hook))
    x = torch.zeros(1, 3, image_size, image_size)
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    return info
