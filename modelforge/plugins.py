from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Callable, Any

@dataclass
class Plugin:
    name: str
    kind: str
    description: str
    version: str = "0.1"

REGISTRY: dict[str, Plugin] = {}

def register(name: str, kind: str, description: str, version: str = "0.1"):
    REGISTRY[name] = Plugin(name, kind, description, version)

def catalog():
    return [asdict(v) for v in REGISTRY.values()]

for item in [
    ("activation_probe","probe","Activation statistics and top-unit capture"),
    ("gradient_probe","probe","Gradient-flow statistics"),
    ("latency_probe","probe","Per-module timing"),
    ("ablate","intervention","Zero a module output or selected units"),
    ("scale","intervention","Scale module output"),
    ("noise","intervention","Inject Gaussian noise"),
    ("clamp","intervention","Clamp activation ranges"),
    ("detach","intervention","Stop gradient flow at a module boundary"),
    ("mask","intervention","Mask selected output units/channels"),
    ("freeze","intervention","Freeze a module's parameters for the run"),
]: register(*item)
