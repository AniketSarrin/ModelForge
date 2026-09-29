from __future__ import annotations
from dataclasses import dataclass, asdict
import time
import torch
import torch.nn as nn


@dataclass
class TrainResult:
    steps: int
    loss_start: float
    loss_end: float
    trainable_params: int
    elapsed_seconds: float
    output_shape: list[int]

    def to_dict(self): return asdict(self)


def synthetic_train(model: nn.Module, steps: int = 3, batch_size: int = 2, image_size: int = 224, num_classes: int = 1000, device: str = "cpu") -> TrainResult:
    model = model.to(device)
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    losses = []
    start = time.perf_counter()
    out = None
    for _ in range(steps):
        x = torch.randn(batch_size, 3, image_size, image_size, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)
        opt.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        opt.step()
        losses.append(float(loss.detach().cpu()))
    elapsed = time.perf_counter() - start
    return TrainResult(
        steps=steps,
        loss_start=losses[0],
        loss_end=losses[-1],
        trainable_params=sum(p.numel() for p in params),
        elapsed_seconds=round(elapsed, 3),
        output_shape=list(out.shape) if out is not None else [],
    )
