from __future__ import annotations
from pathlib import Path
import tempfile, zipfile, textwrap
import torch
import torch.nn as nn

EXPORT_DIR = Path(tempfile.gettempdir()) / "modelforge_exports"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def export_onnx(model: nn.Module, image_size: int = 96, name: str = "model") -> Path:
    path = EXPORT_DIR / f"{name}.onnx"
    x = torch.randn(1,3,image_size,image_size)
    model.eval()
    torch.onnx.export(model, x, str(path), input_names=["input"], output_names=["output"], opset_version=17, dynamo=False)
    return path


def python_source(model_name: str, architecture: str, image_size: int = 224) -> str:
    """Generate a complete runnable torchvision model script.

    The script can run without weights for architecture validation, or load the
    weights.pth included by ModelForge's Python bundle export.
    """
    return textwrap.dedent(f'''\
        """ModelForge export

        Source model: {model_name}
        Architecture: {architecture}

        Run:
            python model.py --weights weights.pth

        Or validate the architecture without a checkpoint:
            python model.py
        """
        from __future__ import annotations
        import argparse
        from pathlib import Path
        import torch
        from torchvision import models

        ARCHITECTURE = {architecture!r}

        def build_model() -> torch.nn.Module:
            if not hasattr(models, ARCHITECTURE):
                raise RuntimeError(f"torchvision.models has no architecture {{ARCHITECTURE!r}}")
            constructor = getattr(models, ARCHITECTURE)
            return constructor(weights=None)

        def extract_state_dict(obj):
            if isinstance(obj, dict):
                for key in ("state_dict", "model_state_dict", "model", "module"):
                    value = obj.get(key)
                    if isinstance(value, dict):
                        obj = value
                        break
            if not isinstance(obj, dict):
                raise RuntimeError("Checkpoint does not contain a state_dict")
            state = {{}}
            for key, value in obj.items():
                if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                    continue
                clean = key
                for prefix in ("module.", "model.", "net."):
                    if clean.startswith(prefix):
                        clean = clean[len(prefix):]
                state[clean] = value
            if not state:
                raise RuntimeError("No tensor weights found in checkpoint")
            return state

        def load_weights(model: torch.nn.Module, path: str | Path) -> None:
            path = Path(path)
            if path.suffix.lower() == ".safetensors":
                try:
                    from safetensors.torch import load_file
                except ImportError as exc:
                    raise RuntimeError("pip install safetensors to load this file") from exc
                state = load_file(str(path), device="cpu")
            else:
                state = extract_state_dict(torch.load(path, map_location="cpu", weights_only=True))
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print("Missing keys:", missing)
            if unexpected:
                print("Unexpected keys:", unexpected)

        def main() -> None:
            parser = argparse.ArgumentParser(description="Run a ModelForge-exported model")
            parser.add_argument("--weights", type=str, default=None, help=".pth/.pt/.safetensors checkpoint")
            parser.add_argument("--image-size", type=int, default={int(image_size)})
            parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
            args = parser.parse_args()

            model = build_model()
            if args.weights:
                load_weights(model, args.weights)
            model = model.to(args.device).eval()

            x = torch.randn(1, 3, args.image_size, args.image_size, device=args.device)
            with torch.inference_mode():
                output = model(x)

            print(model)
            if isinstance(output, torch.Tensor):
                print("Output shape:", tuple(output.shape))
                print("Output dtype:", output.dtype)
                if output.ndim == 2:
                    probabilities = torch.softmax(output[0].float(), dim=-1)
                    values, indices = torch.topk(probabilities, min(5, probabilities.numel()))
                    print("Top outputs:")
                    for rank, (prob, index) in enumerate(zip(values.tolist(), indices.tolist()), 1):
                        print(f"  {{rank}}. class {{index}}: {{prob:.4%}}")
            else:
                print("Output type:", type(output).__name__)

        if __name__ == "__main__":
            main()
    ''')


def export_python_bundle(model: nn.Module, model_name: str, architecture: str, image_size: int = 224) -> Path:
    """Create a self-contained source + current state-dict bundle."""
    safe_arch = ''.join(c if c.isalnum() or c in ('-', '_') else '_' for c in architecture)
    bundle = EXPORT_DIR / f"modelforge_{safe_arch}_python.zip"
    work = Path(tempfile.mkdtemp(prefix="modelforge_py_"))
    source_path = work / "model.py"
    weights_path = work / "weights.pth"
    readme_path = work / "README.txt"
    source_path.write_text(python_source(model_name, architecture, image_size), encoding="utf-8")
    torch.save(model.state_dict(), weights_path)
    readme_path.write_text(
        "ModelForge runnable Python export\n\n"
        "Install:\n  pip install torch torchvision\n\n"
        "Run:\n  python model.py --weights weights.pth\n\n"
        "The weights file is the exact state_dict of the model that was open when this bundle was exported.\n",
        encoding="utf-8",
    )
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(source_path, "model.py")
        zf.write(weights_path, "weights.pth")
        zf.write(readme_path, "README.txt")
    return bundle
