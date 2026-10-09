# SPDX-License-Identifier: GPL-3.0-only
"""Native F12 overlays, prepared once from an existing INT8 ConvRot model."""
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import comfy.ops
import comfy.quant_ops
import comfy.sd
import comfy.utils
import comfy_kitchen as ck
import comfy_kitchen.backends.cuda as cuda_backend
import comfy_kitchen.backends.cuda._C as native
from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout as Layout, QuantizedTensor

FORMAT = "h3-lbh-f12-cache-1"
LEGACY_FORMAT = "h3-streaming-v2-cache-1"
LAYOUT = "TensorCoreConvRotW4A4Layout"
BLOCKS = (19, 22, 23, 24, 25, 26, 27, 28, 29, 47, 48, 49)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def identity(path):
    path = Path(path)
    with path.open("rb") as handle:
        size = int.from_bytes(handle.read(8), "little")
        require(0 < size <= min(path.stat().st_size - 8, 64 * 1024**2), "Invalid safetensors header")
        header = handle.read(size)
    return {"size": path.stat().st_size, "header_sha256": hashlib.sha256(header).hexdigest()}


def cache_file(directory, row):
    directory = Path(directory).resolve()
    path = (directory / row["file"]).resolve()
    require(path.parent == directory, "Cache shard must be directly inside its directory")
    require(path.is_file() and path.stat().st_size == row["file_size"], "Missing or incomplete F12 shard: " + path.name)
    return path


def manifest(directory, source=None):
    data = json.loads((Path(directory) / "manifest.json").read_text(encoding="utf-8"))
    require(data.get("complete") is True, "F12 cache is incomplete; preserve it and choose another directory")
    indices = [row["block"] for row in data["fc1"]]
    if data.get("format") == FORMAT:
        require(indices == list(BLOCKS), "F12 cache has the wrong block selection")
    else:
        require(data.get("format") == LEGACY_FORMAT and indices == list(range(50)), "Unsupported F12 cache format")
    if source is not None:
        require(data["model"]["identity"] == identity(source), "Source model does not match the F12 cache")
    selected = [row for row in data["fc1"] if row["block"] in BLOCKS]
    for row in selected:
        cache_file(directory, row)
    return {**data, "fc1": selected}


def check_environment():
    require("convrot_w4a4" in comfy.quant_ops.QUANT_ALGOS, "ComfyUI lacks native ConvRot W4A4 serialization")
    require("disable_dynamic" in inspect.signature(comfy.sd.load_diffusion_model_state_dict).parameters,
            "ComfyUI lacks the required Dynamic VRAM loader API")
    require(hasattr(native, "cutlass_int4_dequant"), "Native ConvRot INT4 backend is missing")
    require(torch.cuda.is_available(), "This F12 preset requires NVIDIA CUDA")
    device = torch.device("cuda", torch.cuda.current_device())
    supported = getattr(cuda_backend, "_cuda_device_supports_native_int4_mma", None)
    require(supported is not None and supported(torch.empty(0, device=device)), "This GPU/backend would select the INT8 fallback")
    require(ck.sol_attn_is_available(device), "Compiled Sol-Attn is unavailable on this GPU")
    return {"torch": torch.__version__, "comfy-kitchen": importlib.metadata.version("comfy-kitchen"),
            "gpu": torch.cuda.get_device_name(device), "layout": LAYOUT}


def validate_fc1_state(state, index):
    prefix = f"blocks.{index}.mlp.fc1."
    require(set(state) == {prefix + key for key in ("weight", "weight_scale", "comfy_quant")}, "Invalid F12 shard keys")
    weight, scale, quant = (state[prefix + key] for key in ("weight", "weight_scale", "comfy_quant"))
    require(weight.dtype == torch.int8 and tuple(weight.shape) == (28672, 2688), "Invalid F12 packed weight")
    require(scale.dtype == torch.float32 and tuple(scale.shape) == (28672,), "Invalid F12 scales")
    require(quant.dtype == torch.uint8 and quant.numel() < 4096, "Invalid F12 quantization metadata")
    config = json.loads(quant.numpy().tobytes())
    require(config.get("format") == "convrot_w4a4" and config.get("convrot_groupsize") == 256
            and config.get("linear_dtype", "int4") == "int4", "Unsupported F12 quantization parameters")


def _linear(state):
    layer = comfy.ops.mixed_precision_ops(compute_dtype=torch.bfloat16).Linear(5376, 28672, bias=False, device="cpu")
    layer.load_state_dict(dict(state), strict=True)
    return layer


def prepare(source, directory, progress=None):
    source, directory = Path(source), Path(directory)
    if directory.exists():
        return manifest(directory, source)
    check_environment()
    report = {"format": FORMAT, "complete": False, "blocks": list(BLOCKS),
              "model": {"identity": identity(source), "sha256": sha256(source)}, "fc1": []}
    device = torch.device("cuda", torch.cuda.current_device())
    generator = torch.Generator(device=device).manual_seed(43)
    x = torch.randn(1024, 5376, device=device, dtype=torch.bfloat16, generator=generator)
    directory.mkdir(parents=True, exist_ok=False)
    manifest_path = directory / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with safe_open(str(source), framework="pt", device="cpu") as original:
        keys = set(original.keys())
        for completed, index in enumerate(BLOCKS, 1):
            prefix = f"blocks.{index}.mlp.fc1."
            state = {key[len(prefix):]: original.get_tensor(key) for key in keys if key.startswith(prefix)}
            require(set(state) == {"weight", "weight_scale", "comfy_quant"}, "Expected native INT8 ConvRot FC1")
            layer = _linear(state)
            require(layer.weight._layout_cls == "TensorWiseINT8Layout" and layer.weight._params.convrot,
                    "Source FC1 must be native INT8 ConvRot")
            logical = layer.weight.dequantize()
            require(logical.dtype == torch.bfloat16 and bool(torch.isfinite(logical).all()), "Invalid logical FC1")
            packed, params = Layout.quantize(logical.to(device), convrot_groupsize=256, stochastic_rounding=0)
            runtime = QuantizedTensor(packed, LAYOUT, params)
            cpu = runtime.to("cpu")
            layer.weight = torch.nn.Parameter(cpu, requires_grad=False)
            layer.quant_format, layer.layout_type = "convrot_w4a4", LAYOUT
            native_state = {prefix + key: value.detach().contiguous() for key, value in layer.state_dict().items()}
            validate_fc1_state(native_state, index)
            path = directory / f"block_{index:02d}.safetensors"
            save_file(native_state, str(path), metadata={"format": FORMAT, "block": str(index)})
            saved = load_file(str(path), device="cpu")
            require(all(torch.equal(value, saved[key]) for key, value in native_state.items()), "F12 save/reload mismatch")
            restored = _linear({key[len(prefix):]: value for key, value in saved.items()})
            require(all(getattr(restored.weight._params, key) == getattr(cpu._params, key)
                        for key in ("orig_dtype", "orig_shape", "convrot_groupsize", "quant_group_size", "linear_dtype", "transposed")),
                    "F12 layout restoration mismatch")
            reference = torch.nn.functional.linear(x, runtime)
            restored.to(device)
            actual = restored(x)
            restored.to("cpu")
            require(restored.weight.device.type == "cpu" and restored.weight._params.scale.device.type == "cpu", "F12 CPU offload failed")
            restored.to(device)
            again = restored(x)
            require(bool(torch.isfinite(actual).all()) and torch.equal(reference, actual) and torch.equal(actual, again),
                    "F12 native offload/reload mismatch")
            report["fc1"].append({"block": index, "file": path.name, "file_size": path.stat().st_size,
                                  "sha256": sha256(path), "native_roundtrip_exact": True})
            manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            if progress is not None:
                progress(completed, len(BLOCKS))
            del layer, restored, logical, packed, runtime, cpu, native_state, saved, actual, again, reference, state
    report["complete"] = True
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return manifest(directory, source)
