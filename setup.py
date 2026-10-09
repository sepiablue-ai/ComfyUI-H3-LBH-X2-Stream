"""Use an existing ComfyUI Python; no ComfyUI launch, model download or core edits."""
import argparse
import asyncio
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent
DEPENDENCY_NAME = "ComfyUI-H3-X2-Stream"
DEPENDENCY_COMMIT = "eb148a3d0f91a273668bbe027bfcbafb85f2c5e8"
MODELS = {
    "diffusion_models": "minimax_h3_fused_refdelta_r1024_turbo8_mystic07_int8_convrot.safetensors",
    "text_encoders": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
    "video_vae": "minimax_h3_video_vae_int8_convrot.safetensors",
    "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
    "x2_vae": "MiniMax-H3-X2-Detail-v1.safetensors",
    "latent_upscale_models": "minimax_h3_latent_upscaler_3d_fp16.safetensors",
}


def sha(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def package_files():
    excluded = {".git", "__pycache__", ".pytest_cache", "tmp", "results"}
    return [path for path in ROOT.rglob("*") if path.is_file()
            and not excluded.intersection(path.relative_to(ROOT).parts)
            and path.suffix not in {".pyc", ".mp4", ".safetensors", ".pth"}]


def install_package(comfy_root):
    destination = comfy_root / "custom_nodes" / ROOT.name
    if destination.resolve() == ROOT.resolve():
        return destination
    files = package_files()
    if destination.exists():
        for source in files:
            existing = destination / source.relative_to(ROOT)
            if not existing.is_file() or sha(existing) != sha(source):
                raise RuntimeError("Installed package differs; preserve it and install this checkout manually: " + str(destination))
        return destination
    destination.mkdir(parents=True, exist_ok=False)
    for source in files:
        target = destination / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    return destination


def install_dependency(comfy_root):
    destination = comfy_root / "custom_nodes" / DEPENDENCY_NAME
    if destination.exists():
        return destination
    url = f"https://codeload.github.com/sepiablue-ai/{DEPENDENCY_NAME}/zip/{DEPENDENCY_COMMIT}"
    request = urllib.request.Request(url, headers={"User-Agent": "H3-LBH-X2-Setup"})
    with urllib.request.urlopen(request, timeout=60) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    prefix = f"{DEPENDENCY_NAME}-{DEPENDENCY_COMMIT}/"
    members = []
    for member in archive.infolist():
        if not member.filename.startswith(prefix):
            raise RuntimeError("Unexpected dependency archive root")
        relative = Path(member.filename[len(prefix):])
        if relative.is_absolute() or ".." in relative.parts or (member.external_attr >> 16) & 0o170000 == 0o120000:
            raise RuntimeError("Unsafe dependency archive path")
        target = (destination / relative).resolve()
        if not target.is_relative_to(destination.resolve()):
            raise RuntimeError("Dependency archive escapes its directory")
        members.append((member, target))
    destination.mkdir(parents=True, exist_ok=False)
    for member, target in members:
        if member.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(member))
    return destination


def import_package(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "__init__.py", submodule_search_locations=[str(ROOT)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("comfy_argument", nargs="?", type=Path, help="ComfyUI folder passed by setup.bat")
    parser.add_argument("python_argument", nargs="?", help="Python selected by setup.bat")
    parser.add_argument("--comfy-root", type=Path)
    parser.add_argument("--check", action="store_true", help="Read-only environment and model check")
    parser.add_argument("--install-dependencies", action="store_true", help="Install missing X2 Stream and requirements into this dedicated Python")
    parser.add_argument("--prepare-f12", action="store_true", help="Prepare 12 FC1 shards once; stop ComfyUI first")
    parser.add_argument("--cache", default="h3_lbh_x2/f12", help="Existing cache directory, or new models/h3_lbh_x2 directory")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for this process only")
    parser.add_argument("--extra-model-paths", type=Path, help="Additional native ComfyUI model-path YAML")
    parser.add_argument("--gpu-check", action="store_true", help="Small native W4A4 offload/reload check; no video generation")
    args = parser.parse_args()
    if args.check and (args.install_dependencies or args.prepare_f12):
        parser.error("--check cannot be combined with installation or preparation")
    selected_root = args.comfy_root or args.comfy_argument or os.environ.get("H3_LBH_SETUP_ROOT")
    if not selected_root:
        parser.error("Specify the ComfyUI folder")
    comfy_root = Path(selected_root).resolve()
    if not (comfy_root / "main.py").is_file() or not (comfy_root / "comfy/sd.py").is_file():
        parser.error("--comfy-root must contain ComfyUI main.py and comfy/sd.py")
    portable = Path(sys.executable).resolve().parent.name.lower() == "python_embeded"
    if sys.prefix == sys.base_prefix and not portable:
        parser.error("Use the existing ComfyUI venv or portable Python; global/system Python is refused")
    if args.install_dependencies:
        dependency = install_dependency(comfy_root)
        for requirements in (ROOT / "requirements.txt", dependency / "requirements.txt"):
            subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "-r", str(requirements)], check=True)
    dependency = comfy_root / "custom_nodes" / DEPENDENCY_NAME
    if not (dependency / "nodes.py").is_file():
        raise RuntimeError("X2 Stream is missing; install it or use --install-dependencies")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ["PYTHONIOENCODING"] = "utf-8"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(comfy_root))
    sys.argv = [sys.argv[0], "--disable-pinned-memory"]
    import comfy.options
    comfy.options.enable_args_parsing()
    import folder_paths
    from utils.extra_config import load_extra_path_config
    config = comfy_root / "extra_model_paths.yaml"
    if config.is_file():
        load_extra_path_config(str(config))
    if args.extra_model_paths:
        load_extra_path_config(str(args.extra_model_paths.resolve()))
    module = import_package("h3_lbh_setup_package")
    cache_module = sys.modules[module.H3LBHF12Loader.__module__].cache
    environment = cache_module.check_environment()
    import nodes
    asyncio.run(nodes.init_extra_nodes(init_custom_nodes=False, init_api_nodes=False))
    if "BlockSparseAttention" not in nodes.NODE_CLASS_MAPPINGS:
        raise RuntimeError("ComfyUI lacks native BlockSparseAttention")
    found = {}
    for category, filename in MODELS.items():
        folder = "vae" if category.endswith("vae") else category
        candidates = folder_paths.get_filename_list(folder)
        matches = [filename] if filename in candidates else [name for name in candidates if Path(name).name == filename]
        if category == "latent_upscale_models" and not matches:
            alias = "minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors"
            matches = [alias] if alias in candidates else [name for name in candidates if Path(name).name == alias]
        if not matches:
            raise FileNotFoundError(f"Model missing from {folder}: {filename}; see README_ja.md")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple model matches in {folder}; select the intended file in ComfyUI: {filename}")
        found[category] = folder_paths.get_full_path_or_raise(folder, matches[0])
    cache_path = Path(args.cache)
    if not cache_path.is_absolute():
        cache_path = Path(folder_paths.models_dir) / cache_path
    cache_path = cache_path.resolve()
    if args.prepare_f12:
        if not cache_path.exists() and not cache_path.is_relative_to((Path(folder_paths.models_dir) / "h3_lbh_x2").resolve()):
            raise ValueError("New caches must be under this ComfyUI's models/h3_lbh_x2")
        cache_module.prepare(found["diffusion_models"], cache_path,
                             progress=lambda current, total: print(f"F12 {current}/{total}", flush=True))
    cache_report = "missing; the F12 Loader can prepare it on first use"
    if cache_path.exists():
        data = cache_module.manifest(cache_path, found["diffusion_models"])
        for row in data["fc1"]:
            if cache_module.sha256(cache_module.cache_file(cache_path, row)) != row["sha256"]:
                raise RuntimeError("F12 shard SHA-256 mismatch: " + row["file"])
        cache_report = "12 selected shards verified; no Gate is used"
    if args.gpu_check:
        import torch
        from comfy_kitchen.tensor.convrot_w4a4 import TensorCoreConvRotW4A4Layout, QuantizedTensor
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(43)
        weight = torch.randn(1024, 5376, device=device, dtype=torch.bfloat16, generator=generator)
        packed, params = TensorCoreConvRotW4A4Layout.quantize(weight)
        quantized = QuantizedTensor(packed, cache_module.LAYOUT, params)
        x = torch.randn(1024, 5376, device=device, dtype=torch.bfloat16, generator=generator)
        a = torch.nn.functional.linear(x, quantized)
        b = torch.nn.functional.linear(x, quantized.to("cpu").to(device))
        if not torch.equal(a, b) or not bool(torch.isfinite(a).all()):
            raise RuntimeError("Native W4A4 offload/reload check failed")
    installed = str(comfy_root / "custom_nodes" / ROOT.name) if args.check else str(install_package(comfy_root))
    if not args.check:
        for source in (ROOT / "examples/assets").glob("*"):
            if not source.is_file():
                continue
            target = Path(folder_paths.get_input_directory()) / source.name
            if target.exists() and sha(target) != sha(source):
                raise RuntimeError("Existing sample image differs; preserve it: " + str(target))
            if not target.exists():
                shutil.copy2(source, target)
    print(json.dumps({"status": "check_passed" if args.check else "installed", "python": sys.executable,
        "comfy_root": str(comfy_root), "package": installed, "environment": environment,
        "cache": str(cache_path), "cache_status": cache_report, "models": found,
        "gpu_smoke_check": bool(args.gpu_check), "video_generations": 0,
        "model_downloads": 0, "core_files_changed": False}, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
