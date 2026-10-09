# SPDX-License-Identifier: GPL-3.0-only
"""Keep the measured native F12 overlays and ComfyUI's dynamic model loading."""
import logging
from pathlib import Path

import folder_paths
import comfy.model_management as mm
import comfy.sd
import comfy.utils

from . import cache

log = logging.getLogger(__name__)


def load_variant(source, directory, model_options=None):
    data = cache.manifest(directory, source)
    state, metadata = comfy.utils.load_torch_file(str(source), return_metadata=True)
    for row in data["fc1"]:
        shard = comfy.utils.load_torch_file(str(cache.cache_file(directory, row)))
        cache.validate_fc1_state(shard, row["block"])
        state.update(shard)
    model = comfy.sd.load_diffusion_model_state_dict(state, model_options=model_options or {}, metadata=metadata)
    cache.require(model is not None, "H3 F12 model loading failed")
    blocks = model.get_model_object("diffusion_model").blocks
    cache.require(len(blocks) == 50, "Expected the 50-block MiniMax H3 DiT")
    for index, block in enumerate(blocks):
        weight = block.mlp.fc1.weight
        expected = cache.LAYOUT if index in cache.BLOCKS else "TensorWiseINT8Layout"
        cache.require(weight._layout_cls == expected and weight.device.type == "cpu", "Incorrect F12 layout or backing device")
        if index in cache.BLOCKS:
            params = weight._params
            cache.require(params.convrot_groupsize == 256 and params.quant_group_size == 64
                          and params.linear_dtype == "int4" and not params.transposed, "Incorrect native F12 parameters")
        for layer in (block.mlp.fc2, block.attn.qkv_proj, block.attn.out_proj):
            cache.require(layer.weight._layout_cls == "TensorWiseINT8Layout", "FC2/QKV/OutProj must remain INT8")
    model.cached_patcher_init = (load_variant, (str(source), str(directory), model_options))
    log.info("[H3 LBH F12] Loaded native W4A4 FC1 in blocks %s", cache.BLOCKS)
    return model


class H3LBHF12Loader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "unet_name": (folder_paths.get_filename_list("diffusion_models"),),
            "cache_directory": ("STRING", {"default": "h3_lbh_x2/f12"}),
            "prepare_if_missing": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load"
    CATEGORY = "H3 LBH X2 Stream"
    DESCRIPTION = "Use native W4A4 only in the validated 12 FC1 blocks. First-time preparation writes 12 local shards; no downloads."

    def load(self, unet_name, cache_directory, prepare_if_missing):
        cache.check_environment()
        source = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        directory = Path(cache_directory)
        if not directory.is_absolute():
            directory = Path(folder_paths.models_dir) / directory
        directory = directory.resolve()
        if not directory.exists():
            if not prepare_if_missing:
                raise FileNotFoundError("F12 cache missing; enable preparation or run setup.bat with --prepare-f12")
            owned = (Path(folder_paths.models_dir) / "h3_lbh_x2").resolve()
            if not directory.is_relative_to(owned):
                raise ValueError("New F12 caches must be inside this ComfyUI's models/h3_lbh_x2")
            mm.unload_all_models()
            mm.soft_empty_cache()
            progress = comfy.utils.ProgressBar(len(cache.BLOCKS))
            cache.prepare(source, directory, progress=progress.update_absolute)
            mm.soft_empty_cache()
        return (load_variant(source, directory),)
