# SPDX-License-Identifier: GPL-3.0-only
"""The validated LBH upscale and CONST continuation; audio continues natively."""
import gc

import torch

import comfy.model_management as mm
from comfy.nested_tensor import NestedTensor
import folder_paths

from .vendor import lbh_3d as lbh


def renoise_video(model, noise, sigmas, latent):
    video, audio = latent["samples"].unbind()
    mask = NestedTensor([
        torch.ones((video.shape[0], 1, *video.shape[2:]), device=video.device, dtype=torch.float32),
        torch.zeros((audio.shape[0], 1, *audio.shape[2:]), device=audio.device, dtype=torch.float32),
    ])
    masked = {**latent, "noise_mask": mask}
    generated = list(noise.generate_noise(masked).unbind())
    generated[1] = torch.zeros_like(generated[1])
    generated = [member * member_mask.to(device=member.device, dtype=member.dtype).expand_as(member)
                 for member, member_mask in zip(generated, mask.unbind())]
    samples = latent["samples"]
    if any(torch.count_nonzero(member) > 0 for member in samples.unbind()):
        samples = model.get_model_object("process_latent_in")(samples)
    sampling = model.get_model_object("model_sampling")
    mixed = []
    for member, member_noise in zip(samples.unbind(), generated):
        current = sampling.noise_scaling(sigmas[0], member_noise, member)
        current = sampling.inverse_noise_scaling(sigmas[0], current)
        mixed.append(current)
    samples = model.get_model_object("process_latent_out")(NestedTensor(mixed))
    samples = NestedTensor([torch.nan_to_num(member, nan=0.0, posinf=0.0, neginf=0.0)
                            for member in samples.unbind()])
    result = {**masked, "samples": samples.cpu(), "noise_mask": mask.cpu()}
    gc.collect()
    mm.soft_empty_cache()
    return result


class H3LBHVideoHandoff:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "samples": ("LATENT",), "model": ("MODEL",), "noise": ("NOISE",), "sigmas": ("SIGMAS",),
            "upscaler_name": (folder_paths.get_filename_list("latent_upscale_models"),),
            "width": ("INT", {"default": 544, "min": 32, "max": 8192, "step": 32}),
            "height": ("INT", {"default": 960, "min": 32, "max": 8192, "step": 32}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "upscale"
    CATEGORY = "H3 LBH X2 Stream"
    DESCRIPTION = "Upscale the clean low-pass video with LBH, then prepare the remaining CONST sigmas. Connect the low sampler's intermediate audio through native Separate/Concat AV."

    def upscale(self, samples, model, noise, sigmas, upscaler_name, width, height):
        if len(sigmas) < 2 or sigmas.device.type != "cpu":
            raise ValueError("Connect the remaining original CPU sigma schedule from SplitSigmas")
        video, audio = samples["samples"].unbind()
        path = folder_paths.get_full_path_or_raise("latent_upscale_models", upscaler_name)
        mm.free_memory(4 * 1024**3, lbh._resolve_device("cuda"))
        resized = lbh.MinimaxH3LatentUpscaler3D.execute(
            latent={"samples": video}, model_name=path,
            mode={"mode": lbh.UpscaleMode.TARGET_DIMENSIONS, "width": width, "height": height},
            align=32, enable_chunking=False, device="cuda", precision="fp16",
        )[0]["samples"]
        if tuple(resized.shape[-2:]) != (height // 16, width // 16):
            raise ValueError("LBH target dimensions must align to the H3 32px grid")
        upscaled = samples.copy()
        upscaled.pop("noise_mask", None)
        upscaled["samples"] = NestedTensor([resized, audio])
        return (renoise_video(model, noise, sigmas, upscaled),)
