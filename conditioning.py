# SPDX-License-Identifier: GPL-3.0-only
"""The measured shared high-image TE and two upfront VAE keyframes."""
import numpy as np
from PIL import Image, ImageOps
import torch

import node_helpers
from comfy_extras.nodes_minimax_h3 import _empty_av_latent


def fit_image(image, width, height):
    pixels = (image[0, ..., :3].detach().cpu().numpy() * 255).round().clip(0, 255).astype(np.uint8)
    fitted = ImageOps.fit(Image.fromarray(pixels), (width, height), Image.Resampling.LANCZOS, centering=(0.5, 0.5))
    return torch.from_numpy(np.asarray(fitted).copy()).to(dtype=torch.float32).div_(255).unsqueeze(0)


def keyframe_condition(positive, latent):
    conditioned = node_helpers.conditioning_set_values(
        positive, {"minimax_keyframes": [{"resolved_frame_index": 0, "latent": latent}]})
    for _, metadata in conditioned:
        metadata["minimax_token_tags"] = metadata["minimax_token_tags"].clone()
    return conditioned


class H3LBHImageCondition:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip": ("CLIP",), "vae": ("VAE",), "first_frame": ("IMAGE",),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True}),
            "low_width": ("INT", {"default": 384, "min": 32, "max": 8192, "step": 32}),
            "low_height": ("INT", {"default": 672, "min": 32, "max": 8192, "step": 32}),
            "high_width": ("INT", {"default": 544, "min": 32, "max": 8192, "step": 32}),
            "high_height": ("INT", {"default": 960, "min": 32, "max": 8192, "step": 32}),
            "length": ("INT", {"default": 243, "min": 5, "max": 3600, "step": 17}),
        }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("low_positive", "high_positive", "low_latent")
    FUNCTION = "encode"
    CATEGORY = "H3 LBH X2 Stream"

    def encode(self, clip, vae, first_frame, prompt, low_width, low_height, high_width, high_height, length):
        high_image = fit_image(first_frame, high_width, high_height)
        low_image = fit_image(first_frame, low_width, low_height)
        tokens = clip.tokenize(prompt, images=[high_image])
        positive = clip.encode_from_tokens_scheduled(tokens)
        high_keyframe = vae.encode(high_image)
        low_keyframe = vae.encode(low_image)
        low_positive = keyframe_condition(positive, low_keyframe)
        high_positive = keyframe_condition(positive, high_keyframe)
        low_latent, _ = _empty_av_latent(low_width, low_height, length)
        return low_positive, high_positive, low_latent
