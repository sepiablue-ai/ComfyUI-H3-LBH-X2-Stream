"""The validated automatic 1/8 H3 FFN chunk policy.

FFN chunking adapts ComfyUI-H3-X2-Stream/ffn.py and kijai/ComfyUI-KJNodes
MiniMaxChunkFeedForward (GPL-3.0), retaining the native ComfyUI SwiGLU op.
"""

from types import MethodType

import torch
import comfy.ops






class H3LBHAutoFFN:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "low_chunks": ("INT", {"default": 1, "min": 1, "max": 64}),
            "high_chunks": ("INT", {"default": 8, "min": 1, "max": 64}),
            "split_rows": ("INT", {"default": 8192, "min": 256, "max": 262144, "step": 256}),
            "seq_threshold": ("INT", {"default": 4096, "min": 256, "max": 262144, "step": 256}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "H3 LBH X2 Stream"

    def patch(self, model, low_chunks, high_chunks, split_rows, seq_threshold):
        patched = model.clone()
        blocks = patched.get_model_object("diffusion_model.blocks")
        if not blocks or any(not hasattr(block, "mlp") or not hasattr(block.mlp, "fc2") for block in blocks):
            raise ValueError("H3 Resolution Chunk FFN requires a MiniMax H3 diffusion model.")

        def forward(module, x):
            rows = x.shape[0]
            chunks = low_chunks if rows <= split_rows else high_chunks
            if rows <= seq_threshold or chunks == 1:
                return comfy.ops.linear_input_act(module.fc2, module.fc1(x), "swiglu")
            parts = torch.chunk(x, chunks, dim=0)
            result = torch.empty_like(x)
            offset = 0
            for part in parts:
                result[offset:offset + len(part)] = comfy.ops.linear_input_act(module.fc2, module.fc1(part), "swiglu")
                offset += len(part)
            return result

        for index, block in enumerate(blocks):
            patched.add_object_patch(
                f"diffusion_model.blocks.{index}.mlp.forward",
                MethodType(forward, block.mlp),
            )
        return (patched,)




NODE_CLASS_MAPPINGS = {"H3LBHAutoFFN": H3LBHAutoFFN}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LBHAutoFFN": "H3 LBH Automatic FFN Chunks"}
