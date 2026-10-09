# SPDX-License-Identifier: GPL-3.0-only
"""MiniMax H3: F12, bounded QKV reuse, LBH 3+1 and X2 streaming output."""
from .loader import H3LBHF12Loader
from .conditioning import H3LBHImageCondition
from .ffn import H3LBHAutoFFN
from .qkv import H3LBHQKVReuse
from .padding import H3LBHPadConditions
from .handoff import H3LBHVideoHandoff

NODE_CLASS_MAPPINGS = {
    "H3LBHF12Loader": H3LBHF12Loader,
    "H3LBHImageCondition": H3LBHImageCondition,
    "H3LBHAutoFFN": H3LBHAutoFFN,
    "H3LBHQKVReuse": H3LBHQKVReuse,
    "H3LBHPadConditions": H3LBHPadConditions,
    "H3LBHVideoHandoff": H3LBHVideoHandoff,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LBHF12Loader": "H3 LBH F12 Model Loader",
    "H3LBHImageCondition": "H3 LBH Shared Image Conditioning",
    "H3LBHAutoFFN": "H3 LBH Automatic FFN Chunks",
    "H3LBHQKVReuse": "H3 LBH QKV Reuse (336 MiB)",
    "H3LBHPadConditions": "H3 LBH Pad Visual Conditions",
    "H3LBHVideoHandoff": "H3 LBH Video Upscale + Continue",
}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
