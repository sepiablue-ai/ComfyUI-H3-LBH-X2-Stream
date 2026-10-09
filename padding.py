"""Task-local H3 model patch for odd spatial condition latents."""

from comfy.ldm.common_dit import pad_to_patch_size


class _PaddedVisualConditionRows:
    def __init__(self, original, patch_size):
        self.original = original
        self.patch_size = patch_size

    def __call__(self, payload, device):
        padded_payload = payload.copy()
        padded_payload["cond_video_latents"] = [
            pad_to_patch_size(latent, self.patch_size)
            for latent in payload.get("cond_video_latents", [])
        ]
        return self.original(padded_payload, device)


class H3LBHPadConditions:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "H3 LBH X2 Stream"

    def patch(self, model):
        diffusion_model = model.get_model_object("diffusion_model")
        original = model.get_model_object("diffusion_model._cond_video_rows")
        patched = model.clone()
        patched.add_object_patch(
            "diffusion_model._cond_video_rows",
            _PaddedVisualConditionRows(original, diffusion_model.patch_size),
        )
        return (patched,)


NODE_CLASS_MAPPINGS = {"H3LBHPadConditions": H3LBHPadConditions}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3LBHPadConditions": "H3 LBH Pad Visual Conditions"
}
