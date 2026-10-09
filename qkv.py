# SPDX-License-Identifier: GPL-3.0-only
"""Bounded, call-local QKV reuse for the native ComfyUI MiniMax-H3 SLA path.

Native sparse attention, statistics, sinks and producer ordering are delegated
unchanged. No global/module monkeypatch, activation copy, synchronization or
precision change is made. Importing this module itself needs only the stdlib.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path


MIB = 1024 * 1024
MAX_CACHE_BYTES = 336 * MIB
CHUNK_ROWS = 4096
LOW_ROWS_THRESHOLD = 8192
QKV_WIDTH = 21504


class BoundedQKVProjection:
    """Wrap one attention call's projector, bypassing retained second-pass rows.

The native factory still iterates the original x slices. Cache hits return the
same BF16 tensor that the first producer consumed, without invoking projector.
Keys include input storage pointer and shape/stride/dtype/device metadata. The
cache has no lifetime outside this wrapper and is released after the expected
last projector invocation, with an additional mandatory caller finally release.
"""

    def __init__(self, projection, total_rows, enabled=True, cold=True,
                 max_cache_bytes=MAX_CACHE_BYTES):
        if not 0 < int(max_cache_bytes) <= MAX_CACHE_BYTES:
            raise ValueError("QKV cache must be positive and at most 336 MiB")
        if int(total_rows) <= 0:
            raise ValueError("QKV total_rows must be positive")
        self.projection = projection
        self.total_rows = int(total_rows)
        self.enabled = bool(enabled)
        self.cold = bool(cold)
        self.max_cache_bytes = int(max_cache_bytes)
        self.chunk_count = (self.total_rows + CHUNK_ROWS - 1) // CHUNK_ROWS
        self.keep_chunks = (self.chunk_count if self.total_rows <= LOW_ROWS_THRESHOLD
                            else min(2, self.chunk_count))
        self.expected_calls = self.chunk_count * (2 if self.cold else 1)
        self.cache = {}
        self.live_cache_bytes = 0
        self.peak_cache_bytes = 0
        self.cache_stores = 0
        self.cache_hits = 0
        self.hit_rows = 0
        self.projected_chunks = 0
        self.calls = 0
        self.key_misses = 0
        self.budget_skips = 0
        self.release_calls = 0
        self.released_bytes = 0
        self.released = False

    @staticmethod
    def _key(x):
        is_inference = getattr(x, "is_inference", None)
        # ComfyUI's execution context creates inference tensors, for which
        # accessing _version raises instead of returning a version counter.
        version = (None if is_inference is not None and is_inference()
                   else getattr(x, "_version", None))
        return (x.data_ptr(), tuple(x.shape), tuple(x.stride()),
                str(x.dtype), str(x.device), version)

    def __call__(self, x):
        if self.released:
            raise RuntimeError("QKV projection wrapper used after release")
        index = self.calls
        if index >= self.expected_calls:
            raise RuntimeError("Unexpected extra QKV producer pass")
        chunk_index = index % self.chunk_count
        pass_index = index // self.chunk_count
        expected_rows = min(CHUNK_ROWS, self.total_rows - chunk_index * CHUNK_ROWS)
        if tuple(x.shape) != (expected_rows, 5376):
            raise RuntimeError("QKV input slice differs from native H3 chunk contract")
        key = self._key(x)
        self.calls += 1
        entry = self.cache.get(chunk_index) if pass_index == 1 else None
        if entry is not None and entry[0] == key:
            out = entry[1]
            self.cache_hits += 1
            self.hit_rows += expected_rows
        else:
            if entry is not None:
                self.key_misses += 1
            out = self.projection(x)
            self.projected_chunks += 1
            if self.enabled and self.cold and pass_index == 0 and chunk_index < self.keep_chunks:
                if tuple(out.shape) != (expected_rows, QKV_WIDTH) or out.element_size() != 2:
                    raise RuntimeError("QKV cache requires native 2-byte [M,21504] output")
                size = int(out.numel()) * int(out.element_size())
                if self.live_cache_bytes + size <= self.max_cache_bytes:
                    self.cache[chunk_index] = (key, out, size)
                    self.cache_stores += 1
                    self.live_cache_bytes += size
                    self.peak_cache_bytes = max(self.peak_cache_bytes, self.live_cache_bytes)
                else:
                    self.budget_skips += 1
        if self.calls == self.expected_calls:
            # The current out remains alive for the native generator/producer.
            # Earlier cached chunks are no longer needed. CUDA's stream-aware
            # allocator governs their storage lifetime, just as in the baseline.
            self.release()
        return out

    def release(self):
        self.release_calls += 1
        self.released_bytes += self.live_cache_bytes
        self.cache.clear()
        self.live_cache_bytes = 0
        self.released = True

    def snapshot(self):
        return {
            "rows": self.total_rows, "cold": self.cold, "enabled": self.enabled,
            "chunk_rows": CHUNK_ROWS, "chunks_per_pass": self.chunk_count,
            "expected_projector_invocations": self.expected_calls,
            "projector_invocations": self.calls, "projected_chunks": self.projected_chunks,
            "cache_stores": self.cache_stores, "cache_hits": self.cache_hits,
            "hit_rows": self.hit_rows, "key_misses": self.key_misses,
            "budget_skips": self.budget_skips, "max_cache_bytes": self.max_cache_bytes,
            "peak_cache_bytes": self.peak_cache_bytes,
            "live_cache_bytes": self.live_cache_bytes, "released_bytes": self.released_bytes,
            "release_calls": self.release_calls, "released": self.released,
        }


class _AttentionProxy:
    def __init__(self, attention, projection):
        self._attention = attention
        self.qkv_proj = projection

    def __getattr__(self, name):
        return getattr(self._attention, name)




def _closure(function):
    return inspect.getclosurevars(function).nonlocals


def _native_owner(function, expected_source):
    """Resolve the exact module instance that created Comfy's block closure.

Comfy loads extra nodes under an absolute-path sys.modules key. Importing the
same file by its package name creates a second class/code identity, so it must
not be used to recognize or rebuild the already installed SLA patch.
"""
    if not inspect.isfunction(function):
        raise ValueError("QKV reuse must follow native BlockSparseAttention")
    owner = sys.modules.get(function.__module__)
    if owner is None or owner.__dict__ is not function.__globals__:
        raise ValueError("Sparse-attention closure has no registered native owner")
    source = getattr(owner, "__file__", None)
    if source is None or Path(source).resolve() != Path(expected_source).resolve():
        raise ValueError("Sparse-attention closure does not come from the installed native source")
    for name in ("SparseAttnPatch", "make_h3_block_patch", "make_attention_override",
                 "install_override", "h3_eligible", "h3_sparse_attention"):
        if not hasattr(owner, name):
            raise ValueError("Native sparse-attention module is missing its required interface")
    return owner


def _make_block_patch(native, block, block_index, patch, audit):
    def attention(h, rope_freqs=None, transformer_options=None):
        options = {} if transformer_options is None else transformer_options
        n = int(h.shape[0])
        key = (block_index, n, tuple(options.get("uuids", ())))
        cold = patch.pooled.get(key) is None
        if audit.active_calls:
            raise RuntimeError("QKV bounded reuse requires serial H3 attention calls")
        projection = BoundedQKVProjection(block.attn.qkv_proj, n, audit.enabled,
                                          cold, audit.cache_bytes)
        audit.active_calls += 1
        try:
            result = native.h3_sparse_attention(
                _AttentionProxy(block.attn, projection), h, rope_freqs,
                options, patch, block_index)
            return result
        finally:
            projection.release()
            audit.active_calls -= 1

    def block_patch(args, extra):
        if native.h3_eligible(block.attn, args["img"], args["rope_freqs"],
                              args["transformer_options"], patch, block_index):
            args = {**args, "attention": attention}
        return extra["original_block"](args)

    return block_patch


class H3LBHQKVReuse:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "enabled": ("BOOLEAN", {"default": True}),
            "cache_mib": ("INT", {"default": 336, "min": 1, "max": 336}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "H3 LBH X2 Stream"

    def patch(self, model, enabled, cache_mib):
        import comfy.patcher_extension

        cache_bytes = int(cache_mib) * MIB
        if not 0 < cache_bytes <= MAX_CACHE_BYTES:
            raise ValueError("QKV cache is limited to 336 MiB")
        blocks = model.get_model_object("diffusion_model.blocks")
        if len(blocks) != 50:
            raise ValueError("QKV reuse requires the 50-block MiniMax-H3 DiT")
        options = model.model_options["transformer_options"]
        replacements = options.get("patches_replace", {}).get("dit", {})
        native_source = (Path(comfy.patcher_extension.__file__).resolve().parent.parent
                         / "comfy_extras" / "nodes_sparse_attention.py")
        native = _native_owner(replacements.get(("double_block", 0)), native_source)
        source_patch = None
        for index, block in enumerate(blocks):
            qkv = block.attn.qkv_proj
            if (getattr(qkv, "_forward_hooks", None)
                    or getattr(qkv, "_forward_pre_hooks", None)
                    or getattr(qkv, "weight_function", None)
                    or getattr(qkv, "bias_function", None)):
                raise ValueError("QKV reuse cannot bypass projection hooks or dynamic weight functions")
            existing = replacements.get(("double_block", index))
            if _native_owner(existing, native_source) is not native:
                raise ValueError("Expected one native module owner for all 50 SLA block closures")
            captured = _closure(existing)
            current = captured.get("patch")
            if not isinstance(current, native.SparseAttnPatch):
                raise ValueError("Unrecognized sparse-attention block patch")
            expected = native.make_h3_block_patch(block, index, current)
            if (existing.__code__ is not expected.__code__ or captured.get("block") is not block
                    or captured.get("block_index") != index):
                raise ValueError("QKV reuse cannot replace additional/foreign block wrappers")
            if source_patch is None:
                source_patch = current
            elif current is not source_patch:
                raise ValueError("Expected one shared native SLA patch for all 50 blocks")
        if (source_patch.vsa or abs(source_patch.topk_ratio - 0.05) > 1e-12
                or source_patch.sink_conditioning != "exact_kv_and_rows"
                or source_patch.min_tokens not in (4096, 12288) or source_patch.extra_tokens != 256
                or source_patch.dense_blocks):
            raise ValueError("QKV reuse requires SLA5 with exact audio rows, min_tokens4096/12288 and extra_tokens256")
        override = options.get("optimized_attention_override")
        expected_override = native.make_attention_override(source_patch, None)
        if not inspect.isfunction(override) or override.__code__ is not expected_override.__code__:
            raise ValueError("QKV reuse requires native SLA as the current attention override")
        captured_override = _closure(override)
        if captured_override.get("patch") is not source_patch:
            raise ValueError("Native block and attention override patches differ")

        patched = model.clone()
        patched.model_options = patched.model_options.copy()
        transformer_options = patched.model_options["transformer_options"].copy()
        patched.model_options["transformer_options"] = transformer_options
        transformer_options["optimized_attention_override"] = captured_override["previous"]
        patch = native.SparseAttnPatch(
            tau=source_patch.tau, topk_ratio=source_patch.topk_ratio, vsa=False,
            sigma_start=source_patch.sigma_start, sigma_end=source_patch.sigma_end,
            min_tokens=source_patch.min_tokens, dense_blocks=set(source_patch.dense_blocks),
            sink_conditioning=source_patch.sink_conditioning,
            extra_tokens=source_patch.extra_tokens, verbose=source_patch.verbose)
        audit = _CacheState(bool(enabled), cache_bytes)
        native.install_override(patch, transformer_options)
        callbacks = comfy.patcher_extension.CallbacksMP
        patched.remove_callbacks_with_key(callbacks.ON_PREPARE_STATE, "block_sparse_attention")
        patched.remove_callbacks_with_key(callbacks.ON_CLEANUP, "block_sparse_attention")
        patched.add_callback_with_key(
            callbacks.ON_PREPARE_STATE, "block_sparse_attention",
            lambda model_patcher, timestep, model_options: native.install_override(
                patch, model_options["transformer_options"]))

        def cleanup(model_patcher):
            patch.reset()

        patched.add_callback_with_key(callbacks.ON_CLEANUP, "block_sparse_attention", cleanup)
        for index, block in enumerate(blocks):
            patched.set_model_patch_replace(
                _make_block_patch(native, block, index, patch, audit), "dit", "double_block", index)
        return (patched,)


class _CacheState:
    def __init__(self, enabled, cache_bytes):
        self.enabled = enabled
        self.cache_bytes = cache_bytes
        self.active_calls = 0


NODE_CLASS_MAPPINGS = {"H3LBHQKVReuse": H3LBHQKVReuse}
NODE_DISPLAY_NAME_MAPPINGS = {"H3LBHQKVReuse": "H3 LBH QKV Reuse (336 MiB)"}
