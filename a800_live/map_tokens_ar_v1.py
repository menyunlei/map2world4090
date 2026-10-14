"""Static map-token conditioning for the official DreamX causal AR model.

Inference-only port of map2world4090's post-text-projection token prefix.
This module performs no launches and changes no model weights. The official AR
pipeline is positive-only: its configuration's guidance_scale does not execute
classifier-free guidance and the previous negative prompt is not consumed.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor):
    raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


@torch.no_grad()
def encode_map_tokens_ar(vae_wrapper, patch_embedding, map_path,
                         target_size=(640, 384), map_strength=0.6):
    """Return [1,16,3072] BF16 tokens and immutable input/encoding metadata.

    target_size is (width,height), matching the original PIL LANCZOS map path.
    The native wrapper returns normalized [B,T,C,H,W] floating-point mu; it is
    rearranged and cast before LayerNorm to preserve the original algorithm.
    No second mean/std normalization or posterior sampling is performed.
    """
    if float(map_strength) != 0.6:
        raise ValueError("This fixed port preserves map_strength=0.6; tuning is not enabled")
    width, height = map(int, target_size)
    if width <= 0 or height <= 0 or width % 32 or height % 32:
        raise ValueError("Map width and height must be positive multiples of32")
    weight = patch_embedding.weight
    if weight.dtype != torch.bfloat16:
        raise ValueError("The fixed map port requires native BF16 patch_embedding weights")
    if tuple(weight.shape[:2]) != (3072, 48):
        raise ValueError(f"Unexpected patch_embedding channels: {tuple(weight.shape)}")
    if tuple(patch_embedding.kernel_size) != (1, 2, 2) or tuple(patch_embedding.stride) != (1, 2, 2):
        raise ValueError("The fixed token port requires native patch_size/stride(1,2,2)")
    pixels = np.asarray(
        Image.open(map_path).convert("RGB").resize((width, height), Image.Resampling.LANCZOS),
        dtype=np.float32,
    ) / 255.0
    pixels = torch.from_numpy(2.0 * pixels - 1.0).permute(2, 0, 1)[None, :, None]
    pixels = pixels.to(device=weight.device, dtype=torch.bfloat16)
    # The native encoder clears its own encoder cache. Clear wrapper/decode state
    # at this isolated setup boundary, never in the middle of a streamed route.
    vae_wrapper.model.clear_cache()
    latent_btchw = vae_wrapper.encode_to_latent(pixels)
    latent = latent_btchw.permute(0, 2, 1, 3, 4).contiguous().to(torch.bfloat16)
    expected_shape = (1, 48, 1, height // 16, width // 16)
    if tuple(latent.shape) != expected_shape or not bool(torch.isfinite(latent).all()):
        raise ValueError(f"Invalid native normalized map latent: {tuple(latent.shape)}")
    latent_hash = tensor_sha256(latent)
    normalized = F.layer_norm(latent, latent.shape[1:])
    pooled = F.adaptive_avg_pool3d(normalized, (1, 8, 8))
    tokens = patch_embedding(pooled).flatten(2).transpose(1, 2).contiguous() * float(map_strength)
    if tuple(tokens.shape) != (1, 16, 3072) or tokens.dtype != torch.bfloat16:
        raise ValueError(f"Invalid projected map token contract: {tuple(tokens.shape)}, {tokens.dtype}")
    if not bool(torch.isfinite(tokens).all()):
        raise ValueError("Map tokens contain nonfinite values")
    vae_wrapper.model.clear_cache()
    metadata = {
        "method": "native normalized map mu -> BF16 LayerNorm -> pool1x8x8 -> loaded AR patch_embedding ->16tokens ->0.6",
        "map_source": str(Path(map_path).resolve()),
        "map_image_sha256": file_sha256(map_path),
        "resize_width_height": [width, height],
        "resize_filter": "PIL LANCZOS",
        "input_pixel_shape": list(pixels.shape),
        "input_pixel_sha256": tensor_sha256(pixels),
        "native_wrapper_latent_layout": "BTCHW",
        "normalized_mu_layout_before_layernorm": "BCTHW",
        "normalized_mu_shape": list(latent.shape),
        "normalized_mu_sha256": latent_hash,
        "layernorm_shape": list(normalized.shape),
        "layernorm_sha256": tensor_sha256(normalized),
        "pool_shape": list(pooled.shape),
        "pool_sha256": tensor_sha256(pooled),
        "tokens_shape": list(tokens.shape),
        "tokens_sha256": tensor_sha256(tokens),
        "precision": "bfloat16",
        "map_strength": float(map_strength),
        "token_count": 16,
        "patch_embedding_source": "loaded official autoregressive checkpoint; token values need not equal5B-Cam",
        "static_map_for_all_chunks": True,
        "new_training_or_weight_changes": False,
        "official_ar_cfg_executed": False,
        "official_ar_negative_prompt_consumed": False,
    }
    return tokens.detach(), metadata


class StaticMapTokenHooks:
    """Add the map prefix after text_embedding, retaining native cross-KV reuse.

    First map-conditioned invocation verifies all cached prefixes by rerunning
    each block's unchanged K/V projection on the SAME combined context shape.
    This avoids comparing different GEMM shapes and imposes setup-only overhead.
    audit() records that overhead separately from the conditioning contract.
    """
    def __init__(self, model, tokens, metadata=None, expected_dim=3072,
                 expected_heads=24, expected_blocks=30, text_len=512):
        if tuple(tokens.shape[:2]) != (1, 16) or tokens.shape[2] != expected_dim:
            raise ValueError("Map hook token shape contract failed")
        if len(model.blocks) != expected_blocks:
            raise ValueError("Map hook transformer block count mismatch")
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("Nonfinite map tokens")
        self.model = model
        self.tokens = tokens.detach()
        self.metadata = dict(metadata or {})
        self.expected_dim = int(expected_dim)
        self.expected_heads = int(expected_heads)
        self.expected_blocks = int(expected_blocks)
        self.text_len = int(text_len)
        self.enabled = True
        self.handles = []
        self.seen_caches = {}
        self.pending_contexts = {}
        self.records = {str(i): {"calls": [], "cache_prefix_verified": False} for i in range(expected_blocks)}
        for index, block in enumerate(model.blocks):
            attention = block.cross_attn
            if attention.num_heads != expected_heads or attention.dim != expected_dim:
                raise ValueError(f"Cross-attention contract mismatch in block{index}")
            self.handles.append(attention.register_forward_pre_hook(self._pre_hook(index), with_kwargs=True))
            self.handles.append(attention.register_forward_hook(self._post_hook(index), with_kwargs=True))

    @staticmethod
    def _argument(args, kwargs, name, position, default=None):
        return args[position] if len(args) > position else kwargs.get(name, default)

    def _pre_hook(self, index):
        def hook(module, args, kwargs):
            context = self._argument(args, kwargs, "context", 1)
            lengths = self._argument(args, kwargs, "context_lens", 2)
            cache = self._argument(args, kwargs, "crossattn_cache", 3)
            if context is None or context.ndim != 3:
                raise ValueError("Native projected text context must have shape[B,L,D]")
            if tuple(context.shape[1:]) != (self.text_len, self.expected_dim) or lengths is not None:
                raise ValueError("Expected native padded512-token projected text context and context_lens=None")
            if cache is not None:
                self.seen_caches[index] = cache
            record = self.records[str(index)]
            record["calls"].append({"map_enabled": self.enabled, "native_text_tokens": self.text_len,
                                    "condition_tokens": self.text_len + 16 if self.enabled else self.text_len,
                                    "cache_was_initialized": bool(cache.get("is_init", False)) if cache else False})
            if not self.enabled:
                return args, kwargs
            prefix = self.tokens.to(device=context.device, dtype=context.dtype).expand(context.shape[0], -1, -1)
            combined = torch.cat([prefix, context], dim=1)
            if combined.shape[1] != self.text_len + 16:
                raise ValueError("Map prefix must produce528 context tokens")
            if not record["cache_prefix_verified"]:
                if cache is not None and cache.get("is_init", False):
                    raise RuntimeError("Stale cross-attention cache: reset is_init before installing/changing the map")
                if not torch.equal(combined[:, :16], prefix) or not torch.equal(combined[:, 16:], context):
                    raise AssertionError("Map prefix or original projected text tail was changed")
                if cache is not None:
                    self.pending_contexts[index] = combined.detach()
            args = list(args)
            kwargs = dict(kwargs)
            if len(args) > 1:
                args[1] = combined
            else:
                kwargs["context"] = combined
            return tuple(args), kwargs
        return hook

    def _post_hook(self, index):
        @torch.no_grad()
        def hook(module, args, kwargs, output):
            context = self.pending_contexts.pop(index, None)
            if context is None:
                return output
            cache = self._argument(args, kwargs, "crossattn_cache", 3)
            if cache is None or not cache.get("is_init", False):
                raise AssertionError("Native cross-KV cache did not initialize")
            batch = context.shape[0]
            heads, dim = module.num_heads, module.head_dim
            expected_k = module.norm_k(module.k(context)).view(batch, -1, heads, dim)
            expected_v = module.v(context).view(batch, -1, heads, dim)
            if tuple(cache["k"].shape) != (batch, self.text_len + 16, heads, dim):
                raise AssertionError("Native cache did not retain all528 map+text tokens")
            k_error = float((cache["k"][:, :16].float() - expected_k[:, :16].float()).abs().max())
            v_error = float((cache["v"][:, :16].float() - expected_v[:, :16].float()).abs().max())
            if not torch.equal(cache["k"][:, :16], expected_k[:, :16]) or not torch.equal(cache["v"][:, :16], expected_v[:, :16]):
                raise AssertionError(f"Cross-KV map prefix differs from native projections: K{k_error}, V{v_error}")
            if not torch.equal(cache["k"][:, 16:], expected_k[:, 16:]) or not torch.equal(cache["v"][:, 16:], expected_v[:, 16:]):
                raise AssertionError("Cached projected text tail changed")
            self.records[str(index)].update({"cache_prefix_verified": True, "cached_shape": list(cache["k"].shape),
                                             "prefix_k_max_abs": k_error, "prefix_v_max_abs": v_error,
                                             "text_tail_kv_verified": True})
            return output
        return hook

    def reset_cross_attention(self, caches=None):
        """Invalidate text/map K/V only; caller controls video/PRoPE route history."""
        caches = list(self.seen_caches.values()) if caches is None else list(caches)
        for cache in caches:
            cache["is_init"] = False

    def set_enabled(self, enabled, caches=None):
        if bool(enabled) != self.enabled:
            self.reset_cross_attention(caches)
            self.enabled = bool(enabled)

    def remove(self):
        self.reset_cross_attention()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def audit(self):
        return {
            "map_metadata": self.metadata,
            "injection_position": "all blocks cross_attn input, AFTER native4096->3072 text_embedding",
            "native_text_context_length": self.text_len,
            "map_conditioned_context_length": self.text_len + 16,
            "expected_blocks": self.expected_blocks,
            "expected_heads": self.expected_heads,
            "all_block_cache_prefixes_verified": all(r["cache_prefix_verified"] for r in self.records.values()),
            "first_map_call_projection_verification_overhead": True,
            "blocks": self.records,
            "new_trainable_parameters": 0,
            "official_ar_cfg_executed": False,
            "official_ar_negative_prompt_consumed": False,
        }

    def save_audit(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".partial")
        temporary.write_text(json.dumps(self.audit(), indent=2))
        temporary.replace(path)


def install_map_token_hooks(model, tokens, metadata=None):
    return StaticMapTokenHooks(model, tokens, metadata)


@torch.no_grad()
def unit_test_native_cross_attention_cpu(native_cross_attention_class):
    """Isolated synthetic CPU test using the actual native attention class.

    CPU SDPA substitutes only the CUDA FlashAttention kernel during this test
    and is restored in finally. Native Q/K/V/norm/output/cache code is unchanged.
    Run before inference or in a separate process; this is not a GPU benchmark.
    """
    device = torch.device("cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(739241)
        attention = native_cross_attention_class(96, 3, (-1, -1), True, 1e-6).to(device).eval()
        context = torch.randn(1, 512, 96, device=device)
        query = torch.randn(1, 2, 96, device=device)
        tokens = torch.randn(1, 16, 96, device=device) * 0.6
        namespace = SimpleNamespace(blocks=[SimpleNamespace(cross_attn=attention)])
        globals_ = native_cross_attention_class.forward.__globals__
        kernel = globals_["flash_attention"]

        def cpu_attention(q, k, v, k_lens=None, **kwargs):
            if k_lens is not None:
                raise AssertionError("The native test expects no context-length truncation")
            return F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), dropout_p=0.0).transpose(1, 2)

        globals_["flash_attention"] = cpu_attention
        hooks = None
        try:
            unconditioned = attention(query, context, None, crossattn_cache=None)
            weight_hashes_before = {k: tensor_sha256(v) for k, v in attention.state_dict().items()}
            hooks = StaticMapTokenHooks(namespace, tokens, expected_dim=96, expected_heads=3, expected_blocks=1)
            cache = {"k": torch.zeros(1, 512, 3, 32), "v": torch.zeros(1, 512, 3, 32), "is_init": False}
            conditioned = attention(query, context, None, crossattn_cache=cache)
            conditioned_cached = attention(query, context, None, crossattn_cache=cache)
            if not torch.equal(conditioned, conditioned_cached):
                raise AssertionError("Static map/text cached output differs from initialized output")
            if torch.equal(conditioned, unconditioned):
                raise AssertionError("Map prefix has no effect in the synthetic native attention test")
            hooks.set_enabled(False)
            disabled = attention(query, context, None, crossattn_cache=cache)
            if not torch.equal(disabled, unconditioned) or cache["k"].shape[1] != 512:
                raise AssertionError("Disabled hook did not restore the unchanged native text context/output")
            hooks.set_enabled(True)
            reenabled = attention(query, context, None, crossattn_cache=cache)
            if not torch.equal(reenabled, conditioned) or cache["k"].shape[1] != 528:
                raise AssertionError("Map reenable/cache invalidation changed the fixed conditioned output")
            if weight_hashes_before != {k: tensor_sha256(v) for k, v in attention.state_dict().items()}:
                raise AssertionError("Map hooks changed native model weights")
            audit = hooks.audit()
            return {"status": "PASSED", "test": "actual native WanCrossAttention, synthetic CPU96dim3heads512text16map",
                    "cpu_kernel_only": "SDPA substituted for CUDA FlashAttention; restored after test",
                    "cached_output_exact": True, "disabled_output_exact": True, "reenabled_output_exact": True,
                    "model_weights_unchanged": True, "map_context_has_effect": True, "hook_audit": audit,
                    "real_checkpoint_compatibility_verified_by_this_cpu_test": False}
        finally:
            if hooks is not None:
                hooks.remove()
            globals_["flash_attention"] = kernel
