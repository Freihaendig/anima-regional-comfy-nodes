import copy
import logging
from typing import Any

import torch

try:
    import comfy.hooks as comfy_hooks
except Exception:  # pragma: no cover - compatibility path for older ComfyUI builds
    comfy_hooks = None

from .regional_core import (
    build_bias_template,
    build_cond_uncond_gated_bias,
    discover_patch_size,
    mask_to_token_weights,
    normalize_region_weights,
)

LOGGER = logging.getLogger(__name__)


def _validate_single_conditioning(name: str, conditioning: Any) -> tuple[torch.Tensor, dict]:
    if conditioning is None:
        raise ValueError(f"Conditioning '{name}' is required")
    if not isinstance(conditioning, list) or len(conditioning) != 1:
        raise ValueError(
            f"Conditioning '{name}' must contain exactly one entry for MVP. "
            f"Got {type(conditioning)} with length {len(conditioning) if isinstance(conditioning, list) else 'N/A'}"
        )

    entry = conditioning[0]
    if not isinstance(entry, (list, tuple)) or len(entry) != 2:
        raise ValueError(f"Conditioning '{name}' has an unexpected structure")

    cross_attn, options = entry
    if not torch.is_tensor(cross_attn):
        raise ValueError(f"Conditioning '{name}' cross-attn must be a tensor")
    if cross_attn.ndim != 3:
        raise ValueError(f"Conditioning '{name}' cross-attn must have shape [B, T, D], got {tuple(cross_attn.shape)}")
    if not isinstance(options, dict):
        raise ValueError(f"Conditioning '{name}' options must be a dict")

    return cross_attn, options


def _flatten_token_tensor(value: Any) -> torch.Tensor | None:
    if not torch.is_tensor(value):
        return None
    if value.ndim == 1:
        return value
    if value.ndim == 2 and value.shape[0] == 1:
        return value.reshape(-1)
    return None


def _segment_token_length(cross_attn: torch.Tensor, options: dict) -> int:
    t5xxl_ids = _flatten_token_tensor(options.get("t5xxl_ids"))
    if t5xxl_ids is not None and t5xxl_ids.numel() > 0:
        return int(t5xxl_ids.numel())
    return int(cross_attn.shape[1])


def _merge_segment_options(
    ordered_data: list[tuple[str, torch.Tensor, dict]],
    chosen_options: dict,
) -> dict:
    merged = copy.deepcopy(chosen_options)

    keys_to_concat = ("t5xxl_ids", "t5xxl_weights", "attention_mask")
    for key in keys_to_concat:
        chunks = []
        all_have_key = True
        for _, _, options in ordered_data:
            flat = _flatten_token_tensor(options.get(key))
            if flat is None:
                all_have_key = False
                break
            chunks.append(flat)

        if not all_have_key or not chunks:
            continue

        cat = torch.cat(chunks, dim=0)
        if key == "attention_mask":
            cat = cat.unsqueeze(0)
        merged[key] = cat

    return merged


def _collect_segments(region_1, region_2, region_3, region_4, base, base_position: str):
    region_segments = []
    for name, cond in [
        ("region_1", region_1),
        ("region_2", region_2),
        ("region_3", region_3),
        ("region_4", region_4),
    ]:
        if cond is not None:
            region_segments.append((name, cond))

    if not region_segments:
        raise ValueError("At least one region conditioning is required")

    if base is None:
        return region_segments

    if base_position == "first":
        return [("base", base)] + region_segments
    return region_segments + [("base", base)]


def anima_regional_override(orig_attention_fn, q, k, v, heads, **kwargs):
    transformer_options = kwargs.get("transformer_options", {})
    if not transformer_options.get("anima_regional_enabled", False):
        return orig_attention_fn(q, k, v, heads, **kwargs)

    bias_template = transformer_options.get("anima_regional_bias_template")
    if bias_template is None or not torch.is_tensor(bias_template):
        return orig_attention_fn(q, k, v, heads, **kwargs)

    n_q = q.shape[-2]
    n_k = k.shape[-2]
    cross_only = transformer_options.get("anima_regional_cross_only", True)
    is_cross_attn = n_q != n_k
    if cross_only and not is_cross_attn:
        return orig_attention_fn(q, k, v, heads, **kwargs)

    if bias_template.shape[-2:] != (n_q, n_k):
        if transformer_options.get("anima_regional_debug", False):
            LOGGER.warning(
                "anima_regional: bias shape mismatch. expected (..., %s, %s) got %s",
                n_q,
                n_k,
                tuple(bias_template.shape),
            )
        return orig_attention_fn(q, k, v, heads, **kwargs)

    cond_or_uncond = transformer_options.get("cond_or_uncond")
    bias = build_cond_uncond_gated_bias(
        bias_template=bias_template,
        b_total=q.shape[0],
        cond_or_uncond=cond_or_uncond,
    ).to(device=q.device, dtype=q.dtype)

    existing_mask = kwargs.get("mask")
    if existing_mask is None:
        kwargs["mask"] = bias
    elif torch.is_tensor(existing_mask):
        if existing_mask.dtype == torch.bool:
            kwargs["mask"] = existing_mask
        else:
            kwargs["mask"] = existing_mask.to(device=q.device, dtype=q.dtype) + bias
    else:
        kwargs["mask"] = bias

    return orig_attention_fn(q, k, v, heads, **kwargs)


class AnimaRegionalConditioningConcat:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "region_1": ("CONDITIONING",),
                "base_position": (["last", "first"], {"default": "last"}),
            },
            "optional": {
                "region_2": ("CONDITIONING",),
                "region_3": ("CONDITIONING",),
                "region_4": ("CONDITIONING",),
                "base": ("CONDITIONING",),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "REGIONAL_RANGES")
    RETURN_NAMES = ("conditioning_cat", "ranges")
    FUNCTION = "concat"
    CATEGORY = "conditioning/anima_regional"

    def concat(self, region_1, base_position, region_2=None, region_3=None, region_4=None, base=None):
        ordered_segments = _collect_segments(region_1, region_2, region_3, region_4, base, base_position)

        tokens = []
        segment_ranges = []
        ordered_data: list[tuple[str, torch.Tensor, dict]] = []
        chosen_options = None
        position = 0
        embed_dim = None
        batch_size = None

        for name, cond in ordered_segments:
            cross_attn, options = _validate_single_conditioning(name, cond)

            if embed_dim is None:
                batch_size = cross_attn.shape[0]
                embed_dim = cross_attn.shape[-1]
            elif cross_attn.shape[0] != batch_size or cross_attn.shape[-1] != embed_dim:
                raise ValueError(
                    f"Conditioning '{name}' shape mismatch: expected batch={batch_size}, dim={embed_dim}, got {tuple(cross_attn.shape)}"
                )

            if name == "base":
                chosen_options = copy.deepcopy(options)
            elif chosen_options is None:
                chosen_options = copy.deepcopy(options)

            ordered_data.append((name, cross_attn, options))
            length = _segment_token_length(cross_attn, options)
            segment_ranges.append({"name": name, "start": position, "end": position + length})
            position += length
            tokens.append(cross_attn)

        if chosen_options is None:
            chosen_options = {}
        else:
            chosen_options = _merge_segment_options(ordered_data, chosen_options)

        concat_cross_attn = torch.cat(tokens, dim=1)
        conditioning_cat = [[concat_cross_attn, chosen_options]]

        base_name = "base" if any(segment["name"] == "base" for segment in segment_ranges) else None
        ranges = {
            "segments": segment_ranges,
            "base_name": base_name,
            "total_tokens": position,
        }
        return conditioning_cat, ranges


class AnimaMaskToTokenGrid:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": ("LATENT",),
                "mask_1": ("MASK",),
                "downscale_mode": (["area", "bilinear"], {"default": "area"}),
                "normalize_overlaps": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "mask_2": ("MASK",),
                "mask_3": ("MASK",),
                "mask_4": ("MASK",),
            },
        }

    RETURN_TYPES = ("REGIONAL_TOKEN_WEIGHTS",)
    RETURN_NAMES = ("token_weights",)
    FUNCTION = "build"
    CATEGORY = "conditioning/anima_regional"

    def build(self, model, latent, mask_1, downscale_mode, normalize_overlaps, mask_2=None, mask_3=None, mask_4=None):
        latent_samples = latent.get("samples")
        if latent_samples is None or not torch.is_tensor(latent_samples):
            raise ValueError("latent['samples'] tensor is required")
        if latent_samples.ndim != 4:
            raise ValueError(f"Expected latent samples shape [B,C,H,W], got {tuple(latent_samples.shape)}")

        diffusion_model = model.get_model_object("diffusion_model")
        patch_size = discover_patch_size(diffusion_model)

        _, _, latent_h, latent_w = latent_samples.shape

        region_masks = {
            "region_1": mask_1,
            "region_2": mask_2,
            "region_3": mask_3,
            "region_4": mask_4,
        }

        weights: dict[str, torch.Tensor] = {}
        h_tok = None
        w_tok = None
        for name, mask in region_masks.items():
            if mask is None:
                continue
            flat, this_h_tok, this_w_tok = mask_to_token_weights(
                mask=mask,
                latent_h=latent_h,
                latent_w=latent_w,
                patch_size=patch_size,
                downscale_mode=downscale_mode,
            )
            weights[name] = flat
            if h_tok is None:
                h_tok = this_h_tok
                w_tok = this_w_tok

        if not weights:
            raise ValueError("At least one region mask is required")

        if normalize_overlaps:
            weights = normalize_region_weights(weights)

        n_tokens = int(h_tok * w_tok)
        token_weights = {
            "N_img_tokens": n_tokens,
            "H_tok": h_tok,
            "W_tok": w_tok,
            "patch_size": patch_size,
            "weights": weights,
        }
        return (token_weights,)


class AnimaBuildRegionalCrossAttnBias:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ranges": ("REGIONAL_RANGES",),
                "token_weights": ("REGIONAL_TOKEN_WEIGHTS",),
                "mode": (["soft_log", "hard"], {"default": "soft_log"}),
                "eps": ("FLOAT", {"default": 1e-4, "min": 1e-8, "max": 1.0, "step": 1e-4}),
                "hard_value": ("FLOAT", {"default": -80.0, "min": -200.0, "max": -1.0, "step": 1.0}),
                "base_always_allowed": ("BOOLEAN", {"default": True}),
                "unmasked_to_base": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("REGIONAL_ATTN_BIAS_TEMPLATE",)
    RETURN_NAMES = ("bias_template",)
    FUNCTION = "build"
    CATEGORY = "conditioning/anima_regional"

    def build(self, ranges, token_weights, mode, eps, hard_value, base_always_allowed, unmasked_to_base):
        bias_template = build_bias_template(
            ranges=ranges,
            token_weights=token_weights,
            mode=mode,
            eps=eps,
            hard_value=hard_value,
            base_always_allowed=base_always_allowed,
            unmasked_to_base=unmasked_to_base,
        )
        return (bias_template,)


class AnimaApplyRegionalAttentionHook:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "bias_template": ("REGIONAL_ATTN_BIAS_TEMPLATE",),
                "apply_to_cross_attn_only": ("BOOLEAN", {"default": True}),
                "debug_shapes": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive_out", "negative_out")
    FUNCTION = "apply"
    CATEGORY = "conditioning/anima_regional"

    def apply(self, positive, negative, bias_template, apply_to_cross_attn_only=True, debug_shapes=False):
        if comfy_hooks is None:
            raise RuntimeError(
                "ComfyUI build is missing comfy.hooks. Update ComfyUI to a recent version that supports "
                "TransformerOptionsHook."
            )
        if not torch.is_tensor(bias_template):
            raise ValueError("bias_template must be a torch tensor")
        if bias_template.ndim != 4:
            raise ValueError(f"bias_template must have shape [B,1,Nq,Nk], got {tuple(bias_template.shape)}")

        transformers_dict = {
            "optimized_attention_override": anima_regional_override,
            "anima_regional_bias_template": bias_template.detach().cpu(),
            "anima_regional_enabled": True,
            "anima_regional_cross_only": bool(apply_to_cross_attn_only),
            "anima_regional_debug": bool(debug_shapes),
        }

        hook_group = comfy_hooks.HookGroup()
        hook_group.add(comfy_hooks.TransformerOptionsHook(transformers_dict=transformers_dict))

        cache = {}
        positive_out = comfy_hooks.set_hooks_for_conditioning(positive, hook_group, append_hooks=True, cache=cache)
        negative_out = comfy_hooks.set_hooks_for_conditioning(negative, hook_group, append_hooks=True, cache=cache)
        return positive_out, negative_out


NODE_CLASS_MAPPINGS = {
    "AnimaRegionalConditioningConcat": AnimaRegionalConditioningConcat,
    "AnimaMaskToTokenGrid": AnimaMaskToTokenGrid,
    "AnimaBuildRegionalCrossAttnBias": AnimaBuildRegionalCrossAttnBias,
    "AnimaApplyRegionalAttentionHook": AnimaApplyRegionalAttentionHook,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnimaRegionalConditioningConcat": "Anima Regional Conditioning Concat",
    "AnimaMaskToTokenGrid": "Anima Mask To Token Grid",
    "AnimaBuildRegionalCrossAttnBias": "Anima Build Regional Cross-Attn Bias",
    "AnimaApplyRegionalAttentionHook": "Anima Apply Regional Attention Hook",
}
