import copy
import logging
import math
import os
import time
from pathlib import Path
from typing import Any

import torch

# ---- File-based debug log (visible without terminal access) ----
_DEBUG_LOG_PATH = Path(__file__).parent / "anima_regional_debug.log"
_debug_enabled = True  # toggled by debug_shapes on the Apply node
_override_call_count = 0
_override_applied_count = 0
_override_skipped_count = 0


def _debug_log(msg: str) -> None:
    """Append a timestamped message to the debug log file."""
    try:
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass


def _debug_log_reset() -> None:
    """Write a separator for a fresh run (preserves previous entries)."""
    try:
        with open(_DEBUG_LOG_PATH, "a") as f:
            f.write(f"\n=== New run ({time.strftime('%Y-%m-%d %H:%M:%S')}) ===\n")
    except Exception:
        pass

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


def _strip_padding(cross_attn: torch.Tensor, options: dict) -> tuple[torch.Tensor, dict, int]:
    """Strip padding tokens from a conditioning segment using attention_mask.

    Many text encoders (e.g. T5) pad every prompt to a fixed min_length (often 512).
    When multiple padded segments are concatenated, the padding tokens dominate the
    cross-attention softmax and dilute the real text signal — causing washed-out,
    impressionistic output.

    Returns (stripped_cross_attn, stripped_options, n_real_tokens).
    """
    mask = options.get("attention_mask")
    if mask is None:
        return cross_attn, options, cross_attn.shape[1]

    mask_flat = _flatten_token_tensor(mask)
    if mask_flat is None:
        return cross_attn, options, cross_attn.shape[1]

    n_real = int(mask_flat.sum().item())
    if n_real <= 0:
        return cross_attn, options, cross_attn.shape[1]  # safety
    if n_real >= cross_attn.shape[1]:
        return cross_attn, options, cross_attn.shape[1]  # no padding

    stripped = cross_attn[:, :n_real, :]

    new_options = copy.copy(options)
    for key in ("t5xxl_ids", "t5xxl_weights"):
        flat = _flatten_token_tensor(options.get(key))
        if flat is not None and flat.numel() >= n_real:
            new_options[key] = flat[:n_real]
    # All remaining tokens are real — attention_mask is no longer needed.
    new_options.pop("attention_mask", None)

    return stripped, new_options, n_real


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


def _get_current_sigma(transformer_options: dict) -> float | None:
    """Extract the current sigma as a plain float from transformer_options."""
    raw = transformer_options.get("sigmas")
    if raw is None:
        return None
    if torch.is_tensor(raw):
        raw = raw.item() if raw.numel() == 1 else raw[0].item()
    return float(raw)


def _should_apply_bias(transformer_options: dict) -> bool:
    """Decide whether the regional bias should be active at the current sigma.

    Two methods, tried in order:
    1. **Sigma thresholds** – set when the user connects the *model* input.
       Uses ``model_sampling.percent_to_sigma`` for accurate mapping.
    2. **Runtime percentage fallback** – used when no model is connected.
       Estimates denoising progress from sigma values via log-space
       interpolation, tracked in the mutable ``_anima_state`` dict.
    """
    current_sigma = _get_current_sigma(transformer_options)

    # --- Method 1: sigma thresholds (from model) ---------------------------
    start_sigma = transformer_options.get("anima_regional_start_sigma")
    end_sigma = transformer_options.get("anima_regional_end_sigma")
    if start_sigma is not None or end_sigma is not None:
        if current_sigma is None:
            return True  # no sigma info → apply conservatively
        if start_sigma is not None and current_sigma > start_sigma:
            return False
        if end_sigma is not None and current_sigma < end_sigma:
            return False
        return True

    # --- Method 2: percentage + runtime sigma tracking ---------------------
    start_pct = transformer_options.get("anima_regional_start_percent", 0.0)
    end_pct = transformer_options.get("anima_regional_end_percent", 1.0)
    if start_pct <= 0.0 and end_pct >= 1.0:
        return True  # no gating requested
    if current_sigma is None:
        return True  # can't gate without sigma

    state = transformer_options.get("_anima_state")
    if state is None:
        return True  # no tracking dict

    # Record the maximum sigma seen (first step provides the highest sigma).
    if "sigma_max" not in state or current_sigma > state["sigma_max"]:
        state["sigma_max"] = current_sigma
    sigma_max = state["sigma_max"]

    # Estimate denoising progress in log-space  (0.0 = start, 1.0 = end)
    SIGMA_MIN_EST = 0.001  # conservative lower bound
    if sigma_max <= SIGMA_MIN_EST or current_sigma <= 0:
        return True

    if current_sigma >= sigma_max:
        progress = 0.0
    elif current_sigma <= SIGMA_MIN_EST:
        progress = 1.0
    else:
        log_range = math.log(sigma_max) - math.log(SIGMA_MIN_EST)
        progress = (math.log(sigma_max) - math.log(current_sigma)) / log_range
        progress = max(0.0, min(1.0, progress))

    return start_pct <= progress <= end_pct


def anima_regional_override(orig_attention_fn, q, k, v, heads, **kwargs):
    global _override_call_count, _override_applied_count, _override_skipped_count
    transformer_options = kwargs.get("transformer_options", {})
    if not transformer_options.get("anima_regional_enabled", False):
        return orig_attention_fn(q, k, v, heads, **kwargs)

    _override_call_count += 1
    debug = transformer_options.get("anima_regional_debug", False)

    # Log first call to prove override IS being triggered
    if _override_call_count == 1:
        _debug_log(f"OVERRIDE CALLED (first call). q={tuple(q.shape)} k={tuple(k.shape)} heads={heads}")

    # ---- Time gating (skip bias during early/late steps) ----
    if not _should_apply_bias(transformer_options):
        _override_skipped_count += 1
        if debug and _override_skipped_count <= 3:
            sigma = _get_current_sigma(transformer_options)
            _debug_log(f"  GATED OUT at sigma={sigma}")
        return orig_attention_fn(q, k, v, heads, **kwargs)

    bias_template = transformer_options.get("anima_regional_bias_template")
    if bias_template is None or not torch.is_tensor(bias_template):
        if _override_call_count <= 2:
            _debug_log(f"  No bias_template found in transformer_options")
        return orig_attention_fn(q, k, v, heads, **kwargs)

    n_q = q.shape[-2]
    n_k = k.shape[-2]
    cross_only = transformer_options.get("anima_regional_cross_only", True)
    is_cross_attn = n_q != n_k
    if cross_only and not is_cross_attn:
        return orig_attention_fn(q, k, v, heads, **kwargs)

    bias_nq = bias_template.shape[-2]
    bias_nk = bias_template.shape[-1]

    # ---- Nq mismatch: can't fix, skip ----
    if bias_nq != n_q:
        if _override_call_count <= 5 or debug:
            _debug_log(
                f"  Nq MISMATCH (unfixable): bias Nq={bias_nq} != attn Nq={n_q}. "
                f"q={tuple(q.shape)} k={tuple(k.shape)}"
            )
        return orig_attention_fn(q, k, v, heads, **kwargs)

    # ---- Nk mismatch handling ----
    # Safe rules:
    # - If model key length is smaller than our concatenated regional text length,
    #   we cannot map region segments reliably -> skip bias for this layer.
    # - If key length is much larger than typical text context, this is likely a
    #   non-text cross-attention path -> skip to avoid destabilizing sampling.
    # - Otherwise, pad (never truncate) to support models that right-pad text to
    #   a fixed context length (for example 512).
    if bias_nk != n_k:
        if n_k < bias_nk:
            if _override_call_count <= 5 or debug:
                _debug_log(
                    f"  Nk MISMATCH (skip): attn Nk={n_k} is smaller than bias Nk={bias_nk}. "
                    "Likely truncated or non-text context."
                )
            return orig_attention_fn(q, k, v, heads, **kwargs)
        if n_k > 2048:
            if _override_call_count <= 5 or debug:
                _debug_log(
                    f"  Nk MISMATCH (skip): attn Nk={n_k} exceeds text-safe threshold; "
                    "likely non-text cross-attention."
                )
            return orig_attention_fn(q, k, v, heads, **kwargs)
        else:
            # Bias is NARROWER than needed — model pads text to fixed length
            # (e.g., Anima pads LLMAdapter output to 512).
            # Pad with minimum bias value to SUPPRESS attention to padding tokens.
            pad_width = n_k - bias_nk
            pad_value = float(bias_template.min().item())
            if _override_call_count <= 3:
                _debug_log(
                    f"  Padding bias Nk: {bias_nk} -> {n_k} (+{pad_width} cols, "
                    f"fill={pad_value:.2f})"
                )
            padding = torch.full(
                (bias_template.shape[0], bias_template.shape[1], bias_nq, pad_width),
                pad_value, dtype=bias_template.dtype, device=bias_template.device,
            )
            bias_template = torch.cat([bias_template, padding], dim=-1)
            # Update the template in transformer_options so subsequent layers
            # within the same step reuse the padded version.
            transformer_options["anima_regional_bias_template"] = bias_template

    cond_or_uncond = transformer_options.get("cond_or_uncond")

    # ---- Device-side caching ----
    # Avoid repeated CPU->GPU transfers of the bias template on every attention call.
    cache_key = ("anima_regional_bias_cache", q.shape[0], id(cond_or_uncond), q.device, q.dtype)
    cached = transformer_options.get("_anima_cache", {}).get(cache_key)
    if cached is not None:
        bias = cached
    else:
        bias = build_cond_uncond_gated_bias(
            bias_template=bias_template,
            b_total=q.shape[0],
            cond_or_uncond=cond_or_uncond,
        ).to(device=q.device, dtype=q.dtype)
        if "_anima_cache" not in transformer_options:
            transformer_options["_anima_cache"] = {}
        transformer_options["_anima_cache"][cache_key] = bias

    _override_applied_count += 1
    if _override_applied_count <= 3:
        sigma = _get_current_sigma(transformer_options)
        _debug_log(
            f"  BIAS APPLIED #{_override_applied_count}: n_q={n_q} n_k={n_k} "
            f"bias_range=[{bias.min().item():.2f}, {bias.max().item():.2f}] sigma={sigma}"
        )

    # ---- Inline attention with regional bias ------------------------------
    # We compute the biased attention *directly* here instead of delegating
    # to any ComfyUI/xformers/PyTorch backend.  This avoids every possible
    # backend incompatibility (xformers can't handle attn_bias on Blackwell;
    # PyTorch SDPA may pick a cuDNN kernel that also fails).
    #
    # Cosmos feeds (B, H, S, D) with ``skip_reshape=True``.
    # Non-Cosmos models may feed (B, S, H*D) with ``skip_reshape=False``.
    skip_reshape = kwargs.get("skip_reshape", False)
    if skip_reshape:
        b, h, sq, d = q.shape
    else:
        b, sq_dim, hd = q.shape
        d = hd // heads
        h = heads
        sq = sq_dim
        q = q.view(b, sq, h, d).permute(0, 2, 1, 3)  # → (B,H,Sq,D)
        k = k.view(b, -1, h, d).permute(0, 2, 1, 3)   # → (B,H,Sk,D)
        v = v.view(b, -1, h, d).permute(0, 2, 1, 3)   # → (B,H,Sk,D)

    scale = d ** -0.5

    # Similarity in float32 for numerical stability
    sim = torch.matmul(q.float(), k.float().transpose(-1, -2)) * scale
    # sim: (B, H, Sq, Sk)

    # Merge existing mask (if any) with our regional bias
    existing_mask = kwargs.get("mask")
    if existing_mask is not None and torch.is_tensor(existing_mask):
        if existing_mask.dtype == torch.bool:
            sim.masked_fill_(~existing_mask, -torch.finfo(sim.dtype).max)
        else:
            sim = sim + existing_mask.float()

    # Apply regional bias — shape (B, 1, Sq, Sk) broadcasts over heads
    sim = sim + bias.float()

    # Softmax + weighted sum
    attn_weights = sim.softmax(dim=-1)
    out = torch.matmul(attn_weights.to(v.dtype), v)
    # out: (B, H, Sq, D)

    if _override_applied_count <= 3:
        has_nan = torch.isnan(out).any().item()
        has_inf = torch.isinf(out).any().item()
        _debug_log(
            f"  attn output: shape={tuple(out.shape)} "
            f"range=[{out.min().item():.4f}, {out.max().item():.4f}] "
            f"nan={has_nan} inf={has_inf}"
        )

    # Return in the format the caller expects
    skip_output_reshape = kwargs.get("skip_output_reshape", False)
    if skip_output_reshape:
        # Caller wants (B, H, Sq, D) — keep as-is
        return out
    else:
        # Caller wants (B, Sq, H*D)
        return out.permute(0, 2, 1, 3).reshape(b, -1, h * d)


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

            # Strip padding tokens produced by fixed-length text encoders (e.g.
            # T5 pads to 512).  This prevents dead padding tokens from dominating
            # the cross-attention softmax and diluting the real text signal.
            n_before = cross_attn.shape[1]
            cross_attn, options, real_length = _strip_padding(cross_attn, options)
            if real_length < n_before:
                _debug_log(
                    f"  Stripped {n_before - real_length} padding tokens "
                    f"from '{name}' ({n_before} -> {real_length})"
                )

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

        # Log segment layout
        _debug_log(f"CONCAT: {len(segment_ranges)} segments, total_tokens={position}")
        for seg in segment_ranges:
            _debug_log(f"  {seg['name']}: tokens [{seg['start']}..{seg['end']})")
        _debug_log(f"  concat shape: {tuple(concat_cross_attn.shape)}")
        opt_keys = [k for k in chosen_options.keys() if k != "pooled_output"]
        _debug_log(f"  options keys: {opt_keys}")

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
        _debug_log(f"MASK_TO_TOKEN: patch_size={patch_size}, latent=({latent_h},{latent_w}), "
                   f"grid=({h_tok},{w_tok}), N_img_tokens={n_tokens}")
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
                "start_percent": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction of denoising steps to SKIP before activating regional bias. "
                               "Early steps form global composition; applying bias too early produces "
                               "impressionistic / noisy output. Recommended range: 0.20 - 0.45.",
                }),
                "end_percent": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Fraction of denoising at which to STOP applying regional bias. "
                               "1.0 means keep it active until the final step.",
                }),
                "debug_shapes": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Master switch. When OFF, passes conditioning through unchanged "
                               "(useful for A/B testing regional vs non-regional output).",
                }),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("positive_out", "negative_out")
    FUNCTION = "apply"
    CATEGORY = "conditioning/anima_regional"

    def apply(self, positive, negative, bias_template,
              enabled=True, apply_to_cross_attn_only=True, start_percent=0.30, end_percent=1.0,
              debug_shapes=False, model=None):
        global _override_call_count, _override_applied_count, _override_skipped_count

        # Reset debug state for each execution
        _override_call_count = 0
        _override_applied_count = 0
        _override_skipped_count = 0
        _debug_log_reset()

        if not enabled:
            _debug_log("DISABLED — passing conditioning through unchanged")
            return positive, negative

        if comfy_hooks is None:
            raise RuntimeError(
                "ComfyUI build is missing comfy.hooks. Update ComfyUI to a recent version that supports "
                "TransformerOptionsHook."
            )
        if not torch.is_tensor(bias_template):
            raise ValueError("bias_template must be a torch tensor")
        if bias_template.ndim != 4:
            raise ValueError(f"bias_template must have shape [B,1,Nq,Nk], got {tuple(bias_template.shape)}")

        _debug_log(f"bias_template shape: {tuple(bias_template.shape)}")
        _debug_log(f"  N_img_tokens(Nq)={bias_template.shape[2]}, N_text_tokens(Nk)={bias_template.shape[3]}")
        _debug_log(f"  bias value range: [{bias_template.min().item():.4f}, {bias_template.max().item():.4f}]")
        _debug_log(f"  start_percent={start_percent}, end_percent={end_percent}")
        _debug_log(f"  cross_only={apply_to_cross_attn_only}, debug={debug_shapes}")
        _debug_log(f"  model connected: {model is not None}")

        # ---- Convert percent → sigma thresholds ----
        start_sigma = None
        end_sigma = None
        if model is not None:
            model_sampling = model.get_model_object("model_sampling")
            if model_sampling is not None and hasattr(model_sampling, "percent_to_sigma"):
                start_sigma = float(model_sampling.percent_to_sigma(start_percent))
                end_sigma = float(model_sampling.percent_to_sigma(end_percent))
                _debug_log(f"  sigma thresholds: start={start_sigma:.4f}, end={end_sigma:.4f}")
            else:
                _debug_log("  WARNING: model has no percent_to_sigma, using runtime estimation")
        else:
            _debug_log("  No model — using runtime sigma estimation for gating")

        transformers_dict = {
            "optimized_attention_override": anima_regional_override,
            "anima_regional_bias_template": bias_template.detach().cpu(),
            "anima_regional_enabled": True,
            "anima_regional_cross_only": bool(apply_to_cross_attn_only),
            "anima_regional_debug": bool(debug_shapes),
            "anima_regional_start_percent": float(start_percent),
            "anima_regional_end_percent": float(end_percent),
            "_anima_state": {},
        }

        if start_sigma is not None:
            transformers_dict["anima_regional_start_sigma"] = start_sigma
        if end_sigma is not None:
            transformers_dict["anima_regional_end_sigma"] = end_sigma

        hook_group = comfy_hooks.HookGroup()
        hook_group.add(comfy_hooks.TransformerOptionsHook(transformers_dict=transformers_dict))

        cache = {}
        positive_out = comfy_hooks.set_hooks_for_conditioning(positive, hook_group, append_hooks=True, cache=cache)
        negative_out = comfy_hooks.set_hooks_for_conditioning(negative, hook_group, append_hooks=True, cache=cache)

        _debug_log("Hook registered. Waiting for sampling to begin...")
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
