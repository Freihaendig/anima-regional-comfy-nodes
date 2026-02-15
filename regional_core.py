import logging
import math
from typing import Any

import torch
import torch.nn.functional as F


def _coerce_patch_size(value: Any) -> int:
    if isinstance(value, (tuple, list)):
        if not value:
            raise ValueError("patch size is empty")
        value = value[0]
    patch_size = int(value)
    if patch_size <= 0:
        raise ValueError(f"invalid patch size: {patch_size}")
    return patch_size


def discover_patch_size(diffusion_model: Any) -> int:
    if hasattr(diffusion_model, "patch_spatial"):
        return _coerce_patch_size(getattr(diffusion_model, "patch_spatial"))

    x_embedder = getattr(diffusion_model, "x_embedder", None)
    if x_embedder is not None:
        if hasattr(x_embedder, "spatial_patch_size"):
            return _coerce_patch_size(getattr(x_embedder, "spatial_patch_size"))
        if hasattr(x_embedder, "patch_size"):
            return _coerce_patch_size(getattr(x_embedder, "patch_size"))

    model_name = diffusion_model.__class__.__name__
    raise ValueError(
        f"Could not determine patch size for diffusion model '{model_name}'. "
        "Expected patch_spatial, x_embedder.spatial_patch_size, or x_embedder.patch_size."
    )


def _normalize_mask_tensor(mask: torch.Tensor) -> torch.Tensor:
    if not torch.is_tensor(mask):
        raise TypeError("mask must be a torch.Tensor")

    if mask.ndim == 2:
        out = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        out = mask[:1].unsqueeze(0)
    elif mask.ndim == 4:
        out = mask[:1, :1]
    else:
        raise ValueError(f"Unsupported mask shape {tuple(mask.shape)}")

    return out.to(dtype=torch.float32)


def mask_to_token_weights(
    mask: torch.Tensor,
    latent_h: int,
    latent_w: int,
    patch_size: int,
    downscale_mode: str,
) -> tuple[torch.Tensor, int, int]:
    if latent_h % patch_size != 0 or latent_w % patch_size != 0:
        raise ValueError(
            f"Latent shape ({latent_h}, {latent_w}) must be divisible by patch size {patch_size}."
        )

    if downscale_mode not in {"area", "bilinear"}:
        raise ValueError(f"Unsupported downscale mode '{downscale_mode}'")

    mask_4d = _normalize_mask_tensor(mask)
    if downscale_mode == "bilinear":
        resized = F.interpolate(mask_4d, size=(latent_h, latent_w), mode="bilinear", align_corners=False)
    else:
        resized = F.interpolate(mask_4d, size=(latent_h, latent_w), mode="area")

    h_tok = latent_h // patch_size
    w_tok = latent_w // patch_size
    pooled = resized.reshape(1, 1, h_tok, patch_size, w_tok, patch_size).mean(dim=(3, 5))
    flat = pooled.reshape(-1).clamp(0.0, 1.0)
    return flat, h_tok, w_tok


def normalize_region_weights(weights: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not weights:
        return {}

    total = torch.zeros_like(next(iter(weights.values())))
    for weight in weights.values():
        total = total + weight

    divisor = torch.where(total > 1.0, total, torch.ones_like(total))
    normalized: dict[str, torch.Tensor] = {}
    for name, weight in weights.items():
        normalized[name] = (weight / divisor).clamp(0.0, 1.0)
    return normalized


def _segment_lookup(ranges: dict[str, Any]) -> dict[str, tuple[int, int]]:
    lookup: dict[str, tuple[int, int]] = {}
    for segment in ranges.get("segments", []):
        name = segment["name"]
        start = int(segment["start"])
        end = int(segment["end"])
        if end <= start:
            raise ValueError(f"Invalid token range for '{name}': [{start}, {end})")
        lookup[name] = (start, end)
    return lookup


def build_bias_template(
    ranges: dict[str, Any],
    token_weights: dict[str, Any],
    mode: str,
    eps: float,
    hard_value: float,
    base_always_allowed: bool,
    unmasked_to_base: bool,
) -> torch.Tensor:
    if mode not in {"soft_log", "hard"}:
        raise ValueError(f"Unsupported mode '{mode}'")

    weights = token_weights.get("weights", {})
    if not weights:
        raise ValueError("token_weights['weights'] is empty")

    n_img_tokens = int(token_weights["N_img_tokens"])
    total_tokens = int(ranges.get("total_tokens", 0))
    if total_tokens <= 0:
        raise ValueError("ranges['total_tokens'] must be > 0")

    lookup = _segment_lookup(ranges)
    base_name = ranges.get("base_name")
    base_range = lookup.get(base_name) if base_name else None

    if mode == "soft_log":
        fill_value = float(math.log(max(eps, 1e-12)))
    else:
        fill_value = float(hard_value)

    bias = torch.full((1, 1, n_img_tokens, total_tokens), fill_value, dtype=torch.float32)

    for region_name, weight in weights.items():
        if weight.numel() != n_img_tokens:
            raise ValueError(
                f"Weight size mismatch for '{region_name}': {weight.numel()} != {n_img_tokens}"
            )

        token_range = lookup.get(region_name)
        if token_range is None:
            continue

        start, end = token_range
        region_width = end - start
        weight = weight.to(dtype=torch.float32)
        if mode == "soft_log":
            per_token = torch.log(weight.clamp_min(eps)).view(1, 1, n_img_tokens, 1)
            bias[:, :, :, start:end] = per_token.expand(1, 1, n_img_tokens, region_width)
        else:
            allowed = (weight > 0.5).view(1, 1, n_img_tokens, 1)
            allowed = allowed.expand(1, 1, n_img_tokens, region_width)
            bias[:, :, :, start:end] = torch.where(
                allowed,
                torch.zeros_like(bias[:, :, :, start:end]),
                torch.full_like(bias[:, :, :, start:end], fill_value),
            )

    if base_range is not None:
        start, end = base_range
        if base_always_allowed:
            bias[:, :, :, start:end] = 0.0
        elif unmasked_to_base:
            coverage = torch.zeros((n_img_tokens,), dtype=torch.float32)
            for region_name, weight in weights.items():
                if region_name in lookup:
                    coverage = coverage + weight.to(dtype=torch.float32)
            uncovered = (coverage <= eps).view(1, 1, n_img_tokens, 1).expand(1, 1, n_img_tokens, end - start)
            base_slice = bias[:, :, :, start:end]
            bias[:, :, :, start:end] = torch.where(uncovered, torch.zeros_like(base_slice), base_slice)

    return bias


def build_cond_uncond_gated_bias(
    bias_template: torch.Tensor,
    b_total: int,
    cond_or_uncond: list[int] | None,
) -> torch.Tensor:
    if bias_template.ndim != 4:
        raise ValueError(f"Expected bias template to have 4 dims, got {bias_template.ndim}")

    if bias_template.shape[0] == 1:
        bias = bias_template.expand(b_total, -1, -1, -1).clone()
    elif bias_template.shape[0] == b_total:
        bias = bias_template.clone()
    else:
        raise ValueError(
            f"Bias template batch dim must be 1 or {b_total}, got {bias_template.shape[0]}"
        )

    if not cond_or_uncond:
        return bias

    num_chunks = len(cond_or_uncond)
    if num_chunks <= 0:
        return bias

    if b_total % num_chunks != 0:
        logging.warning(
            "anima_regional: B_total (%s) is not divisible by cond_or_uncond chunks (%s); "
            "skipping cond/uncond gating",
            b_total,
            num_chunks,
        )
        return bias

    chunk_size = b_total // num_chunks
    for idx, tag in enumerate(cond_or_uncond):
        if int(tag) == 1:
            start = idx * chunk_size
            end = start + chunk_size
            bias[start:end] = 0

    return bias
