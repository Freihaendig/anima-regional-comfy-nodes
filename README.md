# Anima Regional Conditioning (MVP)

This custom node package implements single-pass regional conditioning for Anima/Cosmos/MiniTrainDiT models by injecting an additive cross-attention bias through ComfyUI's `optimized_attention_override` hook.

## What it does

- Keeps stock ComfyUI sampling (`KSampler` and other standard samplers).
- Keeps cond/uncond batched in one forward pass.
- Gates the regional bias to conditional chunks using `transformer_options["cond_or_uncond"]`.
- Uses model patch size (`patch_spatial`) to correctly map masks to image token grid.

## Nodes

1. `AnimaRegionalConditioningConcat`
- Input: up to 4 region conditionings + optional base conditioning.
- Output: concatenated conditioning and token ranges metadata.

2. `AnimaMaskToTokenGrid`
- Input: `MODEL`, `LATENT`, and up to 4 masks.
- Output: per-region token weights aligned to DiT patch tokens.

3. `AnimaBuildRegionalCrossAttnBias`
- Input: ranges + token weights.
- Output: additive attention bias template `(1, 1, N_img_tokens, N_text_tokens)`.

4. `AnimaApplyRegionalAttentionHook`
- Input: positive conditioning, negative conditioning, bias template.
- Output: positive/negative with a shared `TransformerOptionsHook`.

## Minimal workflow wiring

1. Create text conditionings for `region_1`, `region_2`, and optionally `base` using your normal text encoder node.
2. Pass them to `AnimaRegionalConditioningConcat`.
3. Pass `MODEL`, target `LATENT`, and masks into `AnimaMaskToTokenGrid`.
4. Build bias via `AnimaBuildRegionalCrossAttnBias`.
5. Apply hook via `AnimaApplyRegionalAttentionHook` to produce `positive_out` and `negative_out`.
6. Feed `positive_out` and `negative_out` into `KSampler` as usual.

## Example

Prompt-style JSON examples are provided at:
- `custom_nodes/anima_regional/examples/anima_regional_two_regions.json`
- `custom_nodes/anima_regional/examples/anima_regional_left_right_masks.json`
- These workflows build explicit left/right half masks using `SolidMask` + `MaskComposite`.
- Replace those mask nodes with real masks (for example, loaded masks) when needed.

## Recommended settings for first test

- Resolution: around 1MP (for example `1024x1024`)
- Steps: around `40`
- CFG: around `4.5`
- Bias mode: `soft_log`
- Keep `base_always_allowed = true`

## Notes and limitations

- This MVP expects each input conditioning to have exactly one conditioning entry.
- Complex multi-area conditioning stacks are intentionally out of scope for this first version.
- Boundary seams can still happen; soft masks and base tokens allowed globally generally reduce this.

## Deferred alternative

If attention override compatibility becomes an issue on a future ComfyUI backend, a fallback approach is a MultiDiffusion-style sampler node (multi-pass region predictions + masked blend per step). This is not implemented in this MVP.
