# Anima Regional Conditioning (MVP)

This custom node package implements single-pass regional conditioning for Anima/Cosmos/MiniTrainDiT models by injecting an additive cross-attention bias through ComfyUI's `optimized_attention_override` hook.

## Install

### Option A: One command (recommended)

```bash
gh repo clone Freihaendig/anima-regional-comfy-nodes /path/to/ComfyUI/custom_nodes/anima_regional && COMFYUI_DIR=/path/to/ComfyUI bash /path/to/ComfyUI/custom_nodes/anima_regional/install.sh
```

This clones into `ComfyUI/custom_nodes/anima_regional` and installs optional Python deps.

### Option B: Manual

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/Freihaendig/anima-regional-comfy-nodes.git anima_regional
cd anima_regional
python -m pip install -r requirements.txt
```

Then restart ComfyUI.

## What it does

- Keeps stock ComfyUI sampling (`KSampler` and other standard samplers).
- Keeps cond/uncond batched in one forward pass.
- Gates the regional bias to conditional chunks using `transformer_options["cond_or_uncond"]`.
- Uses model patch size (`patch_spatial`) to correctly map masks to image token grid.
- Logs model/sampling types and warns if an Anima/Cosmos-like model is not using AuraFlow sampling patching.

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

1. Load UNet via `UNETLoader`.
2. Patch the model with `ModelSamplingAuraFlow` (`shift=3.0` for Anima examples).
3. Create text conditionings for `region_1`, `region_2`, and optionally `base` using your normal text encoder node.
4. Pass them to `AnimaRegionalConditioningConcat`.
5. Pass the **patched MODEL**, target `LATENT`, and masks into `AnimaMaskToTokenGrid`.
6. Build bias via `AnimaBuildRegionalCrossAttnBias`.
7. Apply hook via `AnimaApplyRegionalAttentionHook` (with the **patched MODEL**) to produce `positive_out` and `negative_out`.
8. Feed `positive_out` and `negative_out` into `KSampler` with the **same patched MODEL**.

## Example

Prompt-style JSON examples are provided at:
- `examples/anima_regional_two_regions.json`
- `examples/anima_regional_left_right_masks.json`
- These workflows build explicit left/right half masks using `SolidMask` + `MaskComposite`.
- Replace those mask nodes with real masks (for example, loaded masks) when needed.

## Recommended settings for first test

- Resolution: around 1MP (for example `1024x1024`)
- Steps: around `40`
- CFG: around `4.5`
- Sampler: start with `er_sde` (or `euler_a`) for parity with official Anima workflows
- Bias mode: `soft_log`
- Bias strength: start around `0.20`-`0.40`
- Attention backend: `orig` (use `basic` only if your backend crashes with additive bias)
- Keep `base_always_allowed = true`

## Troubleshooting noisy split output

If output looks like static/noise with only a faint left-right split:

- Verify your model path is patched: `UNETLoader -> ModelSamplingAuraFlow(shift=3.0) -> KSampler` and use that same patched model for `AnimaMaskToTokenGrid` + `AnimaApplyRegionalAttentionHook`.
- Lower `strength` in `AnimaBuildRegionalCrossAttnBias` (for example `0.20`-`0.40`) if output collapses to static/noise.
- Increase `start_percent` to `0.45`-`0.65` (regional bias starts too early is the most common cause).
- Keep a meaningful `base` prompt and leave `base_always_allowed = true`.
- Enable `debug_shapes = true` in `AnimaApplyRegionalAttentionHook` and inspect
  `anima_regional_debug.log` in this node folder:
  - Confirm `model types:` appears and no sampling mismatch warning is logged.
  - If you see repeated `Nq MISMATCH` or `Nk MISMATCH`, the hook is skipping or hitting
    incompatible attention layers.
  - If no `BIAS APPLIED` lines appear during sampling, regional conditioning is not active.

## Notes and limitations

- This MVP expects each input conditioning to have exactly one conditioning entry.
- Complex multi-area conditioning stacks are intentionally out of scope for this first version.
- Boundary seams can still happen; soft masks and base tokens allowed globally generally reduce this.

## Deferred alternative

If attention override compatibility becomes an issue on a future ComfyUI backend, a fallback approach is a MultiDiffusion-style sampler node (multi-pass region predictions + masked blend per step). This is not implemented in this MVP.
