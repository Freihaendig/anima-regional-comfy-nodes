import math
import sys
import unittest
from pathlib import Path

import torch

CUSTOM_NODES_ROOT = Path(__file__).resolve().parents[2]
if str(CUSTOM_NODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CUSTOM_NODES_ROOT))

from anima_regional_custom_nodes.regional_core import (  # noqa: E402
    build_bias_template,
    build_cond_uncond_gated_bias,
    mask_to_token_weights,
    normalize_region_weights,
)


class RegionalCoreTests(unittest.TestCase):
    def test_mask_to_token_weights_shape_and_order(self):
        mask = torch.zeros((4, 4), dtype=torch.float32)
        mask[:2, :2] = 1.0
        flat, h_tok, w_tok = mask_to_token_weights(mask, latent_h=4, latent_w=4, patch_size=2, downscale_mode="area")
        self.assertEqual((h_tok, w_tok), (2, 2))
        self.assertEqual(flat.shape[0], 4)
        self.assertAlmostEqual(flat[0].item(), 1.0, places=5)
        self.assertAlmostEqual(flat[1].item(), 0.0, places=5)
        self.assertAlmostEqual(flat[2].item(), 0.0, places=5)
        self.assertAlmostEqual(flat[3].item(), 0.0, places=5)

    def test_normalize_overlaps(self):
        weights = {
            "region_1": torch.tensor([1.0, 0.6]),
            "region_2": torch.tensor([1.0, 0.9]),
        }
        normalized = normalize_region_weights(weights)
        sums = normalized["region_1"] + normalized["region_2"]
        self.assertTrue(torch.all(sums <= 1.0 + 1e-6))
        self.assertAlmostEqual(sums[0].item(), 1.0, places=6)

    def test_build_bias_template_soft_log(self):
        ranges = {
            "segments": [
                {"name": "region_1", "start": 0, "end": 2},
                {"name": "region_2", "start": 2, "end": 4},
                {"name": "base", "start": 4, "end": 6},
            ],
            "base_name": "base",
            "total_tokens": 6,
        }
        token_weights = {
            "N_img_tokens": 2,
            "weights": {
                "region_1": torch.tensor([1.0, 0.25]),
                "region_2": torch.tensor([0.5, 1.0]),
            },
        }
        bias = build_bias_template(
            ranges=ranges,
            token_weights=token_weights,
            mode="soft_log",
            eps=1e-4,
            hard_value=-80.0,
            base_always_allowed=True,
            unmasked_to_base=True,
        )
        self.assertEqual(tuple(bias.shape), (1, 1, 2, 6))
        self.assertTrue(torch.allclose(bias[:, :, :, 4:6], torch.zeros_like(bias[:, :, :, 4:6])))
        self.assertAlmostEqual(bias[0, 0, 0, 0].item(), 0.0, places=6)
        self.assertAlmostEqual(bias[0, 0, 1, 1].item(), math.log(0.25), places=6)

    def test_cond_uncond_gating(self):
        template = torch.full((1, 1, 3, 5), -2.0)
        gated = build_cond_uncond_gated_bias(template, b_total=4, cond_or_uncond=[0, 1])
        self.assertEqual(tuple(gated.shape), (4, 1, 3, 5))
        self.assertTrue(torch.all(gated[:2] == -2.0))
        self.assertTrue(torch.all(gated[2:] == 0.0))


class StripPaddingTests(unittest.TestCase):
    """Test padding token stripping."""

    def test_strips_padding_from_cross_attn(self):
        from anima_regional_custom_nodes.nodes import _strip_padding

        cross_attn = torch.randn(1, 512, 64)
        mask = torch.zeros(1, 512)
        mask[0, :10] = 1.0
        options = {"attention_mask": mask, "t5xxl_ids": torch.arange(512)}

        stripped, new_opts, n_real = _strip_padding(cross_attn, options)
        self.assertEqual(n_real, 10)
        self.assertEqual(stripped.shape, (1, 10, 64))
        self.assertTrue(torch.equal(stripped, cross_attn[:, :10, :]))
        self.assertEqual(new_opts["t5xxl_ids"].numel(), 10)
        self.assertNotIn("attention_mask", new_opts)

    def test_noop_without_mask(self):
        from anima_regional_custom_nodes.nodes import _strip_padding

        cross_attn = torch.randn(1, 20, 64)
        options = {"some_key": "val"}

        stripped, new_opts, n_real = _strip_padding(cross_attn, options)
        self.assertEqual(n_real, 20)
        self.assertIs(stripped, cross_attn)

    def test_noop_when_all_real(self):
        from anima_regional_custom_nodes.nodes import _strip_padding

        cross_attn = torch.randn(1, 10, 64)
        mask = torch.ones(1, 10)
        options = {"attention_mask": mask}

        stripped, new_opts, n_real = _strip_padding(cross_attn, options)
        self.assertEqual(n_real, 10)
        self.assertIs(stripped, cross_attn)


class OverrideTimeGatingTests(unittest.TestCase):
    """Test that anima_regional_override respects sigma-based time gating."""

    def _make_qkv(self, n_q, n_k, heads=4, dim_head=8, batch=1):
        """Create dummy QKV in skip_reshape format (B, H, S, D)."""
        q = torch.randn(batch, heads, n_q, dim_head)
        k = torch.randn(batch, heads, n_k, dim_head)
        v = torch.randn(batch, heads, n_k, dim_head)
        return q, k, v

    def _fake_attn(self, call_log):
        def fn(q, k, v, heads, **kwargs):
            call_log.append(kwargs.get("mask"))
            return torch.zeros(q.shape[0], q.shape[1], q.shape[2], v.shape[-1])
        return fn

    def _base_transformer_options(self, bias_template):
        return {
            "anima_regional_enabled": True,
            "anima_regional_bias_template": bias_template,
            "anima_regional_cross_only": True,
            "anima_regional_start_sigma": 5.0,
            "anima_regional_end_sigma": 0.0,
        }

    def test_bias_skipped_when_sigma_too_high(self):
        """During early steps (high sigma), bias should be skipped."""
        from anima_regional_custom_nodes.nodes import anima_regional_override

        n_img, n_text = 4, 6
        q, k, v = self._make_qkv(n_img, n_text)
        bias_template = torch.full((1, 1, n_img, n_text), -5.0)

        call_log = []
        # sigma=10.0 is above start_sigma=5.0 → should skip bias
        transformer_options = self._base_transformer_options(bias_template)
        transformer_options["sigmas"] = torch.tensor([10.0])
        out = anima_regional_override(self._fake_attn(call_log), q, k, v, 4, transformer_options=transformer_options)
        self.assertEqual(tuple(out.shape), (1, 4, n_img, q.shape[-1]))
        self.assertEqual(len(call_log), 1, "Original attention should be called when bias is skipped")
        self.assertIsNone(call_log[-1], "Bias should be skipped when sigma > start_sigma")

    def test_bias_applied_when_sigma_in_range(self):
        """When sigma is within the active range, bias should be applied."""
        from anima_regional_custom_nodes.nodes import anima_regional_override

        n_img, n_text = 4, 6
        q, k, v = self._make_qkv(n_img, n_text)
        bias_template = torch.full((1, 1, n_img, n_text), -5.0)

        call_log = []
        # sigma=3.0 is below start_sigma=5.0 and above end_sigma=0.0 → should apply
        transformer_options = self._base_transformer_options(bias_template)
        transformer_options["sigmas"] = torch.tensor([3.0])
        out = anima_regional_override(self._fake_attn(call_log), q, k, v, 4, transformer_options=transformer_options)
        self.assertEqual(tuple(out.shape), (1, 4, n_img, q.shape[-1]))
        self.assertEqual(len(call_log), 0, "Inline biased attention should bypass original attention when active")

    def test_bias_skipped_when_sigma_below_end(self):
        """After end_sigma, bias should stop being applied."""
        from anima_regional_custom_nodes.nodes import anima_regional_override

        n_img, n_text = 4, 6
        q, k, v = self._make_qkv(n_img, n_text)
        bias_template = torch.full((1, 1, n_img, n_text), -5.0)

        call_log = []
        # sigma=0.01 is below end_sigma=0.5 → should skip
        transformer_options = self._base_transformer_options(bias_template)
        transformer_options["anima_regional_end_sigma"] = 0.5
        transformer_options["sigmas"] = torch.tensor([0.01])
        out = anima_regional_override(self._fake_attn(call_log), q, k, v, 4, transformer_options=transformer_options)
        self.assertEqual(tuple(out.shape), (1, 4, n_img, q.shape[-1]))
        self.assertEqual(len(call_log), 1, "Original attention should be called when bias is out of active range")
        self.assertIsNone(call_log[-1], "Bias should be skipped when sigma < end_sigma")

    def test_bias_skipped_when_nk_smaller_than_bias(self):
        """If attn Nk is shorter than regional text Nk, skip (unsafe to truncate)."""
        from anima_regional_custom_nodes.nodes import anima_regional_override

        n_img, bias_text = 4, 8
        q, k, v = self._make_qkv(n_img, 4)
        bias_template = torch.full((1, 1, n_img, bias_text), -5.0)

        call_log = []
        transformer_options = self._base_transformer_options(bias_template)
        transformer_options["sigmas"] = torch.tensor([3.0])  # active range
        out = anima_regional_override(self._fake_attn(call_log), q, k, v, 4, transformer_options=transformer_options)
        self.assertEqual(tuple(out.shape), (1, 4, n_img, q.shape[-1]))
        self.assertEqual(len(call_log), 1, "Original attention should be called for Nk<bias Nk safety path")
        self.assertIsNone(call_log[-1])


class RuntimeSigmaFallbackTests(unittest.TestCase):
    """Test that time gating works without model via runtime sigma estimation."""

    def test_fallback_skips_early_steps(self):
        from anima_regional_custom_nodes.nodes import _should_apply_bias

        state = {}
        # Step 1: sigma=80 (highest) with start_pct=0.20
        # progress=0.0 → below start_pct → should NOT apply
        opts = {
            "anima_regional_start_percent": 0.20,
            "anima_regional_end_percent": 1.0,
            "_anima_state": state,
            "sigmas": torch.tensor([80.0]),
        }
        self.assertFalse(_should_apply_bias(opts))
        self.assertAlmostEqual(state["sigma_max"], 80.0)

    def test_fallback_applies_at_midpoint(self):
        from anima_regional_custom_nodes.nodes import _should_apply_bias

        state = {"sigma_max": 80.0}
        # sigma=1.0, progress ≈ (ln80 - ln1) / (ln80 - ln0.001) ≈ 4.38/11.29 ≈ 0.39
        opts = {
            "anima_regional_start_percent": 0.20,
            "anima_regional_end_percent": 1.0,
            "_anima_state": state,
            "sigmas": torch.tensor([1.0]),
        }
        self.assertTrue(_should_apply_bias(opts))

    def test_fallback_applies_all_when_no_gating(self):
        from anima_regional_custom_nodes.nodes import _should_apply_bias

        # start=0, end=1 → always apply
        opts = {
            "anima_regional_start_percent": 0.0,
            "anima_regional_end_percent": 1.0,
            "_anima_state": {},
            "sigmas": torch.tensor([80.0]),
        }
        self.assertTrue(_should_apply_bias(opts))

    def test_fallback_respects_end_percent(self):
        from anima_regional_custom_nodes.nodes import _should_apply_bias

        state = {"sigma_max": 80.0}
        # sigma=0.002, progress ≈ (ln80 - ln0.002) / (ln80 - ln0.001) ≈ 10.60/11.29 ≈ 0.94
        # With end_pct=0.80, 0.94 > 0.80 → should NOT apply
        opts = {
            "anima_regional_start_percent": 0.0,
            "anima_regional_end_percent": 0.80,
            "_anima_state": state,
            "sigmas": torch.tensor([0.002]),
        }
        self.assertFalse(_should_apply_bias(opts))


if __name__ == "__main__":
    unittest.main()
