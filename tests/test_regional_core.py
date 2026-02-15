import math
import sys
import unittest
from pathlib import Path

import torch

CUSTOM_NODES_ROOT = Path(__file__).resolve().parents[2]
if str(CUSTOM_NODES_ROOT) not in sys.path:
    sys.path.insert(0, str(CUSTOM_NODES_ROOT))

from anima_regional.regional_core import (  # noqa: E402
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


if __name__ == "__main__":
    unittest.main()
