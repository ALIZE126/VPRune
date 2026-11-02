"""Tests for mathematical and sequence invariants, without model downloads."""
import unittest
import torch

from vprune.core import (
    VPRuneConfig, greedy_dpp_map, visual_only_select, recycle_tokens,
    original_position_ids, prune_visual_tokens,
)


class CoreTests(unittest.TestCase):
    def test_exact_budget_rounding_and_raster_order(self):
        x = torch.randn(1, 256, 32, generator=torch.Generator().manual_seed(42))
        for budget, want in [(0.1, 26), (0.3, 77), (0.6, 154), (0.8, 205), (1., 256)]:
            with self.subTest(budget=budget):
                keep, kernel = visual_only_select(x, budget)
                self.assertEqual(len(keep), want)
                self.assertEqual(len(set(keep.tolist())), want)
                self.assertTrue(bool(torch.all(keep[1:] > keep[:-1])))
                self.assertEqual(kernel.shape, (256, 256))

    def test_cosine_kernel_has_no_quality_or_text_term(self):
        x = torch.tensor([[[3., 0.], [0., 2.], [3., 4.]]])
        _, kernel = visual_only_select(x, 1.)
        torch.testing.assert_close(kernel, torch.tensor([[1., 0., .6], [0., 1., .8], [.6, .8, 1.]]))

    def test_dpp_chooses_diverse_not_duplicate(self):
        kernel = torch.tensor([[1., 1., 0.], [1., 1., 0.], [0., 0., 1.]])
        self.assertEqual(greedy_dpp_map(kernel, 2).tolist(), [0, 2])

    def test_rank_exhaustion_fills_unique_indices(self):
        for kernel in [torch.ones(8, 8), torch.zeros(8, 8)]:
            self.assertEqual(greedy_dpp_map(kernel, 7).tolist(), list(range(7)))

    def test_group_normalization_and_empty_destination(self):
        x = torch.tensor([[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
                           [2., 0., 0.], [4., 0., 0.], [0., 3., 0.]]])
        keep = torch.tensor([0, 1, 2])
        _, kernel = visual_only_select(x, 1.)
        fused = recycle_tokens(x, keep, kernel, gamma=.3, tau=.1)
        # Two equally similar sources -> average 3; one source -> 3; empty -> identity.
        torch.testing.assert_close(fused, torch.tensor([[[1.6, 0., 0.], [0., 1.6, 0.], [0., 0., 1.]]]))
        torch.testing.assert_close(recycle_tokens(x, keep, kernel, gamma=0), x[:, keep], rtol=0, atol=0)

    def test_similarity_softmax_is_per_destination(self):
        x = torch.tensor([[[1., 0.], [0., 1.], [1., 0.], [.8, .6]]])
        keep = torch.tensor([0, 1]); _, kernel = visual_only_select(x, 1.)
        out = recycle_tokens(x, keep, kernel, gamma=1., tau=.2)
        weight = torch.softmax(torch.tensor([5., 4.]), dim=0)
        want = torch.tensor([[[weight[0] + .8 * weight[1], .6 * weight[1]], [0., 1.]]])
        torch.testing.assert_close(out, want)

    def test_positions_keep_prefix_gaps_and_suffix(self):
        ids = original_position_ids(2, torch.tensor([0, 3, 7]), 8, 2)
        self.assertEqual(ids.tolist(), [[0, 1, 2, 5, 9, 10, 11]])

    def test_full_budget_is_exact_identity(self):
        x = torch.randn(1, 16, 8)
        result = prune_visual_tokens(x, VPRuneConfig(budget=1.))
        torch.testing.assert_close(result.features, x, rtol=0, atol=0)
        self.assertEqual(result.keep_indices.tolist(), list(range(16)))

    def test_one_token_and_dtype(self):
        for dtype in [torch.float32, torch.float16]:
            x = torch.ones(1, 10, 4, dtype=dtype)
            result = prune_visual_tokens(x, VPRuneConfig(budget=.01))
            self.assertEqual(result.features.shape, (1, 1, 4))
            self.assertEqual(result.features.dtype, dtype)
            self.assertTrue(bool(torch.isfinite(result.features).all()))

    def test_configuration_rejects_invalid_numbers(self):
        for kw in [dict(budget=0), dict(budget=1.1), dict(budget=float('nan')),
                   dict(gamma=-.1), dict(gamma=1.1), dict(gamma=float('inf')),
                   dict(tau=0), dict(tau=-.1), dict(tau=float('nan')), dict(tau=1e-9)]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                VPRuneConfig(**kw)

    def test_bad_feature_inputs_rejected(self):
        for x in [torch.ones(2, 3, 4), torch.ones(3, 4), torch.ones(1, 0, 4),
                  torch.ones(1, 3, 0), torch.ones(1, 3, 4, dtype=torch.long),
                  torch.full((1, 3, 4), float('nan'))]:
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                prune_visual_tokens(x)

    def test_bad_indices_rejected_instead_of_misaligning_features(self):
        for keep in [torch.tensor([2, 1]), torch.tensor([1, 1]), torch.tensor([0, 8]),
                     torch.tensor([-1, 2]), torch.tensor([1.5, 2.5])]:
            with self.subTest(keep=keep), self.assertRaises(ValueError):
                original_position_ids(2, keep, 8, 3)


if __name__ == '__main__':
    unittest.main()
