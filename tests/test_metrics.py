"""Metrics: shape/dtype invariants, the position-0 convention, analytic values.

CPU-only, synthetic tensors, fast — part of the CI subset.
"""

from __future__ import annotations

import math
import unittest

import torch

from signoff import metrics as M


def _logits(b, t, v, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(b, t, v, generator=g)


class TestChunking(unittest.TestCase):
    def test_gpt2_vocab_reproduces_exp01_chunk(self):
        # exp-01 used a fixed CH = 1024. For GPT-2's vocabulary the budget rule
        # yields 1335, so the cap binds and the extraction is byte-identical.
        self.assertEqual(M.chunk_rows(50257), 1024)

    def test_large_vocab_shrinks_the_chunk(self):
        # gemma-2's 256k vocabulary is why the rule exists at all.
        self.assertLess(M.chunk_rows(256000), 1024)
        self.assertGreaterEqual(M.chunk_rows(256000), 1)

    def test_chunking_is_numerically_inert(self):
        lp, lq = _logits(3, 8, 97, 1), _logits(3, 8, 97, 2)
        toks = torch.randint(0, 97, (3, 8))
        big = M.metrics_from_logits(lp, lq, toks)
        old_cap = M._CHUNK_CAP
        try:
            M._CHUNK_CAP = 3  # force many chunks
            small = M.metrics_from_logits(lp, lq, toks)
        finally:
            M._CHUNK_CAP = old_cap
        for a, b in zip(big, small):
            torch.testing.assert_close(a, b, rtol=0, atol=1e-6)


class TestInvariants(unittest.TestCase):
    def test_shapes_and_dtypes(self):
        lp, lq = _logits(4, 6, 11, 3), _logits(4, 6, 11, 4)
        toks = torch.randint(0, 11, (4, 6))
        m = M.metrics_from_logits(lp, lq, toks)
        for x in m:
            self.assertEqual(x.shape, (4,))
            self.assertEqual(x.dtype, torch.float32)
        # tuple-unpackable in the exp-01 order
        d_mean, d_max, flip, nll = m
        self.assertIs(d_mean, m.d_mean)
        self.assertIs(nll, m.nll)

    def test_float16_inputs_are_computed_in_float32(self):
        lp, lq = _logits(2, 5, 13, 5).half(), _logits(2, 5, 13, 6).half()
        toks = torch.randint(0, 13, (2, 5))
        m = M.metrics_from_logits(lp, lq, toks)
        self.assertEqual(m.d_mean.dtype, torch.float32)

    def test_identical_logits_give_zero_divergence(self):
        lp = _logits(3, 7, 23, 7)
        toks = torch.randint(0, 23, (3, 7))
        m = M.metrics_from_logits(lp, lp.clone(), toks)
        torch.testing.assert_close(m.d_mean, torch.zeros(3), atol=1e-6, rtol=0)
        torch.testing.assert_close(m.d_max, torch.zeros(3), atol=1e-6, rtol=0)
        torch.testing.assert_close(m.flip, torch.zeros(3), atol=0, rtol=0)

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            M.metrics_from_logits(_logits(2, 4, 9), _logits(2, 5, 9), torch.zeros(2, 4).long())
        with self.assertRaises(ValueError):
            M.metrics_from_logits(_logits(2, 4, 9), _logits(2, 4, 9), torch.zeros(2, 3).long())

    def test_rejects_single_position_windows(self):
        with self.assertRaises(ValueError):
            M.metrics_from_logits(_logits(2, 1, 9), _logits(2, 1, 9), torch.zeros(2, 1).long())


class TestPositionZeroConvention(unittest.TestCase):
    """Position 0 is excluded from every metric. This is load-bearing:
    with no BOS it is unconditioned, with a BOS it is the BOS itself."""

    def setUp(self):
        self.lp = _logits(2, 6, 17, 11)
        self.lq = _logits(2, 6, 17, 12)
        self.toks = torch.randint(0, 17, (2, 6))

    def test_divergence_ignores_position_zero(self):
        base = M.metrics_from_logits(self.lp, self.lq, self.toks)
        lq2 = self.lq.clone()
        lq2[:, 0, :] += 100.0  # wreck position 0 only
        moved = M.metrics_from_logits(self.lp, lq2, self.toks)
        torch.testing.assert_close(base.d_mean, moved.d_mean, atol=1e-6, rtol=0)
        torch.testing.assert_close(base.d_max, moved.d_max, atol=1e-6, rtol=0)
        torch.testing.assert_close(base.flip, moved.flip, atol=0, rtol=0)

    def test_nll_uses_the_same_t_minus_one_predictions(self):
        # NLL comes from logits 0..T-2 predicting tokens 1..T-1: the LAST
        # position's logits are never scored (they predict nothing in-window).
        base = M.metrics_from_logits(self.lp, self.lq, self.toks)
        lp2 = self.lp.clone()
        lp2[:, -1, :] += 100.0
        moved = M.metrics_from_logits(lp2, self.lq, self.toks)
        torch.testing.assert_close(base.nll, moved.nll, atol=1e-5, rtol=0)

    def test_per_position_kl_keeps_position_zero(self):
        # the field is returned whole; callers slice [:, 1:]. Localisation
        # diagnostics need the raw field.
        kl = M.per_position_kl(self.lp, self.lq)
        self.assertEqual(kl.shape, (2, 6))
        d_mean = kl[:, 1:].mean(-1)
        torch.testing.assert_close(
            d_mean, M.metrics_from_logits(self.lp, self.lq, self.toks).d_mean,
            atol=1e-6, rtol=0)


class TestAnalyticValues(unittest.TestCase):
    def test_uniform_base_gives_log_v_nll(self):
        V = 32
        lp = torch.zeros(2, 5, V)          # uniform
        toks = torch.randint(0, V, (2, 5))
        m = M.metrics_from_logits(lp, lp.clone(), toks)
        torch.testing.assert_close(m.nll, torch.full((2,), math.log(V)), atol=1e-5, rtol=0)

    def test_two_point_kl_matches_closed_form(self):
        # P = softmax([a, 0]), Q = softmax([b, 0]) on a 2-token vocabulary.
        a, b = 1.0, -0.5
        p = torch.softmax(torch.tensor([a, 0.0]), -1)
        q = torch.softmax(torch.tensor([b, 0.0]), -1)
        want = float((p * (p.log() - q.log())).sum())
        lp = torch.tensor([a, 0.0]).expand(1, 4, 2).contiguous()
        lq = torch.tensor([b, 0.0]).expand(1, 4, 2).contiguous()
        m = M.metrics_from_logits(lp, lq, torch.zeros(1, 4, dtype=torch.long))
        torch.testing.assert_close(m.d_mean, torch.tensor([want]), atol=1e-6, rtol=0)
        torch.testing.assert_close(m.d_max, torch.tensor([want]), atol=1e-6, rtol=0)

    def test_flip_counts_argmax_disagreements(self):
        lp = torch.zeros(1, 5, 3)
        lq = torch.zeros(1, 5, 3)
        lp[0, :, 0] = 1.0                 # base always predicts token 0
        lq[0, :, 1] = 1.0                 # replacement always predicts token 1
        lq[0, 2, :] = lp[0, 2, :]         # ...except position 2
        m = M.metrics_from_logits(lp, lq, torch.zeros(1, 5, dtype=torch.long))
        # positions 1..4 are scored; 3 of those 4 disagree
        self.assertAlmostEqual(float(m.flip[0]), 3 / 4, places=6)


class TestFvuTerms(unittest.TestCase):
    def test_perfect_reconstruction_is_zero_numerator(self):
        y = torch.randn(2, 4, 8)
        num, den = M.fvu_terms(y, y.clone())
        self.assertEqual(num.shape, (2, 4))
        self.assertEqual(den.shape, (2, 4))
        torch.testing.assert_close(num, torch.zeros(2, 4), atol=1e-6, rtol=0)

    def test_mean_prediction_gives_fvu_one(self):
        y = torch.randn(3, 5, 6)
        yhat = y.reshape(-1, 6).mean(0).expand_as(y).contiguous()
        num, den = M.fvu_terms(y, yhat)
        self.assertAlmostEqual(float(num.sum() / den.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
