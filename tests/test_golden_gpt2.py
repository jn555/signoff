"""Golden-number regression: GPT-2-small + Dunefsky transcoders vs experiment 01.

THE POINT.  A fixed token set — committed verbatim in `tests/golden/`, so the
miner, the dataset and the tokenizer are all out of the loop — scored by the
released code must reproduce experiment 01's own per-row numbers. If the
substitution path, the tap, the metric or the position convention drifts, this
fails. The numbers were regenerated from the experiment's checkpoints, never
hand-entered.

TIER.  CPU-only, n=16, all 12 layers, float32. Needs `transformer_lens` and the
GPT-2 + transcoder weights (~2 GB, from the HF cache if already present). It
SKIPS rather than fails when those are unavailable, so a bare checkout still
has a green suite — set SIGNOFF_REQUIRE_GOLDEN=1 in CI to turn a skip into a
failure.
"""

from __future__ import annotations

import json
import os
import unittest

import torch

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden")
REQUIRE = os.environ.get("SIGNOFF_REQUIRE_GOLDEN") == "1"


def _weights_available() -> tuple[bool, str]:
    if os.environ.get("SIGNOFF_SKIP_WEIGHTS") == "1":
        return False, "SIGNOFF_SKIP_WEIGHTS=1"
    try:
        import transformer_lens  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        return False, f"transformer_lens unavailable ({type(e).__name__})"
    return True, ""


_OK, _WHY = _weights_available()


def _load(name):
    with open(os.path.join(GOLDEN, name)) as f:
        return json.load(f)


class TestGoldenFixtures(unittest.TestCase):
    """These run with no weights: the fixtures themselves must stay coherent."""

    def test_fixture_shapes_agree(self):
        toks = _load("gpt2_dunefsky_tokens.json")
        exp = _load("gpt2_dunefsky_expected.json")
        self.assertEqual(len(toks["tokens"]), toks["n"])
        self.assertEqual(len(exp["rows"]), toks["n"])
        self.assertTrue(all(len(r) == toks["seq_len"] for r in toks["tokens"]))
        self.assertEqual([r["row"] for r in exp["rows"]], list(range(toks["n"])))

    def test_tolerances_are_declared_and_flip_is_exact(self):
        exp = _load("gpt2_dunefsky_expected.json")
        t = exp["tolerances"]
        for k in ("d_mean", "d_max", "nll", "flip"):
            self.assertIn(k, t)
        # an argmax comparison has no floating-point excuse
        self.assertEqual(t["flip"], 0.0)
        self.assertLess(t["d_mean"], 1e-2)

    def test_lineage_matches_the_published_exp01_numbers(self):
        # exp-01's headline corpus figures, carried as context for a reader
        lin = _load("gpt2_dunefsky_expected.json")["full_corpus_lineage"]
        self.assertEqual(lin["n"], 5000)
        self.assertAlmostEqual(lin["d_mean_mean"], 1.306, places=2)
        self.assertAlmostEqual(lin["d_mean_p99"], 2.770, places=2)
        self.assertAlmostEqual(lin["tail_ratio_p99_over_median"], 2.27, places=1)


@unittest.skipUnless(_OK or REQUIRE, f"needs real weights: {_WHY}")
class TestGoldenReproduction(unittest.TestCase):
    """The regression proper. ~30 s on a CPU once the weights are cached."""

    @classmethod
    def setUpClass(cls):
        from signoff import adapters

        cls.fx = _load("gpt2_dunefsky_tokens.json")
        cls.exp = _load("gpt2_dunefsky_expected.json")
        cls.toks = torch.tensor(cls.fx["tokens"], dtype=torch.long)
        cls.ad = adapters.get("gpt2-dunefsky", device="cpu")

    def test_gate_i_is_exactly_zero(self):
        # float32, no softcap, no sandwich norm: the layer-major pass must BE
        # the model forward, not merely agree with it.
        r = self.ad.gate_base_vs_base(self.toks, n=4)
        self.assertTrue(r.passed, r.message)
        self.assertLess(r.value, 1e-6)

    def test_gate_ii_matches_a_healthy_suite(self):
        verdict, table = self.ad.gate_fvu(self.toks[:8], list(range(12)), batch=4)
        self.assertTrue(verdict.passed, verdict.message)
        # exp-01/02's per-layer FVU sits well below the mis-tap band everywhere
        self.assertLess(min(v["fvu_global"] for v in table.values()), 0.2)
        self.assertLess(max(v["fvu_global"] for v in table.values()), 0.6)

    def test_per_row_metrics_reproduce_exp01(self):
        from signoff import metrics as M
        from signoff.replacement import Replacement

        rep = Replacement(self.ad, layers="all")
        model = self.ad.model
        lb = self.ad.base_logits(model, self.toks)
        lr = self.ad.replaced_logits(model, self.toks, rep)
        m = M.metrics_from_logits(lb, lr, self.toks)
        tol = self.exp["tolerances"]
        for i, want in enumerate(self.exp["rows"]):
            self.assertAlmostEqual(float(m.d_mean[i]), want["d_mean"], delta=tol["d_mean"],
                                   msg=f"row {i} d_mean")
            self.assertAlmostEqual(float(m.d_max[i]), want["d_max"], delta=tol["d_max"],
                                   msg=f"row {i} d_max")
            self.assertAlmostEqual(float(m.nll[i]), want["nll"], delta=tol["nll"],
                                   msg=f"row {i} nll")
            self.assertEqual(float(m.flip[i]), want["flip"], f"row {i} flip")

    def test_subset_summary_reproduces(self):
        from signoff import metrics as M
        from signoff import stats as S
        from signoff.replacement import Replacement

        rep = Replacement(self.ad, layers="all")
        model = self.ad.model
        m = M.metrics_from_logits(self.ad.base_logits(model, self.toks),
                                  self.ad.replaced_logits(model, self.toks, rep),
                                  self.toks)
        got = S.summarize([float(x) for x in m.d_mean])
        want = self.exp["subset_summary"]["d_mean"]
        for k in ("mean", "p50", "p99", "max"):
            self.assertAlmostEqual(got[k], want[k], delta=self.exp["tolerances"]["d_mean"],
                                   msg=f"subset {k}")

    def test_adapter_contract_is_fully_declared(self):
        c = self.ad.contract()
        self.assertTrue(c["identity"]["model_revision"])
        self.assertTrue(c["identity"]["dict_revision"])
        self.assertIn("ln2.hook_normalized", c["taps"]["input_hook"])
        self.assertIn("hook_mlp_out", c["taps"]["output_hook"])
        self.assertTrue(c["tokenization"]["declared"])
        self.assertIsNone(c["tokenization"]["bos_id"])     # exp-01 mined without BOS
        self.assertTrue(self.ad.verify_provenance().passed)


if __name__ == "__main__":
    unittest.main()
