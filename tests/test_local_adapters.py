"""gemma-scope and Qwen3: declaration coherence (CI) + real gates (local only).

The declaration tests need no weights and run everywhere: they check that the
two adapters declare the OPPOSITE tap conventions, which is the trap gate (ii)
exists to catch, and that their measured dtype findings are recorded rather
than remembered.

The gate tests need the real weights — 7.85 GB of gemma-scope transcoders, or
18.8 GB of Qwen ones, plus a gated Gemma license. They are skipped unless
SIGNOFF_LOCAL=1, and are marked `slow`/`local_only` for pytest.
"""

from __future__ import annotations

import os
import unittest

LOCAL = os.environ.get("SIGNOFF_LOCAL") == "1"

try:  # pytest markers when available; plain unittest otherwise
    import pytest

    slow = pytest.mark.slow
    local_only = pytest.mark.local_only
except Exception:  # pragma: no cover
    def slow(x):
        return x

    def local_only(x):
        return x


class TestDeclarationsWithoutWeights(unittest.TestCase):
    """Imports only — no model, no network."""

    def setUp(self):
        from signoff.adapters import gemma_scope, qwen_mwhanna

        self.gemma = gemma_scope.GemmaScopeAdapter
        self.qwen = qwen_mwhanna.QwenMwhannaAdapter

    def test_both_pin_model_and_dictionary_revisions(self):
        for A in (self.gemma, self.qwen):
            self.assertTrue(A.identity.fully_pinned, A.name)
            self.assertEqual(len(A.identity.model_revision), 40)
            self.assertEqual(len(A.identity.dict_revision), 40)

    def test_the_two_declare_OPPOSITE_tap_conventions(self):
        # This is the whole reason gate (ii) exists. If these ever agree,
        # someone has copied a tap between releases.
        self.assertIn("PRE-GAIN", self.gemma.taps.input_convention)
        self.assertIn("POST-GAIN", self.qwen.taps.input_convention)
        # ...and the write points differ for the same reason
        self.assertIn("hook_mlp_out", self.gemma.taps.output_hook)
        self.assertIn("mlp", self.qwen.taps.output_hook)
        self.assertIn("ln2_post", self.gemma.taps.output_convention)

    def test_opposite_bos_conventions_are_both_declared(self):
        self.assertEqual(self.gemma.tokenization.bos_id, 2)
        self.assertIsNone(self.qwen.tokenization.bos_id)
        for A in (self.gemma, self.qwen):
            self.assertTrue(A.tokenization.declared)
            self.assertTrue(A.tokenization.notes)

    def test_gemma_records_the_measured_dtype_failure(self):
        m = self.gemma.dtype_policy.measured
        self.assertEqual(self.gemma.dtype_policy.default, "float16")
        self.assertIn("FAIL", m["bfloat16"])
        self.assertIn("PASS", m["float16"])

    def test_qwen_does_not_offer_an_unmeasured_dtype(self):
        self.assertNotIn("bfloat16", self.qwen.dtype_policy.allowed)

    def test_canonical_l0_table_is_complete_and_non_degenerate(self):
        from signoff.adapters.gemma_scope import CANONICAL_L0, N_LAYERS, tc_path

        self.assertEqual(sorted(CANONICAL_L0), list(range(N_LAYERS)))
        # the bug this freezes out: the registry's L0 = 5..15 variants
        self.assertGreater(min(CANONICAL_L0.values()), 50)
        # and the rule that generated it
        for L, l0 in CANONICAL_L0.items():
            self.assertLess(abs(l0 - 100), 50, f"layer {L}")
        self.assertIn("average_l0_115", tc_path(0))

    def test_provenance_gate_detects_the_degenerate_drift(self):
        from signoff import gates as G
        from signoff.adapters.gemma_scope import CANONICAL_L0

        drifted = dict(CANONICAL_L0)
        drifted[11], drifted[12] = 5, 15      # what SAELens actually pointed at
        r = G.check_provenance_freeze(CANONICAL_L0, drifted,
                                      what="canonical-L0 variant selection")
        self.assertEqual(r.status, G.FAIL)
        self.assertEqual(r.value, 2.0)

    def test_registry_marks_both_local_tier(self):
        from signoff import adapters

        d = adapters.describe()
        self.assertEqual(d["gemma-scope-2b"]["tier"], "local")
        self.assertEqual(d["qwen3-mwhanna"]["tier"], "local")


@slow
@local_only
@unittest.skipUnless(LOCAL, "needs local weights; set SIGNOFF_LOCAL=1")
class TestGemmaGates(unittest.TestCase):
    def test_gates_i_and_ii_on_a_two_layer_smoke(self):
        import torch

        from signoff import adapters

        ad = adapters.get("gemma-scope-2b")
        tok = ad.tokenizer()
        toks = torch.full((2, 32), ad.tokenization.bos_id, dtype=torch.long)
        toks[:, 1:] = torch.randint(100, 1000, (2, 31))
        self.assertTrue(ad.gate_base_vs_base(toks, n=2).passed)
        verdict, table = ad.gate_fvu(toks, [0, 1], batch=2)
        self.assertTrue(verdict.passed, verdict.message)

    def test_provenance_matches_the_live_listing(self):
        from signoff import adapters

        r = adapters.get("gemma-scope-2b").verify_provenance()
        self.assertTrue(r.passed, r.message)


@slow
@local_only
@unittest.skipUnless(LOCAL, "needs local weights; set SIGNOFF_LOCAL=1")
class TestQwenGates(unittest.TestCase):
    def test_gates_i_and_ii_on_a_two_layer_smoke(self):
        import torch

        from signoff import adapters

        ad = adapters.get("qwen3-mwhanna")
        toks = torch.randint(100, 1000, (2, 32))
        self.assertTrue(ad.gate_base_vs_base(toks, n=2).passed)
        verdict, _ = ad.gate_fvu(toks, [0, 1], batch=2)
        self.assertTrue(verdict.passed, verdict.message)


if __name__ == "__main__":
    unittest.main()
