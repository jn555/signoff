"""The adapter contract, and the cross-layer seam.

Exercised against the synthetic fixture, so it runs with no weights. The real
adapters satisfy the same contract; `test_golden_gpt2.py` checks one of them
against real numbers.
"""

from __future__ import annotations

import unittest

import torch

from signoff import adapters
from signoff import gates as G
from signoff.replacement import (
    NULL_GAUSSIAN,
    NULL_SHUFFLED,
    NULL_TIED,
    ErrorCalibrator,
    PerLayerPlan,
    Replacement,
    ReplacementSpec,
    Site,
    SubstitutionPlan,
    make_noise,
)


class TestRegistry(unittest.TestCase):
    def test_lists_the_worked_adapters_plus_the_fixtures(self):
        self.assertEqual(set(adapters.available()),
                         {"gpt2-dunefsky", "gpt2-circuit", "gemma-scope-2b",
                          "qwen3-mwhanna", "llama32-clt-mntss", "toy", "toy-circuit"})

    def test_describe_needs_no_heavy_imports(self):
        d = adapters.describe()
        self.assertEqual(d["gpt2-dunefsky"]["tier"], "ci")
        self.assertEqual(d["toy"]["tier"], "test")

    def test_unknown_adapter_names_are_a_helpful_error(self):
        with self.assertRaises(KeyError) as cm:
            adapters.get("not-a-real-adapter")
        self.assertIn("available:", str(cm.exception))


class TestDeclaredContract(unittest.TestCase):
    def setUp(self):
        self.ad = adapters.get("toy")

    def test_all_four_declaration_blocks_are_present(self):
        c = self.ad.contract()
        for key in ("identity", "taps", "tokenization", "dtype_policy"):
            self.assertTrue(c[key])
        self.assertTrue(c["taps"]["input_convention"])
        self.assertTrue(c["taps"]["output_convention"])

    def test_unpinned_revisions_fail_the_provenance_gate(self):
        from signoff.adapters.base import Identity

        self.ad.identity = Identity(release="x", model_repo="r", dict_repo="d")
        r = self.ad.verify_provenance()
        self.assertEqual(r.status, G.FAIL)
        self.assertIn("unpinned", r.message)

    def test_dtype_policy_refuses_undeclared_dtypes(self):
        with self.assertRaises(ValueError):
            adapters.get("toy", dtype="bfloat16")

    def test_run_tag_names_everything_that_changes_a_number(self):
        rep_all = Replacement(self.ad, layers="all")
        rep_one = Replacement(self.ad, layers=[0])
        rep_null = Replacement.null(self.ad, "shuffled")
        tags = {self.ad.run_tag(r) for r in (rep_all, rep_one, rep_null)}
        self.assertEqual(len(tags), 3)      # all three are different experiments
        self.assertIn("layers=all", self.ad.run_tag(rep_all))
        self.assertIn("mode=null:shuffled", self.ad.run_tag(rep_null))
        self.assertIn("float32", self.ad.run_tag(rep_all))

    def test_read_only_tap_does_not_perturb_the_pass(self):
        toks, _ = self.ad.synthetic_corpus(n=4, seq_len=8, n_probes=0)
        model = self.ad.model
        clean = self.ad.clean_layer_major_logits(model, toks)
        state, handles = self.ad.tap(model, 1, replace_fn=None)
        try:
            tapped = self.ad.clean_layer_major_logits(model, toks)
        finally:
            for h in handles:
                h.remove()
        torch.testing.assert_close(clean, tapped, atol=0, rtol=0)
        self.assertIn("x", state)
        self.assertIn("y", state)
        self.assertNotIn("yhat", state)     # read-only means read-only

    def test_layer_major_pass_equals_the_model_forward(self):
        toks, _ = self.ad.synthetic_corpus(n=4, seq_len=8, n_probes=0)
        model = self.ad.model
        torch.testing.assert_close(self.ad.clean_layer_major_logits(model, toks),
                                   self.ad.base_logits(model, toks), atol=1e-6, rtol=0)


class TestReplacementSpec(unittest.TestCase):
    def test_rejects_unknown_modes_and_out_of_range_layers(self):
        with self.assertRaises(ValueError):
            ReplacementSpec(mode="teleport")
        with self.assertRaises(ValueError):
            ReplacementSpec(layers=(0, 99)).resolve_layers(4)

    def test_layer_subsets_are_how_localisation_profiles_run(self):
        ad = adapters.get("toy")
        self.assertEqual(Replacement(ad, layers=[2]).layers, [2])
        self.assertEqual(Replacement(ad, layers="all").layers, [0, 1, 2, 3])

    def test_null_factory_spellings(self):
        ad = adapters.get("toy")
        self.assertEqual(Replacement.null(ad, "shuffled").mode, NULL_SHUFFLED)
        self.assertEqual(Replacement.null(ad, "null:gaussian").mode, NULL_GAUSSIAN)
        self.assertTrue(Replacement.null(ad, "tied-shuffle").spec.is_null)
        self.assertFalse(Replacement(ad).spec.is_null)


class TestNullControls(unittest.TestCase):
    def setUp(self):
        self.errors = {L: torch.randn(2, 5, 8, generator=torch.Generator().manual_seed(L))
                       for L in range(3)}

    def test_shuffle_preserves_the_marginal_exactly(self):
        g = torch.Generator().manual_seed(0)
        noise = make_noise(NULL_SHUFFLED, self.errors, g)
        for L in self.errors:
            a = self.errors[L].reshape(-1, 8).norm(dim=-1).sort().values
            b = noise[L].reshape(-1, 8).norm(dim=-1).sort().values
            torch.testing.assert_close(a, b, atol=1e-6, rtol=0)

    def test_tied_shuffle_uses_one_permutation_for_every_layer(self):
        g = torch.Generator().manual_seed(0)
        tied = make_noise(NULL_TIED, self.errors, g)
        # the same row index moved to the same place at every layer
        src0 = self.errors[0].reshape(-1, 8)
        got0 = tied[0].reshape(-1, 8)
        perm = [int((src0 == got0[i]).all(-1).nonzero()[0]) for i in range(got0.shape[0])]
        for L in (1, 2):
            src = self.errors[L].reshape(-1, 8)
            got = tied[L].reshape(-1, 8)
            torch.testing.assert_close(got, src[torch.tensor(perm)], atol=1e-6, rtol=0)

    def test_independent_shuffle_does_not(self):
        g = torch.Generator().manual_seed(0)
        indep = make_noise(NULL_SHUFFLED, self.errors, g)
        src0, got0 = self.errors[0].reshape(-1, 8), indep[0].reshape(-1, 8)
        perm0 = [int((src0 == got0[i]).all(-1).nonzero()[0]) for i in range(got0.shape[0])]
        src1, got1 = self.errors[1].reshape(-1, 8), indep[1].reshape(-1, 8)
        perm1 = [int((src1 == got1[i]).all(-1).nonzero()[0]) for i in range(got1.shape[0])]
        self.assertNotEqual(perm0, perm1)

    def test_gaussian_null_needs_calibrated_moments(self):
        with self.assertRaises(ValueError):
            make_noise(NULL_GAUSSIAN, self.errors, torch.Generator().manual_seed(0))

    def test_calibrator_recovers_the_error_moments(self):
        d = 4
        g = torch.Generator().manual_seed(0)
        mean = torch.tensor([1.0, -2.0, 0.5, 0.0])
        cal = ErrorCalibrator([0], d)
        for _ in range(40):
            e = torch.randn(8, 16, d, generator=g) * 0.5 + mean
            cal.update({0: e}, {0: torch.ones_like(e)})
        m = cal.finalize()
        torch.testing.assert_close(m.mean[0], mean, atol=0.05, rtol=0)
        # a Cholesky factor exists, so the gaussian null is samplable
        self.assertEqual(tuple(m.chol[0].shape), (d, d))
        self.assertGreater(m.stats[0]["trace_cov"], 0)

    def test_null_forward_keeps_the_true_output_and_adds_the_field(self):
        ad = adapters.get("toy")
        toks, _ = ad.synthetic_corpus(n=2, seq_len=8, n_probes=0)
        rep = Replacement.null(ad, "shuffled")
        zero = {L: torch.zeros(2, 8, ad.d_model) for L in range(ad.n_layers)}
        out = ad.replaced_logits(ad.model, toks, rep, noise=zero)
        # zero noise == the clean model, exactly: the null keeps the TRUE output
        torch.testing.assert_close(out, ad.base_logits(ad.model, toks), atol=1e-6, rtol=0)

    def test_null_forward_refuses_without_a_noise_field(self):
        ad = adapters.get("toy")
        toks, _ = ad.synthetic_corpus(n=2, seq_len=8, n_probes=0)
        with self.assertRaises(ValueError):
            ad.replaced_logits(ad.model, toks, Replacement.null(ad, "shuffled"))


class TestCrossLayerSeam(unittest.TestCase):
    """v0.1's artifacts are per-layer; the next generation is not.

    An adapter supplies its own plan, so a cross-layer artifact (a CLT reading
    once and writing into several downstream layers) needs no change to the
    runner, the metrics, the gates or the report.
    """

    def test_default_plan_is_per_layer_and_sites_are_layer_ids(self):
        ad = adapters.get("toy")
        plan = Replacement(ad, layers=[1, 2]).plan()
        self.assertIsInstance(plan, PerLayerPlan)
        self.assertEqual([s.id for s in plan.sites()], ["1", "2"])

    def test_site_ids_distinguish_read_from_write(self):
        self.assertEqual(Site(3, 3).id, "3")
        self.assertEqual(Site(3, 7).id, "3->7")

    def test_an_adapter_can_override_the_plan(self):
        calls = {}

        class CrossLayerPlan(SubstitutionPlan):
            def sites(self):
                return [Site(1, 2), Site(1, 3)]

            def replaced_logits(self, adapter, model, toks, **kw):
                calls["ran"] = True
                return adapter.base_logits(model, toks)

        class CltAdapter(type(adapters.get("toy"))):
            def substitution_plan(self, spec):
                return CrossLayerPlan(spec, self.n_layers)

        ad = CltAdapter()
        toks, _ = ad.synthetic_corpus(n=2, seq_len=8, n_probes=0)
        rep = Replacement(ad, layers="all")
        plan = rep.plan()
        self.assertEqual([s.id for s in plan.sites()], ["1->2", "1->3"])
        ad.replaced_logits(ad.model, toks, rep)
        self.assertTrue(calls["ran"])

    def test_gate_ii_accepts_cross_layer_site_keys(self):
        r = G.check_fvu_sanity({"1->2": 0.3, "1->3": 0.4})
        self.assertTrue(r.passed)


if __name__ == "__main__":
    unittest.main()
