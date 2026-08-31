"""Restore/freeze policies: the local-replacement-model construction, on CPU.

The object published attribution graphs are drawn on is not "the transcoders
substituted in". It is the LOCAL REPLACEMENT MODEL: transcoders PLUS an error
node per site PLUS attention patterns and normalisation scales frozen from the
real forward. Experiment 06 built it; this file is what keeps it built.

The load-bearing test is `TestLRMIdentity`: the construction reproduces the base
model BY CONSTRUCTION, so a non-zero difference is a bug, not a result. Every
other test here exists because that one has a blind spot — the identity is
achieved by the error nodes ALONE, so it cannot see a freeze that silently did
nothing, and `TestFreezeEfficacy` is the positive control that can.

CPU-only, no weights, ~1 s. The fixture is `toy-circuit`, which is the only
fixture with real attention to freeze.
"""

from __future__ import annotations

import unittest

import torch

from signoff import adapters, gates as G
from signoff import replacement as R
from signoff.adapters import base as AB


def _fixture(defect: float = 0.35):
    ad = adapters.get("toy-circuit", defect=defect)
    toks, _meta = ad.synthetic_corpus(n=4, seq_len=8, seed=0, n_probes=0)
    return ad, toks


def _logits(ad, replacement, toks, cache=None):
    return ad.replaced_logits(ad.model, toks, replacement, clean_cache=cache)


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


# ------------------------------------------------------------------ policies


class TestFreezePolicy(unittest.TestCase):
    def test_the_four_rungs_are_named_and_distinct(self):
        rungs = [R.NO_RESTORE, R.ERROR_NODES_ONLY, R.FROZEN_CONTEXT_ONLY,
                 R.LOCAL_REPLACEMENT_MODEL]
        self.assertEqual(len({p.tag() for p in rungs}), 4)
        self.assertEqual(R.NO_RESTORE.tag(), "none")
        self.assertEqual(R.LOCAL_REPLACEMENT_MODEL.tag(), "errors+attn+ln")

    def test_only_all_three_is_a_local_replacement_model(self):
        self.assertTrue(R.LOCAL_REPLACEMENT_MODEL.is_local_replacement_model)
        for p in (R.NO_RESTORE, R.ERROR_NODES_ONLY, R.FROZEN_CONTEXT_ONLY,
                  R.FreezePolicy(error_nodes=True, attention=True)):
            self.assertFalse(p.is_local_replacement_model, p.tag())

    def test_error_nodes_alone_install_no_forward_hooks(self):
        # the distinction the plan branches on: an error node is written at the
        # ordinary substitution site, so it needs no attention/LN hook at all
        self.assertFalse(R.ERROR_NODES_ONLY.freezes_forward)
        self.assertTrue(R.ERROR_NODES_ONLY.needs_cache)
        self.assertTrue(R.FROZEN_CONTEXT_ONLY.freezes_forward)

    def test_a_policy_is_hashable_so_the_spec_stays_hashable(self):
        spec = R.ReplacementSpec(freeze=R.LOCAL_REPLACEMENT_MODEL)
        self.assertIsInstance(hash(spec), int)

    def test_a_null_control_may_not_also_restore_the_real_error(self):
        # the nulls delete structure from the error field; an error node puts the
        # real error back. Composing them would silently measure neither.
        with self.assertRaises(ValueError) as cm:
            R.ReplacementSpec(mode=R.NULL_SHUFFLED, freeze=R.ERROR_NODES_ONLY)
        self.assertIn("opposite interventions", str(cm.exception))

    def test_describe_names_the_construction(self):
        self.assertIn("LOCAL REPLACEMENT MODEL",
                      R.LOCAL_REPLACEMENT_MODEL.describe())
        self.assertIn("RECOMPUTED", R.NO_RESTORE.describe())


class TestPolicyProvenanceInRunTags(unittest.TestCase):
    """The four rungs replace the same layers with the same dictionaries in the
    same dtype. Without the policy in the tag they would be indistinguishable to
    the identity guard, which would then merge a skeleton's rows into an LRM's."""

    def test_each_rung_gets_its_own_run_tag(self):
        ad, _ = _fixture()
        tags = {p.tag(): ad.run_tag(R.Replacement(ad, freeze=p))
                for p in (R.NO_RESTORE, R.ERROR_NODES_ONLY, R.FROZEN_CONTEXT_ONLY,
                          R.LOCAL_REPLACEMENT_MODEL)}
        self.assertEqual(len(set(tags.values())), 4, tags)

    def test_restoring_nothing_leaves_every_pre_existing_tag_unchanged(self):
        # a regression guard with teeth: every run tag written before policies
        # existed must still mean exactly what it said, or the identity guard
        # would reject every cached record in the workspace
        ad, _ = _fixture()
        self.assertNotIn("freeze=", ad.run_tag(R.Replacement(ad)))
        self.assertIn(":freeze=errors+attn+ln",
                      ad.run_tag(R.Replacement.local_replacement_model(ad)))

    def test_the_named_constructor_is_the_full_policy(self):
        ad, _ = _fixture()
        self.assertTrue(
            R.Replacement.local_replacement_model(ad).freeze.is_local_replacement_model)


# --------------------------------------------------------------------- cache


class TestCleanRunCache(unittest.TestCase):
    def test_error_nodes_are_held_in_float32_whatever_the_stream_dtype(self):
        # not decoration: the identity `TC(x) + e == y` holds exactly only if the
        # subtraction and the addition are lossless on the 16-bit values involved
        c = R.CleanRunCache()
        y = torch.randn(2, 3, 8, dtype=torch.float16)
        recon = torch.randn(2, 3, 8, dtype=torch.float16)
        e = c.set_error(0, y, recon)
        self.assertEqual(e.dtype, torch.float32)
        self.assertEqual(c.error[0].dtype, torch.float32)

    def test_the_float32_error_node_reconstructs_a_16_bit_output_exactly(self):
        # the whole reason for the widening, as one assertion
        y = (torch.randn(4, 5, 16) / 3).half()
        recon = (torch.randn(4, 5, 16) / 3).half()
        e = R.CleanRunCache().set_error(0, y, recon)
        rebuilt = (recon.float() + e).to(y.dtype)
        self.assertTrue(torch.equal(rebuilt, y))

    def test_freeing_one_layer_leaves_the_others(self):
        c = R.CleanRunCache()
        for L in (0, 1):
            c.attn_pattern[L] = torch.zeros(1, 2, 3, 3)
            c.ln_scale[(L, "ln1")] = torch.zeros(1, 3, 1)
            c.set_error(L, torch.zeros(1, 3, 4), torch.zeros(1, 3, 4))
        c.free(0)
        self.assertEqual(c.layers, [1])
        self.assertNotIn((0, "ln1"), c.ln_scale)
        self.assertIn((1, "ln1"), c.ln_scale)
        c.free()
        self.assertEqual(c.layers, [])

    def test_nbytes_reports_a_real_footprint(self):
        c = R.CleanRunCache()
        c.attn_pattern[0] = torch.zeros(2, 4, 8, 8, dtype=torch.float16)
        self.assertEqual(c.nbytes(), 2 * 4 * 8 * 8 * 2)

    def test_a_missing_constant_refuses_rather_than_recomputing(self):
        # a policy that quietly degrades to "recompute it" is a DIFFERENT mode
        # wearing the requested mode's name — the exact confusion exp-06 removes
        c = R.CleanRunCache()
        with self.assertRaises(KeyError) as cm:
            c.require(R.ERROR_NODES_ONLY, 0)
        self.assertIn("skeleton wearing its name", str(cm.exception))
        with self.assertRaises(KeyError):
            c.require(R.FROZEN_CONTEXT_ONLY, 0)


# ----------------------------------------------------- THE LOAD-BEARING GATE


class TestLRMIdentity(unittest.TestCase):
    """The local replacement model must BE the base model on its own prompt.

    Not a measurement — a construction check. `TC(x_clean) + (y_clean -
    TC(x_clean)) == y_clean` pins the substituted stream to the clean trajectory
    at every layer, so by induction the logits reproduce the model.
    """

    def setUp(self):
        self.ad, self.toks = _fixture()
        self.base = self.ad.base_logits(self.ad.model, self.toks)
        self.cache = R.capture_clean_run(self.ad, self.ad.model, self.toks)

    def test_the_local_replacement_model_reproduces_the_base_model(self):
        lrm = _logits(self.ad, R.Replacement.local_replacement_model(self.ad),
                      self.toks, self.cache)
        self.assertLess(_maxdiff(lrm, self.base), 1e-4)

    def test_and_the_skeleton_alone_does_not_come_close(self):
        # without this the test above would pass on a fixture whose dictionaries
        # are so good that every rung agrees, and would be measuring nothing
        skel = _logits(self.ad, R.Replacement(self.ad), self.toks)
        self.assertGreater(_maxdiff(skel, self.base), 100 * 1e-4)

    def test_error_nodes_alone_already_achieve_the_identity(self):
        # stated as a test because it is the fact that makes the LRM gate blind
        # to a dead freeze, and therefore the reason `freeze-efficacy` exists
        se = _logits(self.ad, R.Replacement(self.ad, freeze=R.ERROR_NODES_ONLY),
                     self.toks, self.cache)
        self.assertLess(_maxdiff(se, self.base), 1e-4)

    def test_freezing_alone_does_not_achieve_it(self):
        sa = _logits(self.ad, R.Replacement(self.ad, freeze=R.FROZEN_CONTEXT_ONLY),
                     self.toks, self.cache)
        self.assertGreater(_maxdiff(sa, self.base), 1e-3)

    def test_freezing_with_the_real_mlps_is_a_no_op_on_a_clean_stream(self):
        # the wiring control. A frozen value EQUALS the recomputed one when the
        # stream is clean, so this must reproduce the model exactly — and it
        # fails loudly if a pattern or scale is written with the wrong shape,
        # the wrong head order, or from the wrong batch row.
        bf = _logits(self.ad, R.Replacement(self.ad, layers=[],
                                            freeze=R.FROZEN_CONTEXT_ONLY),
                     self.toks, self.cache)
        self.assertLess(_maxdiff(bf, self.base), 1e-5)

    def test_a_corrupted_error_node_breaks_the_identity_and_the_gate_says_so(self):
        # the gate has teeth: seed the exact bug its diagnosis names third (a
        # cache from a different batch) and confirm the construction stops
        # reproducing the model
        bad = R.capture_clean_run(self.ad, self.ad.model, self.toks)
        bad.error[1] = bad.error[1] + 0.5
        lrm = _logits(self.ad, R.Replacement.local_replacement_model(self.ad),
                      self.toks, bad)
        self.assertGreater(_maxdiff(lrm, self.base), 1e-3)

    def test_a_freezing_run_without_a_cache_refuses(self):
        with self.assertRaises(ValueError) as cm:
            _logits(self.ad, R.Replacement.local_replacement_model(self.ad), self.toks)
        self.assertIn("silently be the skeleton", str(cm.exception))


class TestFreezeEfficacy(unittest.TestCase):
    """The positive control the identity gate cannot be.

    A freeze is implemented by returning a cached tensor from a hook. If the
    library stops routing that value through the hook, the hook still registers,
    still fires, and does nothing — and the LRM gate still passes.
    """

    def setUp(self):
        self.ad, self.toks = _fixture()
        self.cache = R.capture_clean_run(self.ad, self.ad.model, self.toks)

    def test_freezing_measurably_changes_a_substituted_forward(self):
        skel = _logits(self.ad, R.Replacement(self.ad), self.toks)
        frozen = _logits(self.ad, R.Replacement(self.ad, freeze=R.FROZEN_CONTEXT_ONLY),
                         self.toks, self.cache)
        self.assertGreater(_maxdiff(frozen, skel), 1e-4)

    def test_the_freeze_hooks_actually_fire(self):
        fires: dict = {}
        model = self.ad.model
        for L in range(self.ad.n_layers):
            hs = self.ad.freeze_tap(model, L, self.cache, R.FROZEN_CONTEXT_ONLY,
                                    fires=fires)
            try:
                self.ad.block(model, L, self.ad.embed(model, self.toks))
            finally:
                for h in hs:
                    h.remove()
        self.assertEqual(fires.get("attn_pattern"), self.ad.n_layers)
        self.assertEqual(fires.get("ln1"), self.ad.n_layers)
        self.assertEqual(fires.get("ln2"), self.ad.n_layers)

    def test_a_dead_freeze_is_caught_by_the_verdict_function(self):
        r = G.check_freeze_efficacy(0.0)
        self.assertFalse(r.passed)
        self.assertIn("no-op", r.message)

    def test_a_hook_that_never_fired_is_caught_even_with_a_difference(self):
        r = G.check_freeze_efficacy(1.0, hook_fires={"attn_pattern": 0, "ln1": 3},
                                    expected_fires={"attn_pattern": 3, "ln1": 3})
        self.assertFalse(r.passed)
        self.assertIn("attn_pattern: 0 vs 3", r.message)

    def test_a_live_freeze_passes(self):
        r = G.check_freeze_efficacy(0.42, hook_fires={"attn_pattern": 3},
                                    expected_fires={"attn_pattern": 3})
        self.assertTrue(r.passed)


# --------------------------------------------------------------- the verdicts


class TestLRMIdentityVerdict(unittest.TestCase):
    def test_an_exact_reproduction_passes(self):
        r = G.check_lrm_base_identity(0.0, dtype="float16")
        self.assertTrue(r.passed)

    def test_a_real_gap_fails_and_says_the_object_is_wrong(self):
        r = G.check_lrm_base_identity(0.5, dtype="float16")
        self.assertFalse(r.passed)
        self.assertIn("different object", r.message)
        self.assertTrue(r.spec.diagnosis)

    def test_the_tolerance_is_taken_from_the_measured_dtype_floor(self):
        # "within fp16 tolerance" means nothing until something says what fp16
        # costs on this model; gate (i) on the same tokens is that something
        r = G.check_lrm_base_identity(2e-4, dtype="float16", base_vs_base_kl_max=1e-4)
        self.assertTrue(r.passed)
        self.assertAlmostEqual(r.tolerance, 1e-3)
        strict = G.check_lrm_base_identity(2e-4, dtype="float16")
        self.assertFalse(strict.passed)

    def test_non_finite_logits_are_an_overflow_not_a_construction_error(self):
        r = G.check_lrm_base_identity(float("inf"), all_finite=False)
        self.assertFalse(r.passed)
        self.assertIn("overflow", r.message)

    def test_the_identity_residual_is_reported_separately(self):
        # it shares every source of dtype error with the LRM, so a non-zero value
        # localises the bug to the construction rather than to the arithmetic
        r = G.check_lrm_base_identity(1e-6, dtype="float16", identity_residual=0.0)
        self.assertIn("identity residual", r.message)
        self.assertEqual(r.detail["identity_residual"], 0.0)


class TestRequireLRMIdentity(unittest.TestCase):
    """Non-blocking in the registry, mandatory for a run that claims the name."""

    def test_an_unrun_gate_blocks_a_local_replacement_model_report(self):
        rep = G.GateReport()
        with self.assertRaises(G.GateFailure):
            R.require_lrm_identity(rep)

    def test_a_failed_gate_blocks(self):
        rep = G.GateReport()
        rep.record(G.check_lrm_base_identity(0.9, dtype="float16"))
        with self.assertRaises(G.GateFailure):
            R.require_lrm_identity(rep)

    def test_a_passed_gate_lets_the_run_through(self):
        rep = G.GateReport()
        rep.record(G.check_lrm_base_identity(0.0, dtype="float16"))
        R.require_lrm_identity(rep)

    def test_it_does_not_block_an_ordinary_substitution_run(self):
        # the gate is UNRUN for every run that restores nothing, and `require()`
        # must stay meaningful for those
        rep = G.GateReport()
        self.assertNotIn(G.LRM_IDENTITY_GATE,
                         [r.id for r in rep.blocking_failures])
        self.assertNotIn(G.FREEZE_EFFICACY_GATE,
                         [r.id for r in rep.blocking_failures])

    def test_a_dead_freeze_blocks_a_frozen_mode_report(self):
        # the companion refusal, and the one that protects the LABELS: the
        # identity gate passes on error nodes alone, so without this a dead
        # freeze would ship rows labelled "attention frozen" that are not
        rep = G.GateReport()
        rep.record(G.check_lrm_base_identity(0.0, dtype="float16"))
        rep.record(G.check_freeze_efficacy(0.0))
        R.require_lrm_identity(rep)                 # the identity is fine...
        with self.assertRaises(G.GateFailure):      # ...and the labels are not
            R.require_freeze_efficacy(rep)

    def test_a_live_freeze_lets_the_run_through(self):
        rep = G.GateReport()
        rep.record(G.check_freeze_efficacy(0.42))
        R.require_freeze_efficacy(rep)


# ------------------------------------------------------------ the declaration


class TestFreezeCapabilities(unittest.TestCase):
    def test_an_adapter_that_cannot_freeze_refuses_a_frozen_run(self):
        # a frozen run on an adapter that cannot freeze is a RECOMPUTED run
        # wearing the wrong name; refusing is the only honest option
        ad = adapters.get("toy")                       # no attention at all
        self.assertFalse(ad.freeze_capabilities().attention)
        with self.assertRaises(NotImplementedError) as cm:
            ad.freeze_tap(ad.model, 0, R.CleanRunCache(), R.FROZEN_CONTEXT_ONLY)
        self.assertIn("wearing the wrong name", str(cm.exception))

    def test_but_it_still_supports_the_error_node_rung(self):
        # error nodes are written at the ordinary substitution site, so they need
        # nothing the plain contract does not already provide
        ad = adapters.get("toy")
        toks, _ = ad.synthetic_corpus(n=4, seq_len=8, seed=0, n_probes=0)
        cache = R.capture_clean_run(ad, ad.model, toks)
        se = ad.replaced_logits(ad.model, toks,
                                R.Replacement(ad, freeze=R.ERROR_NODES_ONLY),
                                clean_cache=cache)
        self.assertLess(_maxdiff(se, ad.base_logits(ad.model, toks)), 1e-4)

    def test_declared_sites_must_be_real_sites(self):
        with self.assertRaises(ValueError):
            AB.FreezeCapabilities(ln_sites=("ln_middle",))

    def test_gemma_declares_all_four_sandwich_norms(self):
        # gemma-2 is sandwich-normed: a run that froze only the pre-norms would
        # leave half its normalisation recomputing and still call itself frozen
        from signoff.adapters import gemma_scope

        caps = gemma_scope.GemmaScopeAdapter.freeze_capabilities_spec
        self.assertTrue(caps.attention)
        self.assertEqual(set(caps.ln_sites),
                         {"ln1", "ln1_post", "ln2", "ln2_post", "final"})

    def test_the_capability_declaration_reaches_the_report(self):
        ad, _ = _fixture()
        self.assertIn("freeze_capabilities", ad.contract())

    def test_the_seam_methods_are_stamped_as_optional_overrides(self):
        for name in ("capture_tap", "freeze_tap", "final_norm_tap"):
            self.assertIn(name, AB.OPTIONAL_OVERRIDES)


if __name__ == "__main__":
    unittest.main()
