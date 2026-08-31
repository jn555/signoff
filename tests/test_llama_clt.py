"""The circuit-tracer CROSS-LAYER adapter: declarations (CI) + the cross-layer path.

Two halves.

DECLARATIONS — no weights, no network.  They check that the adapter declares a
THIRD tap convention (`hook_resid_mid`, not normalised at all, against
gemma-scope's pre-gain and qwen's post-gain), that its dtype table claims
nothing it has not measured, and that the artifact's shape/footprint arithmetic
matches the released checkpoint headers.

THE CROSS-LAYER PATH — exercised on the synthetic toy fixture, so it runs on a
CPU with no weights.  `CrossLayerPlan` and the overridden `measure_fvu` are the
REAL code; only the dictionaries and the model are toys.  The two tests that
matter are the degenerate reduction (a CLT whose off-diagonal planes are zero
must reproduce `PerLayerPlan` exactly) and upstream-only dependence (adding a
read layer must not move the FVU of any write layer above it).

The real-weight gates are `SIGNOFF_LOCAL=1` only and, as of 2026-08-31, cannot
run on any host without HF access to the gated `meta-llama/Llama-3.2-1B` and
~20 GB of free disk.  See `llama_clt.py`, TIER.
"""

from __future__ import annotations

import os
import unittest

import torch

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


# ------------------------------------------------------------- declarations


class TestDeclarationsWithoutWeights(unittest.TestCase):
    def setUp(self):
        from signoff.adapters import llama_clt

        self.M = llama_clt
        self.A = llama_clt.LlamaCltAdapter

    def test_pins_model_and_dictionary_revisions(self):
        self.assertTrue(self.A.identity.fully_pinned)
        self.assertEqual(len(self.A.identity.model_revision), 40)
        self.assertEqual(len(self.A.identity.dict_revision), 40)
        self.assertEqual(self.A.identity.dict_repo, "mntss/clt-llama-3.2-1b-524k")

    def test_declares_a_THIRD_tap_convention(self):
        # gemma-scope: PRE-GAIN normalised. qwen3-mwhanna: POST-GAIN normalised.
        # This artifact reads the residual stream itself. If these ever collapse
        # to two conventions, someone has copied a tap between releases.
        from signoff.adapters import gemma_scope, qwen_mwhanna

        self.assertEqual(self.A.taps.input_hook, "blocks.{L}.hook_resid_mid")
        self.assertIn("NOT NORMALISED", self.A.taps.input_convention)
        self.assertIn("PRE-GAIN", gemma_scope.GemmaScopeAdapter.taps.input_convention)
        self.assertIn("POST-GAIN", qwen_mwhanna.QwenMwhannaAdapter.taps.input_convention)
        conventions = {
            self.A.taps.input_hook,
            gemma_scope.GemmaScopeAdapter.taps.input_hook,
            qwen_mwhanna.QwenMwhannaAdapter.taps.input_hook,
        }
        self.assertEqual(len(conventions), 3)
        # same WRITE point as gemma even though the read point differs
        self.assertIn("hook_mlp_out", self.A.taps.output_hook)

    def test_bos_is_declared_and_opposite_to_qwen(self):
        from signoff.adapters import qwen_mwhanna

        self.assertEqual(self.A.tokenization.bos_id, 128000)
        self.assertIsNone(qwen_mwhanna.QwenMwhannaAdapter.tokenization.bos_id)
        self.assertTrue(self.A.tokenization.declared)
        self.assertTrue(self.A.tokenization.exclude_position_0)

    def test_dtype_table_claims_nothing_it_has_not_measured(self):
        p = self.A.dtype_policy
        self.assertEqual(p.default, "float32")           # gate (iii) is vacuous there
        self.assertNotIn("float16", p.allowed)           # unmeasured, not offered
        for dtype, note in p.measured.items():
            self.assertIn("UNMEASURED", note, dtype)
            self.assertNotIn("PASS", note, dtype)
        self.assertIn("NO gate-(iii) measurement exists", p.notes)

    def test_the_delegation_stamp_lists_all_four_overrides(self):
        from signoff.adapters.base import (
            overridden_measurement_methods,
            self_reported_gates,
        )

        self.assertEqual(overridden_measurement_methods(self.A),
                         ["measure_fvu", "verify_provenance", "substitution_plan", "run_tag"])
        marked = self_reported_gates(self.A)
        for gid in ("ii-fvu-sanity", "provenance-freeze", "identity-guard",
                    "checkpoint-binding"):
            self.assertIn(gid, marked)

    def test_registry_entry(self):
        from signoff import adapters

        d = adapters.describe()
        self.assertEqual(d["llama32-clt-mntss"]["tier"], "local")
        self.assertIn("CROSS-LAYER", d["llama32-clt-mntss"]["description"])

    def test_config_freeze_matches_the_declared_taps(self):
        # The freeze is what verify_provenance re-derives; if it drifted from
        # the TapSpec, the gate would pass while the tap sentence lied.
        f = self.M.CONFIG_FREEZE
        self.assertIn(f["feature_input_hook"], self.A.taps.input_hook)
        self.assertIn(f["feature_output_hook"], self.A.taps.output_hook)
        self.assertEqual(f["model_kind"], "cross_layer_transcoder")
        self.assertEqual(f["model_name"], self.A.identity.model_repo)

    def test_footprint_arithmetic_matches_the_released_headers(self):
        M = self.M
        # 16 read layers x 32768 x 2048 bf16 encoders = 2.15 GB ...
        full = M.footprint_bytes(range(16))
        self.assertEqual(full["n_encoders"], 16)
        # ... and SUM_L (16 - L) = 136 decoder planes = 18.25 GB
        self.assertEqual(full["n_planes"], 136)
        self.assertAlmostEqual(full["resident"] / 1e9, 20.4, places=1)
        self.assertEqual(full["resident"], full["download"])   # a full run needs it all
        # the smallest genuinely cross-layer smoke: 2 encoders + 3 planes
        smoke = M.footprint_bytes([0, 1])
        self.assertEqual((smoke["n_encoders"], smoke["n_planes"]), (2, 3))
        self.assertLess(smoke["resident"], 0.7e9)
        # but W_dec_0 and W_dec_1 still have to land on disk whole
        self.assertGreater(smoke["download"], 4e9)
        # float32 — this adapter's default — is twice the RAM and the same disk
        f32 = M.footprint_bytes([0, 1], bytes_per_param=4)
        self.assertEqual(f32["resident"], 2 * smoke["resident"])
        self.assertEqual(f32["download"], smoke["download"])

    def test_bundle_ids_name_the_read_span(self):
        b = self.M.bundle_id
        self.assertEqual(b([3], 3), "3")            # a lone diagonal site
        self.assertEqual(b([1], 3), "1->3")
        self.assertEqual(b([0, 1, 2], 2), "0..2->2")
        with self.assertRaises(ValueError):
            b([], 2)

    def test_gate_ii_accepts_the_bundle_keys(self):
        from signoff import gates as G

        r = G.check_fvu_sanity({"0": 0.2, "0..1->1": 0.3, "0..2->2": 0.35})
        self.assertTrue(r.passed, r.message)


class TestPlanDeclarations(unittest.TestCase):
    """The plan is constructible with no weights: it touches no model."""

    def setUp(self):
        from signoff.adapters.llama_clt import LlamaCltAdapter

        self.ad = LlamaCltAdapter()

    def test_the_adapter_supplies_a_cross_layer_plan(self):
        from signoff.adapters.llama_clt import CrossLayerPlan
        from signoff.replacement import PerLayerPlan, Replacement

        plan = Replacement(self.ad, layers=[0, 1, 2]).plan()
        self.assertIsInstance(plan, CrossLayerPlan)
        self.assertNotIsInstance(plan, PerLayerPlan)

    def test_sites_are_every_read_write_pair_with_read_before_write(self):
        from signoff.replacement import Replacement

        plan = Replacement(self.ad, layers=[0, 1, 2]).plan()
        self.assertEqual([s.id for s in plan.sites()],
                         ["0", "0->1", "1", "0->2", "1->2", "2"])
        self.assertEqual(list(plan.bundles()),
                         ["0", "0..1->1", "0..2->2"])

    def test_a_prefix_reproduces_the_artifact_and_says_so(self):
        from signoff.replacement import Replacement

        plan = Replacement(self.ad, layers=[0, 1, 2]).plan()
        self.assertTrue(plan.is_prefix)
        self.assertIn("reproduces the artifact exactly", plan.describe())

    def test_a_NON_prefix_set_is_a_truncated_artifact_and_says_so(self):
        from signoff.replacement import Replacement

        plan = Replacement(self.ad, layers=[8, 9]).plan()
        self.assertFalse(plan.is_prefix)
        self.assertIn("TRUNCATED", plan.describe())
        # write layer 9 sums only read layers in the set, not 0..9
        self.assertEqual(plan.contributing(9), [8, 9])

    def test_run_tag_distinguishes_a_prefix_run_from_a_truncated_one(self):
        from signoff.replacement import Replacement

        prefix = self.ad.run_tag(Replacement(self.ad, layers=[0, 1]))
        trunc = self.ad.run_tag(Replacement(self.ad, layers=[8, 9]))
        self.assertIn(":clt=prefix", prefix)
        self.assertIn(":clt=truncated", trunc)
        self.assertNotEqual(prefix, trunc)
        self.assertIn(":clt=prefix", self.ad.run_tag(Replacement(self.ad, layers="all")))

    def test_a_read_layer_cannot_write_upstream(self):
        from signoff.replacement import Replacement

        plan = Replacement(self.ad, layers=[0, 1, 2]).plan()
        self.assertEqual(plan.contributing(0), [0])       # nothing above feeds layer 0
        self.assertEqual(plan.contributing(2), [0, 1, 2])


# ------------------------------------------------- the cross-layer path, on a toy


D = 16  # the toy fixture's d_model


class _ToyCltLayer:
    """A CLT read layer with identity 'features', so the planes ARE the artifact.

    `cross` scales the off-diagonal planes.  At `cross = 0` every W > L plane is
    exactly zero and the CLT degenerates to a per-layer transcoder — which is
    the reduction the tests below check the plan against.
    """

    def __init__(self, layer: int, n_layers: int, cross: float = 0.0, seed: int = 0):
        g = torch.Generator().manual_seed(1000 * seed + layer)
        self.layer = layer
        self.n_layers = n_layers
        self.b_dec = torch.randn(D, generator=g) * 0.01
        self.M = {}
        for W in range(layer, n_layers):
            s = 0.9 if W == layer else cross
            self.M[W] = torch.randn(D, D, generator=g) * (s / D ** 0.5)

    def encode(self, x):
        return x

    def contribution(self, a, W):
        return a @ self.M[W]

    def forward(self, x):
        """The diagonal-only reconstruction — what `PerLayerPlan` would write."""
        return self.contribution(self.encode(x), self.layer) + self.b_dec


def _clt_toy(cross: float = 0.0):
    """The toy adapter, wearing the CLT adapter's real plan and real measure_fvu."""
    from signoff.adapters.llama_clt import CrossLayerPlan, LlamaCltAdapter
    from signoff.adapters.toy import ToyAdapter
    from signoff.replacement import ReplacementSpec

    class _CltToy(ToyAdapter):
        name = "toy-clt"
        substitution_plan = LlamaCltAdapter.substitution_plan   # the real seam
        measure_fvu = LlamaCltAdapter.measure_fvu               # the real measurement

        def load_dictionary(self, layer, device, dtype):
            return _ToyCltLayer(layer, self.n_layers, cross=cross)

    ad = _CltToy()
    assert isinstance(ad.substitution_plan(ReplacementSpec()), CrossLayerPlan)
    return ad


def _perlayer_toy(cross: float = 0.0):
    """The same dictionaries under the DEFAULT per-layer plan, for comparison."""
    from signoff.adapters.toy import ToyAdapter

    class _PlToy(ToyAdapter):
        name = "toy-perlayer"

        def load_dictionary(self, layer, device, dtype):
            return _ToyCltLayer(layer, self.n_layers, cross=cross)

    return _PlToy()


class TestCrossLayerForward(unittest.TestCase):
    def setUp(self):
        from signoff.adapters.toy import ToyAdapter

        self.toks, _ = ToyAdapter().synthetic_corpus(n=4, seq_len=8, n_probes=0)

    def test_zero_off_diagonal_planes_reduce_EXACTLY_to_the_per_layer_plan(self):
        """The degenerate case, and the strongest check on the wiring.

        A CLT whose W > L planes are all zero writes, at every layer, exactly
        what a per-layer transcoder with the same diagonal would write. Two
        independent code paths (`CrossLayerPlan.replaced_logits` and
        `PerLayerPlan.replaced_logits`) must therefore agree. If the plan
        mis-indexed a plane, dropped `b_dec`, or added it once per contributing
        read layer instead of once per write layer, this is what breaks.
        """
        from signoff.replacement import PerLayerPlan, Replacement

        clt, pl = _clt_toy(cross=0.0), _perlayer_toy(cross=0.0)
        rc = Replacement(clt, layers="all")
        rp = Replacement(pl, layers="all")
        self.assertIsInstance(rp.plan(), PerLayerPlan)
        a = clt.replaced_logits(clt.model, self.toks, rc)
        b = pl.replaced_logits(pl.model, self.toks, rp)
        torch.testing.assert_close(a, b, atol=1e-6, rtol=0)

    def test_a_NON_zero_cross_layer_plane_actually_fires(self):
        """The other half: if the off-diagonal planes were being ignored, the
        test above would pass for the wrong reason."""
        from signoff.replacement import Replacement

        clt, pl = _clt_toy(cross=0.6), _perlayer_toy(cross=0.6)
        a = clt.replaced_logits(clt.model, self.toks, Replacement(clt, layers="all"))
        b = pl.replaced_logits(pl.model, self.toks, Replacement(pl, layers="all"))
        self.assertGreater(float((a - b).abs().max()), 1e-3)

    def test_layer_0_is_unaffected_by_the_cross_layer_planes(self):
        """A CLT writes DOWNSTREAM only: nothing feeds write layer 0 but read
        layer 0, so a run restricted to layer 0 cannot see `cross` at all."""
        from signoff.replacement import Replacement

        a = _clt_toy(cross=0.0)
        b = _clt_toy(cross=0.6)
        la = a.replaced_logits(a.model, self.toks, Replacement(a, layers=[0]))
        lb = b.replaced_logits(b.model, self.toks, Replacement(b, layers=[0]))
        torch.testing.assert_close(la, lb, atol=1e-6, rtol=0)

    def test_the_null_forward_still_keeps_the_true_output(self):
        from signoff.replacement import Replacement

        ad = _clt_toy(cross=0.6)
        zero = {L: torch.zeros(4, 8, ad.d_model) for L in range(ad.n_layers)}
        out = ad.replaced_logits(ad.model, self.toks, Replacement.null(ad, "shuffled"),
                                 noise=zero)
        torch.testing.assert_close(out, ad.base_logits(ad.model, self.toks),
                                   atol=1e-6, rtol=0)

    def test_the_null_forward_refuses_without_a_noise_field(self):
        from signoff.replacement import Replacement

        ad = _clt_toy(cross=0.6)
        with self.assertRaises(ValueError):
            ad.replaced_logits(ad.model, self.toks, Replacement.null(ad, "shuffled"))


class TestCrossLayerFvu(unittest.TestCase):
    def setUp(self):
        from signoff.adapters.toy import ToyAdapter

        self.toks, _ = ToyAdapter().synthetic_corpus(n=4, seq_len=8, n_probes=0)

    def test_fvu_is_keyed_by_write_layer_bundle_and_names_its_reads(self):
        ad = _clt_toy(cross=0.3)
        table = ad.measure_fvu(self.toks, [0, 1, 2], batch=2)
        self.assertEqual(sorted(table), ["0", "0..1->1", "0..2->2"])
        self.assertEqual(table["0..2->2"]["n_read_layers"], 3)
        self.assertEqual(table["0..2->2"]["write_layer"], 2)
        for row in table.values():
            self.assertIn("fvu_global", row)
            self.assertIn("fvu_token_median", row)

    def test_adding_a_read_layer_does_not_move_an_upstream_write_layer(self):
        """Upstream-only dependence, measured rather than assumed.

        FVU at write layer 0 sums over read layers <= 0, so it must be identical
        whether the run covers {0} or {0, 1, 2}. This is what makes a PREFIX run
        a faithful partial audit of the artifact.
        """
        ad = _clt_toy(cross=0.3)
        one = ad.measure_fvu(self.toks, [0], batch=2)
        three = ad.measure_fvu(self.toks, [0, 1, 2], batch=2)
        self.assertAlmostEqual(one["0"]["fvu_global"], three["0"]["fvu_global"], places=6)

    def test_a_truncated_set_measures_a_DIFFERENT_artifact(self):
        """The flip side, and the reason `run_tag` stamps `clt=truncated`:
        write layer 2 read alone is missing the contributions of read layers
        0 and 1, so its FVU is not the artifact's FVU at layer 2."""
        ad = _clt_toy(cross=0.6)
        trunc = ad.measure_fvu(self.toks, [2], batch=2)
        prefix = ad.measure_fvu(self.toks, [0, 1, 2], batch=2)
        self.assertNotAlmostEqual(trunc["2"]["fvu_global"],
                                  prefix["0..2->2"]["fvu_global"], places=3)

    def test_fvu_does_not_depend_on_the_batch_size(self):
        """The global-mean denominator is taken over the whole subset, exactly
        as the base method does it — a per-batch mean would not be."""
        ad = _clt_toy(cross=0.3)
        a = ad.measure_fvu(self.toks, [0, 1], batch=1)
        b = ad.measure_fvu(self.toks, [0, 1], batch=4)
        for k in a:
            self.assertAlmostEqual(a[k]["fvu_global"], b[k]["fvu_global"], places=5)

    def test_the_measured_reconstruction_IS_the_one_the_forward_writes(self):
        """Ties the two paths together.

        `measure_fvu` reconstructs on a CLEAN pass and `replaced_logits` on the
        substituted one, so they only have to agree where nothing upstream was
        replaced — which is exactly the set `{0}`. There, the `yhat` the forward
        captures must reproduce the FVU the gate reports, bit for bit. This is
        what would catch the two paths drifting apart (a bias added in one and
        not the other, or the contributions summed in a different dtype).
        """
        from signoff import stats as St
        from signoff.replacement import Replacement

        ad = _clt_toy(cross=0.6)
        cap: dict = {}
        rep = Replacement(ad, layers=[0])
        ad.replaced_logits(ad.model, self.toks, rep, capture=cap)
        self.assertEqual(sorted(cap), ["0"])
        y = cap["0"]["y"].float()
        yhat = cap["0"]["yhat"].float()
        flat = y.reshape(-1, ad.d_model)
        num = (yhat - y).pow(2).sum(-1)
        den = (flat - flat.mean(0)).pow(2).sum(-1).reshape(num.shape)
        by_hand = St.dual_fvu(num, den)
        measured = ad.measure_fvu(self.toks, [0], batch=4)["0"]
        self.assertAlmostEqual(by_hand["fvu_global"], measured["fvu_global"], places=6)

    def test_gate_ii_runs_on_the_cross_layer_table(self):
        ad = _clt_toy(cross=0.3)
        verdict, table = ad.gate_fvu(self.toks, [0, 1, 2], batch=2)
        self.assertEqual(sorted(table), ["0", "0..1->1", "0..2->2"])
        self.assertIn(verdict.status, ("pass", "fail"))
        self.assertTrue(verdict.detail["per_site"])


# ------------------------------------------------------- real weights, local only


@slow
@local_only
@unittest.skipUnless(LOCAL, "needs gated meta-llama weights + ~20 GB of CLT; "
                            "set SIGNOFF_LOCAL=1")
class TestLlamaCltGates(unittest.TestCase):
    """Blocked as of 2026-08-31 on TWO things, both recorded in llama_clt.py TIER:
    `meta-llama/Llama-3.2-1B` is gated (403 for this host's token) and the
    dictionary is 20.4 GB.  `{0, 1}` is the smallest FAITHFUL smoke — a prefix,
    so its FVU at both write layers is the artifact's own."""

    def test_provenance_matches_the_live_release(self):
        from signoff import adapters

        r = adapters.get("llama32-clt-mntss").verify_provenance()
        self.assertTrue(r.passed, r.message)

    def test_gates_i_and_ii_on_a_two_layer_PREFIX_smoke(self):
        from signoff import adapters
        from signoff.adapters.llama_clt import footprint_bytes

        ad = adapters.get("llama32-clt-mntss", dtype="bfloat16")
        fp = ad.footprint([0, 1])
        print(f"\n  footprint: {fp['resident'] / 1e9:.2f} GB resident (bf16), "
              f"{fp['download'] / 1e9:.2f} GB on disk, plus ~2.5 GB of model")
        self.assertIsNotNone(footprint_bytes([0, 1]))
        toks = torch.full((2, 32), ad.tokenization.bos_id, dtype=torch.long)
        toks[:, 1:] = torch.randint(100, 1000, (2, 31))
        self.assertTrue(ad.gate_base_vs_base(toks, n=2).passed)
        verdict, table = ad.gate_fvu(toks, [0, 1], batch=2)
        self.assertEqual(sorted(table), ["0", "0..1->1"])
        self.assertTrue(verdict.passed, verdict.message)


if __name__ == "__main__":
    unittest.main()
