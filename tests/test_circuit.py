"""Circuit mode, on a CPU, with no weights.

Everything here runs against `adapters.toy_circuit()` — a deterministic fixture
with real multi-head attention.  The questions these tests answer are the ones
that decide whether a circuit faithfulness number means anything:

  * does the keep-set actually control which heads are ablated?
  * is the ablation VALUE the checker (does changing the policy change the
    answer), and is it stamped?
  * does the plan refuse the ways it can silently be wrong — no values, mixed
    sequence lengths, an uncalibrated group, a stale digest?
  * does the seam hold — same runner, same metrics, same gates, and a clean
    circuit-mode pass that is still bit-exactly the model's own forward?
"""

from __future__ import annotations

import unittest

import torch

from signoff import adapters, circuit as CIR, gates as G, metrics as M
from signoff.adapters import toy as toy_mod
from signoff.replacement import CIRCUIT, PerLayerPlan, Replacement, ReplacementSpec

N_L, N_H = toy_mod.C_N_LAYERS, toy_mod.C_N_HEADS


def _tiny_circuit(name="toy-claim", heads=((0, 0), (1, 1), (2, 2)), mlps=None):
    return CIR.CircuitSpec.from_classes(
        name, {"kept": [list(h) for h in heads]},
        n_layers=N_L, n_heads=N_H, mlps=mlps,
        source="synthetic fixture", notes="not a claim about anything",
    )


def _toks(n=4, T=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, toy_mod.C_D_VOCAB, (n, T), generator=g)


def _mean_values(ad, circ, toks, *, key=0, calibration=None):
    """Calibrate mean z (and MLP out) over `toks`, read-only, grouped under one key."""
    model = ad.model
    zs: dict[int, torch.Tensor] = {}
    ys: dict[int, torch.Tensor] = {}
    resid = ad.embed(model, toks)
    for L in range(ad.n_layers):
        zst, zh = ad.head_tap(model, L, write_fn=None)
        mst, mh = ad.tap(model, L, replace_fn=None)
        try:
            resid = ad.block(model, L, resid)
        finally:
            for h in list(zh) + list(mh):
                h.remove()
        zs[L] = zst["z"].mean(0)          # (T, H, dh)
        ys[L] = mst["y"].mean(0)          # (T, d_model)
    return CIR.MeanAblationValues(
        {key: zs}, {key: ys},
        calibration=calibration or dict(distribution="synthetic", n=int(toks.shape[0]),
                                        seed=0, grouping="single-group"),
    )


class TestCircuitSpec(unittest.TestCase):
    def test_keep_set_determines_the_ablated_set(self):
        c = _tiny_circuit()
        self.assertEqual(c.n_kept_heads, 3)
        self.assertEqual(len(c.ablated_heads()), N_L * N_H - 3)
        self.assertEqual(c.ablated_heads_at(0), [h for h in range(N_H) if h != 0])
        # mlps=None keeps every MLP, so no layer is touched on the MLP side
        self.assertEqual(c.ablated_mlps(), [])
        self.assertEqual(c.touched_layers(), list(range(N_L)))

    def test_a_full_keep_set_touches_nothing(self):
        c = CIR.CircuitSpec.from_classes(
            "everything", {"all": [[L, h] for L in range(N_L) for h in range(N_H)]},
            n_layers=N_L, n_heads=N_H)
        self.assertEqual(c.ablated_heads(), [])
        self.assertEqual(c.touched_layers(), [])

    def test_mlp_keep_set_is_expressible(self):
        c = _tiny_circuit(mlps=[0])
        self.assertEqual(c.ablated_mlps(), [1, 2])

    def test_the_digest_tracks_membership_and_nothing_else(self):
        a = _tiny_circuit()
        b = _tiny_circuit(name="renamed-but-same-heads")
        # the name IS part of identity: two claims with the same heads but
        # different provenance are different claims
        self.assertNotEqual(a.digest(), b.digest())
        c = CIR.CircuitSpec.from_classes(
            "toy-claim", {"kept": [[0, 0], [1, 1], [2, 2]]},
            n_layers=N_L, n_heads=N_H, notes="different note, same claim")
        self.assertEqual(a.digest(), c.digest())

    def test_classes_that_disagree_with_the_keep_set_are_rejected(self):
        with self.assertRaises(ValueError):
            CIR.CircuitSpec(name="x", n_layers=N_L, n_heads=N_H,
                            heads=((0, 0), (1, 1)), mlps=tuple(range(N_L)),
                            classes=(("a", ((0, 0),)),))

    def test_out_of_range_heads_are_rejected(self):
        with self.assertRaises(ValueError):
            _tiny_circuit(heads=((0, 0), (N_L, 0)))


class TestReplacementWiring(unittest.TestCase):
    def test_circuit_mode_and_a_spec_must_come_together(self):
        with self.assertRaises(ValueError):
            ReplacementSpec(mode=CIRCUIT)
        with self.assertRaises(ValueError):
            ReplacementSpec(mode="substitute", circuit=_tiny_circuit())

    def test_the_layer_set_is_derived_from_the_keep_set(self):
        ad = adapters.toy_circuit()
        c = _tiny_circuit(heads=[(L, h) for L in range(N_L) for h in range(N_H)
                                 if L != 1])
        r = Replacement.circuit(ad, c)
        self.assertEqual(r.layers, [1])
        self.assertFalse(r.spec.is_null)
        self.assertTrue(r.spec.is_circuit)

    def test_the_adapter_picks_the_plan_from_the_spec(self):
        ad = adapters.toy_circuit()
        self.assertIsInstance(Replacement.circuit(ad, _tiny_circuit()).plan(),
                              CIR.CircuitPlan)
        self.assertIsInstance(Replacement(ad, layers="all").plan(), PerLayerPlan)

    def test_the_run_tag_separates_two_circuits_over_the_same_layers(self):
        ad = adapters.toy_circuit()
        a = Replacement.circuit(ad, _tiny_circuit(heads=((0, 0), (1, 1), (2, 2))))
        b = Replacement.circuit(ad, _tiny_circuit(heads=((0, 1), (1, 2), (2, 3))))
        self.assertEqual(a.layers, b.layers)
        self.assertNotEqual(ad.run_tag(a), ad.run_tag(b))

    def test_an_adapter_without_a_head_tap_says_so(self):
        ad = adapters.get("toy")            # no attention at all
        with self.assertRaises(NotImplementedError) as cm:
            ad.head_tap(ad.model, 0)
        self.assertIn("circuit mode", str(cm.exception))
        self.assertFalse(ad.contract()["circuit_capable"])
        self.assertTrue(adapters.toy_circuit().contract()["circuit_capable"])


class TestCircuitForward(unittest.TestCase):
    def setUp(self):
        self.ad = adapters.toy_circuit()
        self.toks = _toks()
        self.cal = _toks(n=16, T=8, seed=7)
        self.values = _mean_values(self.ad, None, self.cal)

    def _logits(self, circ, values=None, keys=None):
        r = Replacement.circuit(self.ad, circ)
        return r.plan().replaced_logits(
            self.ad, self.ad.model, self.toks,
            ablation=(self.values if values is None else values),
            keys=keys or [0] * int(self.toks.shape[0]))

    def test_keeping_everything_reproduces_the_model_exactly(self):
        # the strongest statement of "the plan is the model plus interventions":
        # with an empty ablated set the circuit forward IS the clean forward.
        circ = CIR.CircuitSpec.from_classes(
            "everything", {"all": [[L, h] for L in range(N_L) for h in range(N_H)]},
            n_layers=N_L, n_heads=N_H)
        ref = self.ad.base_logits(self.ad.model, self.toks)
        got = self._logits(circ)
        self.assertLess(float(M.per_position_kl(ref, got)[:, 1:].max()), 1e-9)

    def test_ablating_everything_is_not_the_model(self):
        circ = CIR.CircuitSpec.from_classes("nothing", {"none": []},
                                            n_layers=N_L, n_heads=N_H)
        ref = self.ad.base_logits(self.ad.model, self.toks)
        got = self._logits(circ)
        self.assertGreater(float(M.per_position_kl(ref, got)[:, 1:].max()), 1e-3)

    def test_only_the_named_heads_survive(self):
        """The keep-set must be the thing doing the work, not the layer list.

        Two circuits that touch the SAME layers but keep different heads must
        produce different logits — otherwise the head index is decorative.
        """
        a = self._logits(_tiny_circuit(heads=((0, 0), (1, 1), (2, 2))))
        b = self._logits(_tiny_circuit(heads=((0, 3), (1, 3), (2, 3))))
        self.assertGreater(float((a - b).abs().max()), 1e-4)

    def test_ablating_an_mlp_changes_the_answer(self):
        heads = [(L, h) for L in range(N_L) for h in range(N_H)]
        keep_all_mlp = self._logits(_tiny_circuit(heads=heads, mlps=None))
        drop_mlp1 = self._logits(_tiny_circuit(heads=heads, mlps=[0, 2]))
        self.assertGreater(float((keep_all_mlp - drop_mlp1).abs().max()), 1e-4)

    def test_the_forward_leaves_no_hooks_behind(self):
        self._logits(_tiny_circuit())
        ref = self.ad.base_logits(self.ad.model, self.toks)
        again = self.ad.base_logits(self.ad.model, self.toks)
        self.assertTrue(torch.equal(ref, again))
        for b in self.ad.model.blocks:
            self.assertIsNone(b.attn.write_fn)
            self.assertIsNone(b.attn.z_state)
            self.assertIsNone(b.replace_fn)

    def test_it_refuses_to_run_without_ablation_values(self):
        r = Replacement.circuit(self.ad, _tiny_circuit())
        with self.assertRaises(ValueError) as cm:
            r.plan().replaced_logits(self.ad, self.ad.model, self.toks, ablation=None)
        self.assertIn("not a value", str(cm.exception))

    def test_an_uncalibrated_group_is_an_error_not_a_guess(self):
        with self.assertRaises(KeyError) as cm:
            self._logits(_tiny_circuit(), keys=["never-calibrated"] * 4)
        self.assertIn("does not cover this batch", str(cm.exception))


class TestPolicyIsTheChecker(unittest.TestCase):
    """Miller et al. 2407.08734: the ablation policy moves the answer.

    If mean- and resample-ablation gave the same number, reporting both would
    be ceremony.  This test is the fixture-scale statement that they do not.
    """

    def test_mean_and_resample_disagree(self):
        ad = adapters.toy_circuit()
        toks = _toks()
        circ = _tiny_circuit()
        mean_v = _mean_values(ad, circ, _toks(n=16, T=8, seed=7))

        cf = _toks(n=4, T=8, seed=99)
        model = ad.model
        zs, ys = {}, {}
        resid = ad.embed(model, cf)
        for L in range(ad.n_layers):
            zst, zh = ad.head_tap(model, L, write_fn=None)
            mst, mh = ad.tap(model, L, replace_fn=None)
            try:
                resid = ad.block(model, L, resid)
            finally:
                for h in list(zh) + list(mh):
                    h.remove()
            zs[L], ys[L] = zst["z"], mst["y"]
        res_v = CIR.ResampleAblationValues(
            zs, ys, cf_tokens=cf,
            calibration=dict(distribution="synthetic", seed=99, n=4))

        plan = Replacement.circuit(ad, circ).plan()
        a = plan.replaced_logits(ad, model, toks, ablation=mean_v, keys=[0] * 4)
        b = plan.replaced_logits(ad, model, toks, ablation=res_v, keys=[0] * 4)
        self.assertGreater(float((a - b).abs().max()), 1e-4)
        self.assertNotEqual(mean_v.digest(), res_v.digest())

    def test_resample_values_are_per_example_and_say_so(self):
        ad = adapters.toy_circuit()
        cf = _toks(n=4, T=8, seed=99)
        v = CIR.ResampleAblationValues(
            {0: torch.zeros(4, 8, N_H, toy_mod.C_D_HEAD)},
            cf_tokens=cf, calibration=dict(distribution="synthetic"))
        with self.assertRaises(ValueError) as cm:
            v.z_for(0, keys=[0] * 8)
        self.assertIn("PER EXAMPLE", str(cm.exception))


class TestAblationProvenanceGate(unittest.TestCase):
    def setUp(self):
        ad = adapters.toy_circuit()
        self.values = _mean_values(ad, None, _toks(n=16, T=8, seed=7))

    def test_stamped_values_pass(self):
        r = CIR.check_ablation_provenance(self.values, declared_policy=CIR.MEAN)
        self.assertTrue(r.passed)
        self.assertIn(self.values.digest(), r.message)

    def test_no_values_at_all_fails(self):
        self.assertEqual(CIR.check_ablation_provenance(None).status, G.FAIL)

    def test_a_policy_mismatch_fails(self):
        r = CIR.check_ablation_provenance(self.values, declared_policy=CIR.RESAMPLE)
        self.assertEqual(r.status, G.FAIL)
        self.assertIn("declares policy", r.message)

    def test_a_drifted_calibration_set_fails_under_a_frozen_digest(self):
        r = CIR.check_ablation_provenance(self.values, expected_digest="0000deadbeef0000")
        self.assertEqual(r.status, G.FAIL)
        self.assertIn("moved under a cached stamp", r.message)

    def test_values_without_a_calibration_descriptor_fail(self):
        bare = CIR.MeanAblationValues({0: {0: torch.zeros(2, N_H, 4)}}, calibration={})
        self.assertEqual(CIR.check_ablation_provenance(bare).status, G.FAIL)

    def test_the_digest_moves_when_the_values_move(self):
        a = self.values.digest()
        other = _mean_values(adapters.toy_circuit(), None, _toks(n=16, T=8, seed=8))
        self.assertNotEqual(a, other.digest())

    def test_enforcement_raises_the_same_refusal_a_blocking_gate_would(self):
        rep = G.GateReport()
        # unrun: the shared registry marks this gate non-blocking (it is N/A to
        # a dictionary artifact), so `require()` alone would let it through...
        self.assertNotIn("ablation-provenance",
                         [r.id for r in rep.blocking_failures])
        # ...and the circuit path closes that door explicitly.
        with self.assertRaises(G.GateFailure):
            CIR.require_ablation_provenance(rep)
        rep.record(CIR.check_ablation_provenance(self.values))
        CIR.require_ablation_provenance(rep)          # no raise

    def test_every_gate_still_has_a_measurement_owner(self):
        from signoff.adapters import base as AB

        self.assertEqual(set(AB.GATE_MEASURED_BY), set(G.GATE_SPECS))


class TestTaskMetrics(unittest.TestCase):
    def test_logit_diff_is_the_difference_of_two_logits_at_the_end(self):
        lg = torch.zeros(2, 3, 10)
        lg[0, -1, 4] = 2.5
        lg[0, -1, 7] = 0.5
        lg[1, -1, 4] = -1.0
        lg[1, -1, 7] = 1.0
        d = M.logit_diff(lg, torch.tensor([4, 4]), torch.tensor([7, 7]))
        self.assertAlmostEqual(float(d[0]), 2.0, places=5)
        self.assertAlmostEqual(float(d[1]), -2.0, places=5)

    def test_per_row_end_positions_are_supported(self):
        lg = torch.zeros(2, 4, 10)
        lg[0, 1, 3] = 1.0
        lg[1, 3, 3] = 5.0
        d = M.logit_diff(lg, torch.tensor([3, 3]), torch.tensor([0, 0]),
                         at=torch.tensor([1, 3]))
        self.assertAlmostEqual(float(d[0]), 1.0, places=5)
        self.assertAlmostEqual(float(d[1]), 5.0, places=5)

    def test_ratio_of_means_is_not_the_mean_of_ratios(self):
        # the confusion this function exists to prevent: the paper's headline
        # 87% is a ratio of means, and a heavy left tail pulls the per-example
        # mean far below it.
        base = torch.tensor([4.0, 4.0, 4.0, 0.5])
        rep = torch.tensor([4.0, 4.0, 4.0, -2.0])
        f = M.faithfulness(rep, base)
        self.assertAlmostEqual(f["ratio_of_means"], 10.0 / 12.5, places=5)
        self.assertAlmostEqual(f["mean_per_example"], (1 + 1 + 1 - 4) / 4, places=5)
        self.assertLess(f["mean_per_example"], f["ratio_of_means"])

    def test_rows_the_model_cannot_do_are_excluded_not_exploded(self):
        f = M.faithfulness(torch.tensor([1.0, 1.0]), torch.tensor([2.0, 0.0]))
        self.assertEqual(f["n_degenerate"], 1)
        self.assertAlmostEqual(f["mean_per_example"], 0.5, places=5)
        self.assertTrue(torch.isnan(f["per_example"][1]))


if __name__ == "__main__":
    unittest.main()
