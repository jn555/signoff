"""Gates: verdict logic, and the refusal contract.

The refusal paths are tested as first-class behaviour — a failed or UNRUN
blocking gate must stop report emission. "Nothing was emitted" is the product.

CPU-only, no weights.
"""

from __future__ import annotations

import unittest

from signoff import gates as G


class TestGateCatalogue(unittest.TestCase):
    def test_every_gate_documents_the_bug_it_caught(self):
        # The bug field is why the gate is worth its runtime; it is printed on
        # failure and rendered in the README table.
        self.assertGreaterEqual(len(G.GATE_SPECS), 7)
        for gid, spec in G.GATE_SPECS.items():
            self.assertEqual(gid, spec.id)
            self.assertTrue(spec.title)
            for field in (spec.checks, spec.bug, spec.diagnosis):
                self.assertTrue(field and len(field) > 40, f"{gid}: thin {field!r}")

    def test_expected_gates_are_present(self):
        self.assertEqual(
            set(G.GATE_SPECS),
            {"i-base-vs-base", "ii-fvu-sanity", "iii-fp32-replay",
             "iii-prime-paired-bound", "identity-guard", "provenance-freeze",
             "bos-declaration", "checkpoint-binding"},
        )

    def test_only_the_fallback_gate_is_non_blocking(self):
        non_blocking = [g for g, s in G.GATE_SPECS.items() if not s.blocking]
        self.assertEqual(non_blocking, ["iii-prime-paired-bound"])


class TestBaseVsBase(unittest.TestCase):
    def test_exact_reproduction_passes(self):
        r = G.check_base_vs_base(0.0, dtype="float32")
        self.assertTrue(r.passed)
        self.assertEqual(r.tolerance, 1e-6)

    def test_the_gemma_softcap_bug_fails(self):
        # 2-ulp logit error from upcasting before the softcap: KL 3.3e-3.
        r = G.check_base_vs_base(3.3e-3, dtype="bfloat16")
        self.assertEqual(r.status, G.FAIL)
        self.assertIn("softcap", r.spec.diagnosis + r.spec.bug)

    def test_tolerance_is_per_dtype(self):
        self.assertTrue(G.check_base_vs_base(5e-5, dtype="float16").passed)
        self.assertFalse(G.check_base_vs_base(5e-5, dtype="float32").passed)

    def test_non_finite_fails_regardless_of_kl(self):
        r = G.check_base_vs_base(0.0, dtype="float16", all_finite=False)
        self.assertEqual(r.status, G.FAIL)
        self.assertIn("overflow", r.message)


class TestFvuSanity(unittest.TestCase):
    def test_healthy_suite_passes(self):
        r = G.check_fvu_sanity({0: 0.10, 1: 0.22, 2: 0.31})
        self.assertTrue(r.passed)

    def test_mistap_signature_fails(self):
        # FVU ~ 1 at EVERY site is the wiring fault, not a weak dictionary.
        r = G.check_fvu_sanity({0: 0.98, 1: 1.01, 2: 0.995})
        self.assertEqual(r.status, G.FAIL)
        self.assertIn("mis-tap", r.message)

    def test_one_bad_layer_is_a_weak_dictionary_not_a_failure(self):
        r = G.check_fvu_sanity({0: 0.12, 1: 0.99, 2: 0.30})
        self.assertTrue(r.passed)
        self.assertEqual(r.detail["suspicious_sites"], ["1"])

    def test_cross_layer_site_ids_are_accepted(self):
        # the CLT seam: sites are "read->write", not layer indices
        r = G.check_fvu_sanity({"3->5": 0.2, "3->6": 0.4})
        self.assertTrue(r.passed)
        self.assertEqual(sorted(r.detail["per_site"]), ["3->5", "3->6"])

    def test_nan_fails(self):
        self.assertEqual(G.check_fvu_sanity({0: float("nan"), 1: 0.2}).status, G.FAIL)

    def test_no_measurement_is_unrun_not_pass(self):
        self.assertEqual(G.check_fvu_sanity({}).status, G.UNRUN)


class TestFp32Replay(unittest.TestCase):
    def test_measured_float16_passes_and_bfloat16_fails(self):
        # the numbers that forced the gemma dtype switch
        self.assertTrue(G.check_fp32_replay(5.85e-4, 2.02e-3, n=4, dtype="float16").passed)
        bf = G.check_fp32_replay(3.09e-1, 3.82e-2, n=4, dtype="bfloat16")
        self.assertEqual(bf.status, G.FAIL)

    def test_a_clean_replaced_stream_cannot_rescue_a_dirty_base_stream(self):
        # the base stream carries no dictionaries: error there is pure dtype
        r = G.check_fp32_replay(1e-5, 5e-2, n=8)
        self.assertEqual(r.status, G.FAIL)
        self.assertIn("base", r.message)


class TestPairedBoundFallback(unittest.TestCase):
    def test_the_measured_gemma_bound_stands(self):
        r = G.check_paired_bound(0.037, ci=dict(lo=0.014, hi=0.061))
        self.assertTrue(r.passed)
        self.assertIn("paired verdicts stand", r.message)

    def test_a_large_discrepancy_kills_even_the_paired_verdict(self):
        self.assertEqual(G.check_paired_bound(0.20).status, G.FAIL)

    def test_it_is_non_blocking_so_it_never_blocks_a_healthy_run(self):
        rep = G.GateReport()
        self.assertFalse(rep.results["iii-prime-paired-bound"].blocks_report)


class TestIdentityGuard(unittest.TestCase):
    def setUp(self):
        self.meta = {0: dict(doc=5, offset=10), 1: dict(doc=6, offset=20)}

    def test_matching_cache_passes(self):
        cached = {0: dict(doc=5, offset=10, run="fp16"), 1: dict(doc=6, offset=20, run="fp16")}
        self.assertTrue(G.check_identity_guard(cached, self.meta, run_tag="fp16").passed)

    def test_smoke_corpus_resumed_into_a_full_run_is_caught(self):
        cached = {0: dict(doc=999, offset=0, run="fp16")}
        r = G.check_identity_guard(cached, self.meta, run_tag="fp16")
        self.assertEqual(r.status, G.FAIL)
        self.assertIn("DIFFERENT CORPUS", r.message)

    def test_stale_dtype_cache_is_caught(self):
        # the real one: an fp16 run nearly reported bfloat16 divergences
        cached = {0: dict(doc=5, offset=10, run="artifact:bfloat16:layers=all")}
        r = G.check_identity_guard(cached, self.meta, run_tag="artifact:float16:layers=all")
        self.assertEqual(r.status, G.FAIL)
        self.assertIn("bfloat16", r.message)

    def test_empty_cache_is_vacuously_fine(self):
        self.assertTrue(G.check_identity_guard({}, self.meta, run_tag="x").passed)


class TestProvenanceFreeze(unittest.TestCase):
    def test_matching_listing_passes(self):
        self.assertTrue(G.check_provenance_freeze({0: 115, 1: 104}, {0: 115, 1: 104}).passed)

    def test_the_degenerate_l0_drift_is_caught(self):
        # SAELens pointed at L0=5..15 variants for 12 of 26 gemma layers
        r = G.check_provenance_freeze({11: 108, 12: 111}, {11: 5, 12: 15})
        self.assertEqual(r.status, G.FAIL)
        self.assertEqual(r.value, 2.0)
        self.assertEqual(r.detail["differences"]["11"], dict(expected=108, observed=5))

    def test_a_new_layer_appearing_is_also_drift(self):
        self.assertEqual(G.check_provenance_freeze({0: 1}, {0: 1, 1: 2}).status, G.FAIL)


class TestBosDeclaration(unittest.TestCase):
    def test_declared_and_enforced_passes(self):
        self.assertTrue(G.check_bos_declaration(
            declared_bos_id=2, corpus_uses_bos=True, corpus_bos_id=2).passed)
        self.assertTrue(G.check_bos_declaration(
            declared_bos_id=None, corpus_uses_bos=False).passed)

    def test_bos_free_eval_of_a_bos_trained_suite_is_caught(self):
        # corpus NLL 7.005 -> 3.217 nats/token purely from prepending BOS
        r = G.check_bos_declaration(declared_bos_id=2, corpus_uses_bos=False)
        self.assertEqual(r.status, G.FAIL)

    def test_wrong_bos_token_is_caught(self):
        r = G.check_bos_declaration(declared_bos_id=2, corpus_uses_bos=True, corpus_bos_id=1)
        self.assertEqual(r.status, G.FAIL)

    def test_an_undeclared_convention_is_a_failure_not_a_default(self):
        r = G.check_bos_declaration(declared_bos_id=None, corpus_uses_bos=False, declared=False)
        self.assertEqual(r.status, G.FAIL)


class TestRefusalContract(unittest.TestCase):
    """A run that fails a gate refuses to emit a report. This is the product."""

    def _healthy(self):
        rep = G.GateReport()
        rep.record(G.check_base_vs_base(0.0, dtype="float32"))
        rep.record(G.check_fvu_sanity({0: 0.2}))
        rep.record(G.check_fp32_replay(1e-4, 1e-4, n=4))
        rep.record(G.check_identity_guard({}, {}, run_tag="t"))
        rep.record(G.check_provenance_freeze({0: 1}, {0: 1}))
        rep.record(G.check_bos_declaration(declared_bos_id=None, corpus_uses_bos=False))
        # a healthy report is also BOUND: verdicts that are not tied to the
        # configuration that measured them do not clear the binding gate
        rep.bind("cfg0", dict(adapter="test"))
        rep.record(G.check_checkpoint_binding(
            expected_hash="cfg0", expected_digest=rep.results_digest(),
            binding=rep.binding, restored=False))
        return rep

    def test_a_fresh_report_is_all_unrun_and_therefore_blocks(self):
        rep = G.GateReport()
        self.assertFalse(rep.ok)
        # an UNRUN blocking gate is treated exactly like a failed one
        self.assertEqual(len(rep.blocking_failures), 7)
        with self.assertRaises(G.GateFailure):
            rep.require()

    def test_all_gates_cleared_permits_emission(self):
        rep = self._healthy()
        self.assertTrue(rep.ok)
        rep.require()  # must not raise

    def test_one_failed_gate_blocks_everything(self):
        rep = self._healthy()
        rep.record(G.check_fvu_sanity({0: 0.99, 1: 0.99}))
        self.assertFalse(rep.ok)
        with self.assertRaises(G.GateFailure) as cm:
            rep.require("emit a report")
        msg = str(cm.exception)
        self.assertIn("refusing to emit a report", msg)
        self.assertIn("why it exists", msg)     # the bug is printed
        self.assertIn("what to do", msg)        # so is the diagnosis
        self.assertIn("quarantined", msg)

    def test_waiving_is_explicit_and_visible(self):
        rep = self._healthy()
        rep.record(G.check_fp32_replay(0.5, 0.5, n=4, dtype="bfloat16"))
        self.assertFalse(rep.ok)
        rep.waive("iii-fp32-replay", "paired-bound fallback measured at 0.037")
        self.assertTrue(rep.ok)
        d = rep.to_dict()
        waived = [g for g in d["gates"] if g["status"] == G.WAIVED]
        self.assertEqual(len(waived), 1)
        self.assertIn("waived:", waived[0]["message"])

    def test_round_trips_through_json_shaped_dicts(self):
        rep = self._healthy()
        again = G.GateReport.from_dict(rep.to_dict())
        self.assertTrue(again.ok)
        self.assertEqual(again.to_dict()["n_pass"], rep.to_dict()["n_pass"])

    def test_summary_table_has_one_line_per_gate(self):
        lines = G.summary_table(self._healthy()).splitlines()
        self.assertEqual(len(lines), len(G.GATE_SPECS))


if __name__ == "__main__":
    unittest.main()
