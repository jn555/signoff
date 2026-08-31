"""The trust boundary: who measured the numbers, and is this checkpoint ours?

Three bypass paths were demonstrated against v0.1 by a critic. Two of them are
not bugs and cannot be closed — measurement is delegated to the adapter by
design — so the tool's answer is to make the delegation VISIBLE. The third
(report() trusting whatever gates.json it found) is a bug, and is closed.

  1. an adapter overriding a gate to return PASS gets a PASS
     -> not closable; the report STAMPS the override and marks the row
        "self-reported"
  2. an adapter overriding the measurement under a gate
     -> same treatment, same stamp
  3. a stale or hand-edited gates.json licensing a report
     -> CLOSED: verdicts are bound to a config hash plus a results digest, and
        emission refuses on a mismatch or an unbound file

CPU-only, no weights, no network.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import unittest

from signoff import adapters
from signoff import gates as G
from signoff import report as R
from signoff.adapters import base as AB
from signoff.replacement import Replacement
from signoff.runner import Runner


def _read(path: str) -> str:
    with open(path) as f:
        return f.read()


def _load(path: str):
    with open(path) as f:
        return json.load(f)


def _toy_class():
    return type(adapters.get("toy"))


class BypassAdapter(_toy_class()):
    """The critic's scratch bypass: override a gate so it always returns PASS.

    The defect is set to the mis-tap signature (FVU ~ 1 everywhere), so the
    honest adapter FAILS gate (ii) and emits nothing. This one passes.
    """

    name = "toy-bypass"

    def gate_fvu(self, toks, layers, batch: int = 8):
        table = {str(L): dict(fvu_global=0.05, fvu_token_median=0.05, n_tokens=0)
                 for L in layers}
        return G.GateResult(G.GATE_SPECS["ii-fvu-sanity"], G.PASS, 0.05, G.MISTAP_FVU,
                            "reconstruction is excellent, trust me"), table


class LyingContractAdapter(BypassAdapter):
    """...and then tries to hide the override by rewriting its own contract."""

    name = "toy-bypass-liar"

    def contract(self):
        d = super().contract()
        d["overridden_measurement_methods"] = []
        return d


class _Case(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="signoff-trust-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def runner(self, cls=None, out="run", layers="all", **kw):
        ad = cls(**kw) if cls is not None else adapters.get("toy", **kw)
        return ad, Runner(ad, Replacement(ad, layers=layers),
                          out=os.path.join(self.dir, out), echo=False)


# ------------------------------------------------------------ the stamp


class TestOverrideStamp(unittest.TestCase):
    def test_the_reference_adapters_take_over_no_measurement(self):
        from signoff.adapters.gpt2_dunefsky import Gpt2DunefskyAdapter
        from signoff.adapters.qwen_mwhanna import QwenMwhannaAdapter
        from signoff.adapters.toy import ToyAdapter

        for cls in (ToyAdapter, Gpt2DunefskyAdapter, QwenMwhannaAdapter):
            self.assertEqual(AB.overridden_measurement_methods(cls), [], cls.__name__)
            self.assertEqual(AB.self_reported_gates(cls), {}, cls.__name__)

    def test_gemma_declares_its_one_documented_override(self):
        # gemma-scope's release has a variant-SELECTION rule, so the adapter
        # re-derives the canonical-L0 table from the live listing itself. A
        # declared extension point — and still a self-report, so it is listed.
        from signoff.adapters.gemma_scope import GemmaScopeAdapter

        self.assertEqual(AB.overridden_measurement_methods(GemmaScopeAdapter),
                         ["verify_provenance"])
        self.assertEqual(AB.self_reported_gates(GemmaScopeAdapter),
                         {"provenance-freeze": ["verify_provenance"]})

    def test_a_gate_override_is_detected(self):
        self.assertEqual(AB.overridden_measurement_methods(BypassAdapter), ["gate_fvu"])
        self.assertEqual(AB.self_reported_gates(BypassAdapter),
                         {"ii-fvu-sanity": ["gate_fvu"]})

    def test_the_stamp_is_not_read_off_the_adapter(self):
        # an adapter can rewrite its own contract(); the stamp is computed from
        # the CLASS by a module-level function, so rewriting it changes nothing
        # except adding `contract` to the list
        stamp = R.trust_stamp(LyingContractAdapter())
        self.assertEqual(stamp["overridden_measurement_methods"],
                         ["gate_fvu", "contract"])
        self.assertIn("provenance-freeze", stamp["self_reported_gates"])

    def test_required_overrides_are_not_flagged(self):
        # every adapter supplies tap/head/embed/...; flagging them would make
        # the marker meaningless. They are named in the stamp instead.
        for m in AB.REQUIRED_OVERRIDES:
            self.assertNotIn(m, AB.MEASUREMENT_METHODS)
        self.assertIn("tap", AB.REQUIRED_OVERRIDES)
        self.assertIn("head", AB.REQUIRED_OVERRIDES)

    def test_every_gate_has_a_declared_measurement_owner(self):
        self.assertEqual(set(AB.GATE_MEASURED_BY), set(G.GATE_SPECS))


class TestStampReachesTheReport(_Case):
    def _audit(self, cls=None, out="run", **kw):
        ad, run = self.runner(cls, out=out, **kw)
        toks, meta = ad.synthetic_corpus(n=16, seq_len=16, n_probes=4)
        return run, run.audit(n=16, seq_len=16, fvu_n=8, tokens=toks, meta=meta)

    def test_the_bypass_adapter_reports_but_is_marked_self_reported(self):
        # the bypass WORKS — that is the honest finding, and the reason the
        # report has to say who measured what
        run, written = self._audit(BypassAdapter, defect=2.0)
        md = _read(written["markdown"])
        self.assertIn("[self-reported]", md)
        # ...on the FVU row specifically
        fvu_row = [ln for ln in md.splitlines() if ln.startswith("| (ii) FVU sanity")]
        self.assertEqual(len(fvu_row), 1)
        self.assertIn("[self-reported]", fvu_row[0])
        self.assertIn("gate_fvu", md)
        self.assertIn("not tamper-proof", md)

        d = _load(written["json"])
        self.assertEqual(d["trust"]["overridden_measurement_methods"], ["gate_fvu"])
        by_id = {g["id"]: g for g in d["gates"]["gates"]}
        self.assertTrue(by_id["ii-fvu-sanity"]["self_reported"])
        self.assertEqual(by_id["ii-fvu-sanity"]["measured_by_adapter_overrides"],
                         ["gate_fvu"])
        self.assertFalse(by_id["i-base-vs-base"]["self_reported"])

    def test_a_reference_adapter_report_carries_no_marker(self):
        run, written = self._audit()
        md = _read(written["markdown"])
        self.assertNotIn("[self-reported]", md)
        self.assertIn("Adapter-overridden measurement methods: **none**", md)
        # and still refuses to overclaim about what that means
        self.assertIn("not a tamper-proofness claim", md)
        d = _load(written["json"])
        self.assertEqual(d["trust"]["overridden_measurement_methods"], [])
        self.assertEqual(d["config"]["adapter"]["overridden_measurement_methods"], [])

    def test_the_honest_adapter_with_the_same_defect_emits_nothing(self):
        # the control that makes the bypass test mean something
        ad, run = self.runner(defect=2.0, out="honest")
        toks, meta = ad.synthetic_corpus(n=16, seq_len=16, n_probes=4)
        with self.assertRaises(G.GateFailure):
            run.audit(n=16, seq_len=16, fvu_n=8, tokens=toks, meta=meta)
        self.assertFalse(os.path.exists(run.p("report.md")))


# -------------------------------------------------------- checkpoint binding


class TestCheckpointBinding(_Case):
    def _gated(self, out="run", layers="all", **kw):
        ad, run = self.runner(out=out, layers=layers, **kw)
        toks, meta = ad.synthetic_corpus(n=16, seq_len=16, n_probes=4)
        run.mine(tokens=toks, meta=meta)
        run.gates(fvu_n=8, strict=False)
        return ad, run, toks, meta

    def _reattach(self, ad, run, layers="all"):
        return Runner(ad, Replacement(ad, layers=layers), out=run.out, echo=False)

    def _edit_gates_json(self, run, fn):
        d = _load(run.p("gates.json"))
        fn(d)
        with open(run.p("gates.json"), "w") as f:
            json.dump(d, f, indent=2)

    def test_a_clean_run_binds_its_verdicts(self):
        ad, run, _, _ = self._gated()
        b = _load(run.p("gates.json"))["binding"]
        self.assertEqual(b["config_hash"], run.config_hash())
        self.assertEqual(b["results_digest"], run.gate_report.results_digest())
        self.assertIn("dict_revision", b["fingerprint"])
        self.assertTrue(run.gate_report.results["checkpoint-binding"].passed)

    def test_a_matching_checkpoint_still_reports(self):
        # the control: re-attaching to your own run directory must keep working
        ad, run, toks, meta = self._gated()
        again = self._reattach(ad, run)
        again.rows = []
        written = again.report()
        self.assertTrue(os.path.exists(written["markdown"]))

    def test_a_hand_edited_gates_json_is_refused(self):
        # the critic's path: fail a gate, flip the verdict in the file, report
        ad, run, _, _ = self._gated(defect=2.0)
        self.assertFalse(run.gate_report.ok)
        self._edit_gates_json(run, lambda d: [g.update(status="pass") for g in d["gates"]])

        again = self._reattach(ad, run)
        self.assertTrue(all(r.status == G.PASS for r in again.gate_report.ordered
                            if r.spec.id != "checkpoint-binding"))
        with self.assertRaises(G.GateFailure) as cm:
            again.report()
        msg = str(cm.exception)
        self.assertIn("checkpoint binding", msg)
        self.assertIn("edited after it was written", msg)
        self.assertFalse(os.path.exists(again.p("report.md")))
        self.assertFalse(os.path.exists(again.p("report.json")))

    def test_an_unbound_gates_json_is_refused(self):
        ad, run, _, _ = self._gated(defect=2.0)

        def strip(d):
            d.pop("binding", None)
            for g in d["gates"]:
                g["status"] = "pass"

        self._edit_gates_json(run, strip)
        again = self._reattach(ad, run)
        with self.assertRaises(G.GateFailure) as cm:
            again.report()
        self.assertIn("NO configuration binding", str(cm.exception))
        self.assertFalse(os.path.exists(again.p("report.md")))

    def test_a_checkpoint_from_another_configuration_is_refused(self):
        # gates measured on two layers; the report wants to be about all four
        ad, run, _, _ = self._gated(layers=[0, 1])
        self.assertTrue(run.gate_report.ok)
        again = self._reattach(ad, run, layers="all")
        again.rows = []
        with self.assertRaises(G.GateFailure) as cm:
            again.report()
        msg = str(cm.exception)
        self.assertIn("DIFFERENT configuration", msg)
        self.assertIn("replaced_layers", msg)
        self.assertFalse(os.path.exists(again.p("report.md")))

    def test_the_binding_gate_names_its_bug_and_the_fix(self):
        spec = G.GATE_SPECS["checkpoint-binding"]
        self.assertIn("hand-edit", spec.diagnosis)
        self.assertIn("unkeyed", spec.diagnosis)          # honest about its limit
        self.assertIn("decorative", spec.bug)

    def test_a_waiver_does_not_break_the_binding(self):
        # waiving is an in-process, rendered decision, not an edit behind the
        # tool's back — it must not look like tampering
        ad, run, _, _ = self._gated(defect=2.0)
        run.gate_report.waive("ii-fvu-sanity", "known weak fixture; reviewed by hand")
        r = run.verify_checkpoint_binding()
        self.assertTrue(r.passed, r.message)
        self.assertTrue(run.gate_report.ok)


class TestBindingChecker(unittest.TestCase):
    """The pure verdict function, with no runner attached."""

    def _binding(self, **kw):
        d = dict(config_hash="abc", results_digest="dig", fingerprint=dict(dtype="float32"))
        d.update(kw)
        return d

    def test_matching_binding_passes(self):
        r = G.check_checkpoint_binding(expected_hash="abc", expected_digest="dig",
                                       binding=self._binding())
        self.assertTrue(r.passed)

    def test_missing_binding_fails(self):
        for b in (None, {}, {"fingerprint": {}}):
            r = G.check_checkpoint_binding(expected_hash="abc", expected_digest="dig",
                                           binding=b)
            self.assertEqual(r.status, G.FAIL)
            self.assertIn("NO configuration binding", r.message)

    def test_hash_mismatch_names_the_differing_fields(self):
        r = G.check_checkpoint_binding(
            expected_hash="zzz", expected_digest="dig",
            binding=self._binding(), expected_fingerprint=dict(dtype="bfloat16"))
        self.assertEqual(r.status, G.FAIL)
        self.assertIn("dtype", r.message)
        self.assertEqual(r.detail["differing_fields"]["dtype"],
                         dict(when_gated="float32", now="bfloat16"))

    def test_digest_mismatch_is_reported_as_an_edit(self):
        r = G.check_checkpoint_binding(expected_hash="abc", expected_digest="OTHER",
                                       binding=self._binding())
        self.assertEqual(r.status, G.FAIL)
        self.assertIn("edited after it was written", r.message)

    def test_the_digest_ignores_the_binding_gates_own_verdict(self):
        rep = G.GateReport()
        rep.record(G.check_base_vs_base(0.0, dtype="float32"))
        before = rep.results_digest()
        rep.record(G.check_checkpoint_binding(expected_hash="a", expected_digest=before,
                                              binding=dict(config_hash="a")))
        self.assertEqual(rep.results_digest(), before)

    def test_the_digest_moves_when_a_verdict_moves(self):
        rep = G.GateReport()
        rep.record(G.check_fvu_sanity({0: 0.2}))
        a = rep.results_digest()
        rep.record(G.check_fvu_sanity({0: 0.99, 1: 0.99}))
        self.assertNotEqual(rep.results_digest(), a)


if __name__ == "__main__":
    unittest.main()
