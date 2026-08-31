"""The refusal path, end to end, on the synthetic fixture.

"A run that fails a gate refuses to emit a report" is the product. CI asserts
that a failed gate produces NO report file and a nonzero exit — if this test
ever passes vacuously, the tool has stopped being a falsifier and become a
number generator.

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
from signoff.cli import EXIT_GATE, EXIT_OK, main
from signoff.replacement import Replacement
from signoff.runner import Runner


class _Case(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="signoff-test-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def runner(self, **kw):
        ad = adapters.get("toy", **kw)
        return ad, Runner(ad, Replacement(ad, layers="all"),
                          out=os.path.join(self.dir, "run"), echo=False)


class TestHappyPath(_Case):
    def test_full_pipeline_emits_a_gated_report(self):
        ad, run = self.runner()
        toks, meta = ad.synthetic_corpus(n=24, seq_len=16, n_probes=8)
        written = run.audit(n=24, seq_len=16, fvu_n=8, tokens=toks, meta=meta,
                            search_seeds=1, iters=2, coverage=True)
        self.assertTrue(os.path.exists(written["markdown"]))
        self.assertTrue(os.path.exists(written["json"]))
        d = json.load(open(written["json"]))
        self.assertTrue(d["gates"]["ok"])
        self.assertEqual(d["verdict"]["strength"], "L3")     # search ran
        self.assertIn("never_emitted", d["verdict"])
        self.assertTrue(d["coverage"])

    def test_the_report_never_claims_faithfulness(self):
        ad, run = self.runner()
        toks, meta = ad.synthetic_corpus(n=16, seq_len=16, n_probes=4)
        written = run.audit(n=16, seq_len=16, fvu_n=8, tokens=toks, meta=meta)
        md = open(written["markdown"]).read()
        self.assertIn("it cannot return equivalence", md)
        self.assertIn("What this report does not say", md)
        for banned in ("is faithful", "proves equivalence", "verified faithful"):
            self.assertNotIn(banned, md)

    def test_no_text_bodies_reach_the_report(self):
        # the tail table is ids + family labels; the synthetic fixture has no
        # text at all, but the shape is what matters
        ad, run = self.runner()
        toks, meta = ad.synthetic_corpus(n=16, seq_len=16, n_probes=4)
        written = run.audit(n=16, seq_len=16, fvu_n=8, tokens=toks, meta=meta)
        d = json.load(open(written["json"]))
        for row in d["distribution"]["tail_top30"]:
            self.assertNotIn("text", row)
            self.assertNotIn("tokens", row)

    def test_resume_reuses_scored_rows(self):
        ad, run = self.runner()
        toks, meta = ad.synthetic_corpus(n=16, seq_len=16, n_probes=4)
        run.mine(tokens=toks, meta=meta)
        run.gates(fvu_n=8)
        first = run.score()
        again = Runner(ad, Replacement(ad, layers="all"), out=run.out, echo=False)
        again.mine(tokens=toks, meta=meta)
        again.gates(fvu_n=8)
        second = again.score()
        self.assertEqual([r["d_mean"] for r in first], [r["d_mean"] for r in second])


class TestRefusal(_Case):
    def test_a_mistapped_artifact_emits_nothing(self):
        # defect high enough that FVU ~ 1 at every site: the mis-tap signature
        ad, run = self.runner(defect=2.0)
        toks, meta = ad.synthetic_corpus(n=16, seq_len=16, n_probes=4)
        with self.assertRaises(G.GateFailure):
            run.audit(n=16, seq_len=16, fvu_n=8, tokens=toks, meta=meta)
        self.assertFalse(os.path.exists(run.p("report.md")))
        self.assertFalse(os.path.exists(run.p("report.json")))

    def test_quarantined_checkpoints_survive_for_diagnosis(self):
        ad, run = self.runner(defect=2.0)
        toks, meta = ad.synthetic_corpus(n=16, seq_len=16, n_probes=4)
        run.mine(tokens=toks, meta=meta)
        with self.assertRaises(G.GateFailure):
            run.gates(fvu_n=8)
        # the failure is explained on disk, and the FVU table that produced it
        # is kept — you cannot diagnose a mis-tap without the numbers
        self.assertTrue(os.path.exists(run.p("gates.json")))
        self.assertTrue(os.path.exists(run.p("fvu.json")))
        self.assertFalse(os.path.exists(run.p("report.md")))

    def test_report_refuses_when_gates_were_never_run(self):
        ad, run = self.runner()
        toks, meta = ad.synthetic_corpus(n=16, seq_len=16, n_probes=4)
        run.mine(tokens=toks, meta=meta)
        run.score()                      # numbers exist...
        with self.assertRaises(G.GateFailure) as cm:
            run.report()                 # ...and are still not renderable
        self.assertIn("refusing to emit a report", str(cm.exception))
        self.assertFalse(os.path.exists(run.p("report.md")))

    def test_the_failure_message_names_the_bug_and_the_next_step(self):
        ad, run = self.runner(defect=2.0)
        toks, meta = ad.synthetic_corpus(n=16, seq_len=16, n_probes=4)
        run.mine(tokens=toks, meta=meta)
        with self.assertRaises(G.GateFailure) as cm:
            run.gates(fvu_n=8)
        msg = str(cm.exception)
        self.assertIn("tap trap", msg)          # the historical bug
        self.assertIn("pre-gain or post-gain", msg)  # the discriminating follow-up


class TestCliExitCodes(_Case):
    def _args(self, *extra):
        return ["audit", "--adapter", "toy", "--n", "16", "--seq-len", "16",
                "--n-probes", "4", "--fvu-n", "8",
                "--run", os.path.join(self.dir, "cli")] + list(extra)

    def test_clean_run_exits_zero(self):
        self.assertEqual(main(self._args()), EXIT_OK)
        self.assertTrue(os.path.exists(os.path.join(self.dir, "cli", "report.md")))

    def test_gate_failure_exits_two_and_writes_no_report(self):
        import signoff.adapters as A

        real = A.get

        def poisoned(name, **kw):
            return real(name, defect=2.0) if name == "toy" else real(name, **kw)

        A.get = poisoned
        try:
            self.assertEqual(main(self._args()), EXIT_GATE)
        finally:
            A.get = real
        self.assertFalse(os.path.exists(os.path.join(self.dir, "cli", "report.md")))

    def test_adapters_command_lists_the_registry(self):
        self.assertEqual(main(["adapters"]), EXIT_OK)

    def test_gates_explain_prints_every_gate(self):
        self.assertEqual(main(["gates", "--explain"]), EXIT_OK)


if __name__ == "__main__":
    unittest.main()
