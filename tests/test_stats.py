"""Stats: the validity protocol, locked on synthetic rows with known answers.

Every test here corresponds to a correction that changed a headline number in
experiments 02/03.  If one of these fails, the tool has quietly reverted to the
overstated version of a result.

CPU-only, no torch autograd, no weights — part of the CI subset.
"""

from __future__ import annotations

import unittest

import torch

from signoff import stats as S


def row(i, kind, d_mean, nll, pid=None, doc=None, offset=0, family=None):
    r = dict(row=i, kind=kind, d_mean=float(d_mean), nll=float(nll),
             pid=pid, doc=doc, offset=offset)
    if family:
        r["family"] = family
    return r


class TestDescriptive(unittest.TestCase):
    def test_quantiles_are_index_rounded_not_interpolated(self):
        # Locks witness.py's estimator. Switching to linear interpolation would
        # silently move every pinned p99 in the golden tests.
        xs = list(range(1, 101))  # 1..100
        q = S.quantiles(xs, (0.5, 0.9, 0.99))
        self.assertEqual(q["p99"], 99.0)   # index round(0.99*99) = 98 -> value 99
        self.assertEqual(q["p90"], 90.0)   # index round(0.90*99) = 89 -> value 90
        # round(49.5) = 50 under Python's round-half-to-EVEN, so p50 is 51, not
        # 50. That is the estimator the completed runs' pinned numbers use.
        self.assertEqual(q["p50"], 51.0)

    def test_summarize_handles_empty_and_singleton(self):
        e = S.summarize([])
        self.assertEqual(e["n"], 0)
        self.assertNotEqual(e["mean"], e["mean"])  # NaN
        one = S.summarize([2.0])
        self.assertEqual(one["n"], 1)
        self.assertEqual(one["std"], 0.0)
        self.assertEqual(one["p99"], 2.0)

    def test_spearman_is_rank_based_and_tie_averaged(self):
        a = [1, 2, 3, 4, 5]
        b = [10, 20, 30, 40, 50]
        self.assertAlmostEqual(S.spearman(a, b), 1.0, places=5)
        self.assertAlmostEqual(S.spearman(a, [5, 4, 3, 2, 1]), -1.0, places=5)
        # monotone but non-linear: spearman 1, pearson < 1
        c = [1, 2, 4, 8, 1000]
        self.assertAlmostEqual(S.spearman(a, c), 1.0, places=5)
        self.assertLess(S.pearson(a, c), 0.95)

    def test_paired_summary_reports_se_and_sign_rate(self):
        p = S.paired_summary([1.0, 1.0, 1.0, 1.0])
        self.assertEqual(p["n"], 4)
        self.assertAlmostEqual(p["mean"], 1.0)
        self.assertEqual(p["sd"], 0.0)
        self.assertEqual(p["frac_positive"], 1.0)
        mixed = S.paired_summary([1.0, -1.0, 2.0, -2.0])
        self.assertAlmostEqual(mixed["mean"], 0.0)
        self.assertEqual(mixed["frac_positive"], 0.5)


class TestPseudoReplicationGuard(unittest.TestCase):
    """The exp-02 correction that cut a headline by ~3x.

    Windows drawn from the same source text are not independent observations.
    A family effect carried by ONE text looks large at the windowed level and
    collapses at the distinct-text level; both must always be reported.
    """

    def _rows(self):
        rows = []
        # 40 corpus controls spanning the NLL range, all low divergence
        for i in range(40):
            rows.append(row(i, "corpus", d_mean=1.0, nll=2.0 + i * 0.05, doc=1000 + i))
        # one "hot" text contributing 10 windows at high divergence...
        for k in range(10):
            rows.append(row(100 + k, "probe", d_mean=5.0, nll=2.5,
                            pid=f"pile:777:{k * 32}", doc=777, offset=k * 32))
        # ...and three other texts contributing one window each, at control level
        for k, doc in enumerate((801, 802, 803)):
            rows.append(row(200 + k, "probe", d_mean=1.0, nll=2.5,
                            pid=f"pile:{doc}:0", doc=doc))
        return rows

    def test_grouping_levels_deduplicate_by_source_text(self):
        rows = self._rows()
        probes = [r for r in rows if r["kind"] == "probe"]
        lv = S.grouping_levels(probes)
        self.assertEqual(len(lv["windowed"]), 13)
        self.assertEqual(len(lv["distinct_text"]), 4)      # 4 source texts
        self.assertEqual(len(lv["independent_source"]), 13)

    def test_text_id_strips_the_offset(self):
        self.assertEqual(S.text_id(dict(pid="pile:5206:128")), "pile:5206")
        self.assertEqual(S.text_id(dict(pid="file:mit:64")), "file:mit")

    def test_windowed_level_overstates_a_single_text_effect(self):
        out = S.family_test(self._rows())
        w = out["levels"]["windowed"]["paired_diff_nats"]["mean"]
        d = out["levels"]["distinct_text"]["paired_diff_nats"]["mean"]
        # windowed: 10 of 13 pairs carry +4 nats; distinct-text: 1 of 4 does
        self.assertGreater(w, 2.5)
        self.assertLess(d, 1.5)
        self.assertGreater(w, 2 * d)

    def test_all_three_levels_are_always_present(self):
        out = S.family_test(self._rows())
        self.assertEqual(set(out["levels"]), {"windowed", "distinct_text", "independent_source"})

    def test_probe_only_or_corpus_only_returns_none_not_a_number(self):
        only_corpus = [row(i, "corpus", 1.0, 2.0) for i in range(5)]
        self.assertIsNone(S.family_test(only_corpus))

    def test_control_blocklist_shrinks_the_pool(self):
        rows = self._rows()
        blocked = {r["row"] for r in rows if r["kind"] == "corpus" and r["row"] % 2 == 0}
        out = S.family_test(rows, control_eligible=lambda r: r["row"] not in blocked)
        self.assertEqual(out["control_pool"], 20)


class TestNllMatchedPairing(unittest.TestCase):
    def test_pairs_nearest_nll_without_replacement(self):
        probes = [row(0, "probe", 3.0, 1.0, pid="pile:1:0"),
                  row(1, "probe", 3.0, 1.05, pid="pile:2:0")]
        pool = [row(10, "corpus", 1.0, 1.02),
                row(11, "corpus", 1.0, 1.04),
                row(12, "corpus", 1.0, 9.0)]
        pairs = S.greedy_nearest_nll_pairs(probes, pool)
        self.assertEqual(len(pairs), 2)
        self.assertEqual(len({p["ctrl_row"] for p in pairs}), 2)   # no reuse
        self.assertNotIn(12, {p["ctrl_row"] for p in pairs})       # the far one is unused
        for p in pairs:
            self.assertLess(p["nll_gap"], 0.05)

    def test_early_break_matches_a_full_scan(self):
        g = torch.Generator().manual_seed(3)
        probes = [row(i, "probe", 1.0, float(x), pid=f"pile:{i}:0")
                  for i, x in enumerate(torch.rand(12, generator=g) * 5)]
        pool = [row(100 + i, "corpus", 1.0, float(x))
                for i, x in enumerate(torch.rand(30, generator=g) * 5)]
        fast = S.greedy_nearest_nll_pairs(probes, pool)

        # reference: the same greedy procedure with no early break
        used, ref = set(), []
        for pr in sorted(probes, key=lambda r: r["nll"]):
            cands = [c for c in pool if c["row"] not in used]
            best = min(cands, key=lambda c: abs(c["nll"] - pr["nll"]))
            used.add(best["row"])
            ref.append((pr["row"], best["row"]))
        self.assertEqual([(p["probe_row"], p["ctrl_row"]) for p in fast], ref)

    def test_pool_smaller_than_probe_set_pairs_what_it_can(self):
        probes = [row(i, "probe", 1.0, 1.0 + i, pid=f"pile:{i}:0") for i in range(5)]
        pool = [row(100 + i, "corpus", 1.0, 1.0 + i) for i in range(2)]
        self.assertEqual(len(S.greedy_nearest_nll_pairs(probes, pool)), 2)


class TestStratifiedLowNll(unittest.TestCase):
    """analyze03 study_bg6(a): boilerplate is low-NLL, so repeat the paired test
    inside the easy stratum where probe and control are both easy."""

    def test_restricts_both_arms_to_the_low_nll_stratum(self):
        rows = [row(i, "corpus", 1.0, nll=float(i)) for i in range(100)]  # nll 0..99
        # probes at nll 5 (in stratum) and nll 90 (out)
        rows += [row(200, "probe", 4.0, 5.0, pid="pile:1:0"),
                 row(201, "probe", 4.0, 90.0, pid="pile:2:0")]
        out = S.stratified_low_nll_test(rows, q=0.25)
        self.assertTrue(out["available"])
        self.assertEqual(out["stratum_ceiling_nll"], 24.0)   # corpus p25 by index
        self.assertEqual(out["n_probes_in_stratum"], 1)
        self.assertEqual(out["n_probes_total"], 2)
        self.assertEqual(out["n_pool"], 25)
        self.assertEqual(out["paired"]["n"], 1)
        self.assertAlmostEqual(out["paired"]["mean"], 3.0, places=5)

    def test_reports_unavailable_rather_than_nan_when_there_is_no_stratum(self):
        rows = [row(i, "corpus", 1.0, float(i)) for i in range(20)]
        rows += [row(200, "probe", 4.0, 99.0, pid="pile:1:0")]
        out = S.stratified_low_nll_test(rows, q=0.25)
        self.assertFalse(out["available"])

    def test_effect_that_is_purely_an_nll_readout_vanishes_in_the_stratum(self):
        # d = 4 - 0.03*nll for EVERYTHING: no family effect, just an NLL slope.
        rows = [row(i, "corpus", 4.0 - 0.03 * i, float(i)) for i in range(100)]
        rows += [row(200 + k, "probe", 4.0 - 0.03 * (2 + k), float(2 + k),
                     pid=f"pile:{k}:0") for k in range(8)]
        out = S.stratified_low_nll_test(rows, q=0.25)
        self.assertLess(abs(out["paired"]["mean"]), 0.05)


class TestResidualizedTail(unittest.TestCase):
    """analyze03 study_bg6(b): a top-K tail can be a low-NLL readout rather than
    a family effect. Residualise d on a quadratic in NLL and recompose."""

    def _rows(self):
        rows = []
        # 60 rows whose divergence is a pure quadratic function of NLL
        for i in range(60):
            nll = 0.5 + i * 0.1
            rows.append(row(i, "corpus", d_mean=6.0 - 0.8 * nll + 0.05 * nll * nll,
                            nll=nll, family="corpus"))
        # 5 mid-NLL rows with a genuine family bump. The bump is deliberately
        # SMALLER than the NLL-driven spread (corpus d runs 5.61 -> 2.93), so
        # the probes are NOT in the raw top-K — which is the whole point: the
        # raw tail is an NLL readout and only the residual exposes the family.
        for k in range(5):
            nll = 3.0 + k * 0.1
            rows.append(row(500 + k, "probe",
                            d_mean=6.0 - 0.8 * nll + 0.05 * nll * nll + 0.3,
                            nll=nll, pid=f"pile:{k}:0", family="license"))
        return rows

    def test_raw_tail_is_the_nll_readout_and_the_residual_tail_is_the_family(self):
        out = S.residualized_tail(self._rows(), k=5)
        self.assertTrue(out["available"])
        # raw top-5 is dominated by the extreme-NLL corpus rows
        self.assertEqual(out["raw"]["probe_share"], 0.0)
        # residualised top-5 is exactly the bumped family
        self.assertEqual(out["residualized"]["probe_share"], 1.0)
        self.assertEqual(out["residualized"]["composition"], {"license": 5})
        self.assertEqual(out["overlap_with_raw"], 0)

    def test_fit_recovers_the_generating_quadratic(self):
        # The fit is over ALL rows including the bumped family, so it is
        # slightly contaminated by the very effect it is removing — the
        # residualisation is conservative, not exact. Bounds are loose on
        # purpose; `test_polyfit_matches_a_known_polynomial` pins the solver.
        out = S.residualized_tail(self._rows(), k=5)
        b0, b1, b2 = out["beta"]
        self.assertAlmostEqual(b1, -0.8, delta=0.1)
        self.assertAlmostEqual(b2, 0.05, delta=0.02)

    def test_annotates_rows_in_place_for_the_reporter(self):
        rows = self._rows()
        S.residualized_tail(rows, k=5)
        self.assertTrue(all("resid_d_mean" in r for r in rows))

    def test_refuses_on_too_few_rows_and_on_a_singular_design(self):
        self.assertFalse(S.residualized_tail([row(i, "corpus", 1.0, 1.0) for i in range(4)],
                                             k=2)["available"])
        flat = [row(i, "corpus", 1.0, 2.0) for i in range(20)]   # every nll identical
        self.assertFalse(S.residualized_tail(flat, k=5)["available"])

    def test_polyfit_matches_a_known_polynomial(self):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [1.0 + 2.0 * x + 3.0 * x * x for x in xs]
        beta = S.polyfit_least_squares(xs, ys, 2)
        for got, want in zip(beta, (1.0, 2.0, 3.0)):
            self.assertAlmostEqual(got, want, places=4)


class TestDualFvu(unittest.TestCase):
    """run_b_gemma stage_pass: the global and per-token-median estimators are
    DIFFERENT quantities and disagree in sign of the ranking on GPT-2. Quoting
    one alone lets a suite look good by choice of estimator."""

    def test_the_two_estimators_can_rank_two_sites_oppositely(self):
        # site A: most tokens reconstruct well; a few high-variance tokens are awful
        num_a = torch.tensor([[0.0, 0.01, 0.01, 0.01, 900.0]])
        den_a = torch.tensor([[1.0, 1.00, 1.00, 1.00, 1000.0]])
        # site B: every token is mediocre
        num_b = torch.tensor([[0.0, 0.50, 0.50, 0.50, 500.0]])
        den_b = torch.tensor([[1.0, 1.00, 1.00, 1.00, 1000.0]])
        a = S.dual_fvu(num_a, den_a)
        b = S.dual_fvu(num_b, den_b)
        self.assertGreater(a["fvu_global"], b["fvu_global"])              # A looks worse
        self.assertLess(a["fvu_token_median"], b["fvu_token_median"])     # B looks worse
        # ...which is exactly why both are always emitted.

    def test_excludes_position_zero_by_default(self):
        num = torch.tensor([[999.0, 0.0, 0.0, 0.0]])
        den = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
        self.assertAlmostEqual(S.dual_fvu(num, den)["fvu_global"], 0.0, places=6)
        self.assertGreater(S.dual_fvu(num, den, exclude_position_0=False)["fvu_global"], 100)

    def test_perfect_reconstruction_is_zero_both_ways(self):
        num = torch.zeros(2, 5)
        den = torch.ones(2, 5)
        d = S.dual_fvu(num, den)
        self.assertEqual(d["fvu_global"], 0.0)
        self.assertEqual(d["fvu_token_median"], 0.0)
        self.assertEqual(d["n_tokens"], 8)

    def test_rejects_mismatched_shapes(self):
        with self.assertRaises(ValueError):
            S.dual_fvu(torch.zeros(2, 4), torch.zeros(2, 5))


class TestBootstrapAndTailRatios(unittest.TestCase):
    def test_bootstrap_ci_brackets_the_point_estimate_and_is_deterministic(self):
        g = torch.Generator().manual_seed(0)
        xs = [float(x) for x in torch.randn(200, generator=g) * 0.1 + 0.5]
        a = S.bootstrap_ci(xs, seed=7, n_boot=500)
        b = S.bootstrap_ci(xs, seed=7, n_boot=500)
        self.assertEqual(a, b)                       # same seed, same CI
        self.assertLessEqual(a["lo"], a["point"])
        self.assertGreaterEqual(a["hi"], a["point"])
        self.assertAlmostEqual(a["point"], sum(xs) / len(xs), places=6)

    def test_bootstrap_degrades_gracefully_on_tiny_input(self):
        out = S.bootstrap_ci([1.0])
        self.assertEqual(out["n_boot"], 0)
        self.assertEqual(out["point"], 1.0)

    def test_tail_ratio_is_a_diagnostic_shaped_like_the_reported_one(self):
        t = S.tail_ratios([1.0] * 99 + [11.0])
        self.assertAlmostEqual(t["max_over_median"], 11.0, places=5)
        self.assertAlmostEqual(t["frac_above_10x_median"], 0.01, places=6)
        # the threshold is a STRICT inequality: exactly 10x does not count
        self.assertEqual(S.tail_ratios([1.0] * 99 + [10.0])["frac_above_10x_median"], 0.0)


if __name__ == "__main__":
    unittest.main()
