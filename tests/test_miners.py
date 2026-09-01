"""Miners: search determinism, the constraint, seed selection, family detectors.

The greedy miner is exercised against a SYNTHETIC objective — no weights — so
the search logic is tested independently of any model. CPU-only.
"""

from __future__ import annotations

import unittest

import torch

from signoff import miners as Mine
from signoff.metrics import SeqMetrics


def fake_evaluate(target_token: int = 7, vocab: int = 32):
    """An evaluator with a known optimum: divergence rises with occurrences of
    `target_token`, and NLL rises with them too — so the hinge actually bites."""

    def evaluate(batch: torch.Tensor):
        hits = (batch == target_token).float().sum(-1)
        d_mean = 0.1 + 0.5 * hits
        nll = 2.0 + 1.0 * hits
        lb = torch.zeros(batch.shape[0], batch.shape[1], vocab)
        return lb, SeqMetrics(d_mean, d_mean * 2, torch.zeros_like(d_mean), nll)

    return evaluate


class TestObjective(unittest.TestCase):
    def test_hinge_only_penalises_above_the_ceiling(self):
        d = torch.tensor([1.0, 1.0, 1.0])
        nll = torch.tensor([1.0, 3.0, 5.0])
        J = Mine.GreedyMiner.objective(d, nll, lam=2.0, nll_ceiling=3.0)
        self.assertAlmostEqual(float(J[0]), 1.0, places=6)   # under: no penalty
        self.assertAlmostEqual(float(J[1]), 1.0, places=6)   # exactly at: no penalty
        self.assertAlmostEqual(float(J[2]), 1.0 - 4.0, places=6)

    def test_lambda_zero_is_the_unconstrained_arm(self):
        d = torch.tensor([1.0])
        nll = torch.tensor([99.0])
        self.assertAlmostEqual(
            float(Mine.GreedyMiner.objective(d, nll, lam=0.0, nll_ceiling=3.0)[0]), 1.0)


class TestGreedySearch(unittest.TestCase):
    def setUp(self):
        self.seed_tokens = torch.zeros(12, dtype=torch.long)
        self.ev = fake_evaluate()

    def test_is_deterministic_under_seed(self):
        a = Mine.GreedyMiner(iters=6, cands=8, seed=3).search(
            seed_tokens=self.seed_tokens, evaluate=self.ev, d_vocab=32,
            nll_ceiling=10.0, arm="unconstrained", lam=0.0)
        b = Mine.GreedyMiner(iters=6, cands=8, seed=3).search(
            seed_tokens=self.seed_tokens, evaluate=self.ev, d_vocab=32,
            nll_ceiling=10.0, arm="unconstrained", lam=0.0)
        self.assertEqual(a.tokens, b.tokens)
        self.assertEqual(a.d_mean, b.d_mean)
        self.assertEqual([e["pos"] for e in a.edits], [e["pos"] for e in b.edits])

    def test_different_seeds_explore_differently(self):
        a = Mine.GreedyMiner(iters=6, cands=8, seed=1).search(
            seed_tokens=self.seed_tokens, evaluate=self.ev, d_vocab=32,
            nll_ceiling=10.0, lam=0.0)
        b = Mine.GreedyMiner(iters=6, cands=8, seed=2).search(
            seed_tokens=self.seed_tokens, evaluate=self.ev, d_vocab=32,
            nll_ceiling=10.0, lam=0.0)
        self.assertNotEqual([e["pos"] for e in a.edits], [e["pos"] for e in b.edits])

    def test_unconstrained_arm_climbs_freely(self):
        r = Mine.GreedyMiner(iters=20, cands=16, seed=0).search(
            seed_tokens=self.seed_tokens, evaluate=self.ev, d_vocab=32,
            nll_ceiling=2.5, arm="unconstrained", lam=0.0)
        self.assertGreater(r.d_mean, r.seed_d_mean)
        self.assertGreater(r.nll, 2.5)          # it left the distribution, as designed

    def test_constrained_arm_respects_the_perplexity_hinge(self):
        free = Mine.GreedyMiner(iters=20, cands=16, seed=0).search(
            seed_tokens=self.seed_tokens, evaluate=self.ev, d_vocab=32,
            nll_ceiling=2.5, arm="unconstrained", lam=0.0)
        tied = Mine.GreedyMiner(iters=20, cands=16, seed=0).search(
            seed_tokens=self.seed_tokens, evaluate=self.ev, d_vocab=32,
            nll_ceiling=2.5, arm="constrained", lam=5.0)
        # the constraint costs divergence; that gap is the reportable quantity
        self.assertLessEqual(tied.d_mean, free.d_mean)
        self.assertLess(tied.nll, free.nll)

    def test_never_edits_position_zero(self):
        r = Mine.GreedyMiner(iters=25, cands=8, seed=5).search(
            seed_tokens=self.seed_tokens, evaluate=self.ev, d_vocab=32,
            nll_ceiling=99.0, lam=0.0)
        self.assertTrue(all(e["pos"] >= 1 for e in r.edits))
        # position 0 is excluded from every metric, so editing it is a no-op
        self.assertEqual(r.tokens[0], int(self.seed_tokens[0]))

    def test_early_stopping_is_off_by_default(self):
        m = Mine.GreedyMiner(iters=10, cands=4, seed=0)
        self.assertIsNone(m.patience)
        # a hopeless objective still runs the full budget rather than declaring
        # "no witness found" early
        flat = lambda b: (torch.zeros(b.shape[0], b.shape[1], 32),  # noqa: E731
                          SeqMetrics(torch.zeros(b.shape[0]), torch.zeros(b.shape[0]),
                                     torch.zeros(b.shape[0]), torch.zeros(b.shape[0])))
        r = m.search(seed_tokens=self.seed_tokens, evaluate=flat, d_vocab=32,
                     nll_ceiling=1.0, lam=0.0)
        self.assertEqual(r.iters_run, 10)
        self.assertEqual(r.n_edits, 0)

    def test_trajectory_records_every_iteration(self):
        r = Mine.GreedyMiner(iters=7, cands=4, seed=0).search(
            seed_tokens=self.seed_tokens, evaluate=self.ev, d_vocab=32,
            nll_ceiling=5.0, lam=2.0)
        self.assertEqual(len(r.trajectory), 8)          # the seed + 7 iterations
        self.assertEqual(r.trajectory[0]["it"], -1)
        self.assertIn("accepted", r.trajectory[1])

    def test_proposals_are_half_model_half_uniform(self):
        m = Mine.GreedyMiner(cands=10, seed=0)
        g = torch.Generator().manual_seed(0)
        _, cand, src = m.propose(self.seed_tokens, torch.zeros(12, 32), g, 32)
        self.assertEqual(len(cand), 10)
        self.assertEqual(src.count("model"), 5)
        self.assertEqual(src.count("uniform"), 5)


def disagreeing_evaluate(mean_token: int = 3, max_token: int = 7, vocab: int = 32):
    """d_mean and d_max climb on DIFFERENT tokens, so the two objectives are
    distinguishable: a search can only look good on the metric it targeted."""

    def evaluate(batch: torch.Tensor):
        hits_mean = (batch == mean_token).float().sum(-1)
        hits_max = (batch == max_token).float().sum(-1)
        d_mean = 0.1 + 0.5 * hits_mean
        d_max = 0.1 + 0.5 * hits_max
        nll = 2.0 + 0.5 * (hits_mean + hits_max)
        lb = torch.zeros(batch.shape[0], batch.shape[1], vocab)
        return lb, SeqMetrics(d_mean, d_max, torch.zeros_like(d_mean), nll)

    return evaluate


class TestObjectiveMetric(unittest.TestCase):
    """The d_max objective option (added for experiment 09's preregistered arm)."""

    def setUp(self):
        self.seed_tokens = torch.zeros(12, dtype=torch.long)

    def test_default_is_the_extracted_d_mean_search(self):
        m = Mine.GreedyMiner(iters=4, cands=8, seed=0)
        self.assertEqual(m.objective_metric, "d_mean")
        self.assertEqual(m.name_full, "greedy")
        r = m.search(seed_tokens=self.seed_tokens, evaluate=fake_evaluate(),
                     d_vocab=32, nll_ceiling=99.0, lam=0.0)
        self.assertEqual(r.miner, "greedy")
        self.assertEqual(r.params["objective_metric"], "d_mean")

    def test_unknown_objective_metric_is_refused(self):
        with self.assertRaises(ValueError):
            Mine.GreedyMiner(objective_metric="flip")

    def test_d_max_objective_climbs_d_max_not_d_mean(self):
        ev = disagreeing_evaluate()
        r_max = Mine.GreedyMiner(iters=25, cands=16, seed=0,
                                 objective_metric="d_max").search(
            seed_tokens=self.seed_tokens, evaluate=ev, d_vocab=32,
            nll_ceiling=99.0, lam=0.0)
        r_mean = Mine.GreedyMiner(iters=25, cands=16, seed=0).search(
            seed_tokens=self.seed_tokens, evaluate=ev, d_vocab=32,
            nll_ceiling=99.0, lam=0.0)
        # each arm moves its own metric...
        self.assertGreater(r_max.d_max, r_max.seed_d_max)
        self.assertGreater(r_mean.d_mean, r_mean.seed_d_mean)
        # ...and beats the other arm ON that metric
        self.assertGreater(r_max.d_max, r_mean.d_max)
        self.assertGreater(r_mean.d_mean, r_max.d_mean)

    def test_d_max_arm_respects_the_perplexity_hinge(self):
        ev = disagreeing_evaluate()
        free = Mine.GreedyMiner(iters=20, cands=16, seed=0,
                                objective_metric="d_max").search(
            seed_tokens=self.seed_tokens, evaluate=ev, d_vocab=32,
            nll_ceiling=2.5, arm="unconstrained", lam=0.0)
        tied = Mine.GreedyMiner(iters=20, cands=16, seed=0,
                                objective_metric="d_max").search(
            seed_tokens=self.seed_tokens, evaluate=ev, d_vocab=32,
            nll_ceiling=2.5, arm="constrained", lam=5.0)
        self.assertLessEqual(tied.d_max, free.d_max)
        self.assertLess(tied.nll, free.nll)

    def test_d_max_arm_is_distinguishable_in_the_record(self):
        r = Mine.GreedyMiner(iters=3, cands=4, seed=0,
                             objective_metric="d_max").search(
            seed_tokens=self.seed_tokens, evaluate=fake_evaluate(), d_vocab=32,
            nll_ceiling=99.0, lam=0.0)
        self.assertEqual(r.miner, "greedy[d_max]")
        self.assertEqual(r.params["objective_metric"], "d_max")
        d = r.to_dict()
        self.assertIn("seed_d_max", d)
        self.assertEqual(d["seed_d_max"], r.seed_d_max)
        # the trajectory now carries d_max at the seed and at every iteration
        self.assertTrue(all("d_max" in t for t in r.trajectory))

    def test_d_max_objective_is_deterministic_under_seed(self):
        kw = dict(seed_tokens=self.seed_tokens, evaluate=disagreeing_evaluate(),
                  d_vocab=32, nll_ceiling=10.0, lam=0.0)
        a = Mine.GreedyMiner(iters=6, cands=8, seed=3, objective_metric="d_max").search(**kw)
        b = Mine.GreedyMiner(iters=6, cands=8, seed=3, objective_metric="d_max").search(**kw)
        self.assertEqual(a.tokens, b.tokens)
        self.assertEqual(a.d_max, b.d_max)


class TestSeedSelection(unittest.TestCase):
    def test_interleaves_tail_and_median_seeds(self):
        rows = [dict(row=i, d_mean=float(i), nll=1.0) for i in range(20)]
        seeds = Mine.pick_seeds(rows, n_tail=3, n_median=3)
        self.assertEqual([k for k, _ in seeds],
                         ["tail", "median", "tail", "median", "tail", "median"])
        # a run cut short after two seeds still covers both kinds
        self.assertEqual({k for k, _ in seeds[:2]}, {"tail", "median"})

    def test_tail_seeds_are_the_highest_divergence_rows(self):
        rows = [dict(row=i, d_mean=float(i), nll=1.0) for i in range(20)]
        seeds = Mine.pick_seeds(rows, n_tail=2, n_median=0)
        self.assertEqual([r["row"] for _, r in seeds], [19, 18])

    def test_handles_an_empty_row_set(self):
        self.assertEqual(Mine.pick_seeds([], 3, 3), [])


class TestDetectors(unittest.TestCase):
    """These are DETECTORS over user-supplied text, not shipped boilerplate."""

    def test_keyword_groups_fire_on_their_own_names(self):
        self.assertIn("copyright", Mine.kw_hits("Copyright 2019 Someone"))
        self.assertIn("w3c", Mine.kw_hits("published by the W3C"))
        self.assertEqual(Mine.kw_hits("the cat sat on the mat"), [])

    def test_classifier_prioritises_license_over_code(self):
        # license text embedded in a code comment must label as license
        self.assertEqual(
            Mine.classify_witness("# This file is licensed under the terms above\nimport os"),
            "license")

    def test_classifier_labels_whitespace_soup(self):
        self.assertEqual(Mine.classify_witness("a\t\t\t\t\t\t\t\t\t b \n\n\n\n\n c"),
                         "css_whitespace")

    def test_classifier_falls_through_to_other(self):
        self.assertEqual(Mine.classify_witness("the quick brown fox jumped over"), "other")

    def test_best_window_picks_the_densest_window(self):
        vocab = {0: "hello ", 1: "copyright 2020 ", 2: "warranty license "}
        ids = [0, 0, 1, 2, 0, 0]
        decode = lambda w: "".join(vocab[i] for i in w)  # noqa: E731
        off, win, hits = Mine.best_window(ids, decode, seq_len=2, stride=1)
        self.assertEqual(off, 2)
        self.assertEqual(sorted(hits), ["copyright", "license", "warranty"])

    def test_corpus_signature_names_what_makes_a_corpus_different(self):
        sig = Mine.corpus_signature(dict(model="m", seq_len=64, seed=0, n_corpus=10,
                                         n_probes=2, bos=True))
        self.assertEqual(set(sig), {"model", "seq_len", "seed", "n_corpus", "n_probes", "bos"})
        other = Mine.corpus_signature(dict(model="m", seq_len=64, seed=0, n_corpus=99,
                                           n_probes=2, bos=True))
        self.assertNotEqual(sig, other)   # a smoke corpus is not a full corpus


if __name__ == "__main__":
    unittest.main()
