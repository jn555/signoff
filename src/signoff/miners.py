"""Witness generation: corpus sweep, probe re-windowing, constrained search.

PROVENANCE.  Extracted from
  experiments/01-divergence-witnesses/witness.py            (build_corpus)
  experiments/01-divergence-witnesses/run_stage_b.py        (the constrained
                                                             greedy coordinate
                                                             ascent, dual arms)
  experiments/03-mechanism-and-validity/validity_common.py  (KW_GROUPS, kw_hits,
                                                             best_window, probe
                                                             re-windowing)

TWO KINDS OF WITNESS HUNTING, and they answer different questions:

  CORPUS SWEEP — "where does this replacement already fail, on text the model
  was built for?"  Deterministic seeded windowing, one window per document.
  The tail of this distribution is the honest headline.

  CONSTRAINED SEARCH — "can I MANUFACTURE a failure without leaving the
  distribution?"  Greedy coordinate ascent on
        J = d_mean - lambda * max(0, nll - nll_p99_corpus)
  The hinge is the whole point: an unconstrained search trivially finds
  gibberish with enormous divergence and proves nothing. Both arms are run —
  constrained (lambda > 0) and unconstrained (lambda = 0) — because the
  unconstrained arm is the control that tells you what the constraint cost.

  Early stopping is OFF by default (`patience=None`). Stopping a constrained
  arm early biases it toward a false "no witness found", which is precisely the
  conclusion this tool must never reach by accident.

NO PROBE OR LICENSE TEXT BODY is stored, logged or shipped anywhere in this
module.  Probe corpora are referenced by (source, doc, offset) and matched
KEYWORD GROUPS; the regexes below are DETECTORS, not boilerplate.  A user
supplies their own probe texts locally; nothing is redistributed.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import torch
import torch.nn.functional as F

# ------------------------------------------------------------ keyword groups
# Verbatim from validity_common.py (which took them from exp-02's
# run_b_family.py).  DETECTORS, not text bodies.
KW_GROUPS = {
    "license": r"\blicen[sc]e[sd]?\b",
    "warranty": r"\bwarrant(?:y|ies|ed|ies)\b",
    "merchantability": r"\bmerchantability\b",
    "liability": r"\bliabilit(?:y|ies)\b|\bliable\b",
    "redistribution": r"\bredistribut\w*",
    "copyright": r"\bcopyright\b|\(c\)\s*\d{4}|©\s*\d{4}",
    "as_is": r"\bAS\s+IS\b",
    "rights_reserved": r"\ball rights reserved\b",
    "permission_grant": r"\bhereby grant\w*\b|\bpermission is hereby\b",
    "gutenberg": r"Project Gutenberg",
    "rfc2119": r"RFC\s*2119|\bMUST NOT\b|\bSHALL NOT\b|\bSHOULD NOT\b|"
               r"\bREQUIRED\b.{0,40}\bRECOMMENDED\b",
    "w3c": r"\bW3C\b",
    "fsf_gpl": r"GNU (?:General|Lesser) Public License|Free Software Foundation",
    "disclaimer": r"\bdisclaim\w*\b|\bIN NO EVENT\b|\bWITHOUT WARRANT",
    "terms": r"\bterms and conditions\b|\bthis agreement\b|\bthe Software\b",
}
KW_RE = {k: re.compile(v, re.I) for k, v in KW_GROUPS.items()}

# The control blocklist: family-like text is excluded from the control pool, or
# the paired test compares the family against itself.  From common02.py.
LICENSE_RE = re.compile(
    r"licen[sc]e|warrant|liabilit|copyright|GNU General Public|"
    r"permission is hereby|redistribut|MERCHANTABILITY",
    re.I,
)
CSS_RE = re.compile(
    r"background-|font-(family|size|weight)|margin\s*:|padding\s*:|border\s*:|"
    r"#[0-9a-fA-F]{6}\b|<div|<td|<tr|&nbsp|no-repeat|text-align|<span|px;|</",
)
CODE_RE = re.compile(
    r"\bdef |\bclass |\bimport |\bfunction\b|=>|\bpublic \b|\bstatic \b|\bvoid \b|"
    r"SELECT |#include|printf|\breturn\b|\bvar \b|\bconst \b|\}\n|;\n|::|\bnamespace\b|"
    r"\busing \b|\bmov\b|\bpush\b",
)


def kw_hits(text: str) -> list[str]:
    return sorted(k for k, r in KW_RE.items() if r.search(text))


def classify_witness(text: str) -> str:
    """Coarse family LABEL for a witness.  Verbatim from common02.py.

    Priority matters: license text is often embedded in code comments, so the
    license test runs first; the CSS/whitespace test runs before the code test
    because HTML/CSS soup trips several code tokens.

    Returns a label — never the text.  This is how the tail table is reportable.
    """
    if LICENSE_RE.search(text):
        return "license"
    ws = sum(c in " \t\n\r" for c in text) / max(len(text), 1)
    if ws > 0.30 or len(CSS_RE.findall(text)) >= 3:
        return "css_whitespace"
    if CODE_RE.search(text):
        return "code"
    return "other"


def best_window(ids: Sequence[int], decode, seq_len: int, stride: int = 32):
    """The window of `ids` matching the most keyword groups.  From validity_common."""
    best = None
    for off in range(0, max(1, len(ids) - seq_len + 1), stride):
        w = ids[off : off + seq_len]
        if len(w) < seq_len:
            break
        h = kw_hits(decode(w))
        if best is None or len(h) > len(best[2]):
            best = (off, w, h)
    return best


# ------------------------------------------------------------- corpus sweep


def build_corpus(tokenizer, n_seqs: int, seq_len: int = 64, seed: int = 0,
                 bos_id: int | None = None, dataset: str = "NeelNanda/pile-10k",
                 split: str = "train"):
    """`n_seqs x seq_len` token windows.  Verbatim from witness.py::build_corpus.

    One non-overlapping window per document from a deterministic seeded offset,
    documents walked in order until `n_seqs` are collected; a second pass takes
    a second window from long documents if the first pass came up short.

    `bos_id` prepends BOS, so a window is `[BOS] + (seq_len-1)` REAL tokens and
    is still `seq_len` wide.  Every metric already drops position 0, so no
    definition changes — only the real-token count, recorded in the meta as
    `real_tokens_per_window`.  This is NOT cosmetic: BOS-free evaluation of a
    BOS-trained suite moved corpus NLL by 3.8 nats.
    """
    from datasets import load_dataset

    real_len = seq_len - 1 if bos_id is not None else seq_len
    ds = load_dataset(dataset, split=split)
    g = torch.Generator().manual_seed(seed)
    seqs, meta = [], []
    for i in range(len(ds)):
        if len(seqs) >= n_seqs:
            break
        text = ds[i]["text"]
        if not text or len(text) < 200:
            continue
        ids = tokenizer(text, truncation=True, max_length=4096)["input_ids"]
        if len(ids) < real_len + 1:
            continue
        off = int(torch.randint(0, len(ids) - real_len, (1,), generator=g).item())
        seqs.append(ids[off : off + real_len])
        meta.append(dict(doc=i, offset=off,
                         pile_set=(ds[i].get("meta") or {}).get("pile_set_name")))
    if len(seqs) < n_seqs:
        for i in range(len(ds)):
            if len(seqs) >= n_seqs:
                break
            text = ds[i]["text"]
            if not text or len(text) < 2000:
                continue
            ids = tokenizer(text, truncation=True, max_length=8192)["input_ids"]
            if len(ids) < 2 * real_len + 1:
                continue
            off = int(torch.randint(0, len(ids) - real_len, (1,), generator=g).item())
            seqs.append(ids[off : off + real_len])
            meta.append(dict(doc=i, offset=off,
                             pile_set=(ds[i].get("meta") or {}).get("pile_set_name"),
                             pass2=True))
    toks = torch.tensor(seqs[:n_seqs], dtype=torch.long)
    meta = meta[:n_seqs]
    if bos_id is not None:
        toks = torch.cat(
            [torch.full((toks.shape[0], 1), int(bos_id), dtype=torch.long), toks], 1)
    assert toks.shape[1] == seq_len, (tuple(toks.shape), seq_len)
    return toks, meta


def mine_probes(tokenizer, *, probe_dir: str | None = None,
                pile_docs: Sequence[int] = (), seq_len: int = 64,
                per_file: int = 10, n_probes: int = 175, bos_id: int | None = None,
                stride: int = 24, dataset: str = "NeelNanda/pile-10k"):
    """Re-window a user-supplied probe family under THIS tokenizer.

    From validity_common.py::mine.  Re-windowing under the target tokenizer is
    what makes a probe family comparable across model families: the same TEXT,
    not the same token ids, is the thing being held fixed.

    NO PROBE TEXT IS SHIPPED WITH THIS PACKAGE.  `probe_dir` points at local
    `.txt` files the user provides; `pile_docs` are document ids in a public
    dataset.  Outputs reference (source, doc, offset) and matched keyword
    groups only.

    Sources are round-robined so that any truncated prefix stays representative.
    """
    real_len = seq_len - 1 if bos_id is not None else seq_len
    decode = tokenizer.decode
    probes: list[dict] = []

    if probe_dir and os.path.isdir(probe_dir):
        for fn in sorted(os.listdir(probe_dir)):
            if not fn.endswith(".txt"):
                continue
            stem = fn[:-4]
            path = os.path.join(probe_dir, fn)
            ids = tokenizer(open(path, encoding="utf-8", errors="replace").read(),
                            truncation=True, max_length=8192)["input_ids"]
            taken = 0
            for off in range(0, max(1, len(ids) - real_len + 1), stride):
                if taken >= per_file:
                    break
                w = ids[off : off + real_len]
                if len(w) < real_len:
                    break
                h = kw_hits(decode(w))
                if not h:
                    continue
                probes.append(dict(pid=f"file:{stem}:{off}", source="probefile", doc=stem,
                                   offset=off, kw=h, tokens=[int(x) for x in w]))
                taken += 1

    if pile_docs:
        from datasets import load_dataset

        ds = load_dataset(dataset, split="train")
        for i in pile_docs:
            ids = tokenizer(ds[int(i)]["text"], truncation=True, max_length=2048)["input_ids"]
            if len(ids) < real_len:
                continue
            bw = best_window(ids, decode, real_len)
            if bw is None or not bw[2]:
                continue
            probes.append(dict(pid=f"pile:{int(i)}:{bw[0]}", source="pile", doc=int(i),
                               offset=int(bw[0]), kw=bw[2],
                               pile_set=(ds[int(i)].get("meta") or {}).get("pile_set_name"),
                               tokens=[int(x) for x in bw[1]]))

    buckets: dict[str, list] = {}
    for p in probes:
        buckets.setdefault(p["source"], []).append(p)
    ordered, k = [], 0
    while len(ordered) < len(probes):
        for s in sorted(buckets):
            if k < len(buckets[s]):
                ordered.append(buckets[s][k])
        k += 1
    return ordered[:n_probes]


def corpus_signature(meta: dict) -> dict:
    """`MINE_SIG` — the corpus identity every downstream checkpoint carries.

    From run_b_gemma.py.  This is what stops a smoke run's artifacts from being
    resumed into a full run.
    """
    return dict(model=meta.get("model"), seq_len=meta.get("seq_len"),
                seed=meta.get("seed"), n_corpus=meta.get("n_corpus"),
                n_probes=meta.get("n_probes"), bos=bool(meta.get("bos")))


# ------------------------------------------------------------ search miners


@dataclass
class SearchResult:
    """One (seed, arm) search.  Tokens are kept; TEXT is the caller's business."""

    seed_row: int
    seed_kind: str
    arm: str
    lam: float
    iters_run: int
    seed_d_mean: float
    seed_nll: float
    d_mean: float
    d_max: float
    flip: float
    nll: float
    objective: float
    n_edits: int
    #: The seed's d_max, recorded since experiment 09 made d_max a targetable
    #: objective — a d_max arm's movement cannot be read off a record that only
    #: kept the seed's d_mean.  NaN in records written before the field existed.
    seed_d_max: float = float("nan")
    edits: list[dict] = field(default_factory=list)
    tokens: list[int] = field(default_factory=list)
    trajectory: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    #: Which miner produced this.  Two miners' finals are only comparable if
    #: you can tell them apart in the checkpoint, and experiment 07 had to
    #: reconstruct exp-01's search budget from a LOG LINE — see `n_evals`.
    miner: str = "greedy"
    #: The budget, RECORDED rather than reconstructed downstream: candidate
    #: forward evaluations of the (base, replacement) pair, including the
    #: seed's own.  This is the `m` a return-level test charges the run.
    n_evals: int = 0
    #: Gradient (forward+backward) passes.  Zero for a gradient-free miner; a
    #: separate column because it is a different unit of compute, and a
    #: head-to-head that hides it is not budget-matched.
    n_grad: int = 0
    #: The miner's own hyperparameters, so a record is self-describing.
    params: dict = field(default_factory=dict)
    #: Best point VISITED, which equals the final only under monotone
    #: acceptance.  Recorded always, so a non-monotone arm is reported at its
    #: best rather than wherever its last step happened to land.
    best: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dict(seed_row=self.seed_row, seed_kind=self.seed_kind, arm=self.arm,
                    lam=self.lam, iters_run=self.iters_run,
                    seed_d_mean=self.seed_d_mean, seed_d_max=self.seed_d_max,
                    seed_nll=self.seed_nll,
                    d_mean=self.d_mean, d_max=self.d_max, flip=self.flip, nll=self.nll,
                    objective=self.objective, n_edits=self.n_edits, edits=self.edits,
                    tokens=self.tokens, trajectory=self.trajectory, seconds=self.seconds,
                    miner=self.miner, n_evals=self.n_evals, n_grad=self.n_grad,
                    params=self.params, best=self.best)


class Miner:
    """propose-edit -> evaluate -> accept.

    v0.1 shipped `GreedyMiner`; v0.2 adds `GCGMiner`, the gradient-guided one.
    They differ in `propose` and in nothing else that touches a reported number:
    both hill-climb the same J, both evaluate every proposal EXACTLY, and both
    return the same `SearchResult`.  That is what makes an oracle-strength
    head-to-head a controlled comparison rather than two different experiments.

    `propose` returns `(position, candidate_tokens, sources)`.  `position` is an
    int for a single-position miner (greedy edits one position per iteration,
    all candidates at that position) and a LongTensor of per-candidate positions
    for a multi-position miner (GCG spreads its batch over the window).  Callers
    inside this module build the candidate batch accordingly; `apply_candidates`
    handles both shapes.
    """

    name = "miner"

    def propose(self, current: torch.Tensor, base_logits: torch.Tensor,
                generator: torch.Generator, d_vocab: int):
        raise NotImplementedError

    def search(self, *args, **kwargs) -> SearchResult:
        raise NotImplementedError


def apply_candidates(current: torch.Tensor, position, cand_tok: torch.Tensor
                     ) -> torch.Tensor:
    """`(T,) -> (n_cands, T)` with one token substituted per row.

    `position` is an int (every candidate at the same position) or a (n_cands,)
    LongTensor (one position per candidate).
    """
    batch = current.unsqueeze(0).repeat(len(cand_tok), 1)
    if isinstance(position, torch.Tensor) and position.dim() > 0:
        batch[torch.arange(len(cand_tok)), position.long()] = cand_tok
    else:
        batch[:, int(position)] = cand_tok
    return batch


class GreedyMiner(Miner):
    """Perplexity-constrained greedy coordinate ascent.  From run_stage_b.py.

    Each iteration picks ONE position (uniform over positions 1..T-1), proposes
    `cands` replacement tokens — half sampled from the base model's own
    predictive distribution at that position, half uniform over the vocabulary —
    evaluates all of them in one batch, and accepts the best if it improves J.

    The half-and-half proposal matters: model-sampled candidates keep the string
    plausible (they are what the constrained arm can actually use), uniform
    candidates are what find the pathological tokens.

    `objective_metric` selects WHICH divergence the hinged objective climbs.
    PROVENANCE: the search extracted from run_stage_b.py targeted `d_mean` only
    (the default here, which reproduces it bit-for-bit).  Experiment 09
    (experiments/09-moderator-decider/SPEC.md, a committed preregistration)
    needed the same constrained search targeted at `d_max` — the bounded-fit
    metric of exp-07's gemma envelope — so the divergence term became
    selectable.  Nothing else changes: same hinge, same proposal distribution,
    same acceptance rule, and all four metrics are recorded either way.
    `SearchResult.miner` distinguishes the arms ("greedy" vs "greedy[d_max]").
    """

    name = "greedy"

    #: divergences the objective may climb; both are per-sequence SeqMetrics
    #: fields, so nothing about evaluation or recording depends on the choice
    OBJECTIVE_METRICS = ("d_mean", "d_max")

    def __init__(self, iters: int = 20, cands: int = 32, lam: float = 2.0,
                 patience: int | None = None, seed: int = 0,
                 objective_metric: str = "d_mean"):
        self.iters = int(iters)
        self.cands = int(cands)
        self.lam = float(lam)
        # None = off. Early stopping biases the constrained arm toward a false
        # "refuted"; run_stage_b.py defaults it to 999 for the same reason.
        self.patience = patience
        self.seed = int(seed)
        if objective_metric not in self.OBJECTIVE_METRICS:
            raise ValueError(
                f"objective_metric must be one of {self.OBJECTIVE_METRICS}, got "
                f"{objective_metric!r}")
        self.objective_metric = objective_metric

    @property
    def name_full(self) -> str:
        """"greedy" for the extracted d_mean search; "greedy[d_max]" otherwise,
        so two arms' finals can never be confused in a checkpoint."""
        if self.objective_metric == "d_mean":
            return self.name
        return f"{self.name}[{self.objective_metric}]"

    def propose(self, current: torch.Tensor, base_logits: torch.Tensor,
                generator: torch.Generator, d_vocab: int):
        """Returns (position, candidate_tokens, sources)."""
        p = int(torch.randint(1, current.shape[0], (1,), generator=generator).item())
        n_half = self.cands // 2
        probs = F.softmax(base_logits[p - 1].float(), dim=-1).cpu()
        samp = torch.multinomial(probs, n_half, replacement=True, generator=generator)
        unif = torch.randint(0, d_vocab, (self.cands - n_half,), generator=generator)
        return p, torch.cat([samp, unif]), ["model"] * n_half + ["uniform"] * (self.cands - n_half)

    @staticmethod
    def objective(d_mean: torch.Tensor, nll: torch.Tensor, lam: float,
                  nll_ceiling: float) -> torch.Tensor:
        """J = d - lambda * max(0, nll - nll_ceiling).  Verbatim from
        run_stage_b.py; the first argument is whichever divergence
        `objective_metric` selects (named `d_mean` for the extracted default)."""
        return d_mean - lam * torch.clamp(nll - nll_ceiling, min=0.0)

    def search(self, *, seed_tokens: torch.Tensor, evaluate: Callable,
               d_vocab: int, nll_ceiling: float, lam: float | None = None,
               arm: str = "constrained", seed_row: int = -1, seed_kind: str = "tail",
               index: int = 0) -> SearchResult:
        """Run one (seed, arm).

        `evaluate(batch_tokens) -> (base_logits, SeqMetrics)` is the ONLY
        model-specific part, so this miner is testable against a synthetic
        objective with no weights at all.
        """
        lam = self.lam if lam is None else float(lam)
        g = torch.Generator(device="cpu").manual_seed(
            self.seed * 100003 + index * 17 + int(lam * 7))
        cur = seed_tokens.clone()
        ts = time.time()
        lb, m = evaluate(cur.unsqueeze(0))
        cur_dm, cur_dx = float(m.d_mean[0]), float(m.d_max[0])
        cur_fl, cur_nl = float(m.flip[0]), float(m.nll[0])
        cur_J = float(self.objective(
            getattr(m, self.objective_metric), m.nll, lam, nll_ceiling)[0])
        # d_max joined the trajectory when it became targetable (experiment 09):
        # a d_max arm's per-iteration movement is otherwise unrecoverable.
        traj = [dict(it=-1, J=cur_J, d_mean=cur_dm, d_max=cur_dx, nll=cur_nl,
                     accepted=True)]
        edits: list[dict] = []
        stall = 0
        n_evals = 1
        for it in range(self.iters):
            p, cand_tok, src = self.propose(cur, lb[0], g, d_vocab)
            batch = apply_candidates(cur, p, cand_tok)
            n_evals += int(batch.shape[0])
            lbc, mc = evaluate(batch)
            Jc = self.objective(
                getattr(mc, self.objective_metric), mc.nll, lam, nll_ceiling)
            k = int(Jc.argmax())
            accepted = float(Jc[k]) > cur_J + 1e-6
            if accepted:
                edits.append(dict(it=it, pos=p, old=int(cur[p]), new=int(cand_tok[k]),
                                  src=src[k], J=float(Jc[k])))
                cur = batch[k].clone()
                cur_J = float(Jc[k])
                cur_dm, cur_dx = float(mc.d_mean[k]), float(mc.d_max[k])
                cur_fl, cur_nl = float(mc.flip[k]), float(mc.nll[k])
                # clone: a view would pin the whole candidate logit tensor
                lb = lbc[k : k + 1].clone()
                stall = 0
            else:
                stall += 1  # lb still corresponds to cur, which is unchanged
            traj.append(dict(it=it, pos=p, J=cur_J, d_mean=cur_dm, d_max=cur_dx,
                             nll=cur_nl, accepted=accepted))
            del lbc
            if self.patience is not None and stall >= self.patience:
                break
        return SearchResult(
            seed_row=seed_row, seed_kind=seed_kind, arm=arm, lam=lam,
            iters_run=len(traj) - 1, seed_d_mean=float(traj[0]["d_mean"]),
            seed_d_max=float(traj[0]["d_max"]),
            seed_nll=float(traj[0]["nll"]), d_mean=cur_dm, d_max=cur_dx, flip=cur_fl,
            nll=cur_nl, objective=cur_J, n_edits=len(edits), edits=edits,
            tokens=[int(x) for x in cur], trajectory=traj, seconds=time.time() - ts,
            miner=self.name_full, n_evals=n_evals, n_grad=0,
            params=dict(iters=self.iters, cands=self.cands, patience=self.patience,
                        seed=self.seed, accept="improve",
                        objective_metric=self.objective_metric,
                        proposal="half model-sampled, half uniform, one position"),
            # monotone acceptance: the final IS the best point visited
            best=dict(it=(edits[-1]["it"] if edits else -1), objective=cur_J,
                      d_mean=cur_dm, d_max=cur_dx, flip=cur_fl, nll=cur_nl),
        )


class GCGMiner(Miner):
    """Gradient-guided coordinate ascent — GCG (Zou et al., arXiv 2307.15043),
    ported to a MITER objective.

    THE PORT IS THE INTERESTING PART.  Canonical GCG differentiates a single
    model's loss on a target string.  Here the objective is a DIVERGENCE between
    two models, one of which is the substituted one, so `dJ/d(one-hot)` has to
    flow through both forward passes of the miter (`gradients.GradientOracle`).
    Everything downstream is standard GCG:

      1. rank substitutions at every editable position by the linearised
         `dJ/dX[p, v]`;
      2. keep each position's `topk`;
      3. sample `cands` (position, token) pairs from that shortlist — SPREAD
         over the window, unlike the greedy miner's one-position-per-iteration;
      4. evaluate all of them EXACTLY, in one batch, and accept the best.

    Step 4 is why a wrong gradient costs efficiency and never correctness: no
    number this miner reports is ever the linearisation's, always the exact
    evaluation's.

    TWO KNOBS EXIST BECAUSE THEY CONFOUND A HEAD-TO-HEAD, and both default to
    the setting that makes GCG differ from `GreedyMiner` in the PROPOSAL ONLY:

      `accept="improve"` (default) hill-climbs exactly as the greedy miner does.
      Canonical GCG takes the batch argmax unconditionally (`accept="always"`),
      which is a different search — it can walk downhill to escape a local
      optimum, and at the 20-iteration budget this tool inherits from experiment
      01 that is a hyperparameter, not an improvement.  Either way `best`
      records the best point visited.

      `guidance="gradient"` (default) is GCG.  `guidance="random"` keeps the
      multi-position proposal and throws the gradient away — the ablation that
      separates "gradient guidance helps" from "spreading edits over the window
      helps", which are otherwise measured together and reported as one.
    """

    name = "gcg"

    def __init__(self, iters: int = 20, cands: int = 32, topk: int = 256,
                 lam: float = 2.0, patience: int | None = None, seed: int = 0,
                 gradient: Callable | None = None, accept: str = "improve",
                 guidance: str = "gradient"):
        if accept not in ("improve", "always"):
            raise ValueError(f"accept must be 'improve' or 'always', got {accept!r}")
        if guidance not in ("gradient", "random"):
            raise ValueError(f"guidance must be 'gradient' or 'random', got {guidance!r}")
        self.iters = int(iters)
        self.cands = int(cands)
        self.topk = int(topk)
        self.lam = float(lam)
        # None = off, for the reason in GreedyMiner: stopping a constrained arm
        # early biases it toward a false "no witness found".
        self.patience = patience
        self.seed = int(seed)
        self.gradient = gradient
        self.accept = accept
        self.guidance = guidance

    @property
    def name_full(self) -> str:
        return f"gcg[{self.guidance},{self.accept}]"

    #: Same expression as the greedy miner's, referenced rather than re-typed so
    #: the two arms cannot drift apart.
    objective = staticmethod(GreedyMiner.objective)

    def propose(self, current: torch.Tensor, base_logits: torch.Tensor,
                generator: torch.Generator, d_vocab: int, *,
                grad: torch.Tensor | None = None):
        """`(positions, tokens, sources)` — positions is a (cands,) LongTensor.

        Position 0 is never editable: every metric drops it, so an edit there is
        a wasted evaluation (the greedy miner excludes it for the same reason).
        The CURRENT token at each position is excluded from its own shortlist —
        proposing it would spend an evaluation on a no-op.
        """
        T = int(current.shape[0])
        pos = torch.randint(1, T, (self.cands,), generator=generator)
        if self.guidance == "random" or grad is None:
            tok = torch.randint(0, int(d_vocab), (self.cands,), generator=generator)
            return pos, tok, ["uniform"] * self.cands
        g = grad.detach().float().clone()                       # (T, d_vocab)
        g[torch.arange(T), current.cpu().long()] = float("-inf")  # no self-swaps
        k = max(1, min(self.topk, int(d_vocab)))
        shortlist = g[1:].topk(k, dim=-1).indices               # (T-1, k)
        pick = torch.randint(0, k, (self.cands,), generator=generator)
        tok = shortlist[pos - 1, pick]
        return pos, tok, ["gradient-topk"] * self.cands

    def search(self, *, seed_tokens: torch.Tensor, evaluate: Callable,
               d_vocab: int, nll_ceiling: float, gradient: Callable | None = None,
               lam: float | None = None, arm: str = "constrained",
               seed_row: int = -1, seed_kind: str = "tail",
               index: int = 0) -> SearchResult:
        """Run one (seed, arm).  Same contract as `GreedyMiner.search`.

        `gradient(tokens, lam=..., nll_ceiling=...) -> (T, d_vocab)` is the
        oracle; it may instead be given to the constructor, which is how a
        `Runner` (whose `search()` does not know about gradients) drives this
        miner.  In `guidance="gradient"` mode a missing oracle is an ERROR, not
        a fallback to random proposals: a GCG arm that silently degraded to
        random search would still be reported as GCG.
        """
        lam = self.lam if lam is None else float(lam)
        grad_fn = gradient or self.gradient
        if grad_fn is None and self.guidance == "gradient":
            raise ValueError(
                "GCGMiner needs a gradient oracle (see gradients.gradient_fn); pass "
                "`gradient=` to the constructor or to search(). Refusing to fall back "
                "to random proposals, which would be a different search under this "
                "miner's name.")
        g = torch.Generator(device="cpu").manual_seed(
            self.seed * 100003 + index * 17 + int(lam * 7))
        cur = seed_tokens.clone()
        ts = time.time()
        lb, m = evaluate(cur.unsqueeze(0))
        cur_dm, cur_dx = float(m.d_mean[0]), float(m.d_max[0])
        seed_dx = cur_dx
        cur_fl, cur_nl = float(m.flip[0]), float(m.nll[0])
        cur_J = float(self.objective(m.d_mean, m.nll, lam, nll_ceiling)[0])
        traj = [dict(it=-1, J=cur_J, d_mean=cur_dm, nll=cur_nl, accepted=True)]
        edits: list[dict] = []
        best = dict(it=-1, objective=cur_J, d_mean=cur_dm, d_max=cur_dx, flip=cur_fl,
                    nll=cur_nl)
        best_tokens = cur.clone()
        n_evals, n_grad, stall = 1, 0, 0
        for it in range(self.iters):
            grad = None
            if self.guidance == "gradient":
                grad = grad_fn(cur, lam=lam, nll_ceiling=nll_ceiling)
                n_grad += 1
            pos, cand_tok, src = self.propose(cur, lb[0], g, d_vocab, grad=grad)
            batch = apply_candidates(cur, pos, cand_tok)
            n_evals += int(batch.shape[0])
            lbc, mc = evaluate(batch)
            Jc = self.objective(mc.d_mean, mc.nll, lam, nll_ceiling)
            k = int(Jc.argmax())
            improved = float(Jc[k]) > cur_J + 1e-6
            accepted = improved or self.accept == "always"
            if accepted:
                edits.append(dict(it=it, pos=int(pos[k]), old=int(cur[int(pos[k])]),
                                  new=int(cand_tok[k]), src=src[k], J=float(Jc[k])))
                cur = batch[k].clone()
                cur_J = float(Jc[k])
                cur_dm, cur_dx = float(mc.d_mean[k]), float(mc.d_max[k])
                cur_fl, cur_nl = float(mc.flip[k]), float(mc.nll[k])
                # clone: a view would pin the whole candidate logit tensor
                lb = lbc[k : k + 1].clone()
            if improved:
                stall = 0
            else:
                stall += 1
            if cur_J > best["objective"]:
                best = dict(it=it, objective=cur_J, d_mean=cur_dm, d_max=cur_dx,
                            flip=cur_fl, nll=cur_nl)
                best_tokens = cur.clone()
            traj.append(dict(it=it, pos=int(pos[k]), J=cur_J, d_mean=cur_dm,
                             nll=cur_nl, accepted=accepted))
            del lbc
            if self.patience is not None and stall >= self.patience:
                break
        return SearchResult(
            seed_row=seed_row, seed_kind=seed_kind, arm=arm, lam=lam,
            iters_run=len(traj) - 1, seed_d_mean=float(traj[0]["d_mean"]),
            seed_d_max=seed_dx,
            seed_nll=float(traj[0]["nll"]), d_mean=cur_dm, d_max=cur_dx, flip=cur_fl,
            nll=cur_nl, objective=cur_J, n_edits=len(edits), edits=edits,
            tokens=[int(x) for x in cur], trajectory=traj, seconds=time.time() - ts,
            miner=self.name_full, n_evals=n_evals, n_grad=n_grad,
            params=dict(iters=self.iters, cands=self.cands, topk=self.topk,
                        patience=self.patience, seed=self.seed, accept=self.accept,
                        guidance=self.guidance,
                        proposal=("top-k of dJ/d(one-hot), positions spread over the "
                                  "window" if self.guidance == "gradient" else
                                  "uniform tokens, positions spread over the window")),
            best=dict(best, tokens=[int(x) for x in best_tokens]),
        )


def pick_seeds(rows: Sequence[dict], n_tail: int = 10, n_median: int = 10):
    """Tail + median seeds, INTERLEAVED.  From run_stage_b.py.

    Interleaving is deliberate: a run cut short still covers both seed kinds,
    and "the search only improves already-bad windows" is exactly the artifact
    a median-seeded arm exists to rule out.
    """
    order = sorted(rows, key=lambda r: -r["d_mean"])
    tail = order[:n_tail]
    if rows:
        med_val = sorted(r["d_mean"] for r in rows)[len(rows) // 2]
        med = sorted(rows, key=lambda r: abs(r["d_mean"] - med_val))[:n_median]
    else:
        med = []
    seeds: list[tuple[str, dict]] = []
    for k in range(max(len(tail), len(med))):
        if k < len(tail):
            seeds.append(("tail", tail[k]))
        if k < len(med):
            seeds.append(("median", med[k]))
    return seeds


ARMS = (("constrained", None), ("unconstrained", 0.0))
