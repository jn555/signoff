"""Divergence metrics: the measurement half of the miter.

PROVENANCE.  Extracted (verbatim estimators) from
  experiments/01-divergence-witnesses/witness.py::metrics_from_logits
  experiments/03-mechanism-and-validity/common03.py::per_position_kl
Those two files are the reference implementations; the experiment directories
are immutable and this module must reproduce their numbers exactly.

Three conventions are baked in and must not be "cleaned up":

1. FULL-VOCABULARY softmax in float32.  Not a top-k or sampled approximation:
   the tail of the distribution is exactly where a replacement model is allowed
   to be wrong, so truncating it would hide witnesses.
2. POSITION 0 IS EXCLUDED from every metric.  With no BOS the first position is
   unconditioned; with a BOS it is the BOS itself.  Either way it is not a
   prediction, and including it moves every number.  `d_mean`/`d_max`/`flip`
   score positions 1..T-1; `nll` uses logits 0..T-2 predicting tokens 1..T-1
   (the same T-1 predictions, indexed from the other side).
3. CHUNKED over rows, with the chunk sized from the vocabulary.  At d_vocab
   256k a single (B*T, V) log-softmax is hundreds of MB; the chunk keeps peak
   memory flat.  Chunking is numerically inert — every row is independent.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

#: Position 0 never contributes to a metric.  See the module docstring.
EXCLUDE_POSITION_0 = True

#: Peak-bytes budget for one chunk of float32 log-probabilities.  1<<26 floats
#: = ~256 MB of log-probs per chunk (verbatim from common03.per_position_kl).
_CHUNK_BUDGET = 1 << 26

#: Row cap, verbatim from witness.metrics_from_logits (CH = 1024).  For GPT-2's
#: 50257-token vocabulary the budget rule yields 1335, so the cap binds and the
#: chunk is exactly exp-01's 1024 — the extraction is numerically identical.
_CHUNK_CAP = 1024


def chunk_rows(d_vocab: int) -> int:
    """Rows of logits to log-softmax at a time, for a `d_vocab`-wide vocabulary."""
    return max(1, min(_CHUNK_CAP, _CHUNK_BUDGET // max(int(d_vocab), 1)))


class SeqMetrics(NamedTuple):
    """Per-sequence divergence.  Tuple-unpackable in the exp-01 order.

    d_mean : mean over positions t>=1 of KL(P_base(.|x_<t) || P_replacement(.|x_<t))
    d_max  : max of the same per-position KL field
    flip   : fraction of positions t>=1 where the argmax token differs
    nll    : BASE-model mean per-token NLL of the actual next token.  This is a
             property of the input, not of the replacement — it is what the
             constrained search is constrained by, and what family tests match on.
    """

    d_mean: torch.Tensor
    d_max: torch.Tensor
    flip: torch.Tensor
    nll: torch.Tensor


@torch.no_grad()
def per_position_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(P||Q) per (batch, position), float32, chunked.  Returns (B, T).

    Position 0 is NOT dropped here — callers slice `[:, 1:]`.  Returning the
    full field is what per-position localisation diagnostics need.

    Extracted from common03.py::per_position_kl.
    """
    B, T, V = logits_p.shape
    lp = logits_p.reshape(B * T, V).float()
    lq = logits_q.reshape(B * T, V).float()
    kl = torch.empty(B * T, device=lp.device, dtype=torch.float32)
    ch = chunk_rows(V)
    for s in range(0, B * T, ch):
        e = min(s + ch, B * T)
        logp = F.log_softmax(lp[s:e], dim=-1)
        logq = F.log_softmax(lq[s:e], dim=-1)
        kl[s:e] = (logp.exp() * (logp - logq)).sum(-1)
    return kl.view(B, T)


@torch.no_grad()
def metrics_from_logits(
    logits_p: torch.Tensor, logits_q: torch.Tensor, toks: torch.Tensor
) -> SeqMetrics:
    """The four per-sequence metrics, in float32, over positions 1..T-1.

    logits_p : base model            (B, T, V)
    logits_q : replacement model     (B, T, V)
    toks     : the tokens they were both run on (B, T)

    Extracted from witness.py::metrics_from_logits.  One fused pass computes
    the KL, the base NLL and both argmaxes so the (B*T, V) log-softmax is
    materialised once.
    """
    if logits_p.shape != logits_q.shape:
        raise ValueError(f"logit shape mismatch: {tuple(logits_p.shape)} vs {tuple(logits_q.shape)}")
    if toks.shape != logits_p.shape[:2]:
        raise ValueError(f"tokens {tuple(toks.shape)} do not match logits {tuple(logits_p.shape)}")
    B, T, V = logits_p.shape
    if T < 2:
        raise ValueError("need at least 2 positions: position 0 is excluded by convention")
    lp = logits_p.reshape(B * T, V).float()
    lq = logits_q.reshape(B * T, V).float()
    kl = torch.empty(B * T, device=lp.device, dtype=torch.float32)
    nll_tok = torch.empty(B * T, device=lp.device, dtype=torch.float32)
    argp = torch.empty(B * T, device=lp.device, dtype=torch.long)
    argq = torch.empty(B * T, device=lp.device, dtype=torch.long)
    # the last column is padding: it is dropped by the [:, :-1] slice below
    tgt_flat = torch.cat(
        [toks[:, 1:], torch.zeros(B, 1, dtype=toks.dtype, device=toks.device)], dim=1
    ).reshape(B * T)
    ch = chunk_rows(V)
    for s in range(0, B * T, ch):
        e = min(s + ch, B * T)
        a, b = lp[s:e], lq[s:e]
        logp = F.log_softmax(a, dim=-1)
        logq = F.log_softmax(b, dim=-1)
        kl[s:e] = (logp.exp() * (logp - logq)).sum(-1)
        nll_tok[s:e] = -logp.gather(-1, tgt_flat[s:e].unsqueeze(-1)).squeeze(-1)
        argp[s:e] = a.argmax(-1)
        argq[s:e] = b.argmax(-1)
    kl = kl.view(B, T)
    kl_w = kl[:, 1:]  # exclude position 0
    d_mean = kl_w.mean(-1)
    d_max = kl_w.max(-1).values
    flip = (argp.view(B, T)[:, 1:] != argq.view(B, T)[:, 1:]).float().mean(-1)
    # NLL: logits at 0..T-2 predict tokens 1..T-1
    nll = nll_tok.view(B, T)[:, :-1].mean(-1)
    return SeqMetrics(d_mean, d_max, flip, nll)


@torch.no_grad()
def fvu_terms(y_true: torch.Tensor, y_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-token (numerator, denominator) of fraction-of-variance-unexplained.

    numerator   : ||y_hat - y_true||^2 per token
    denominator : ||y_true - mean(y_true)||^2 per token, against the mean over
                  ALL tokens in the batch (the "global mean" convention of
                  run_b_gemma.py::stage_pass, which accumulates exact
                  denominators against the true global mean rather than a
                  per-chunk one).

    `stats.dual_fvu` turns these into the two estimators that disagree.
    """
    y_true = y_true.float()
    y_hat = y_hat.float()
    flat_t = y_true.reshape(-1, y_true.shape[-1])
    num = (y_hat - y_true).pow(2).sum(-1)
    den = (flat_t - flat_t.mean(0)).pow(2).sum(-1).reshape(num.shape)
    return num, den
