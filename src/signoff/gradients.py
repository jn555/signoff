"""The gradient oracle: dJ/d(one-hot input) through the WHOLE miter.

NEW in v0.2 (experiment 09).  Nothing here is extracted from experiments 01-03 —
those ran a gradient-free search — so every convention below is stated rather
than inherited, and every one of them is CHECKED against the existing no-grad
measurement path before a search is allowed to use it.

WHAT IT COMPUTES.  For one token window x and the same objective the greedy
miner hill-climbs,

    J(x) = d_mean(x) - lambda * max(0, nll(x) - nll_ceiling)

it returns `dJ/dX` where `X` is the (T, d_vocab) one-hot matrix whose rows are
the tokens.  `dJ/dX[p, v]` is the first-order estimate of what happens to J if
position p were token v — which is exactly the ranking signal GCG (Zou et al.,
arXiv 2307.15043) uses to shortlist substitutions.  It is a LINEARISATION and
nothing more: every shortlisted candidate is still evaluated exactly, so a bad
gradient costs search efficiency, never correctness of a reported number.

WHY THE WHOLE MITER.  `d_mean` is KL(base || replacement), so the gradient has
to flow through BOTH forward passes.  That is the part that makes this more than
a standard GCG port: the derivative of a *divergence between two models* w.r.t.
the input, where one of the two models is the substituted one.  Two traps came
out of building it, both now guarded:

  * a `@torch.no_grad()` on `Dictionary.forward` severs the graph at the
    dictionary and the search is then steered by the skip connections alone.
    Silent, and it looks like a working GCG.  `check_dictionary_is_differentiable`
    refuses instead.
  * the base pass here is the LAYER-MAJOR twin (embed -> blocks -> head), not
    `model(toks)`, because the one-hot embedding has to be injected at the top.
    Gate (i) (`gate_base_vs_base`) is what licenses treating the two as the same
    model, and `verify()` re-checks it locally on the actual seed.

SCOPE (v1, deliberately narrow).  Plain SUBSTITUTE mode with no freeze policy —
i.e. the object experiments 01/07 measured.  Null modes, circuit mode and any
`FreezePolicy` are REFUSED, not approximated: a frozen forward's constants are
detached from the input by construction, so a gradient taken through it would be
a different function than the one the evaluator scores.
"""

from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from . import metrics as M
from .replacement import SUBSTITUTE, PerLayerPlan, Replacement

#: Max abs logit discrepancy tolerated between the differentiable twin and the
#: no-grad measurement path, per dtype.  Looser than gate (i)'s 1e-6 because the
#: one-hot matmul reassociates the embedding lookup into a 50257-term sum; the
#: measured value on GPT-2 float32 is recorded in every verification record, so
#: drift shows up as a number rather than as a passing tolerance.
TWIN_TOL = {"float32": 1e-3, "float16": 1e-1, "bfloat16": 1e-1}

#: Max abs discrepancy tolerated on the two SCALARS the objective is built from
#: (d_mean and nll, both in nats).  These are what the search actually compares,
#: so they get their own, tighter budget.
SCALAR_TOL_NATS = 1e-4


class GradientUnavailable(RuntimeError):
    """The gradient oracle cannot be built for this (adapter, replacement).

    Raised rather than degraded.  A GCG miner whose gradient silently became a
    constant is a random search wearing a gradient's name, and it would report
    "no witness found" with the same confidence as a real one.
    """


def onehot(toks: torch.Tensor, d_vocab: int, dtype=torch.float32) -> torch.Tensor:
    """`(B, T) -> (B, T, d_vocab)` one-hot, in a differentiable dtype."""
    return F.one_hot(toks.long(), int(d_vocab)).to(dtype)


def objective_terms(logits_base: torch.Tensor, logits_repl: torch.Tensor,
                    toks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable `(d_mean, nll)`, in the exact conventions of `metrics.py`.

    Position 0 is excluded from both; `nll` uses logits 0..T-2 against tokens
    1..T-1.  Unlike `metrics.metrics_from_logits` this is NOT chunked (the
    gradient needs the graph, and a search window is a single short row) and it
    computes only the two terms the objective uses.  `verify()` pins it against
    the chunked estimator, which is the one every reported number comes from.
    """
    logp = F.log_softmax(logits_base.float(), dim=-1)
    logq = F.log_softmax(logits_repl.float(), dim=-1)
    kl = (logp.exp() * (logp - logq)).sum(-1)                     # (B, T)
    d_mean = kl[:, 1:].mean(-1)
    tgt = toks[:, 1:].unsqueeze(-1)
    nll = -logp[:, :-1].gather(-1, tgt).squeeze(-1).mean(-1)
    return d_mean, nll


def hinged_objective(d_mean: torch.Tensor, nll: torch.Tensor, lam: float,
                     nll_ceiling: float) -> torch.Tensor:
    """`J = d_mean - lambda * max(0, nll - ceiling)` — the same expression the
    greedy miner maximises (`GreedyMiner.objective`), kept differentiable."""
    return d_mean - float(lam) * torch.clamp(nll - float(nll_ceiling), min=0.0)


class GradientOracle:
    """`dJ/d(one-hot)` for one (adapter, replacement) pair.

    Callable as `oracle(tokens, lam=..., nll_ceiling=...) -> (T, d_vocab)`,
    which is the signature `GCGMiner` expects.  Holds no state beyond the
    adapter's own model and a verification record.
    """

    def __init__(self, adapter, replacement: Replacement | None = None, *,
                 model=None, verify: bool = True):
        self.adapter = adapter
        self.replacement = replacement or Replacement(adapter, layers="all")
        self._model = model
        self.verify_on_first_call = bool(verify)
        self.verification: dict[str, Any] | None = None
        self.n_calls = 0
        self.seconds = 0.0
        self._check_supported()

    # ------------------------------------------------------------- refusals

    def _check_supported(self) -> None:
        spec = self.replacement.spec
        plan = self.replacement.plan()
        if spec.mode != SUBSTITUTE:
            raise GradientUnavailable(
                f"the gradient oracle supports {SUBSTITUTE!r} only, not {spec.mode!r}. "
                "A null control's error field and a circuit's ablation values are "
                "constants w.r.t. the input, so a gradient taken through them would "
                "be of a different function than the one the evaluator scores.")
        if spec.freeze.restores_anything:
            raise GradientUnavailable(
                f"the gradient oracle cannot run under freeze policy "
                f"{spec.freeze.tag()!r}: frozen attention patterns, frozen LN scales "
                "and error nodes are all detached constants captured from a clean "
                "run, so d/dx of the frozen forward is not d/dx of the forward the "
                "witness would be scored on.")
        if not isinstance(plan, PerLayerPlan):
            raise GradientUnavailable(
                f"the gradient oracle's differentiable twin implements "
                f"{PerLayerPlan.__name__} only; this replacement uses "
                f"{type(plan).__name__}. A cross-layer plan needs its own twin — "
                "write it next to the plan, not here.")
        self.layers = set(plan.layers)
        self.n_layers = int(self.adapter.n_layers)

    @property
    def model(self):
        if self._model is None:
            self._model = self.adapter.model
        return self._model

    # -------------------------------------------------------- the two passes

    def _forward_pair(self, resid0: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Clean and substituted logits from ONE shared embedding tensor.

        Sharing `resid0` is what makes the KL's gradient correct: both passes
        must be functions of the same one-hot, or the derivative of their
        difference is missing a term.
        """
        ad, model = self.adapter, self.model
        resid = resid0
        for L in range(self.n_layers):
            resid = ad.block(model, L, resid)
        logits_base = ad.head(model, resid)

        resid = resid0
        for L in range(self.n_layers):
            if L not in self.layers:
                resid = ad.block(model, L, resid)
                continue
            d = ad.dictionary(L)
            state, handles = ad.tap(model, L, replace_fn=lambda x, y, _d=d: _d.forward(x))
            try:
                resid = ad.block(model, L, resid)
            finally:
                for h in handles:
                    h.remove()
        logits_repl = ad.head(model, resid)
        return logits_base, logits_repl

    def _embed(self, oh: torch.Tensor, toks: torch.Tensor) -> torch.Tensor:
        return self.adapter.embed_from_onehot(self.model, oh, toks)

    # ------------------------------------------------------------- the call

    def __call__(self, tokens: torch.Tensor, *, lam: float,
                 nll_ceiling: float) -> torch.Tensor:
        """`dJ/dX` for one window.  Returns `(T, d_vocab)` on the CPU.

        CPU because the only consumer is a top-k over positions, done once per
        iteration, and keeping it off the accelerator makes the miner's
        proposal arithmetic identical on every device.
        """
        if self.verify_on_first_call and self.verification is None:
            self.verify(tokens, lam=lam, nll_ceiling=nll_ceiling)
        ts = time.time()
        toks = tokens.reshape(1, -1).to(self.adapter.device).long()
        oh = onehot(toks, self.adapter.d_vocab, self.adapter.dtype).requires_grad_(True)
        with torch.enable_grad():
            resid0 = self._embed(oh, toks)
            lb, lr = self._forward_pair(resid0)
            d_mean, nll = objective_terms(lb, lr, toks)
            J = hinged_objective(d_mean, nll, lam, nll_ceiling)[0]
            (grad,) = torch.autograd.grad(J, oh)
        self.n_calls += 1
        self.seconds += time.time() - ts
        return grad[0].detach().float().cpu()

    # ------------------------------------------------------- the local gates

    def check_dictionary_is_differentiable(self) -> dict[str, Any]:
        """Does a gradient actually reach the dictionary's parameters' input?

        The failure this catches is a `@torch.no_grad()` on `Dictionary.forward`:
        the substituted logits then depend on the input only through the skip
        connections, the search still "works", and every ranking it produces is
        about the wrong function.
        """
        L = sorted(self.layers)[0]
        d = self.adapter.dictionary(L)
        x = torch.zeros(1, 2, self.adapter.d_model, device=self.adapter.device,
                        dtype=self.adapter.dtype, requires_grad=True)
        with torch.enable_grad():
            y = d.forward(x)
        ok = bool(getattr(y, "requires_grad", False))
        if not ok:
            raise GradientUnavailable(
                f"{type(d).__name__}.forward at layer {L} returns a tensor with no "
                "graph — it is almost certainly decorated `@torch.no_grad()`. The "
                "gradient would then be taken through the skip connections alone "
                "and the search would be silently steered by the wrong function.")
        return dict(layer=int(L), dictionary=type(d).__name__, differentiable=True)

    def verify(self, tokens: torch.Tensor, *, lam: float = 0.0,
               nll_ceiling: float = 0.0) -> dict[str, Any]:
        """Pin the differentiable twin against the no-grad measurement path.

        Four checks on the ACTUAL seed the search is about to start from:
          1. the dictionary is differentiable (above);
          2. `embed_from_onehot(one_hot(x)) == embed(x)` — the one-hot
             reconstruction is the same embedding, not a similar one;
          3. the twin's base and substituted logits match `base_logits` /
             `replaced_logits` within `TWIN_TOL`;
          4. the twin's `(d_mean, nll)` match `metrics.metrics_from_logits`
             within `SCALAR_TOL_NATS`.
        Check 4 is the one that matters most: it is the statement that the
        gradient is of the objective the search's own accept test uses.
        """
        ad = self.adapter
        tol = TWIN_TOL.get(ad.dtype_name, 1e-2)
        rec: dict[str, Any] = dict(tolerance_logits=tol,
                                   tolerance_scalars_nats=SCALAR_TOL_NATS,
                                   dtype=ad.dtype_name, device=str(ad.device))
        rec["dictionary"] = self.check_dictionary_is_differentiable()

        toks = tokens.reshape(1, -1).to(ad.device).long()
        oh = onehot(toks, ad.d_vocab, ad.dtype)
        with torch.no_grad():
            emb_onehot = self._embed(oh, toks)
            emb_direct = ad.embed(self.model, toks)
            rec["embed_max_abs_diff"] = float((emb_onehot - emb_direct).abs().max())
            lb_twin, lr_twin = self._forward_pair(emb_onehot)
            lb_ref = ad.base_logits(self.model, toks)
            lr_ref = ad.replaced_logits(self.model, toks, self.replacement)
            rec["base_logits_max_abs_diff"] = float((lb_twin - lb_ref).abs().max())
            rec["replaced_logits_max_abs_diff"] = float((lr_twin - lr_ref).abs().max())
            dm_twin, nll_twin = objective_terms(lb_twin, lr_twin, toks)
            m = M.metrics_from_logits(lb_ref, lr_ref, toks)
            rec["d_mean_twin"] = float(dm_twin[0])
            rec["d_mean_reference"] = float(m.d_mean[0])
            rec["nll_twin"] = float(nll_twin[0])
            rec["nll_reference"] = float(m.nll[0])
            rec["d_mean_abs_diff"] = abs(rec["d_mean_twin"] - rec["d_mean_reference"])
            rec["nll_abs_diff"] = abs(rec["nll_twin"] - rec["nll_reference"])

        failures = []
        if rec["embed_max_abs_diff"] > tol:
            failures.append(
                f"embed_from_onehot disagrees with embed by "
                f"{rec['embed_max_abs_diff']:.3e} (> {tol:.0e}) — this adapter's "
                "embedding is probably not additive in W_E; override "
                "`embed_from_onehot`")
        for k in ("base_logits_max_abs_diff", "replaced_logits_max_abs_diff"):
            if rec[k] > tol:
                failures.append(f"{k} = {rec[k]:.3e} exceeds {tol:.0e}")
        for k in ("d_mean_abs_diff", "nll_abs_diff"):
            if rec[k] > SCALAR_TOL_NATS:
                failures.append(f"{k} = {rec[k]:.3e} nats exceeds "
                                f"{SCALAR_TOL_NATS:.0e}")
        rec["passed"] = not failures
        rec["failures"] = failures
        self.verification = rec
        if failures:
            raise GradientUnavailable(
                "the differentiable twin does not reproduce the measurement path, so "
                "its gradient is not the gradient of the objective the search scores: "
                + "; ".join(failures))
        return rec

    def stats(self) -> dict[str, Any]:
        return dict(n_calls=self.n_calls, seconds=round(self.seconds, 3),
                    seconds_per_call=(round(self.seconds / self.n_calls, 4)
                                      if self.n_calls else None),
                    verification=self.verification)


def gradient_fn(adapter, replacement: Replacement | None = None, *, model=None,
                verify: bool = True) -> GradientOracle:
    """`gradients.gradient_fn(ad, repl)` -> the callable `GCGMiner` wants."""
    return GradientOracle(adapter, replacement, model=model, verify=verify)


__all__ = ["GradientOracle", "GradientUnavailable", "gradient_fn", "hinged_objective",
           "objective_terms", "onehot", "TWIN_TOL", "SCALAR_TOL_NATS"]
