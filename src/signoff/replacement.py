"""What gets substituted, and the null controls that keep the answer honest.

PROVENANCE.  Extracted from
  experiments/01-divergence-witnesses/witness.py   (_hooks_for / logits_replaced)
  experiments/03-mechanism-and-validity/gemma_artifact.py (tap: read-only vs replacing)
  experiments/02-witness-anatomy/common02.py       (hooks_add_noise)
  experiments/02-witness-anatomy/run_a_null.py     (calibrate / make_noise — the
                                                    three nulls that separated a
                                                    2.8x superadditive composition
                                                    from plain depth geometry)

THE NULLS ARE BUILT IN ON PURPOSE.  "Replacing 12 layers diverges 2.8x more than
the sum of replacing each alone" sounds like a claim about error STRUCTURE.  It
is only a claim about structure if an error field of the same size, without the
structure, does not do the same thing.  The three nulls each delete one
property while preserving the others:

  shuffled      per-layer independent permutation of that batch's own real
                errors across (sequence, position).  The injected multiset is
                exactly the real multiset, so the marginal is preserved
                EXACTLY; input-dependence and cross-layer alignment are gone.
  tied-shuffle  ONE permutation shared by every layer.  Marginal preserved
                exactly AND cross-layer alignment preserved; only
                input-dependence is gone.  This is what isolates whether an
                effect needs errors to be correlated across layers.
  gaussian      g_L ~ N(mean(e_L), Cov(e_L)) with the FULL empirical covariance
                (Cholesky), independent across layers and positions.

A null run is a first-class run: same corpus, same metrics, same gates.  The
comparison it licenses is the whole point of having it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

SUBSTITUTE = "substitute"
NULL_SHUFFLED = "null:shuffled"
NULL_TIED = "null:tied-shuffle"
NULL_GAUSSIAN = "null:gaussian"

MODES = (SUBSTITUTE, NULL_SHUFFLED, NULL_TIED, NULL_GAUSSIAN)

#: Human-readable, printed in the report so a reader knows what was controlled.
MODE_DESCRIPTIONS = {
    SUBSTITUTE: "the dictionary's own reconstruction is written into the stream",
    NULL_SHUFFLED: "real errors, permuted independently per layer (marginal preserved "
                   "exactly; input-dependence and cross-layer alignment destroyed)",
    NULL_TIED: "real errors, ONE permutation shared across layers (marginal and "
               "cross-layer alignment preserved; input-dependence destroyed)",
    NULL_GAUSSIAN: "Gaussian errors with the layer's empirical mean and FULL covariance "
                   "(independent across layers and positions)",
}


@dataclass(frozen=True)
class ReplacementSpec:
    """A declarative description of the substituted forward.

    `layers=None` means every layer the adapter has dictionaries for.  A subset
    is how single-layer localisation profiles are run (replace layer L alone,
    for each L) — the same object, thirteen configurations.
    """

    mode: str = SUBSTITUTE
    layers: tuple[int, ...] | None = None
    seed: int = 0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown replacement mode {self.mode!r}; expected one of {MODES}")

    @property
    def is_null(self) -> bool:
        return self.mode != SUBSTITUTE

    def resolve_layers(self, n_layers: int) -> list[int]:
        if self.layers is None:
            return list(range(n_layers))
        bad = [L for L in self.layers if not (0 <= L < n_layers)]
        if bad:
            raise ValueError(f"layers out of range for a {n_layers}-layer model: {bad}")
        return sorted(set(int(L) for L in self.layers))

    def describe(self, n_layers: int | None = None) -> str:
        n = len(self.layers) if self.layers is not None else n_layers
        where = "all layers" if self.layers is None else f"layers {list(self.layers)}"
        return f"{self.mode} on {where}" + (f" ({n} replaced)" if n else "")

    def to_dict(self) -> dict[str, Any]:
        return dict(mode=self.mode, layers=(list(self.layers) if self.layers else None),
                    seed=self.seed, description=MODE_DESCRIPTIONS[self.mode])


class Replacement:
    """A replacement spec bound to an adapter.

    Holds no model state: the adapter owns weights and hooks.  This object is
    what a Runner is configured with, and what the report prints as "the claim
    under test".
    """

    def __init__(self, adapter, layers: Iterable[int] | str | None = "all",
                 mode: str = SUBSTITUTE, seed: int = 0):
        if isinstance(layers, str):
            if layers != "all":
                raise ValueError("layers must be 'all', None, or an iterable of ints")
            layers = None
        self.adapter = adapter
        self.spec = ReplacementSpec(
            mode=mode,
            layers=(None if layers is None else tuple(sorted(set(int(x) for x in layers)))),
            seed=int(seed),
        )

    @classmethod
    def null(cls, adapter, kind: str = "shuffled", layers: Iterable[int] | str | None = "all",
             seed: int = 0) -> "Replacement":
        """`Replacement.null(ad, "shuffled" | "tied-shuffle" | "gaussian")`."""
        mode = kind if kind.startswith("null:") else f"null:{kind}"
        return cls(adapter, layers=layers, mode=mode, seed=seed)

    @property
    def mode(self) -> str:
        return self.spec.mode

    @property
    def layers(self) -> list[int]:
        return self.spec.resolve_layers(self.adapter.n_layers)

    def plan(self) -> "SubstitutionPlan":
        """The object that actually runs the substituted forward.

        The ADAPTER chooses it, so a cross-layer artifact supplies a
        cross-layer plan without any change here, in the runner, or in the
        metrics.  See the substitution-plan section below.
        """
        return self.adapter.substitution_plan(self.spec)

    def __repr__(self) -> str:
        return f"<Replacement {self.spec.describe(self.adapter.n_layers)} on {self.adapter.name}>"

    def to_dict(self) -> dict[str, Any]:
        d = self.spec.to_dict()
        d["resolved_layers"] = self.layers
        d["adapter"] = self.adapter.name
        return d


# --------------------------------------------------------------- null fields


@dataclass
class ErrorMoments:
    """Per-layer mean and covariance of the reconstruction error, in float64.

    Accumulated exactly (sum and sum-of-outer-products), never streamed as a
    running estimate: the Gaussian null is only "covariance-matched" if the
    covariance is the real one.  From run_a_null.py::calibrate.
    """

    d_model: int
    layers: tuple[int, ...]
    mean: dict[int, torch.Tensor] = field(default_factory=dict)
    chol: dict[int, torch.Tensor] = field(default_factory=dict)
    stats: dict[int, dict[str, float]] = field(default_factory=dict)
    n_positions: int = 0


class ErrorCalibrator:
    """Accumulate exact error moments over a held-out calibration set.

    The calibration set MUST be disjoint from the evaluation set — otherwise the
    null is fitted to the very sequences it is being compared on.  The caller
    supplies disjoint index sets; `run_a_null.py` asserts the disjointness and
    so does the Runner.
    """

    def __init__(self, layers: Sequence[int], d_model: int):
        self.layers = tuple(int(L) for L in layers)
        self.d_model = int(d_model)
        self._s1 = {L: torch.zeros(self.d_model, dtype=torch.float64) for L in self.layers}
        self._s2 = {L: torch.zeros(self.d_model, self.d_model, dtype=torch.float64)
                    for L in self.layers}
        self._ne = {L: 0.0 for L in self.layers}
        self._ny = {L: 0.0 for L in self.layers}
        self._count = 0

    @torch.no_grad()
    def update(self, errors: dict[int, torch.Tensor], outputs: dict[int, torch.Tensor]) -> None:
        """One batch: errors[L] and true outputs[L], both (B, T, d_model)."""
        first = None
        for L in self.layers:
            e = errors[L].reshape(-1, self.d_model)
            # .cpu() before .double(): MPS has no float64
            self._s1[L] += e.sum(0).cpu().double()
            self._s2[L] += (e.T @ e).cpu().double()
            self._ne[L] += float(e.norm(dim=-1).sum())
            self._ny[L] += float(outputs[L].reshape(-1, self.d_model).norm(dim=-1).sum())
            if first is None:
                first = e.shape[0]
        self._count += int(first or 0)

    @torch.no_grad()
    def finalize(self, device="cpu") -> ErrorMoments:
        if self._count == 0:
            raise RuntimeError("no calibration batches were accumulated")
        m = ErrorMoments(d_model=self.d_model, layers=self.layers, n_positions=self._count)
        eye = torch.eye(self.d_model, dtype=torch.float64)
        for L in self.layers:
            mu = self._s1[L] / self._count
            cov = self._s2[L] / self._count - torch.outer(mu, mu)
            cov = 0.5 * (cov + cov.T)
            ridge = 1e-8 * float(torch.diagonal(cov).mean())
            chol = None
            for _ in range(8):
                try:
                    chol = torch.linalg.cholesky(cov + ridge * eye)
                    break
                except Exception:
                    ridge = max(ridge * 100, 1e-12)
            if chol is None:
                raise RuntimeError(f"layer {L}: error covariance not positive-definite "
                                   f"even with ridge {ridge}")
            m.mean[L] = mu.float().to(device)
            m.chol[L] = chol.float().to(device)
            m.stats[L] = dict(
                mean_norm=float(mu.norm()),
                trace_cov=float(torch.diagonal(cov).sum()),
                rms_err=float((torch.diagonal(cov).sum() + mu.dot(mu)) ** 0.5),
                mean_err_norm=self._ne[L] / self._count,
                mean_output_norm=self._ny[L] / self._count,
                rel_err=self._ne[L] / max(self._ny[L], 1e-9),
                ridge=ridge,
            )
        return m


@torch.no_grad()
def make_noise(
    mode: str, errors: dict[int, torch.Tensor], generator: torch.Generator,
    moments: ErrorMoments | None = None, device=None,
) -> dict[int, torch.Tensor]:
    """The input-independent error field for one batch.  From run_a_null.py::make_noise.

    `errors[L]` is (B, T, d_model) — this batch's REAL reconstruction errors,
    which the shuffle nulls permute and the Gaussian null ignores.
    """
    layers = sorted(errors)
    ref = errors[layers[0]]
    B, T, D = ref.shape
    device = ref.device if device is None else device
    if mode == NULL_SHUFFLED:
        return {L: errors[L].reshape(-1, D)[
            torch.randperm(B * T, generator=generator).to(device)].reshape(B, T, D)
            for L in layers}
    if mode == NULL_TIED:
        p = torch.randperm(B * T, generator=generator).to(device)
        return {L: errors[L].reshape(-1, D)[p].reshape(B, T, D) for L in layers}
    if mode == NULL_GAUSSIAN:
        if moments is None:
            raise ValueError("the gaussian null needs calibrated ErrorMoments")
        out = {}
        for L in layers:
            z = torch.randn(B * T, D, generator=generator).to(device)
            out[L] = (moments.mean[L] + z @ moments.chol[L].T).reshape(B, T, D)
        return out
    raise ValueError(f"{mode!r} is not a null mode; expected one of "
                     f"{(NULL_SHUFFLED, NULL_TIED, NULL_GAUSSIAN)}")


def null_replace_fn(noise_L: torch.Tensor):
    """A `replace_fn` that keeps the TRUE sublayer output and adds a noise field.

    This is the null forward of experiment 02 study A: the real MLP output is
    kept and an input-independent error field of matched marginal is added, in
    place of the dictionary's input-dependent error.  Compare
    common02.py::hooks_add_noise, which does the same thing as a raw hook.
    """

    def fn(x_unused, y_true: torch.Tensor) -> torch.Tensor:
        return y_true + noise_L.to(y_true.dtype)

    return fn


# ------------------------------------------------------- substitution plans
# THE CROSS-LAYER SEAM.
#
# v0.1's artifacts are per-layer: one dictionary reads layer L's sublayer input
# and writes layer L's sublayer output.  The next artifact generation is NOT —
# a cross-layer transcoder (CLT, as used by the open circuit-tracer stack)
# reads once at layer L and writes contributions into SEVERAL downstream
# layers.  So the substituted forward is owned by a PLAN object rather than by
# a `for L in layers` loop baked into the adapter base class: adding a CLT
# adapter means adding a plan, not rewriting the runner, the gates or the
# metrics.  A `Site` is deliberately (read_layer, write_layer) rather than a
# single index for the same reason.


@dataclass(frozen=True)
class Site:
    """One substitution site: where a dictionary reads, and where it writes.

    For a per-layer artifact `read_layer == write_layer`.  For a cross-layer
    artifact one read feeds many sites, and `id` is what gate (ii) keys its
    per-site FVU on.
    """

    read_layer: int
    write_layer: int

    @property
    def id(self) -> str:
        return (str(self.read_layer) if self.read_layer == self.write_layer
                else f"{self.read_layer}->{self.write_layer}")

    def __str__(self) -> str:
        return self.id


class SubstitutionPlan:
    """Owns the substituted forward pass.

    Subclass contract:
      * `sites()`      — every (read, write) pair this plan touches.
      * `replaced_logits(adapter, model, toks, **kw)` — the forward.

    The adapter supplies the primitives (`embed`, `block`, `tap`, `head`); the
    plan decides the wiring.  `PerLayerPlan` is the only one v0.1 ships.
    """

    def __init__(self, spec: ReplacementSpec, n_layers: int):
        self.spec = spec
        self.n_layers = int(n_layers)
        self.layers = spec.resolve_layers(self.n_layers)

    def sites(self) -> list[Site]:
        raise NotImplementedError

    def describe(self) -> str:
        return f"{type(self).__name__}: {self.spec.describe(self.n_layers)}"

    def replaced_logits(self, adapter, model, toks, **kwargs):
        raise NotImplementedError


class PerLayerPlan(SubstitutionPlan):
    """One dictionary per layer, read and write at the same layer.

    The forward is layer-major (embed -> blocks -> head) and is exactly the
    loop gate (i) verifies against the model's own forward, which is why the
    replaced pass is trustworthy at all.
    """

    def sites(self) -> list[Site]:
        return [Site(L, L) for L in self.layers]

    @torch.no_grad()
    def replaced_logits(self, adapter, model, toks, *, noise=None, capture=None):
        """Layer-major substituted forward.  Returns logits (B, T, V).

        `noise` is required for null modes: {layer: (B, T, d_model)} added to
        the TRUE sublayer output in place of the dictionary's error.
        `capture` (optional dict) receives per-site tap state for FVU.
        """
        if self.spec.is_null and noise is None:
            raise ValueError(f"{self.spec.mode} needs a per-layer noise field; "
                             "compute it from this batch's real errors first")
        replaced = set(self.layers)
        resid = adapter.embed(model, toks)
        for L in range(self.n_layers):
            if L not in replaced:
                resid = adapter.block(model, L, resid)
                continue
            if self.spec.is_null:
                fn = null_replace_fn(noise[L])
            else:
                d = adapter.dictionary(L)
                fn = lambda x, y, _d=d: _d.forward(x)  # noqa: E731
            state, handles = adapter.tap(model, L, replace_fn=fn)
            try:
                resid = adapter.block(model, L, resid)
            finally:
                for h in handles:
                    h.remove()
            if capture is not None:
                capture[Site(L, L).id] = state
        return adapter.head(model, resid)
