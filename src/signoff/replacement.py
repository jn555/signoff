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

------------------------------------------- RESTORE AND FREEZE POLICIES (06)
A substituted forward that RECOMPUTES everything is not the object published
attribution graphs are drawn on.  The production construction — the "local
replacement model" — additionally (a) adds an ERROR NODE per site, the clean
run's unexplained residual `e_L = mlp_out_clean - TC(mlp_in_clean)`, written as
a CONSTANT, and (b) FREEZES the attention patterns and the normalisation
denominators at their clean-run values.  `FreezePolicy` names which of those
restorations are in force; `CleanRunCache` holds what they restore.  The rungs
of the ladder are ordinary `Replacement`s differing only in their policy, so the
runner, the metrics, the gates and the run tag treat them identically.

THE LOAD-BEARING FACT, and the reason `lrm-base-identity` is a gate: with error
nodes present the substituted stream is PINNED to the clean trajectory, because
`TC(x_clean) + (y_clean - TC(x_clean)) == y_clean` exactly.  By induction every
residual — and therefore every logit — reproduces the base model.  Freezing does
not change that value at the clean point (a frozen pattern EQUALS the recomputed
one when the stream is already clean); freezing is what makes the linearisation
a graph is read off well-defined.  Two consequences worth stating, because both
are easy to get wrong:

  * A local replacement model that does NOT reproduce the base model on its own
    prompt is misconstructed.  That is a construction check, not a measurement,
    which is why it is a gate and why it must run before anything is reported.
  * The identity is achieved by the error nodes ALONE.  So the LRM gate cannot,
    by itself, detect a freeze that silently did nothing.  Freeze EFFICACY needs
    its own positive control — a frozen run must DIFFER from the recomputed run
    of the same substitution — and `check_freeze_efficacy` is that control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import torch

SUBSTITUTE = "substitute"
CIRCUIT = "circuit"
NULL_SHUFFLED = "null:shuffled"
NULL_TIED = "null:tied-shuffle"
NULL_GAUSSIAN = "null:gaussian"

MODES = (SUBSTITUTE, CIRCUIT, NULL_SHUFFLED, NULL_TIED, NULL_GAUSSIAN)

#: Human-readable, printed in the report so a reader knows what was controlled.
MODE_DESCRIPTIONS = {
    SUBSTITUTE: "the dictionary's own reconstruction is written into the stream",
    CIRCUIT: "a claimed CIRCUIT is the replacement: the keep-set of heads/MLPs runs "
             "intact, everything else is replaced by a declared, stamped ablation "
             "value, and everything downstream is recomputed (see `circuit.py`)",
    NULL_SHUFFLED: "real errors, permuted independently per layer (marginal preserved "
                   "exactly; input-dependence and cross-layer alignment destroyed)",
    NULL_TIED: "real errors, ONE permutation shared across layers (marginal and "
               "cross-layer alignment preserved; input-dependence destroyed)",
    NULL_GAUSSIAN: "Gaussian errors with the layer's empirical mean and FULL covariance "
                   "(independent across layers and positions)",
}


# ------------------------------------------------- restore / freeze policies


#: The LN sites a block can expose.  "ln1"/"ln2" are the pre-sublayer norms every
#: decoder has; "ln1_post"/"ln2_post" exist only on sandwich-norm models (gemma-2),
#: and "final" is the model's last norm before the unembed.  An adapter declares
#: which of these it can freeze; the policy never assumes.
LN_SITES = ("ln1", "ln1_post", "ln2", "ln2_post", "final")


@dataclass(frozen=True)
class FreezePolicy:
    """Which parts of the real forward a substituted run RESTORES.

    Three independent switches, because the point of experiment 06 was to stop
    attributing a gap to `{error nodes + frozen attention + frozen LN}` jointly
    and measure the rungs separately:

      error_nodes  add `e_L = mlp_out_clean - TC(mlp_in_clean)` as a CONSTANT at
                   the write site.  This is what makes the substituted stream
                   reproduce the base model (see the module docstring).
      attention    write the clean run's post-softmax attention PATTERNS instead
                   of recomputing them on the corrupted stream.
      layernorm    write the clean run's normalisation SCALES (the denominators)
                   instead of recomputing them.

    Frozen, so a `ReplacementSpec` carrying one stays hashable.
    """

    error_nodes: bool = False
    attention: bool = False
    layernorm: bool = False

    @property
    def restores_anything(self) -> bool:
        return self.error_nodes or self.attention or self.layernorm

    @property
    def freezes_forward(self) -> bool:
        """True when a hook must be installed on attention or the norms."""
        return self.attention or self.layernorm

    @property
    def needs_cache(self) -> bool:
        return self.restores_anything

    @property
    def is_local_replacement_model(self) -> bool:
        """All three: the production attribution-graph construction."""
        return self.error_nodes and self.attention and self.layernorm

    def tag(self) -> str:
        """Stable, stamped into run tags.  `none` for a plain skeleton run."""
        on = [n for n, v in (("errors", self.error_nodes), ("attn", self.attention),
                             ("ln", self.layernorm)) if v]
        return "+".join(on) if on else "none"

    def describe(self) -> str:
        if not self.restores_anything:
            return ("nothing restored: the interpretable skeleton alone, with attention "
                    "and normalisation RECOMPUTED on the substituted stream")
        parts = []
        if self.error_nodes:
            parts.append("error nodes (clean-run mlp_out - TC(mlp_in), added as constants)")
        if self.attention:
            parts.append("attention patterns frozen from the clean run")
        if self.layernorm:
            parts.append("normalisation scales frozen from the clean run")
        head = ("the LOCAL REPLACEMENT MODEL: " if self.is_local_replacement_model else "")
        return head + "; ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return dict(error_nodes=self.error_nodes, attention=self.attention,
                    layernorm=self.layernorm, tag=self.tag(),
                    is_local_replacement_model=self.is_local_replacement_model,
                    description=self.describe())


#: The four rungs of experiment 06's ladder, named once so no caller spells a
#: policy out by hand and gets it subtly wrong.
NO_RESTORE = FreezePolicy()
ERROR_NODES_ONLY = FreezePolicy(error_nodes=True)
FROZEN_CONTEXT_ONLY = FreezePolicy(attention=True, layernorm=True)
LOCAL_REPLACEMENT_MODEL = FreezePolicy(error_nodes=True, attention=True, layernorm=True)


class CleanRunCache:
    """What a frozen forward restores: one clean run's per-layer constants.

    Deliberately a plain mutable container rather than a tensor blob: a
    layer-major pass (the only shape that fits a 26-layer gemma next to a
    transcoder on a 16 GB machine) fills and FREES one layer at a time, so the
    cache must support `free(layer)` and report its own footprint.

    Everything is stored exactly as the hooks saw it — same device, same dtype,
    same batch order.  A cache is bound to ONE token batch; `tokens_digest`
    records which, so a mismatched cache is caught rather than silently
    restoring another batch's attention.
    """

    def __init__(self, tokens_digest: str | None = None):
        self.tokens_digest = tokens_digest
        #: layer -> the clean post-softmax pattern (B, n_heads, T, T)
        self.attn_pattern: dict[int, torch.Tensor] = {}
        #: (layer, site) -> the clean normalisation scale (B, T, 1)
        self.ln_scale: dict[tuple[int, str], torch.Tensor] = {}
        #: layer -> the clean dictionary input / true sublayer output
        self.mlp_in: dict[int, torch.Tensor] = {}
        self.mlp_out: dict[int, torch.Tensor] = {}
        #: layer -> the ERROR NODE, `mlp_out - TC(mlp_in)`, in float32
        self.error: dict[int, torch.Tensor] = {}
        #: the final norm's clean scale, keyed separately because it is captured
        #: at head time rather than inside a block
        self.final_scale: torch.Tensor | None = None

    # ------------------------------------------------------------- filling

    def set_error(self, layer: int, mlp_out_clean: torch.Tensor,
                  reconstruction_clean: torch.Tensor) -> torch.Tensor:
        """`e_L = mlp_out_clean - TC(mlp_in_clean)`, held in float32.

        FLOAT32 IS NOT DECORATION.  The identity that makes an LRM reproduce the
        base model is `TC(x) + e_L == y_clean`, and it holds EXACTLY only if the
        subtraction and the addition happen wide enough to be lossless on the
        16-bit values involved.  Computed and stored in the working dtype, the
        gate comes back with a small non-zero residual per layer that compounds
        over depth and is indistinguishable from a real construction bug.
        """
        e = mlp_out_clean.detach().float() - reconstruction_clean.detach().float()
        self.error[int(layer)] = e
        return e

    # -------------------------------------------------------------- freeing

    def free(self, layer: int | None = None) -> None:
        """Drop one layer's constants, or everything.  Called after the layer's
        modes have run — a 26-layer cache held whole is the thing that does not
        fit next to the model."""
        if layer is None:
            self.attn_pattern.clear()
            self.ln_scale.clear()
            self.mlp_in.clear()
            self.mlp_out.clear()
            self.error.clear()
            self.final_scale = None
            return
        L = int(layer)
        self.attn_pattern.pop(L, None)
        self.mlp_in.pop(L, None)
        self.mlp_out.pop(L, None)
        self.error.pop(L, None)
        for key in [k for k in self.ln_scale if k[0] == L]:
            self.ln_scale.pop(key, None)

    # ------------------------------------------------------------ reporting

    @property
    def layers(self) -> list[int]:
        return sorted(set(self.attn_pattern) | set(self.error) | set(self.mlp_out)
                      | {k[0] for k in self.ln_scale})

    def nbytes(self) -> int:
        tensors: list[torch.Tensor] = [
            *self.attn_pattern.values(), *self.ln_scale.values(), *self.mlp_in.values(),
            *self.mlp_out.values(), *self.error.values(),
        ]
        if self.final_scale is not None:
            tensors.append(self.final_scale)
        return sum(int(t.numel()) * int(t.element_size()) for t in tensors)

    def require(self, policy: FreezePolicy, layer: int, ln_sites: Sequence[str] = ()) -> None:
        """Raise unless this cache holds what `policy` needs at `layer`.

        A missing constant must not degrade into "recompute it": that would be a
        silently different mode, which is the exact confusion experiment 06
        exists to remove.
        """
        L = int(layer)
        if policy.error_nodes and L not in self.error:
            raise KeyError(
                f"freeze policy asks for an error node at layer {L} but the clean-run "
                f"cache has none (cached: {sorted(self.error)}). An LRM whose error "
                f"nodes are silently skipped is just the skeleton wearing its name.")
        if policy.attention and L not in self.attn_pattern:
            raise KeyError(
                f"freeze policy asks for a frozen attention pattern at layer {L} but "
                f"the clean-run cache has none (cached: {sorted(self.attn_pattern)})")
        if policy.layernorm:
            missing = [s for s in ln_sites if (L, s) not in self.ln_scale]
            if missing:
                raise KeyError(
                    f"freeze policy asks for frozen LN scales at layer {L} but the "
                    f"clean-run cache is missing {missing}")

    def __repr__(self) -> str:
        return (f"<CleanRunCache layers={self.layers} "
                f"{self.nbytes() / 1e6:.1f} MB digest={self.tokens_digest}>")


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
    #: A `circuit.CircuitSpec` when `mode == CIRCUIT`, else None.  Typed loosely
    #: to keep `circuit.py` (which imports this module) out of the import cycle;
    #: it is frozen and hashable, so this dataclass stays hashable too.
    circuit: Any | None = None
    #: Which parts of the real forward this run RESTORES.  The default restores
    #: nothing, which is what every pre-experiment-06 run did and what every
    #: existing run tag therefore keeps meaning.
    freeze: FreezePolicy = NO_RESTORE

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"unknown replacement mode {self.mode!r}; expected one of {MODES}")
        if self.freeze.restores_anything and self.is_null:
            raise ValueError(
                f"a null control cannot also restore {self.freeze.tag()}: the nulls exist "
                "to delete structure from the error field, and an error node puts the real "
                "error back. Those are opposite interventions, not composable ones.")
        if (self.mode == CIRCUIT) != (self.circuit is not None):
            raise ValueError(
                "circuit mode and a CircuitSpec go together: mode="
                f"{self.mode!r} with circuit={'set' if self.circuit is not None else 'None'}. "
                "A circuit run whose keep-set is not recorded is not auditable.")

    @property
    def is_null(self) -> bool:
        return self.mode.startswith("null:")

    @property
    def is_circuit(self) -> bool:
        return self.mode == CIRCUIT

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
        base = f"{self.mode} on {where}" + (f" ({n} replaced)" if n else "")
        if self.circuit is not None:
            base += f" [circuit {self.circuit.name}, digest {self.circuit.digest()}]"
        if self.freeze.restores_anything:
            base += f" [restores {self.freeze.tag()}]"
        return base

    def to_dict(self) -> dict[str, Any]:
        return dict(mode=self.mode, layers=(list(self.layers) if self.layers else None),
                    seed=self.seed, description=MODE_DESCRIPTIONS[self.mode],
                    circuit=(self.circuit.to_dict() if self.circuit is not None else None),
                    freeze=self.freeze.to_dict())


class Replacement:
    """A replacement spec bound to an adapter.

    Holds no model state: the adapter owns weights and hooks.  This object is
    what a Runner is configured with, and what the report prints as "the claim
    under test".
    """

    def __init__(self, adapter, layers: Iterable[int] | str | None = "all",
                 mode: str = SUBSTITUTE, seed: int = 0, circuit: Any | None = None,
                 freeze: FreezePolicy = NO_RESTORE):
        if isinstance(layers, str):
            if layers != "all":
                raise ValueError("layers must be 'all', None, or an iterable of ints")
            layers = None
        self.adapter = adapter
        self.spec = ReplacementSpec(
            mode=mode,
            layers=(None if layers is None else tuple(sorted(set(int(x) for x in layers)))),
            seed=int(seed),
            circuit=circuit,
            freeze=freeze,
        )

    @classmethod
    def null(cls, adapter, kind: str = "shuffled", layers: Iterable[int] | str | None = "all",
             seed: int = 0) -> "Replacement":
        """`Replacement.null(ad, "shuffled" | "tied-shuffle" | "gaussian")`."""
        mode = kind if kind.startswith("null:") else f"null:{kind}"
        return cls(adapter, layers=layers, mode=mode, seed=seed)

    @classmethod
    def circuit(cls, adapter, circuit, seed: int = 0) -> "Replacement":
        """`Replacement.circuit(ad, IOI_CIRCUIT)` — a claimed circuit as the miter's M̂.

        The substituted layer set is DERIVED from the keep-set (every layer with
        an ablated head or MLP), so the run tag and the gate fingerprint name the
        layers actually intervened in rather than a hand-typed list that could
        drift away from the circuit.
        """
        return cls(adapter, layers=circuit.touched_layers(), mode=CIRCUIT, seed=seed,
                   circuit=circuit)

    @classmethod
    def local_replacement_model(cls, adapter, layers: Iterable[int] | str | None = "all",
                                seed: int = 0) -> "Replacement":
        """The production attribution-graph construction, by name.

        `Replacement.local_replacement_model(ad)` is the object a published graph
        is drawn on: transcoders everywhere, error nodes restored, attention and
        normalisation frozen from the real forward.  It must reproduce the base
        model on its own prompt — see `gates.check_lrm_base_identity`.
        """
        return cls(adapter, layers=layers, mode=SUBSTITUTE, seed=seed,
                   freeze=LOCAL_REPLACEMENT_MODEL)

    @property
    def mode(self) -> str:
        return self.spec.mode

    @property
    def freeze(self) -> FreezePolicy:
        return self.spec.freeze

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
    def replaced_logits(self, adapter, model, toks, *, noise=None, capture=None,
                        clean_cache=None):
        """Layer-major substituted forward.  Returns logits (B, T, V).

        `noise` is required for null modes: {layer: (B, T, d_model)} added to
        the TRUE sublayer output in place of the dictionary's error.
        `capture` (optional dict) receives per-site tap state for FVU.
        `clean_cache` is required by any non-trivial `FreezePolicy`: a
        `CleanRunCache` from THIS batch's clean run, supplying the error nodes
        and the frozen attention / normalisation values.
        """
        if self.spec.is_null and noise is None:
            raise ValueError(f"{self.spec.mode} needs a per-layer noise field; "
                             "compute it from this batch's real errors first")
        pol = self.spec.freeze
        if pol.needs_cache and clean_cache is None:
            raise ValueError(
                f"freeze policy {pol.tag()!r} needs a CleanRunCache from this batch's "
                "clean run; running without one would silently be the skeleton. See "
                "`capture_clean_run`.")
        replaced = set(self.layers)
        resid = adapter.embed(model, toks)
        for L in range(self.n_layers):
            handles: list = []
            if pol.freezes_forward:
                clean_cache.require(pol, L, ln_sites=adapter.ln_sites_for(L))
                handles += list(adapter.freeze_tap(model, L, clean_cache, pol))
            try:
                if L not in replaced:
                    resid = adapter.block(model, L, resid)
                    continue
                if self.spec.is_null:
                    fn = null_replace_fn(noise[L])
                else:
                    d = adapter.dictionary(L)
                    if pol.error_nodes:
                        clean_cache.require(pol, L)
                        fn = error_node_replace_fn(d, clean_cache.error[L])
                    else:
                        fn = lambda x, y, _d=d: _d.forward(x)  # noqa: E731
                state, tap_handles = adapter.tap(model, L, replace_fn=fn)
                handles += list(tap_handles)
                resid = adapter.block(model, L, resid)
                if capture is not None:
                    capture[Site(L, L).id] = state
            finally:
                for h in handles:
                    h.remove()
        final_handles = list(adapter.final_norm_tap(model, clean_cache, freeze=True)) \
            if pol.layernorm else []
        try:
            return adapter.head(model, resid)
        finally:
            for h in final_handles:
                h.remove()


def error_node_replace_fn(dictionary, error_L: torch.Tensor):
    """`TC(x) + e_L` — the write an ERROR NODE performs, in float32.

    The width matters and is the whole trick (see `CleanRunCache.set_error`):
    `e_L` was formed as `y_clean - TC(x_clean)` in float32, so when the stream is
    still clean this returns `y_clean` EXACTLY, and the substituted forward is
    pinned to the real one.  Do the same arithmetic in float16 and each layer
    leaks an ulp, the leak compounds over depth, and the LRM identity gate comes
    back with a residual that looks like a bug and is only rounding.
    """

    def fn(x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        recon = dictionary.forward(x)
        return (recon.float() + error_L.to(recon.device).float()).to(y_true.dtype)

    return fn


# ------------------------------------------------------- the clean-run cache


@torch.no_grad()
def capture_clean_run(adapter, model, toks, layers: Sequence[int] | None = None, *,
                      with_errors: bool = True, tokens_digest: str | None = None,
                      cache: "CleanRunCache | None" = None) -> "CleanRunCache":
    """One clean forward, recorded: attention patterns, LN scales, error nodes.

    This is the reference implementation of the local-replacement-model
    construction, and it holds the WHOLE cache — which is right for a small
    model and wrong for a 26-layer gemma sharing 16 GB with a transcoder.  A
    memory-bound caller should instead drive `adapter.capture_tap` /
    `CleanRunCache.set_error` layer by layer and `free(L)` as it goes; the
    constants are layer-local, so nothing is lost by doing so.
    """
    layers = list(range(adapter.n_layers)) if layers is None else sorted(int(L) for L in layers)
    cache = cache if cache is not None else CleanRunCache(tokens_digest=tokens_digest)
    resid = adapter.embed(model, toks)
    for L in range(adapter.n_layers):
        _cap_state, cap_handles = adapter.capture_tap(model, L, cache)
        tap_state, tap_handles = adapter.tap(model, L, replace_fn=None)   # read-only
        try:
            resid = adapter.block(model, L, resid)
        finally:
            for h in (*cap_handles, *tap_handles):
                h.remove()
        if L in layers:
            cache.mlp_in[L] = tap_state["x"].detach()
            cache.mlp_out[L] = tap_state["y"].detach()
            if with_errors:
                cache.set_error(L, tap_state["y"],
                                adapter.dictionary(L).forward(tap_state["x"]))
    handles = adapter.final_norm_tap(model, cache, freeze=False)
    try:
        adapter.head(model, resid)
    finally:
        for h in handles:
            h.remove()
    return cache


def require_lrm_identity(report) -> None:
    """Refuse to proceed unless the LRM reproduced the base model.

    The same shape as `circuit.require_ablation_provenance`, and for the same
    reason: the gate is non-blocking in the shared registry because it is N/A to
    a run that restores nothing, so a run that DOES claim to be a local
    replacement model has to demand it explicitly.  A failure here is not a
    weak result — it means the object the measurements were taken on is not the
    object they are being reported as.
    """
    from . import gates as G

    r = report.results.get(G.LRM_IDENTITY_GATE)
    if r is None or r.status not in (G.PASS, G.WAIVED):
        raise G.GateFailure(
            [r] if r is not None else [G.GateResult(G.GATE_SPECS[G.LRM_IDENTITY_GATE])],
            "report local-replacement-model measurements")


def require_freeze_efficacy(report) -> None:
    """Refuse to report FROZEN numbers unless the freeze was shown to be live.

    The companion refusal to `require_lrm_identity`, and the one that actually
    protects the labels: the identity gate is satisfied by the error nodes alone,
    so a freeze that silently did nothing passes it, and every row labelled
    "attention frozen" would then be the recomputed run under a false name. A run
    that reports a frozen mode has to demand this.
    """
    from . import gates as G

    r = report.results.get(G.FREEZE_EFFICACY_GATE)
    if r is None or r.status not in (G.PASS, G.WAIVED):
        raise G.GateFailure(
            [r] if r is not None else [G.GateResult(G.GATE_SPECS[G.FREEZE_EFFICACY_GATE])],
            "report measurements from a frozen-context mode")
