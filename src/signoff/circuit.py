"""Circuit mode: a claimed CIRCUIT as the replacement model.

WHAT THIS IS FOR.  v0.1's miter substitutes a *dictionary* (SAE / transcoder)
for a sublayer.  A published attention *circuit* — "these 26 heads implement
indirect-object identification" (Wang et al., arXiv:2211.00593) — is the same
kind of claim with a different substitution: keep the named components intact,
knock everything else out, and assert the result still does the task.  The
field calls the resulting number `faithfulness`; in equivalence-checking terms
it is exactly a miter, and it has never been hunted adversarially.

    M     the model
    C     the claimed circuit: a KEEP-SET of (layer, head) pairs, plus a
          keep-set of MLP layers
    M̂     the replacement: M with every component outside C replaced by an
          ABLATION VALUE, and everything downstream recomputed

THE ABLATION VALUE IS NOT INCIDENTAL — IT IS THE CHECKER.  Miller, Chughtai &
Saunders (arXiv:2407.08734) showed that circuit faithfulness numbers move a lot
with the ablation policy, so this module makes the policy a first-class,
declared, hash-stamped object and ships TWO of them:

  mean      each ablated component is replaced by its MEAN activation over a
            declared calibration distribution, grouped so that grammatical
            role is held constant.  This is Wang et al.'s own convention
            (their §2.1: mean-ablation over p_ABC, "we compute the mean of a
            node across samples of the same template").
  resample  each ablated component is replaced by its activation on ONE
            counterfactual sample drawn per example from the same calibration
            distribution.  This is the Adversarial Circuit Evaluation
            (arXiv:2407.15166) / ACDC convention.

Both are run, both are reported per-cell, and disagreement between them is a
finding rather than a nuisance.

WHAT IS HASH-STAMPED, AND WHY.  A mean vector is a number that came from
somewhere.  If the calibration set drifts — different templates, different
seed, different size — every faithfulness number moves and nothing in the
output would have said so.  `AblationValues.digest()` covers the actual values
(mean tensors, or the counterfactual token ids) together with the descriptor of
the set they came from, and gate `ablation-provenance` refuses a run whose
values are unstamped or whose stamp does not match the values on hand.

WHY THE GATE IS NON-BLOCKING IN THE SHARED REGISTRY.  It is N/A to a dictionary
artifact, and a gate that is permanently UNRUN-and-blocking for four of five
adapters would make `require()` meaningless.  Circuit runs enforce it
explicitly with `require_ablation_provenance()`, which raises the same
`GateFailure`.  There is no path that reports circuit numbers past a bad stamp.

RELATION TO THE CROSS-LAYER SEAM.  `CircuitPlan` is a `SubstitutionPlan`, so it
plugs into the same place a CLT plan does: the runner, the metrics and the
gates do not know a circuit from a transcoder.  The one thing an adapter must
add is `head_tap()` — a per-head write point (`hook_z` on TransformerLens) —
because a dictionary tap replaces a whole sublayer and a circuit replaces
individual heads inside one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Mapping, Sequence

import torch

from . import gates as G
from .replacement import Site, SubstitutionPlan

# --------------------------------------------------------------- policies

MEAN = "mean"
RESAMPLE = "resample"
POLICIES = (MEAN, RESAMPLE)

POLICY_DESCRIPTIONS = {
    MEAN: "ablated components take their MEAN activation over a declared calibration "
          "distribution, averaged within a declared grouping (template) so that "
          "grammatical role is held constant — Wang et al. 2022 §2.1",
    RESAMPLE: "ablated components take their activation on ONE counterfactual sample "
              "drawn per example from the same calibration distribution — the "
              "Adversarial Circuit Evaluation / ACDC convention",
}


# ------------------------------------------------------------- the keep-set


@dataclass(frozen=True)
class CircuitSpec:
    """A claimed circuit: what is KEPT.  Everything else is ablated.

    Frozen and all-tuples so it is hashable and can live inside a
    `ReplacementSpec`, which the run tag and the gate fingerprint are computed
    from.  `classes` carries the paper's own grouping (name movers, induction,
    ...) purely so the report can print it; it is NOT the source of truth for
    membership — `heads` is, and `__post_init__` checks the two agree.
    """

    name: str
    n_layers: int
    n_heads: int
    #: KEPT attention heads, as (layer, head).
    heads: tuple[tuple[int, int], ...]
    #: KEPT MLP layers.  All of them, for a claim that does not touch MLPs.
    mlps: tuple[int, ...]
    #: Where this keep-set came from, verbatim enough to check.
    source: str = ""
    #: Named groups, for the report: (class_name, ((layer, head), ...)).
    classes: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = ()
    #: Resolutions of source ambiguity.  Printed; not decoration.
    notes: str = ""

    def __post_init__(self):
        for L, h in self.heads:
            if not (0 <= L < self.n_layers and 0 <= h < self.n_heads):
                raise ValueError(f"head ({L}, {h}) out of range for a "
                                 f"{self.n_layers}x{self.n_heads} model")
        if len(set(self.heads)) != len(self.heads):
            raise ValueError("duplicate heads in the keep-set")
        for L in self.mlps:
            if not (0 <= L < self.n_layers):
                raise ValueError(f"MLP layer {L} out of range")
        if self.classes:
            flat = [hd for _, hds in self.classes for hd in hds]
            if sorted(flat) != sorted(self.heads):
                raise ValueError(
                    "the named classes and the keep-set disagree: "
                    f"{len(flat)} classified vs {len(self.heads)} kept")
            if len(set(flat)) != len(flat):
                raise ValueError("a head appears in two classes")

    # ------------------------------------------------------------ membership

    @property
    def n_kept_heads(self) -> int:
        return len(self.heads)

    def ablated_heads(self) -> list[tuple[int, int]]:
        keep = set(self.heads)
        return [(L, h) for L in range(self.n_layers) for h in range(self.n_heads)
                if (L, h) not in keep]

    def ablated_heads_at(self, layer: int) -> list[int]:
        keep = {h for L, h in self.heads if L == layer}
        return [h for h in range(self.n_heads) if h not in keep]

    def ablated_mlps(self) -> list[int]:
        keep = set(self.mlps)
        return [L for L in range(self.n_layers) if L not in keep]

    def touched_layers(self) -> list[int]:
        """Layers the substituted forward has to intervene in."""
        ab_mlp = set(self.ablated_mlps())
        return [L for L in range(self.n_layers)
                if self.ablated_heads_at(L) or L in ab_mlp]

    # ------------------------------------------------------------- identity

    def digest(self) -> str:
        blob = json.dumps(dict(
            name=self.name, n_layers=self.n_layers, n_heads=self.n_heads,
            heads=sorted(map(list, self.heads)), mlps=sorted(self.mlps),
        ), sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return dict(
            name=self.name, n_layers=self.n_layers, n_heads=self.n_heads,
            n_kept_heads=len(self.heads), n_ablated_heads=len(self.ablated_heads()),
            kept_heads=[list(x) for x in sorted(self.heads)],
            kept_mlps=sorted(self.mlps), ablated_mlps=self.ablated_mlps(),
            touched_layers=self.touched_layers(),
            classes={k: [list(x) for x in v] for k, v in self.classes},
            source=self.source, notes=self.notes, digest=self.digest(),
        )

    @classmethod
    def from_classes(cls, name: str, classes: Mapping[str, Sequence[Sequence[int]]],
                     *, n_layers: int, n_heads: int, mlps: Iterable[int] | None = None,
                     source: str = "", notes: str = "") -> "CircuitSpec":
        """Build from the paper's own class -> head-list table.

        `mlps=None` means EVERY MLP is kept, which is the right default for a
        claim that says in so many words that it does not intervene on MLPs.
        """
        cl = tuple((str(k), tuple(sorted((int(a), int(b)) for a, b in v)))
                   for k, v in classes.items())
        heads = tuple(sorted({hd for _, hds in cl for hd in hds}))
        keep_mlps = tuple(range(n_layers)) if mlps is None else tuple(sorted(set(map(int, mlps))))
        return cls(name=name, n_layers=int(n_layers), n_heads=int(n_heads), heads=heads,
                   mlps=keep_mlps, source=source, classes=cl, notes=notes)


# ------------------------------------------------------------ ablation values
# What gets written into an ablated slot, and the stamp that says where it came
# from.  Both implementations expose the same three things, so the plan does not
# branch on policy and the gate does not either.


class AblationValues:
    """Base: values for one batch, plus their provenance stamp.

    `keys` is the per-row calibration group (for `mean`, the group whose mean
    is used; for `resample`, unused but recorded).  A batch is required to be
    uniform in sequence length — the calibration grouping is what guarantees
    that, and a silent broadcast across different lengths would be a bug the
    numbers would not show.
    """

    kind = "abstract"

    def z_for(self, layer: int, keys: Sequence[Hashable]) -> torch.Tensor:
        """(B, T, n_heads, d_head) of head-output values to write at `layer`."""
        raise NotImplementedError

    def mlp_for(self, layer: int, keys: Sequence[Hashable]) -> torch.Tensor:
        """(B, T, d_model) of MLP-output values to write at `layer`."""
        raise NotImplementedError

    def provenance(self) -> dict[str, Any]:
        raise NotImplementedError

    def digest(self) -> str:
        raise NotImplementedError


def _digest_tensors(chunks: Iterable[torch.Tensor], extra: Mapping[str, Any]) -> str:
    """sha256 over the VALUES themselves plus the descriptor of where they came from.

    float64 on CPU before hashing: the same calibration set must digest the
    same whether it was accumulated on mps or cpu, and a float32 bit pattern is
    not stable across those.  (float64 rounding of a float32 value is exact, so
    this loses nothing.)
    """
    h = hashlib.sha256()
    h.update(json.dumps(dict(extra), sort_keys=True, default=str).encode())
    for t in chunks:
        h.update(t.detach().to("cpu", torch.float64).contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


class MeanAblationValues(AblationValues):
    """Mean activations over a declared calibration set, grouped.

    `z_means[key][layer]` is (T, n_heads, d_head); `mlp_means[key][layer]` is
    (T, d_model).  The grouping is the caller's declaration — for a templated
    task it is the template id, which is what makes "the mean of a node across
    samples of the same template" (Wang et al. §2.1) literal rather than
    approximate.
    """

    kind = MEAN

    def __init__(self, z_means: Mapping[Hashable, Mapping[int, torch.Tensor]],
                 mlp_means: Mapping[Hashable, Mapping[int, torch.Tensor]] | None = None,
                 *, calibration: Mapping[str, Any]):
        self.z_means = {k: dict(v) for k, v in z_means.items()}
        self.mlp_means = {k: dict(v) for k, v in (mlp_means or {}).items()}
        self.calibration = dict(calibration)
        self._digest: str | None = None

    def _stack(self, table, layer: int, keys: Sequence[Hashable], what: str) -> torch.Tensor:
        try:
            parts = [table[k][layer] for k in keys]
        except KeyError as e:
            raise KeyError(
                f"no {what} mean for group {e.args[0]!r} at layer {layer}: the "
                f"calibration set does not cover this batch. Calibrate over every "
                f"group the evaluation uses, or the run is measuring a mean that was "
                f"never computed.") from None
        shapes = {tuple(p.shape) for p in parts}
        if len(shapes) != 1:
            raise ValueError(
                f"mixed shapes {sorted(shapes)} in one batch at layer {layer}: batches "
                f"must be uniform in sequence length. Group the evaluation by "
                f"calibration key (template), which is uniform by construction.")
        return torch.stack(parts, 0)

    def z_for(self, layer: int, keys: Sequence[Hashable]) -> torch.Tensor:
        return self._stack(self.z_means, layer, keys, "head")

    def mlp_for(self, layer: int, keys: Sequence[Hashable]) -> torch.Tensor:
        return self._stack(self.mlp_means, layer, keys, "MLP")

    def digest(self) -> str:
        if self._digest is None:
            chunks = []
            for table in (self.z_means, self.mlp_means):
                for k in sorted(table, key=repr):
                    for L in sorted(table[k]):
                        chunks.append(table[k][L])
            self._digest = _digest_tensors(chunks, dict(
                kind=self.kind, calibration=self.calibration,
                groups=sorted(map(repr, self.z_means)),
                layers=sorted({L for v in self.z_means.values() for L in v}),
            ))
        return self._digest

    def provenance(self) -> dict[str, Any]:
        return dict(
            kind=self.kind, description=POLICY_DESCRIPTIONS[MEAN],
            calibration=dict(self.calibration), digest=self.digest(),
            n_groups=len(self.z_means),
            layers=sorted({L for v in self.z_means.values() for L in v}),
            has_mlp_means=bool(self.mlp_means),
        )


class ResampleAblationValues(AblationValues):
    """Activations from ONE counterfactual sample per example.

    Built by running the model on a counterfactual batch with read-only head
    taps, so the values are a real forward of a real (declared-distribution)
    input rather than an average of many.  The stamp covers the counterfactual
    TOKEN IDS, which is the thing that would silently drift if the sampler or
    its seed changed.
    """

    kind = RESAMPLE

    def __init__(self, z: Mapping[int, torch.Tensor],
                 mlp: Mapping[int, torch.Tensor] | None = None, *,
                 cf_tokens: torch.Tensor, calibration: Mapping[str, Any]):
        self.z = dict(z)
        self.mlp = dict(mlp or {})
        self.cf_tokens = cf_tokens.detach().to("cpu", torch.long)
        self.calibration = dict(calibration)
        self._digest: str | None = None

    def _get(self, table, layer: int, keys: Sequence[Hashable], what: str) -> torch.Tensor:
        if layer not in table:
            raise KeyError(
                f"no resampled {what} activation cached for layer {layer}: the "
                f"counterfactual pass did not tap it. Cache every touched layer.")
        v = table[layer]
        if v.shape[0] != len(keys):
            raise ValueError(
                f"resample batch mismatch at layer {layer}: {v.shape[0]} counterfactual "
                f"rows for {len(keys)} evaluation rows. Resample values are PER EXAMPLE "
                f"and must be rebuilt for each batch.")
        return v

    def z_for(self, layer: int, keys: Sequence[Hashable]) -> torch.Tensor:
        return self._get(self.z, layer, keys, "head")

    def mlp_for(self, layer: int, keys: Sequence[Hashable]) -> torch.Tensor:
        return self._get(self.mlp, layer, keys, "MLP")

    def digest(self) -> str:
        if self._digest is None:
            self._digest = _digest_tensors(
                [self.cf_tokens.to(torch.float64)],
                dict(kind=self.kind, calibration=self.calibration,
                     layers=sorted(self.z), shape=list(self.cf_tokens.shape)))
        return self._digest

    def provenance(self) -> dict[str, Any]:
        return dict(
            kind=self.kind, description=POLICY_DESCRIPTIONS[RESAMPLE],
            calibration=dict(self.calibration), digest=self.digest(),
            n_counterfactuals=int(self.cf_tokens.shape[0]),
            seq_len=int(self.cf_tokens.shape[1]),
            layers=sorted(self.z), has_mlp_values=bool(self.mlp),
        )


# --------------------------------------------------------------- the plan


class CircuitPlan(SubstitutionPlan):
    """Layer-major forward with every out-of-circuit component ablated.

    The forward is the same shape as `PerLayerPlan`'s — embed, blocks, head,
    with the adapter's own primitives — which is what makes gate (i) (the clean
    layer-major pass reproduces `model(toks)` exactly) meaningful for a circuit
    run too: it is the SAME loop with the taps removed.

    Downstream components are recomputed, not patched: ablating head (L, h)
    changes the residual stream that layer L+1 reads, and every later head's
    attention pattern is recomputed from it.  That is the "knockout" semantics
    of the claim being tested (Wang et al. §2.1), not a patch of the direct
    path only.
    """

    def __init__(self, spec, n_layers: int, circuit: CircuitSpec):
        super().__init__(spec, n_layers)
        if circuit.n_layers != n_layers:
            raise ValueError(f"circuit declares {circuit.n_layers} layers, adapter has "
                             f"{n_layers}")
        self.circuit = circuit

    def sites(self) -> list[Site]:
        return [Site(L, L) for L in self.circuit.touched_layers()]

    def describe(self) -> str:
        c = self.circuit
        return (f"CircuitPlan: keep {c.n_kept_heads}/{c.n_layers * c.n_heads} heads and "
                f"{len(c.mlps)}/{c.n_layers} MLPs ({c.name}, digest {c.digest()}); "
                f"ablate the rest")

    @torch.no_grad()
    def replaced_logits(self, adapter, model, toks, *, ablation: AblationValues,
                        keys: Sequence[Hashable] | None = None, capture=None):
        if ablation is None:
            raise ValueError(
                "circuit mode needs an AblationValues: 'knocked out' is not a value, and "
                "which value is used is the checker (see Miller et al. 2407.08734)")
        B = int(toks.shape[0])
        keys = list(keys) if keys is not None else [0] * B
        if len(keys) != B:
            raise ValueError(f"{len(keys)} calibration keys for {B} rows")
        ab_mlp = set(self.circuit.ablated_mlps())
        resid = adapter.embed(model, toks)
        for L in range(self.n_layers):
            heads = self.circuit.ablated_heads_at(L)
            if not heads and L not in ab_mlp:
                resid = adapter.block(model, L, resid)
                continue
            handles: list[Any] = []
            state: dict[str, Any] = {}
            if heads:
                z_new = ablation.z_for(L, keys)
                idx = torch.tensor(heads, dtype=torch.long)

                def write_z(z, _z=z_new, _i=idx):
                    if z.shape[:2] != _z.shape[:2]:
                        raise ValueError(
                            f"ablation values {tuple(_z.shape)} do not match the head "
                            f"activations {tuple(z.shape)} — a batch/length mismatch")
                    out = z.clone()
                    out[:, :, _i, :] = _z[:, :, _i, :].to(z.device, z.dtype)
                    return out

                st, hs = adapter.head_tap(model, L, write_fn=write_z)
                state["heads"] = st
                handles += list(hs)
            if L in ab_mlp:
                y_new = ablation.mlp_for(L, keys)

                def write_mlp(x_unused, y_true, _y=y_new):
                    return _y.to(y_true.device, y_true.dtype)

                st2, hs2 = adapter.tap(model, L, replace_fn=write_mlp)
                state["mlp"] = st2
                handles += list(hs2)
            try:
                resid = adapter.block(model, L, resid)
            finally:
                for h in handles:
                    h.remove()
            if capture is not None:
                capture[Site(L, L).id] = state
        return adapter.head(model, resid)


# ----------------------------------------------------------- the gate hook


def check_ablation_provenance(values: AblationValues | None, *,
                              expected_digest: str | None = None,
                              declared_policy: str | None = None) -> G.GateResult:
    """Gate `ablation-provenance`.  The values must be stamped, and the stamp must fit.

    Three failures, all of which produce numbers that look fine:
      1. no values at all — a "circuit" run that silently zero-ablated;
      2. an undeclared policy — mean and resample give different faithfulness
         (Miller et al. 2407.08734), so an unlabelled number is not comparable
         with anything;
      3. a stamp that does not match the values on hand — the calibration set
         drifted (different templates, seed or size) under a cached digest.
    """
    spec = G.GATE_SPECS["ablation-provenance"]
    if values is None:
        return G.GateResult(spec, G.FAIL, None, None,
                            "no ablation values: nothing says what the knocked-out "
                            "components were replaced BY", {})
    prov = values.provenance()
    digest = values.digest()
    detail = dict(provenance=prov, digest=digest, expected_digest=expected_digest,
                  declared_policy=declared_policy)
    if values.kind not in POLICIES:
        return G.GateResult(spec, G.FAIL, None, None,
                            f"ablation policy {values.kind!r} is not one of {POLICIES}",
                            detail)
    if declared_policy is not None and declared_policy != values.kind:
        return G.GateResult(spec, G.FAIL, None, None,
                            f"the run declares policy {declared_policy!r} but the values "
                            f"are {values.kind!r}", detail)
    if not prov.get("calibration"):
        return G.GateResult(spec, G.FAIL, None, None,
                            "the ablation values carry no calibration descriptor — a mean "
                            "vector with no declared set behind it is not provenance",
                            detail)
    if expected_digest is not None and str(expected_digest) != str(digest):
        return G.GateResult(spec, G.FAIL, None, None,
                            f"ablation values digest {digest} != frozen {expected_digest}: "
                            f"the calibration set moved under a cached stamp", detail)
    return G.GateResult(spec, G.PASS, 0.0, None,
                        f"{values.kind}-ablation values stamped {digest} over a declared "
                        f"calibration set", detail)


def require_ablation_provenance(report: G.GateReport) -> None:
    """Enforce the ablation gate as if it were blocking.

    It is non-blocking in the shared registry because it is N/A to a dictionary
    artifact (see the module docstring).  Every circuit run calls this, so
    there is no path from a bad stamp to a reported number.
    """
    r = report.results.get("ablation-provenance")
    if r is None or r.status not in (G.PASS, G.WAIVED):
        raise G.GateFailure([r] if r is not None
                            else [G.GateResult(G.GATE_SPECS["ablation-provenance"])],
                           "report circuit-mode numbers")


__all__ = [
    "MEAN", "RESAMPLE", "POLICIES", "POLICY_DESCRIPTIONS",
    "CircuitSpec", "AblationValues", "MeanAblationValues", "ResampleAblationValues",
    "CircuitPlan", "check_ablation_provenance", "require_ablation_provenance",
]
