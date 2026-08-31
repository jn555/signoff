"""The adapter contract — everything artifact-specific, and nothing else.

PROVENANCE.  The contract is the generalisation of three worked adapters:
  experiments/01-divergence-witnesses/witness.py            (GPT-2 + Dunefsky)
  experiments/03-mechanism-and-validity/gemma_artifact.py   (gemma-2 + gemma-scope;
                                                             the model of what an
                                                             adapter docstring owes
                                                             the reader)
  experiments/03-mechanism-and-validity/run_b_validity.py   (Qwen3 + mwhanna)

THE SEVEN REQUIRED CLAUSES.  Each is a lesson that cost real debugging time.
An adapter that does not satisfy all seven cannot be trusted to produce a
number, and the gates are what enforce them at runtime rather than in prose.

  1. PINNED IDENTITY.  Repo ids AND revisions for model and dictionaries, plus
     a frozen variant-selection expectation that HALTS on repo drift.
  2. DECLARED TAPS, VERIFIED NOT TRUSTED.  Named input/output hook per layer
     with the convention written down — two suites use the SAME hook name with
     OPPOSITE pre/post-gain conventions.  Gate (ii) is the verifier.
  3. TOKENIZATION / BOS DECLARED PER ARTIFACT.  bos_id or None, and position-0
     exclusion.  There is no safe default to guess at.
  4. DTYPE POLICY WITH AN fp32 SPOT-CHECK.  The declared dtype must clear gate
     (iii), or the run downgrades to paired-verdict-only claims with a measured
     bound.
  5. HEAD REPRODUCTION.  `head(resid) -> logits` must match the model's own
     forward bit-for-bit, in the MODEL's dtype.  "More accurate" is wrong here.
  6. BIT-EXACT LAYER-MAJOR REPRODUCTION.  A clean layer-major pass must equal
     `model(toks)` with KL == 0 (gate (i)), all values finite.
  7. WEIGHT-LAYOUT ASSERTS AT LOAD.  Shapes, transposition, threshold
     positivity, config identity — checked at load, not assumed.

CROSS-LAYER ARTIFACTS.  `substitution_plan()` is the seam: the base class
returns a per-layer plan, and an adapter for a cross-layer artifact (a CLT, as
in the circuit-tracer stack) overrides it with a plan that writes into several
downstream layers from one read site.  Nothing in the runner, the metrics or
the gates assumes one-dictionary-per-layer.

THE TRUST BOUNDARY.  This file is where it sits, so it is stated here.  The
adapter is what knows how to run the artifact, so MEASUREMENT IS DELEGATED TO
IT BY DESIGN: the gates in `gates.py` are pure verdict functions over numbers
that adapter code produced.  That is what makes the tool portable to an artifact
this package has never seen, and it means the gates catch AUTHOR ERROR — a
mis-tap, a dtype bug, registry drift — not tampering.  An adapter that overrides
`gate_fvu` to return PASS gets a PASS.

The tool does not pretend otherwise.  It STAMPS the delegation:
`overridden_measurement_methods()` reports which base-class measurement methods
a concrete adapter replaced, the report prints that list, and every gate row
whose measurement was taken over is marked "self-reported".  A reader of a
third-party adapter's report can then see exactly how much of it is the tool's
measurement and how much is the adapter's claim.  See the README's
"Trust model".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Protocol, Sequence

import torch

from .. import gates as G
from .. import metrics as M
from .. import stats as S
from ..replacement import (
    LN_SITES,
    FreezePolicy,
    PerLayerPlan,
    ReplacementSpec,
    Site,
    SubstitutionPlan,
)


# ------------------------------------------------------------- declarations


@dataclass(frozen=True)
class Identity:
    """Clause 1.  What, exactly, is being audited."""

    release: str
    model_repo: str
    dict_repo: str
    model_revision: str | None = None
    dict_revision: str | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(release=self.release, model_repo=self.model_repo,
                    model_revision=self.model_revision, dict_repo=self.dict_repo,
                    dict_revision=self.dict_revision, notes=self.notes)

    @property
    def fully_pinned(self) -> bool:
        return bool(self.model_revision) and bool(self.dict_revision)


@dataclass(frozen=True)
class TapSpec:
    """Clause 2.  Where the dictionary reads and writes, and under WHICH convention.

    `input_convention` / `output_convention` are free text on purpose: the
    dangerous cases are not expressible as an enum (pre-gain vs post-gain
    normalised input; MLP module return value vs the block's additive
    contribution after a second norm).  Write the sentence a reader needs.
    """

    input_hook: str
    output_hook: str
    input_convention: str
    output_convention: str
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return dict(input_hook=self.input_hook, output_hook=self.output_hook,
                    input_convention=self.input_convention,
                    output_convention=self.output_convention, notes=self.notes)


@dataclass(frozen=True)
class TokenizationSpec:
    """Clause 3.  BOS convention and the position-exclusion rule."""

    bos_id: int | None
    declared: bool = True
    exclude_position_0: bool = True
    notes: str = ""

    @property
    def prepend_bos(self) -> bool:
        return self.bos_id is not None

    def to_dict(self) -> dict[str, Any]:
        return dict(bos_id=self.bos_id, prepend_bos=self.prepend_bos,
                    declared=self.declared, exclude_position_0=self.exclude_position_0,
                    notes=self.notes)


@dataclass(frozen=True)
class FreezeCapabilities:
    """What of the real forward this adapter can RESTORE from a clean run.

    DECLARED, like every other clause, because there is no safe default: an
    adapter that cannot reach its attention pattern must say so and be refused a
    frozen run, not quietly hand back a recomputed one under the frozen name.
    An empty capability set is a legitimate declaration — error nodes need no
    hooks beyond the ordinary write site, so an adapter with no attention at all
    still supports the error-node rung.
    """

    attention: bool = False
    #: which of `replacement.LN_SITES` this adapter can freeze
    ln_sites: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self):
        bad = [s for s in self.ln_sites if s not in LN_SITES]
        if bad:
            raise ValueError(f"unknown LN freeze sites {bad}; expected from {LN_SITES}")

    @property
    def block_ln_sites(self) -> tuple[str, ...]:
        """The per-block sites; `final` is captured at head time, not in a block."""
        return tuple(s for s in self.ln_sites if s != "final")

    @property
    def layernorm(self) -> bool:
        return bool(self.ln_sites)

    def supports(self, policy: FreezePolicy) -> bool:
        return ((self.attention or not policy.attention)
                and (self.layernorm or not policy.layernorm))

    def to_dict(self) -> dict[str, Any]:
        return dict(attention=self.attention, ln_sites=list(self.ln_sites),
                    notes=self.notes)


@dataclass(frozen=True)
class DtypePolicy:
    """Clause 4.  The working dtype, what else is allowed, and what was MEASURED."""

    default: str
    allowed: tuple[str, ...] = ("float32", "float16", "bfloat16")
    fp32_replay_tolerance_nats: float = G.FP32_REPLAY_TOL_NATS
    #: dtype -> what a real gate-(iii) run measured, e.g.
    #: {"bfloat16": "replaced KL max 3.09e-1 — FAIL"}
    measured: dict[str, str] = field(default_factory=dict)
    notes: str = ""

    def torch_dtype(self, name: str | None = None):
        name = name or self.default
        if name not in self.allowed:
            raise ValueError(f"dtype {name!r} not allowed by this adapter; "
                             f"allowed: {self.allowed}")
        return dict(float32=torch.float32, float16=torch.float16,
                    bfloat16=torch.bfloat16)[name]

    def to_dict(self) -> dict[str, Any]:
        return dict(default=self.default, allowed=list(self.allowed),
                    fp32_replay_tolerance_nats=self.fp32_replay_tolerance_nats,
                    measured=dict(self.measured), notes=self.notes)


class Dictionary(Protocol):
    """What a loaded dictionary must expose.  Clause 7 lives in its constructor."""

    layer: int

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


# ----------------------------------------------------- the delegation stamp
# See the module docstring's TRUST BOUNDARY paragraph.  These three tables are
# what turns "the adapter could have measured anything" from an unstated
# assumption into a line in the report.


#: Base methods that HAVE a real implementation here and produce a number a
#: gate or the report then quotes.  Overriding one moves that measurement out
#: of framework code and into adapter code.
MEASUREMENT_METHODS: tuple[str, ...] = (
    "base_logits",
    "clean_layer_major_logits",
    "replaced_logits",
    "measure_fvu",
    "gate_base_vs_base",
    "gate_fvu",
    "gate_fp32_replay",
    "gate_bos",
    "verify_provenance",
    "substitution_plan",
    "contract",
    "run_tag",
)

#: Methods the base class does NOT implement.  Every adapter supplies them, so
#: overriding them carries no information — but the whole measurement chain
#: runs through them, which is the irreducible part of the delegation and the
#: reason an empty override list is not a tamper-proofness claim.
REQUIRED_OVERRIDES: tuple[str, ...] = (
    "tokenizer", "load_model", "load_dictionary", "embed", "block", "head", "tap",
)

#: Supplied only by CIRCUIT-capable adapters (see `circuit.py`).  Same status as
#: `REQUIRED_OVERRIDES` — no base implementation, so an override carries no
#: information, but the circuit forward runs through it, so it is named here
#: rather than left unmentioned.
#: Same status as `REQUIRED_OVERRIDES`: no base implementation worth the name, so
#: an override carries no information, but the frozen forward runs through them.
#: `head_tap` is CIRCUIT mode's seam; the other three are the RESTORE/FREEZE seam
#: a local replacement model needs (experiment 06).
OPTIONAL_OVERRIDES: tuple[str, ...] = ("head_tap", "capture_tap", "freeze_tap",
                                       "final_norm_tap")

#: gate id -> the measurement methods whose override makes that gate's verdict
#: self-reported rather than framework-measured.  The report marks those rows.
GATE_MEASURED_BY: dict[str, tuple[str, ...]] = {
    "i-base-vs-base": ("gate_base_vs_base", "base_logits", "clean_layer_major_logits"),
    "ii-fvu-sanity": ("gate_fvu", "measure_fvu"),
    "iii-fp32-replay": ("gate_fp32_replay", "base_logits", "replaced_logits"),
    "iii-prime-paired-bound": ("base_logits", "replaced_logits"),
    "identity-guard": ("run_tag",),
    "provenance-freeze": ("verify_provenance", "contract"),
    "bos-declaration": ("gate_bos",),
    "ablation-provenance": ("substitution_plan", "replaced_logits"),
    "checkpoint-binding": ("contract", "run_tag"),
    "lrm-base-identity": ("base_logits", "replaced_logits", "substitution_plan"),
    "freeze-efficacy": ("replaced_logits", "substitution_plan"),
}


def _underlying(f):
    return getattr(f, "__func__", f)


def overridden_measurement_methods(adapter_or_class) -> list[str]:
    """Which security-relevant base methods this concrete adapter REPLACES.

    A module-level function, deliberately not a method: the one thing that
    reports an adapter's overrides must not itself be reachable through the
    object it is reporting on.  Callers (the reporter) pass the adapter in.

    Returns a list in `MEASUREMENT_METHODS` order.  Empty for the reference
    per-layer adapters, except `gemma-scope-2b`, whose release has a
    variant-SELECTION rule and so overrides `verify_provenance` — a declared
    extension point, and still a self-report, which is why it is listed.
    """
    cls = adapter_or_class if isinstance(adapter_or_class, type) else type(adapter_or_class)
    out: list[str] = []
    for name in MEASUREMENT_METHODS:
        base = getattr(ModelAdapter, name, None)
        concrete = getattr(cls, name, None)
        if base is None or concrete is None:
            continue
        if _underlying(concrete) is not _underlying(base):
            out.append(name)
    return out


def self_reported_gates(adapter_or_class) -> dict[str, list[str]]:
    """gate id -> the overridden methods behind its verdict.  Only non-empty entries."""
    over = set(overridden_measurement_methods(adapter_or_class))
    return {gid: [m for m in methods if m in over]
            for gid, methods in GATE_MEASURED_BY.items()
            if any(m in over for m in methods)}


# ------------------------------------------------------------------ adapter


class ModelAdapter:
    """Base class.  Subclasses declare the four specs and implement six methods.

    Required overrides:
        tokenizer()                       -> a HF tokenizer
        load_model(device, dtype)         -> the model
        load_dictionary(layer, device, dtype) -> Dictionary  (clause 7 asserts here)
        embed(model, toks)                -> the residual stream entering block 0
        block(model, layer, resid)        -> one block applied to the residual
        head(model, resid)                -> logits, bit-for-bit as clause 5 requires
        tap(model, layer, replace_fn)     -> (state, handles)

    Optional:
        verify_provenance()               -> GateResult (clause 1; default: pinned-ness)
        substitution_plan(spec)           -> SubstitutionPlan (the cross-layer seam)
    """

    name: str = "unnamed"
    identity: Identity
    taps: TapSpec
    tokenization: TokenizationSpec
    dtype_policy: DtypePolicy
    n_layers: int
    d_model: int
    d_vocab: int
    #: Attention geometry.  Only CIRCUIT-capable adapters need these; a
    #: dictionary adapter never looks inside the attention sublayer.
    n_heads: int | None = None
    d_head: int | None = None
    #: Layers this adapter actually has dictionaries for (default: all).
    dictionary_layers: tuple[int, ...] | None = None

    def __init__(self, device: str | None = None, dtype: str | None = None):
        self.device = device or pick_device()
        self.dtype_name = dtype or self.dtype_policy.default
        self.dtype = self.dtype_policy.torch_dtype(self.dtype_name)
        self._model = None
        self._tok = None
        self._dicts: dict[int, Dictionary] = {}

    # ------------------------------------------------------------ required

    def tokenizer(self):
        raise NotImplementedError

    def load_model(self, device, dtype):
        raise NotImplementedError

    def load_dictionary(self, layer: int, device, dtype) -> Dictionary:
        raise NotImplementedError

    def embed(self, model, toks: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def block(self, model, layer: int, resid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def head(self, model, resid: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def tap(self, model, layer: int, replace_fn: Callable | None = None):
        """Install the declared taps on one layer.

        `replace_fn(x, y_true) -> y_new` receives the dictionary INPUT and the
        TRUE sublayer output, and returns what to write.  Taking both is what
        lets the null controls (`y_true + noise`) share this path with an
        ordinary substitution (`dictionary.forward(x)`).

        With `replace_fn=None` the tap MUST be strictly read-only — every hook
        returns None — so FVU can be measured without perturbing the pass.

        Returns `(state, handles)` where state gets "x", "y" and (when
        replacing) "yhat"; the caller removes the handles.
        """
        raise NotImplementedError

    def head_tap(self, model, layer: int, write_fn: Callable | None = None):
        """OPTIONAL, for CIRCUIT mode.  A per-ATTENTION-HEAD read/write point.

        `tap()` replaces a whole sublayer, which is the granularity a dictionary
        claims at.  A circuit claims at the granularity of individual heads
        inside the attention sublayer, so it needs a separate seam: the
        per-head output `z`, shaped (B, T, n_heads, d_head), BEFORE it is mixed
        by W_O.  Writing there is what "knock out head (L, h)" means, and it is
        the only intervention point at which one head can be changed without
        touching its neighbours.

        `write_fn(z) -> z_new` gets the whole (B, T, n_heads, d_head) tensor and
        returns a whole one; the plan owns which head slots it overwrites.  With
        `write_fn=None` the tap is strictly read-only and records "z" — that is
        the mode the ablation-value calibration runs in.

        Returns `(state, handles)`; the caller removes the handles.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a per-head tap; circuit mode needs "
            f"one (see adapters/base.py::head_tap and circuit.py)")

    # -------------------------------------------------- restore / freeze seam
    # The hooks a LOCAL REPLACEMENT MODEL needs beyond an ordinary substitution:
    # read the clean run's attention patterns and normalisation scales, then
    # write them back on a corrupted stream.  Hook NAMES are artifact-specific
    # (and gemma-2's sandwich norm has four per block where GPT-2 has two), so
    # this is adapter territory; the policy and the cache are not.

    #: Clause 8, by the same logic as clauses 1-7: declared, then refused when
    #: a run asks for more than the declaration.  Default: nothing freezable.
    freeze_capabilities_spec: FreezeCapabilities = FreezeCapabilities()

    def freeze_capabilities(self) -> FreezeCapabilities:
        return self.freeze_capabilities_spec

    def ln_sites_for(self, layer: int) -> tuple[str, ...]:
        """The per-block LN sites frozen at `layer`.  Uniform by default."""
        return self.freeze_capabilities().block_ln_sites

    def _check_freeze_supported(self, policy: FreezePolicy) -> FreezeCapabilities:
        caps = self.freeze_capabilities()
        if not caps.supports(policy):
            want = [n for n, v in (("attention", policy.attention),
                                   ("layernorm", policy.layernorm)) if v]
            raise NotImplementedError(
                f"{type(self).__name__} declares freeze capabilities {caps.to_dict()} and "
                f"cannot freeze {want}. A frozen run on an adapter that cannot freeze is "
                f"a RECOMPUTED run wearing the wrong name — implement `capture_tap` / "
                f"`freeze_tap` and declare them, or run the policy this adapter supports.")
        return caps

    def capture_tap(self, model, layer: int, cache, fires: dict | None = None):
        """Read-only: record `layer`'s clean attention pattern and LN scales.

        Returns `(state, handles)` like `tap()`; the caller removes the handles.
        Writes straight into `cache` (a `replacement.CleanRunCache`) so a
        layer-major pass can capture and free one layer at a time.

        `fires` (optional) counts how many times each site was invoked. That
        count is the STRUCTURAL fact a freeze-efficacy check needs: a norm is not
        necessarily called once per block (TransformerLens calls `ln1` three
        times on a pre-norm block, once each for the query, key and value
        inputs), so the only honest expectation for the frozen forward is the
        one the clean forward just measured on the same architecture.
        """
        if self.freeze_capabilities().attention or self.freeze_capabilities().ln_sites:
            raise NotImplementedError(
                f"{type(self).__name__} declares freeze capabilities but does not "
                f"implement `capture_tap`")
        return {}, ()

    def freeze_tap(self, model, layer: int, cache, policy: FreezePolicy):
        """Write the cached clean values back during a forward on this layer.

        Returns handles only (there is nothing to read back).  The base class
        supports exactly one policy — one that freezes nothing.
        """
        if not policy.freezes_forward:
            return ()
        self._check_freeze_supported(policy)
        raise NotImplementedError(
            f"{type(self).__name__} declares freeze capabilities but does not "
            f"implement `freeze_tap`")

    def final_norm_tap(self, model, cache, freeze: bool = False):
        """Capture (or restore) the FINAL norm's scale, around the head.

        Separate from `capture_tap` because the last normalisation happens at
        head time, not inside a block — and it is the one that decides how a
        corrupted final residual is scaled before the unembed, so a frozen run
        that forgets it is not fully frozen.
        """
        if "final" not in self.freeze_capabilities().ln_sites:
            if freeze:
                raise NotImplementedError(
                    f"{type(self).__name__} does not declare the final norm as freezable, "
                    f"so a layernorm-freezing run cannot be completed through its head")
            return ()
        raise NotImplementedError(
            f"{type(self).__name__} declares the final norm freezable but does not "
            f"implement `final_norm_tap`")

    # ------------------------------------------------------------ provided

    @property
    def model(self):
        if self._model is None:
            self._model = self.load_model(self.device, self.dtype)
        return self._model

    def dictionary(self, layer: int) -> Dictionary:
        if layer not in self._dicts:
            self._dicts[layer] = self.load_dictionary(layer, self.device, self.dtype)
        return self._dicts[layer]

    def available_layers(self) -> list[int]:
        if self.dictionary_layers is None:
            return list(range(self.n_layers))
        return sorted(self.dictionary_layers)

    def free(self) -> None:
        """Drop cached weights.  Needed before any float32 CPU replay stage."""
        self._model = None
        self._dicts.clear()
        import gc

        gc.collect()
        if self.device == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif self.device == "cuda":
            torch.cuda.empty_cache()

    def substitution_plan(self, spec: ReplacementSpec) -> SubstitutionPlan:
        """THE CROSS-LAYER SEAM.  Override for a CLT-style artifact.

        Circuit mode is dispatched here rather than in an adapter override, so
        ANY adapter that supplies `head_tap()` gets circuit mode for free — the
        keep-set travels inside the spec, and the plan is chosen from it.
        """
        if spec.is_circuit:
            from ..circuit import CircuitPlan

            return CircuitPlan(spec, self.n_layers, spec.circuit)
        return PerLayerPlan(spec, self.n_layers)

    @torch.no_grad()
    def base_logits(self, model, toks: torch.Tensor) -> torch.Tensor:
        """The model's OWN forward.  Never the hand-rolled path — gate (i) is
        what licenses treating the two as interchangeable."""
        return model(toks, return_type="logits").float()

    @torch.no_grad()
    def clean_layer_major_logits(self, model, toks: torch.Tensor) -> torch.Tensor:
        """Clause 6: embed -> every block -> head, with nothing replaced."""
        resid = self.embed(model, toks)
        for L in range(self.n_layers):
            resid = self.block(model, L, resid)
        return self.head(model, resid)

    @torch.no_grad()
    def replaced_logits(self, model, toks: torch.Tensor, replacement, **kw) -> torch.Tensor:
        return replacement.plan().replaced_logits(self, model, toks, **kw)

    # ------------------------------------------------------------- gates

    @torch.no_grad()
    def gate_base_vs_base(self, toks: torch.Tensor, n: int = 4) -> G.GateResult:
        """Gate (i).  From run_b_gemma.py::stage_gates."""
        model = self.model
        tk = toks[: min(n, toks.shape[0])].to(self.device)
        ref = self.base_logits(model, tk)
        man = self.clean_layer_major_logits(model, tk)
        finite = bool(torch.isfinite(man).all() and torch.isfinite(ref).all())
        kl = M.per_position_kl(ref, man)[:, 1:]
        return G.check_base_vs_base(
            float(kl.max()) if finite else float("inf"),
            dtype=self.dtype_name, all_finite=finite,
            max_abs_logit_diff=float((ref - man).abs().max()) if finite else None,
        )

    @torch.no_grad()
    def measure_fvu(self, toks: torch.Tensor, layers: Sequence[int],
                    batch: int = 8) -> dict[str, dict[str, float]]:
        """Per-site dual FVU from a READ-ONLY clean pass.

        The denominator is taken against the true global mean over the whole
        subset (run_b_gemma.py::stage_pass buffers `y` for exactly this reason:
        a per-batch mean would make FVU depend on the batch size).
        """
        model = self.model
        n, T = toks.shape
        layers = sorted(layers)
        resid = None
        ys: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
        nums: dict[int, list[torch.Tensor]] = {L: [] for L in layers}
        # layer-major over the whole subset, one layer at a time
        chunks = []
        for i in range(0, n, batch):
            chunks.append(self.embed(model, toks[i : i + batch].to(self.device)).float().cpu())
        for L in range(self.n_layers):
            replacing = L in layers
            for bi, i in enumerate(range(0, n, batch)):
                r = chunks[bi].to(self.device, self.dtype)
                if replacing:
                    state, handles = self.tap(model, L, replace_fn=None)  # read-only
                    try:
                        r = self.block(model, L, r)
                    finally:
                        for h in handles:
                            h.remove()
                    y = state["y"].float()
                    yhat = self.dictionary(L).forward(state["x"]).float()
                    ys[L].append(y.reshape(-1, self.d_model).cpu())
                    nums[L].append((yhat - y).pow(2).sum(-1).cpu())
                    del state, y, yhat
                else:
                    r = self.block(model, L, r)
                chunks[bi] = r.float().cpu()
                del r
        out: dict[str, dict[str, float]] = {}
        for L in layers:
            y_all = torch.cat(ys[L], 0)
            num = torch.cat(nums[L], 0).reshape(n, T)
            den = (y_all - y_all.mean(0)).pow(2).sum(-1).reshape(n, T)
            out[Site(L, L).id] = S.dual_fvu(num, den)
        return out

    @torch.no_grad()
    def gate_fvu(self, toks: torch.Tensor, layers: Sequence[int],
                 batch: int = 8) -> tuple[G.GateResult, dict[str, dict[str, float]]]:
        """Gate (ii).  Returns the verdict AND the per-site table for the report."""
        table = self.measure_fvu(toks, layers, batch=batch)
        verdict = G.check_fvu_sanity({k: v["fvu_global"] for k, v in table.items()})
        return verdict, table

    @torch.no_grad()
    def gate_fp32_replay(self, toks: torch.Tensor, replacement, n: int = 4,
                         device: str = "cpu") -> G.GateResult:
        """Gate (iii).  Replay `n` sequences in float32 and compare BOTH streams.

        The float32 arm loads its OWN model so the two are never co-resident —
        on the gemma run the fp32 arm alone is ~13 GB.  Callers should
        `free()` the working model first, and this gate runs LAST so a failure
        costs nothing already computed.
        """
        if self.dtype_name == "float32":
            return G.GateResult(G.GATE_SPECS["iii-fp32-replay"], G.PASS, 0.0,
                                self.dtype_policy.fp32_replay_tolerance_nats,
                                "working dtype is already float32; no 16-bit gate needed")
        tk = toks[: min(n, toks.shape[0])]
        model16 = self.model
        base16 = self.base_logits(model16, tk.to(self.device))
        rep16 = self.replaced_logits(model16, tk.to(self.device), replacement)
        base16, rep16 = base16.float().cpu(), rep16.float().cpu()
        self.free()

        arm = type(self)(device=device, dtype="float32")
        try:
            m32 = arm.model
            base32 = arm.base_logits(m32, tk.to(device)).float().cpu()
            rep32 = arm.replaced_logits(m32, tk.to(device), _rebind(replacement, arm)).float().cpu()
        finally:
            arm.free()
        kb = M.per_position_kl(base32, base16)[:, 1:]
        kr = M.per_position_kl(rep32, rep16)[:, 1:]
        return G.check_fp32_replay(
            float(kr.max()), float(kb.max()), n=int(tk.shape[0]),
            tolerance=self.dtype_policy.fp32_replay_tolerance_nats, dtype=self.dtype_name,
        )

    def verify_provenance(self) -> G.GateResult:
        """Clause 1.  Default: revisions must be pinned.

        An adapter whose release has a variant-SELECTION rule (gemma-scope's
        canonical-L0) must override this and re-derive the selection from the
        live repository listing, as `gemma_artifact.discover_canonical_l0` does.
        """
        spec = G.GATE_SPECS["provenance-freeze"]
        if not self.identity.fully_pinned:
            return G.GateResult(
                spec, G.FAIL, None, None,
                f"{self.name}: model_revision={self.identity.model_revision!r} "
                f"dict_revision={self.identity.dict_revision!r} — an unpinned revision "
                f"means the artifact under audit is whatever the repo holds today",
                dict(identity=self.identity.to_dict()),
            )
        return G.GateResult(spec, G.PASS, 0.0, None,
                            f"model and dictionary revisions are pinned; this adapter "
                            f"has no variant-selection rule to re-derive",
                            dict(identity=self.identity.to_dict()))

    def gate_bos(self, corpus_meta: dict[str, Any]) -> G.GateResult:
        """Clause 3.  The miner must have obeyed the declared convention."""
        return G.check_bos_declaration(
            declared_bos_id=self.tokenization.bos_id,
            corpus_uses_bos=bool(corpus_meta.get("bos")),
            corpus_bos_id=corpus_meta.get("bos_id"),
            declared=self.tokenization.declared,
        )

    # -------------------------------------------------------------- report

    def contract(self) -> dict[str, Any]:
        """The declared contract, verbatim, for the report's provenance block."""
        return dict(
            name=self.name,
            identity=self.identity.to_dict(),
            taps=self.taps.to_dict(),
            tokenization=self.tokenization.to_dict(),
            dtype_policy=self.dtype_policy.to_dict(),
            n_layers=self.n_layers, d_model=self.d_model, d_vocab=self.d_vocab,
            n_heads=self.n_heads, d_head=self.d_head,
            circuit_capable=(type(self).head_tap is not ModelAdapter.head_tap),
            freeze_capabilities=self.freeze_capabilities().to_dict(),
            dictionary_layers=self.available_layers(),
            device=self.device, dtype=self.dtype_name,
            # the delegation stamp; computed by the module-level function so
            # overriding `contract()` cannot quietly drop it from the report
            overridden_measurement_methods=overridden_measurement_methods(self),
            required_by_contract=list(REQUIRED_OVERRIDES),
        )

    def run_tag(self, replacement) -> str:
        """Stamped into every scored row.

        Must name everything that changes what a `d_mean` MEANS: the artifact,
        the dtype, the replaced layer set, and the BOS convention.  A 2-layer
        smoke and a 26-layer full run produce completely different divergences
        for the same window, and an fp16 run's numbers are not a bfloat16
        run's.  From run_b_gemma.py's RUN_TAG.
        """
        layers = replacement.layers
        lay = "all" if layers == list(range(self.n_layers)) else ",".join(map(str, layers))
        tag = (f"{self.identity.release}:{self.dtype_name}:layers={lay}"
               f":mode={replacement.mode}:bos={int(self.tokenization.prepend_bos)}")
        # Two different keep-sets touch the same layers and would otherwise share
        # a tag; the identity guard would then merge their rows.
        if replacement.spec.circuit is not None:
            tag += f":circuit={replacement.spec.circuit.digest()}"
        # POLICY PROVENANCE.  The four rungs of experiment 06's ladder replace the
        # same layers with the same dictionaries in the same dtype and differ ONLY
        # in what they restore — so without this they would share a run tag, and
        # the identity guard would happily merge a skeleton's rows into a local
        # replacement model's.  Appended only when something is restored, so every
        # tag written before this existed still means what it said.
        if replacement.spec.freeze.restores_anything:
            tag += f":freeze={replacement.spec.freeze.tag()}"
        return tag

    def __repr__(self) -> str:
        return (f"<{type(self).__name__} {self.name} {self.n_layers}L "
                f"d_model={self.d_model} dtype={self.dtype_name} device={self.device}>")


def _rebind(replacement, adapter):
    """A copy of `replacement` bound to another adapter instance (the fp32 arm)."""
    from ..replacement import Replacement

    return Replacement(adapter, layers=replacement.spec.layers,
                       mode=replacement.spec.mode, seed=replacement.spec.seed,
                       circuit=replacement.spec.circuit, freeze=replacement.spec.freeze)


def pick_device() -> str:
    """Verbatim from witness.py::pick_device."""
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
