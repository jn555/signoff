"""Llama-3.2-1B + the circuit-tracer CROSS-LAYER transcoder (`mntss/clt-llama-3.2-1b-524k`).

This is the first adapter for an artifact this package did not produce: it audits
what the open circuit-tracer / attribution-graph ecosystem actually ships today.
It is also the first CROSS-LAYER adapter, so it is the thing the
`Site(read, write)` / `SubstitutionPlan` seam was built for.

=============================================================================
PROVENANCE — where every fact below was read, and when
=============================================================================
Investigated 2026-08-31 by reading the released source and the released
checkpoint headers, not the papers or the README prose.

  [CT]  github.com/decoderesearch/circuit-tracer @ 6018ed8d35e40f2c50062822e8dde422b8e52e2d
        (committed 2026-08-30).  *** The repository MOVED: `safety-research/
        circuit-tracer` now 301-redirects to `decoderesearch/circuit-tracer`.
        Both names resolve; the new one is canonical. ***
        - `circuit_tracer/transcoder/cross_layer_transcoder.py`  (the CLT)
        - `circuit_tracer/transcoder/single_layer_transcoder.py` (the PLT, for contrast)
        - `circuit_tracer/replacement_model/replacement_model_transformerlens.py`
        - `circuit_tracer/utils/hf_utils.py`  (config resolution)
        - `README.md` §"Available transcoders"
  [CFG] huggingface.co/mntss/clt-llama-3.2-1b-524k/raw/main/config.yaml
  [HDR] the safetensors headers of that repo's `W_enc_*.safetensors` /
        `W_dec_*.safetensors`, read by HTTP range request (dtype + shape, no
        weights downloaded).

WHAT circuit-tracer SUPPORTS (README, [CT]).  Two artifact families, both:
  Gemma-2-2b   PLT `mntss/gemma-scope-transcoders`; CLT `mntss/clt-gemma-2-2b-426k`
                                                    and `mntss/clt-gemma-2-2b-2.5M`
  Llama-3.2-1B PLT `mntss/transcoder-Llama-3.2-1B`;  CLT `mntss/clt-llama-3.2-1b-524k`
  Qwen-3       PLT `mwhanna/qwen3-{0.6b,1.7b,4b,8b,14b}-transcoders*`
               (the 0.6B one is already audited by `qwen3-mwhanna`)
  GPT-OSS-20B  CLT `mntss/clt-131k`
  Gemma-3      PLT collection `mwhanna/gemma-scope-2-transcoders-circuit-tracer` (nnsight only)
  Llama-3.1-8B-Instruct  TopK PLT `facebook/crv-8b-instruct-transcoders` (k = 128, hardcoded)

Llama-3.2-1B is the RAM-cheapest model in that list, so it is the first target.
Both of its artifact families are per-MLP substitutions with the SAME hooks
([CFG] for both repos: `feature_input_hook: hook_resid_mid`,
`feature_output_hook: hook_mlp_out`) and they differ in exactly the way that
matters here:

  PLT `mntss/transcoder-Llama-3.2-1B`  — per-layer, d_sae 131072, AND IT HAS A
      SKIP CONNECTION: [HDR] `layer_0.safetensors` carries `W_skip (2048, 2048)`
      alongside W_enc/W_dec/b_enc/b_dec.  Applied as `x @ W_skip.T` [CT
      single_layer_transcoder.py::compute_skip].  No `threshold` key, so
      `load_transcoder` falls through to `activation_function = F.relu` [CT
      single_layer_transcoder.py::load_transcoder].
  CLT `mntss/clt-llama-3.2-1b-524k`    — cross-layer, 16 x 32768 features, NO
      skip connection, JumpReLU.  This adapter.

THE ONE THAT WAS BUILT.  The CLT, because (a) STATUS names the CLT as the
v0.1.1 milestone, (b) a Llama PLT adapter would be structurally a copy of
`qwen_mwhanna.py` with a skip term and would exercise nothing new, and (c) the
cross-layer path is the untested half of this package.

=============================================================================
THE ARTIFACT — layout, verified against the released headers (clause 7)
=============================================================================
[HDR], all BF16 on disk:

    W_enc_{L}.safetensors :  W_enc_{L}   (32768, 2048)   f, d
                             b_enc_{L}   (32768,)
                             b_dec_{L}   (2048,)         <- the WRITE-layer bias
                             threshold_{L} (32768,)      <- JumpReLU
    W_dec_{L}.safetensors :  W_dec_{L}   (32768, 16-L, 2048)   f, j, d

`W_dec_{L}[f, j, :]` is feature f of read layer L writing into **layer L + j**.
So read layer L feeds write layers L..15, and layer 15 feeds only itself; that
is the whole cross-layer structure, and it is why `W_dec_0` is 2.1 GB and
`W_dec_15` is 134 MB.

The forward, transcribed from [CT cross_layer_transcoder.py]:

    a_L      = h * (h > threshold_L),   h = x_L @ W_enc_L.T + b_enc_L
    recon_W  = b_dec_W + SUM_{L <= W} a_L @ W_dec_L[:, W-L, :]

Three details that are easy to get wrong and are asserted or reproduced here:
  * `b_dec` is indexed by the WRITE layer and added ONCE per write layer
    ([CT]::compute_reconstruction adds `b_dec[:, None]` after the index_add),
    NOT once per contributing read layer.
  * the JumpReLU is `h * (h > threshold)`, which does NOT clamp negatives
    ([CT]::apply_activation_function is `features * (features > threshold)`).
    That is identical to SAELens' `relu(h) * (h > threshold)` only while every
    threshold is positive; we reproduce circuit-tracer's form so it matches
    either way, and report the observed threshold minimum.
  * there is NO skip connection in this release.  `load_clt`'s
    `_load_state_dict` never reads a `W_skip` key at all, so every CLT loaded
    through that path is skip-free by construction, and [HDR] confirms no such
    key exists.  (The gemma-scope-2 CLT path *can* carry one, from
    `affine_skip_connection`; a different loader, a different artifact.)

=============================================================================
THE TAP — a THIRD convention, different from both existing adapters
=============================================================================
  INPUT  = `blocks.{L}.hook_resid_mid` — the RAW, UN-NORMALISED residual stream
           after attention and BEFORE ln2.  [CT
           replacement_model_transformerlens.py] installs the feature-input hook
           at `f"blocks.{layer}.{self.feature_input_hook}"` with
           `feature_input_hook = "hook_resid_mid"` [CFG], i.e. the transcoder
           sees the residual stream itself, not any normalised version of it.

           *** This is the third distinct convention in this package's three
           real adapters, and all three are reachable from the same two hook
           NAMES:
               gemma-scope-2b : ln2.hook_normalized  = PRE-gain  normalised
               qwen3-mwhanna  : mlp module input     = POST-gain normalised
               llama32-clt    : hook_resid_mid       = NOT normalised at all
           Gate (ii) is the verifier; a flip reads FVU ~ 1 at every site. ***

  OUTPUT = `blocks.{L}.hook_mlp_out` — the MLP's contribution before the
           residual add.  Llama has no post-MLP norm (no sandwich norm, unlike
           gemma-2), so this IS the block's additive contribution, and the
           write point coincides with gemma's even though the read point does
           not.  Substitution semantics confirmed at [CT
           replacement_model_transformerlens.py:459]: `error_vectors =
           mlp_out_cache - reconstruction`, i.e. the reconstruction stands in
           for `hook_mlp_out` and the residue is the error node.

  MODEL LOADING must be unfolded.  [CT] loads with `fold_ln=False,
  center_writing_weights=False, center_unembed=False`; `center_writing_weights`
  in particular would shift `hook_resid_mid` itself, which is the tap.
  `load_model` asserts the model is unfolded.

=============================================================================
BOS
=============================================================================
circuit-tracer PREPENDS a special token and treats position 0 as an artifact
position: `_prepend_bos_if_needed` inserts `tokenizer.bos_token_id` when the
first token is not already special, with the comment "Prepend a special token
to avoid artifacts at position 0", and attribution zeroes position-0 features
(`zero_positions = slice(0, 1)`) and position-0 error vectors [CT
replacement_model_transformerlens.py:171,295,461].  For Llama-3.2 that token is
`<|begin_of_text|>` = 128000, asserted against the tokenizer on first use.
This is the OPPOSITE declaration from `qwen3-mwhanna` (no BOS at all).

=============================================================================
DTYPE — nothing is measured yet, and this file says so
=============================================================================
`dtype_policy.measured` carries UNMEASURED strings, not results.  No gate (iii)
run exists for this artifact on any host (see TIER), so the only defensible
default is float32, where gate (iii) is vacuous.  bfloat16 is *allowed* — it is
the on-disk dtype and the only way the full artifact fits anywhere — but a
bfloat16 run must clear gate (iii) first, and the precedent is not encouraging:
bfloat16 FAILED gate (iii) by ~4x on gemma-2 (`gemma_scope.py`).  float16 is
not offered: nothing measures it here and the artifact is bfloat16-trained.

=============================================================================
TIER: local-only, and CURRENTLY UNRUNNABLE ON THIS HOST.  Two blockers.
=============================================================================
1. `meta-llama/Llama-3.2-1B` is a GATED repo (`gated: manual`).  Verified
   2026-08-31: this host's HF token gets `GatedRepoError 403` on even
   `config.json`.  Until access is granted, no gate touching the model can run.
2. The dictionary is 20.4 GB of weights: encoders 16 x 134 MB = 2.15 GB,
   decoders SUM_L 32768 x (16-L) x 2048 x 2 B = 18.25 GB.  (The repo's
   `features/*.bin` add ~5 GB more and are visualization data this tool never
   reads.)  A full-artifact run therefore does not fit a 16 GB / 10 GB-free
   laptop, and this adapter is written so that it does not have to:

   FAITHFUL PARTIAL RUNS.  `CrossLayerPlan` treats `Replacement(layers=S)` as
   "S is both the read set and the write set".  Because `recon_W` sums over
   ALL read layers L <= W, the plan reproduces the artifact EXACTLY iff S is a
   PREFIX `{0..k}` — for a prefix, every L <= W that the artifact would sum is
   in S.  For any other S the reconstruction at W is missing real contributions
   and the run measures a TRUNCATED artifact, not the artifact.  That is not
   forbidden (single-layer localisation profiles are a supported use of
   `Replacement`), but it is stamped: `run_tag` carries `clt=prefix` or
   `clt=truncated`, and `plan.describe()` says which.

   A prefix `{0..k}` costs (k+1) encoders and (k+1)(k+2)/2 decoder PLANES of
   134 MB each in bf16 — only the planes the plan can reach are materialised,
   sliced out of the safetensors file rather than loaded whole.  `{0,1}` is
   the smallest genuinely cross-layer smoke: 2 encoders + 3 planes ~ 0.67 GB
   in bf16, on top of ~2.5 GB of model.  (The 2.1 GB `W_dec_0` file must still
   be DOWNLOADED whole; only the RAM is saved.)

=============================================================================
DELEGATION STAMP
=============================================================================
This adapter overrides FOUR base measurement methods, so
`overridden_measurement_methods()` returns
`["measure_fvu", "verify_provenance", "substitution_plan", "run_tag"]` and the
report marks gate (ii), the provenance gate, the identity guard and the
checkpoint binding `[self-reported]`.  That is a real cost and it is paid on
purpose; each override is forced, and each is argued where it is defined:

  * `substitution_plan` — THE SEAM.  Any cross-layer adapter overrides this by
    construction; a `PerLayerPlan` cannot express a CLT.
  * `measure_fvu`       — the base method hardcodes `Site(L, L)` and calls
    `dictionary(L).forward(x)`.  Neither is meaningful when the reconstruction
    at a write layer is a SUM over read layers.
  * `verify_provenance` — the base default only checks that revisions are
    pinned.  Here there is a real drift surface: the release's `config.yaml`
    declares the HOOKS, so an upstream edit to it would silently invalidate
    every tap sentence above.  The override re-derives it from the live repo.
  * `run_tag`           — a prefix run and a truncated run of the same SIZE are
    different experiments (see TIER), and `run_tag` is the field that "must
    name everything that changes what a `d_mean` MEANS".  Leaving it un-stamped
    would buy two fewer `[self-reported]` marks by hiding a real distinction.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch

from .. import gates as G
from .. import stats as S
from ..replacement import ReplacementSpec, Site, SubstitutionPlan, null_replace_fn
from .base import (
    Dictionary,
    DtypePolicy,
    Identity,
    ModelAdapter,
    TapSpec,
    TokenizationSpec,
)

MODEL = "meta-llama/Llama-3.2-1B"
MODEL_REV = "4e20de362430cd3b72f300e6b0f18e50e7166e08"
CLT_REPO = "mntss/clt-llama-3.2-1b-524k"
CLT_REV = "d6ab3ffa8184c9894b2ffb35093c50c098c0f223"

N_LAYERS = 16
D_MODEL = 2048
D_TRANSCODER = 32768           # per read layer; 16 x 32768 = 524288 = the "524k"
D_VOCAB = 128256
BOS_ID = 128000                # <|begin_of_text|>; verified against the tokenizer

#: bf16 bytes, from the released headers.  Used by the footprint helper so a
#: caller can be told what a layer set costs BEFORE anything is downloaded.
ENC_BYTES = D_TRANSCODER * D_MODEL * 2          # 134.2 MB per read layer
PLANE_BYTES = D_TRANSCODER * D_MODEL * 2        # 134.2 MB per (read -> write) plane

#: The release's own config.yaml, frozen.  It is what declares the HOOKS, so a
#: silent upstream edit to it would invalidate this file's tap sentences.
#: `verify_provenance()` re-downloads it and halts on any difference.
CONFIG_FREEZE = {
    "model_name": MODEL,
    "model_kind": "cross_layer_transcoder",
    "feature_input_hook": "hook_resid_mid",
    "feature_output_hook": "hook_mlp_out",
}


def enc_file(layer: int) -> str:
    return f"W_enc_{layer}.safetensors"


def dec_file(layer: int) -> str:
    return f"W_dec_{layer}.safetensors"


def footprint_bytes(layers: Sequence[int], n_layers: int = N_LAYERS,
                    bytes_per_param: int = 2) -> dict[str, int]:
    """What a layer set costs, before anything is loaded.

    `download` is on-disk and always bf16 (2 B) — that is how the release is
    stored.  `resident` scales with `bytes_per_param`, so pass 4 to price a
    float32 run, which is this adapter's default dtype and TWICE the RAM.

    `download` exceeds `resident` on purpose: a decoder PLANE is sliced out of a
    whole `W_dec_L.safetensors`, and that file has to exist on disk first.
    """
    S_ = sorted(set(int(L) for L in layers))
    planes = sum(1 for L in S_ for W in S_ if W >= L)
    scale = bytes_per_param / 2
    resident = int((len(S_) * ENC_BYTES + planes * PLANE_BYTES) * scale)
    download = len(S_) * ENC_BYTES + sum((n_layers - L) * PLANE_BYTES for L in S_)
    return dict(resident=resident, download=download, n_encoders=len(S_),
                n_planes=planes, bytes_per_param=int(bytes_per_param))


def bundle_id(read_layers: Sequence[int], write_layer: int) -> str:
    """The site key gate (ii) is given for one write layer's reconstruction.

    A CLT gives a write layer ONE target and MANY read sites, so there is no
    per-(read, write) FVU to compute: the artifact provides no per-read-layer
    target to take a residual against.  One FVU therefore covers the BUNDLE of
    sites feeding `write_layer`, and the key names the bundle rather than
    pretending to be a single `Site`:

        [3], 3        -> "3"        (a lone diagonal site — an ordinary Site id)
        [0, 1, 2], 2  -> "0..2->2"

    Gate (ii) keys on the string and is indifferent to its shape
    (`check_fvu_sanity` docstring); the report prints it verbatim.
    """
    reads = sorted(set(int(L) for L in read_layers))
    if not reads:
        raise ValueError(f"write layer {write_layer} has no contributing read layer")
    if reads == [int(write_layer)]:
        return Site(int(write_layer), int(write_layer)).id
    if len(reads) == 1:
        return Site(reads[0], int(write_layer)).id
    return f"{reads[0]}..{reads[-1]}->{int(write_layer)}"


class CltLayer(Dictionary):
    """One READ layer of the cross-layer transcoder: its encoder + its planes.

    Not a per-layer dictionary.  It owns the encoder for read layer L, the
    write-layer bias `b_dec_L` (which belongs to L *as a write layer*), and a
    lazily materialised map `write_layer -> W_dec_L[:, write-L, :]`.

    Clause 7 asserts on every tensor at load: shapes, the (f, j, d) decoder
    layout, the plane count `n_layers - L`, and the threshold minimum (reported,
    not required positive — see the module docstring on the JumpReLU form).
    """

    def __init__(self, layer: int, n_layers: int, enc_path: str, dec_path: str,
                 device, dtype, chunk: int = 256):
        from safetensors import safe_open

        self.layer = int(layer)
        self.n_layers = int(n_layers)
        self.dec_path = dec_path
        self.device = device
        self.dtype = dtype
        self.chunk = int(chunk)
        self._planes: dict[int, torch.Tensor] = {}

        L = self.layer
        with safe_open(enc_path, framework="pt") as f:
            keys = set(f.keys())
            for k in (f"W_enc_{L}", f"b_enc_{L}", f"b_dec_{L}", f"threshold_{L}"):
                assert k in keys, f"{enc_file(L)} is missing {k!r}; keys={sorted(keys)}"
            assert "W_skip" not in keys and f"W_skip_{L}" not in keys, (
                f"{enc_file(L)} carries a skip connection; this adapter's forward "
                f"has none and would silently drop it")
            self.W_enc = f.get_tensor(f"W_enc_{L}").to(device, dtype)
            self.b_enc = f.get_tensor(f"b_enc_{L}").to(device, dtype)
            self.b_dec = f.get_tensor(f"b_dec_{L}").to(device, dtype)
            self.threshold = f.get_tensor(f"threshold_{L}").to(device, dtype)

        assert self.W_enc.shape == (D_TRANSCODER, D_MODEL), self.W_enc.shape
        assert self.b_enc.shape == (D_TRANSCODER,), self.b_enc.shape
        assert self.b_dec.shape == (D_MODEL,), self.b_dec.shape
        assert self.threshold.shape == (D_TRANSCODER,), self.threshold.shape

        with safe_open(dec_path, framework="pt") as f:
            assert f"W_dec_{L}" in set(f.keys()), f"{dec_file(L)} is missing W_dec_{L}"
            shape = tuple(f.get_slice(f"W_dec_{L}").get_shape())
        assert shape == (D_TRANSCODER, self.n_layers - L, D_MODEL), (
            f"decoder layout for read layer {L} is {shape}; expected "
            f"(d_transcoder, n_layers - L, d_model) = "
            f"({D_TRANSCODER}, {self.n_layers - L}, {D_MODEL}) — a CLT read layer "
            f"writes into layers L..n_layers-1 and nothing else")
        self.dec_shape = shape
        self.threshold_min = float(self.threshold.min())

    # ------------------------------------------------------------------ parts

    def write_layers(self) -> range:
        """Every layer this read layer can write into."""
        return range(self.layer, self.n_layers)

    def plane(self, write_layer: int) -> torch.Tensor:
        """`W_dec_L[:, write-L, :]`, materialised on first use and cached.

        Slicing the safetensors file is what keeps a prefix run affordable: a
        full `W_dec_0` is 2.1 GB, one plane is 134 MB.
        """
        from safetensors import safe_open

        W = int(write_layer)
        if W not in self._planes:
            if not (self.layer <= W < self.n_layers):
                raise ValueError(
                    f"read layer {self.layer} cannot write into layer {W}: a CLT "
                    f"feature writes only into its own layer and later ones")
            with safe_open(self.dec_path, framework="pt") as f:
                sl = f.get_slice(f"W_dec_{self.layer}")[:, W - self.layer, :]
            self._planes[W] = sl.to(self.device, self.dtype).contiguous()
        return self._planes[W]

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """`h * (h > threshold)` with `h = x @ W_enc.T + b_enc`.

        circuit-tracer's exact form — it does NOT clamp negatives, so it is only
        equal to SAELens' `relu(h) * (h > t)` while `t > 0`.
        """
        sh = x.shape
        xf = x.reshape(-1, sh[-1]).to(self.dtype)
        out = torch.empty(xf.shape[0], D_TRANSCODER, device=xf.device, dtype=self.dtype)
        for s in range(0, xf.shape[0], self.chunk):
            e = min(s + self.chunk, xf.shape[0])
            h = xf[s:e] @ self.W_enc.T + self.b_enc
            out[s:e] = h * (h > self.threshold)
            del h
        return out.reshape(*sh[:-1], D_TRANSCODER)

    @torch.no_grad()
    def contribution(self, acts: torch.Tensor, write_layer: int) -> torch.Tensor:
        """`acts @ W_dec_L[:, write-L, :]`.  NO bias — `b_dec` belongs to the
        write layer and is added once, by the plan."""
        sh = acts.shape
        af = acts.reshape(-1, sh[-1])
        P = self.plane(write_layer)
        out = torch.empty(af.shape[0], D_MODEL, device=af.device, dtype=self.dtype)
        for s in range(0, af.shape[0], self.chunk):
            e = min(s + self.chunk, af.shape[0])
            out[s:e] = af[s:e] @ P
        return out.reshape(*sh[:-1], D_MODEL)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """The `Dictionary` protocol, and a WARNING in code form.

        This is the DIAGONAL-ONLY reconstruction — `b_dec_L + a_L @ plane(L)` —
        i.e. what a CLT truncated to read layer L alone would write at layer L.
        It is NOT the artifact's reconstruction at layer L, which also sums the
        contributions of every read layer above it.  Nothing in this adapter
        calls it: `substitution_plan()` returns `CrossLayerPlan`, and
        `measure_fvu` is overridden.  It exists so the protocol is honoured and
        so `Replacement(layers=[L])` — which IS a truncated run, and is stamped
        as one — does something well-defined rather than something wrong.
        """
        return (self.contribution(self.encode(x), self.layer) + self.b_dec).to(x.dtype)


class CrossLayerPlan(SubstitutionPlan):
    """The substituted forward for a CLT.  `layers` is the read set AND the write set.

    Sites are every `(L, W)` with `L <= W` and both in the set.  See the module
    docstring, "FAITHFUL PARTIAL RUNS": this equals the artifact exactly iff the
    set is a prefix `{0..k}`, and `is_prefix` is what `run_tag` stamps.

    The forward holds ONE activation tensor at a time.  Rather than caching
    `a_L` until the last write layer needs it, each `a_L` is pushed straight
    into a running partial `pending[W]` for every `W >= L` still to come, so
    peak memory is |S| tensors of (B, T, d_model) plus one of
    (B, T, d_transcoder) — not |S| of the latter.
    """

    #: The read set and the write set are the SAME set — `Replacement` carries
    #: one layer list, and a CLT read layer that is not also written would
    #: contribute to nothing in range.  Named for readability at the call sites.
    @property
    def read_layers(self) -> list[int]:
        return list(self.layers)

    @property
    def write_layers(self) -> list[int]:
        return list(self.layers)

    @property
    def is_prefix(self) -> bool:
        """True iff every read layer the artifact would sum at every write layer
        in the set is itself in the set — i.e. the set is `{0..k}`."""
        return self.layers == list(range(len(self.layers)))

    def contributing(self, write_layer: int) -> list[int]:
        return [L for L in self.read_layers if L <= write_layer]

    def sites(self) -> list[Site]:
        return [Site(L, W) for W in self.write_layers for L in self.contributing(W)]

    def bundles(self) -> dict[str, list[int]]:
        """write-layer bundle id -> the read layers inside it."""
        return {bundle_id(self.contributing(W), W): self.contributing(W)
                for W in self.write_layers}

    def describe(self) -> str:
        kind = ("prefix — reproduces the artifact exactly at every write layer"
                if self.is_prefix else
                "TRUNCATED — write layers in this set are missing the contributions "
                "of read layers outside it; this is not the artifact")
        return (f"CrossLayerPlan: {self.spec.describe(self.n_layers)}; "
                f"{len(self.sites())} (read -> write) sites; {kind}")

    @torch.no_grad()
    def replaced_logits(self, adapter, model, toks, *, noise=None, capture=None):
        if self.spec.is_null and noise is None:
            raise ValueError(f"{self.spec.mode} needs a per-layer noise field; "
                             "compute it from this batch's real errors first")
        active = set(self.layers)
        pending: dict[int, torch.Tensor | None] = {W: None for W in self.layers}

        resid = adapter.embed(model, toks)
        for W in range(self.n_layers):
            if W not in active:
                resid = adapter.block(model, W, resid)
                continue

            if self.spec.is_null:
                fn = null_replace_fn(noise[W])
            else:
                def fn(x, y_true, _W=W):
                    # push this read layer's contributions forward FIRST (the
                    # `Wt == _W` term is this layer's own), then consume
                    d = adapter.dictionary(_W)
                    a = d.encode(x)
                    for Wt in self.layers:
                        if Wt >= _W:
                            c = d.contribution(a, Wt)
                            pending[Wt] = c if pending[Wt] is None else pending[Wt] + c
                    del a
                    out = pending[_W] + d.b_dec
                    pending[_W] = None            # freed as soon as it is consumed
                    return out

            state, handles = adapter.tap(model, W, replace_fn=fn)
            try:
                resid = adapter.block(model, W, resid)
            finally:
                for h in handles:
                    h.remove()
            if capture is not None:
                capture[bundle_id(self.contributing(W), W)] = state
        return adapter.head(model, resid)


class LlamaCltAdapter(ModelAdapter):
    """Llama-3.2-1B with all 16 MLP sublayers replaceable by one cross-layer transcoder."""

    name = "llama32-clt-mntss"

    identity = Identity(
        release="Llama-3.2-1B+mntss/clt-llama-3.2-1b-524k (cross-layer, 16 x 32768 features)",
        model_repo=MODEL, model_revision=MODEL_REV,
        dict_repo=CLT_REPO, dict_revision=CLT_REV,
        notes="The cross-layer transcoder used by the open circuit-tracer stack "
              "(decoderesearch/circuit-tracer, formerly safety-research/circuit-tracer). "
              "JumpReLU, no skip connection, b_dec indexed by WRITE layer. Read layer L "
              "writes into layers L..15, so a partial run reproduces the artifact only "
              "on a PREFIX layer set — run_tag stamps which.",
    )
    taps = TapSpec(
        input_hook="blocks.{L}.hook_resid_mid",
        output_hook="blocks.{L}.hook_mlp_out",
        input_convention="NOT NORMALISED: the raw residual stream after attention and "
                         "before ln2. circuit-tracer's config.yaml declares "
                         "feature_input_hook: hook_resid_mid and its replacement model "
                         "hooks blocks.{L}.hook_resid_mid directly",
        output_convention="the MLP's contribution before the residual add — Llama has no "
                          "post-MLP norm, so hook_mlp_out IS the block's additive "
                          "contribution; circuit-tracer defines its error node as "
                          "mlp_out - reconstruction at this hook",
        notes="THE THIRD CONVENTION. gemma-scope reads the PRE-gain normalised input and "
              "qwen3-mwhanna the POST-gain normalised input; this artifact reads neither, "
              "it reads the residual stream itself. Same two hook names, three meanings. "
              "The model must be loaded UNFOLDED — center_writing_weights would move "
              "hook_resid_mid, which is the tap.",
    )
    tokenization = TokenizationSpec(
        bos_id=BOS_ID, declared=True, exclude_position_0=True,
        notes="circuit-tracer prepends a special token when the first token is not already "
              "one ('to avoid artifacts at position 0') and zeroes position-0 features and "
              "error vectors during attribution. For Llama-3.2 that token is "
              "<|begin_of_text|> = 128000. Opposite declaration from qwen3-mwhanna, which "
              "declares no BOS at all.",
    )
    dtype_policy = DtypePolicy(
        default="float32",
        allowed=("float32", "bfloat16"),
        measured={
            "float32": "UNMEASURED — but gate (iii) is vacuous at float32, which is why it "
                       "is the default. It is also circuit-tracer's own default dtype.",
            "bfloat16": "UNMEASURED for this artifact on any host. It is the on-disk dtype "
                        "and the only way the full 20.4 GB fits, but bfloat16 FAILED gate "
                        "(iii) by ~4x on gemma-2 (see gemma_scope.py), so a bfloat16 run "
                        "here must clear gate (iii) before its numbers mean anything.",
        },
        notes="NO gate-(iii) measurement exists for this adapter: the model repo is gated "
              "and this host has no access (see the module docstring, TIER). float16 is "
              "not offered — nothing measures it and the artifact is bfloat16-trained.",
    )
    n_layers = N_LAYERS
    d_model = D_MODEL
    d_vocab = D_VOCAB

    def __init__(self, device=None, dtype=None, clt_chunk: int = 256):
        super().__init__(device=device, dtype=dtype)
        self.clt_chunk = int(clt_chunk)

    # --------------------------------------------------------------- the seam

    def substitution_plan(self, spec: ReplacementSpec) -> CrossLayerPlan:
        return CrossLayerPlan(spec, self.n_layers)

    def footprint(self, layers: Sequence[int]) -> dict[str, int]:
        """RAM and download cost of a layer set, in bytes, priced at THIS
        adapter's working dtype.  Needs no network and no weights."""
        return footprint_bytes(layers, self.n_layers,
                               bytes_per_param=4 if self.dtype_name == "float32" else 2)

    def run_tag(self, replacement) -> str:
        """Base tag plus the fact that decides what a CLT `d_mean` MEANS.

        A truncated set and a prefix set of the same SIZE are different
        experiments: the truncated one is missing real contributions.
        """
        plan = self.substitution_plan(replacement.spec)
        return (super().run_tag(replacement)
                + f":clt={'prefix' if plan.is_prefix else 'truncated'}")

    # ---------------------------------------------------------------- loading

    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
            assert tok.bos_token_id == BOS_ID, (tok.bos_token_id, BOS_ID)
            self._tok = tok
        return self._tok

    def load_model(self, device, dtype):
        """`from_pretrained_no_processing` — circuit-tracer's `fold_ln=False,
        center_writing_weights=False, center_unembed=False`.

        Folding is not a stylistic choice here: `center_writing_weights` shifts
        the residual stream, and the residual stream at `hook_resid_mid` IS the
        transcoder's input.
        """
        from transformer_lens import HookedTransformer

        try:
            m = HookedTransformer.from_pretrained_no_processing(
                MODEL, device=device, dtype=dtype, revision=MODEL_REV)
        except TypeError:
            m = HookedTransformer.from_pretrained_no_processing(
                MODEL, device=device, dtype=dtype)
        m.eval()
        cfg = m.cfg
        assert cfg.n_layers == N_LAYERS, (cfg.n_layers, N_LAYERS)
        assert cfg.d_model == D_MODEL, (cfg.d_model, D_MODEL)
        assert cfg.d_vocab == D_VOCAB, (cfg.d_vocab, D_VOCAB)
        assert cfg.positional_embedding_type == "rotary", cfg.positional_embedding_type
        assert cfg.normalization_type == "RMS", cfg.normalization_type
        # no sandwich norm: that is what makes hook_mlp_out the additive contribution
        assert not cfg.use_normalization_before_and_after, \
            "this model has a post-MLP norm; the CLT's write point would be wrong"
        assert not cfg.parallel_attn_mlp, \
            "parallel attn/mlp: there is no hook_resid_mid to read"
        assert not float(cfg.output_logits_soft_cap or 0), cfg.output_logits_soft_cap
        assert not cfg.post_embedding_ln
        # unfolded, or the tap moved out from under us
        assert hasattr(m.blocks[0].ln2, "w"), \
            "ln2 has no gain: the model was loaded folded, and folding moves hook_resid_mid"
        assert hasattr(m.blocks[0], "hook_resid_mid"), "no hook_resid_mid to read"
        return m

    def load_dictionary(self, layer: int, device, dtype) -> CltLayer:
        """Download this read layer's encoder file and its decoder file.

        The decoder file is downloaded WHOLE (2.1 GB at layer 0, 134 MB at layer
        15) because safetensors slices a local file; only the planes the plan
        reaches are then materialised in RAM.  `footprint()` reports both
        numbers before anything is fetched.
        """
        from huggingface_hub import hf_hub_download

        enc = hf_hub_download(CLT_REPO, enc_file(layer), revision=CLT_REV)
        dec = hf_hub_download(CLT_REPO, dec_file(layer), revision=CLT_REV)
        return CltLayer(layer, self.n_layers, enc, dec, device, dtype, chunk=self.clt_chunk)

    # ------------------------------------------------------------- primitives

    def embed(self, model, toks):
        return model.embed(toks)  # rotary-only: no positional embedding added

    def block(self, model, layer: int, resid):
        return model.blocks[layer](resid)

    def head(self, model, resid):
        """Clause 5.  No logit softcap on Llama, so this is the whole head.
        Llama-3.2-1B ties W_U to W_E; TransformerLens materialises both."""
        return model.unembed(model.ln_final(resid)).float()

    def tap(self, model, layer: int, replace_fn: Callable | None = None):
        """Read at `hook_resid_mid` (raw), write at `hook_mlp_out`.

        With `replace_fn=None` both hooks return None, so the tap is strictly
        read-only and `measure_fvu` can use it on a clean pass.
        """
        block = model.blocks[layer]
        state: dict = {}

        def on_resid_mid(mod, inputs, out):
            state["x"] = out
            return None

        def on_mlp_out(mod, inputs, out):
            state["y"] = out
            if replace_fn is None:
                return None
            yhat = replace_fn(state["x"], out).to(out.dtype)
            state["yhat"] = yhat
            return yhat

        handles = (
            block.hook_resid_mid.register_forward_hook(on_resid_mid),
            block.hook_mlp_out.register_forward_hook(on_mlp_out),
        )
        return state, handles

    # ------------------------------------------------------------------ gates

    @torch.no_grad()
    def measure_fvu(self, toks: torch.Tensor, layers: Sequence[int],
                    batch: int = 8) -> dict[str, dict[str, float]]:
        """Cross-layer FVU.  Overrides the base method, which cannot express this.

        WHY THE OVERRIDE (this is what the report's "adapter-overridden
        measurement methods" line is pointing at).  `ModelAdapter.measure_fvu`
        keys its output on `Site(L, L)` and reconstructs with
        `dictionary(L).forward(x)`.  For a CLT neither holds: the reconstruction
        at write layer W is a SUM over every read layer L <= W, and there is no
        per-(read, write) residual to take — one target, many read sites.  So
        one FVU is reported per WRITE layer, keyed by `bundle_id`, and the
        detail names the read layers inside the bundle.

        Two deliberate deviations from the base implementation, both documented
        rather than hidden:
          * BATCH-MAJOR, not layer-major.  The base version walks layers so only
            one dictionary is resident; a CLT needs every read layer's encoder
            resident anyway, so a single clean forward per batch with read-only
            taps on the whole set is both simpler and cheaper.
          * The clean pass is the MODEL's own forward, not the hand-rolled
            layer-major loop.  Gate (i) is what licenses treating those as
            interchangeable, and it runs first.

        The denominator is the global mean over the whole subset, exactly as the
        base method does it, so FVU does not depend on `batch`.

        ARITHMETIC MATCHES THE FORWARD.  The contributions are summed in the
        DICTIONARY's dtype and `b_dec` is added LAST, which is the order
        `CrossLayerPlan.replaced_logits` uses.  Accumulating the sum in float32
        instead would be more accurate and would describe a reconstruction the
        run does not actually write — the same mistake `gemma_scope.head()`
        documents for the softcap.
        """
        model = self.model
        n, T = toks.shape
        S_ = sorted(set(int(L) for L in layers))
        ys: dict[int, list[torch.Tensor]] = {W: [] for W in S_}
        nums: dict[int, list[torch.Tensor]] = {W: [] for W in S_}

        for i in range(0, n, batch):
            tk = toks[i : i + batch].to(self.device)
            states, handles = {}, []
            for L in S_:
                st, hs = self.tap(model, L, replace_fn=None)   # read-only
                states[L] = st
                handles.extend(hs)
            try:
                self.base_logits(model, tk)
            finally:
                for h in handles:
                    h.remove()

            pending: dict[int, torch.Tensor | None] = {W: None for W in S_}
            for L in S_:                       # one activation tensor at a time
                d = self.dictionary(L)
                a = d.encode(states[L]["x"])
                for W in S_:
                    if W >= L:
                        c = d.contribution(a, W)
                        pending[W] = c if pending[W] is None else pending[W] + c
                del a
            for W in S_:
                y = states[W]["y"].float()
                recon = (pending[W] + self.dictionary(W).b_dec).float()
                ys[W].append(y.reshape(-1, self.d_model).cpu())
                nums[W].append((recon - y).pow(2).sum(-1).cpu())
                del recon
            del states, pending

        out: dict[str, dict[str, float]] = {}
        for W in S_:
            reads = [L for L in S_ if L <= W]
            y_all = torch.cat(ys[W], 0)
            num = torch.cat(nums[W], 0).reshape(n, T)
            den = (y_all - y_all.mean(0)).pow(2).sum(-1).reshape(n, T)
            row = S.dual_fvu(num, den)
            row["n_read_layers"] = len(reads)
            row["write_layer"] = int(W)
            out[bundle_id(reads, W)] = row
        return out

    def verify_provenance(self) -> G.GateResult:
        """Clause 1: re-derive the release's own config.yaml and halt on drift.

        The revisions are pinned, which is all the base method checks.  But this
        release declares its HOOKS in `config.yaml`, and every tap sentence in
        this file is a transcription of that file plus circuit-tracer's loader.
        If mntss re-uploads with a different `feature_input_hook`, the pinned
        revision still resolves and the tap declaration is silently wrong.  This
        gate is what makes that stop the run.

        Also checks the file inventory: 16 encoder files and 16 decoder files,
        no more and no fewer.
        """
        spec = G.GATE_SPECS["provenance-freeze"]
        try:
            import yaml
            from huggingface_hub import HfApi, hf_hub_download

            path = hf_hub_download(CLT_REPO, "config.yaml", revision=CLT_REV)
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
            files = set(HfApi().list_repo_files(CLT_REPO, revision=CLT_REV))
        except Exception as e:
            return G.GateResult(
                spec, G.UNRUN, None, None,
                f"could not re-derive {CLT_REPO}@{CLT_REV[:8]} ({type(e).__name__}); the "
                f"frozen hook declaration is UNVERIFIED, so the taps in this adapter are "
                f"a transcription nothing has checked",
                dict(error=str(e), frozen=dict(CONFIG_FREEZE)))

        observed = {k: cfg.get(k) for k in CONFIG_FREEZE}
        expected = dict(CONFIG_FREEZE)
        for L in range(self.n_layers):
            expected[f"file:{enc_file(L)}"] = True
            expected[f"file:{dec_file(L)}"] = True
            observed[f"file:{enc_file(L)}"] = enc_file(L) in files
            observed[f"file:{dec_file(L)}"] = dec_file(L) in files
        return G.check_provenance_freeze(
            expected, observed,
            what="circuit-tracer config.yaml hook declaration + CLT file inventory")


def llama32_clt_mntss(device=None, dtype=None, **kw) -> LlamaCltAdapter:
    return LlamaCltAdapter(device=device, dtype=dtype, **kw)
