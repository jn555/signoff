"""Qwen3-0.6B + mwhanna low-L0 ReLU transcoders.

PROVENANCE.  Extracted from experiments/03-mechanism-and-validity/run_b_validity.py
(the artifact-specific half: `TC`, `load_tc`, `load_model`, `tap`,
`final_logits`).  The model-agnostic halves of that script are this package's
metrics/stats/runner.

--------------------------------------------------------------------- THE TAP
Read this next to `gemma_scope.py` — the two are the reason gate (ii) exists.

  INPUT  = the MLP module's OWN input, i.e. the POST-GAIN normalised residual.
           These transcoders were trained on the value the MLP actually
           receives, so the tap is a forward PRE-hook on `blocks.L.mlp`.
           *** `blocks.L.ln2.hook_normalized` is the WRONG tap here — it is the
           pre-gain value, which is exactly what the gemma-scope release wants.
           Same hook name, opposite convention. ***

  OUTPUT = the MLP module's return value.  Qwen3 has no post-MLP norm, so that
           IS the block's additive contribution to the residual stream, and
           writing at the module output is correct (on gemma-2 it would miss
           `ln2_post`).

------------------------------------------------------------------ THE SUITE
This suite is SATURATED, and the report should say so rather than average over
it: measured tail ratio p99/median 1.66 (vs 2.27 for a healthy GPT-2 suite) and
flip ~ 0.98 — the replacement disagrees with the base model's argmax on
essentially every position, so "divergence" has little dynamic range left.
That is a finding about the artifact, not a defect of the tool, and it is why
suite health is REPORTED and not gated: three suites is not a criterion.

TIER: local-only.  d_sae is 163840 (160x expansion) — 18.8 GB of transcoder
weights across 28 layers, which is why the reference run evicted each file
after use.
"""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn.functional as F

from .base import (
    Dictionary,
    DtypePolicy,
    Identity,
    ModelAdapter,
    TapSpec,
    TokenizationSpec,
)

MODEL = "Qwen/Qwen3-0.6B"
MODEL_REV = "c1899de289a04d12100db370d81485cdf75e47ca"
TC_REPO = "mwhanna/qwen3-0.6b-transcoders-lowl0"
TC_REV = "28aefe686c09e9bc1a862195b51e145f10868d87"
TC_FILE = "layer_{L}.safetensors"

N_LAYERS = 28
D_MODEL = 1024
D_SAE = 163840
D_VOCAB = 151936


class ReluTranscoder(Dictionary):
    """`relu(x @ W_enc + b_enc) @ W_dec + b_dec`.

    Clause 7, the transposition trap: SAELens'
    `mwhanna_transcoder_huggingface_loader` stores W_enc as (d_sae, d_in) and
    TRANSPOSES ON LOAD — the opposite of both other shipped releases, which
    store (d_in, d_sae).  A missed transpose is a shape error here, which is the
    good case; the asserts make it loud either way.

    `apply_b_dec_to_input` is False, so there is no input pre-bias (unlike the
    Dunefsky transcoders).
    """

    def __init__(self, layer: int, sd: dict, device, dtype, chunk: int = 256):
        self.layer = layer
        self.chunk = chunk
        self.dtype = dtype
        self.W_enc = sd["W_enc"].T.contiguous().to(device, dtype)   # (d_in, d_sae)
        self.b_enc = sd["b_enc"].to(device, dtype)
        self.W_dec = sd["W_dec"].to(device, dtype)                  # (d_sae, d_out)
        self.b_dec = sd["b_dec"].to(device, dtype)
        assert self.W_enc.shape == (D_MODEL, D_SAE), self.W_enc.shape
        assert self.W_dec.shape == (D_SAE, D_MODEL), self.W_dec.shape
        assert self.b_enc.shape == (D_SAE,), self.b_enc.shape
        assert self.b_dec.shape == (D_MODEL,), self.b_dec.shape

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sh = x.shape
        xf = x.reshape(-1, sh[-1]).to(self.dtype)
        out = torch.empty(xf.shape[0], self.W_dec.shape[1], device=xf.device, dtype=self.dtype)
        for s in range(0, xf.shape[0], self.chunk):
            e = min(s + self.chunk, xf.shape[0])
            acts = F.relu(xf[s:e] @ self.W_enc + self.b_enc)
            out[s:e] = acts @ self.W_dec + self.b_dec
            del acts
        return out.reshape(sh).to(x.dtype)


class QwenMwhannaAdapter(ModelAdapter):
    """Qwen3-0.6B with its 28 MLP sublayers replaceable by mwhanna transcoders."""

    name = "qwen3-mwhanna"

    identity = Identity(
        release="Qwen3-0.6B+mwhanna-qwen3-0.6b-transcoders-lowl0",
        model_repo=MODEL, model_revision=MODEL_REV,
        dict_repo=TC_REPO, dict_revision=TC_REV,
        notes="One checkpoint per layer, d_sae 163840 (160x expansion), low-L0 variant. "
              "Measured as a SATURATED suite: tail ratio p99/median 1.66 and flip ~ 0.98.",
    )
    taps = TapSpec(
        input_hook="blocks.{L}.mlp (forward pre-hook)",
        output_hook="blocks.{L}.mlp (forward hook)",
        input_convention="POST-GAIN: the value the MLP module actually receives, which is "
                         "what these transcoders were trained on",
        output_convention="the MLP module's return value — Qwen3 has no post-MLP norm, so "
                          "it IS the block's additive contribution to the residual stream",
        notes="THE TAP TRAP, from the other side. `ln2.hook_normalized` is the WRONG tap "
              "for this release and the RIGHT one for gemma-scope. Gate (ii) is the "
              "verifier: a flip reads FVU ~ 1 at every layer.",
    )
    tokenization = TokenizationSpec(
        bos_id=None, declared=True, exclude_position_0=True,
        notes="No BOS: Qwen3's tokenizer defines no bos_token_id and the reference run "
              "mined raw windows. Declared explicitly rather than defaulted — the same "
              "choice is WRONG for the BOS-trained gemma-scope suite.",
    )
    dtype_policy = DtypePolicy(
        default="float16",
        allowed=("float32", "float16"),
        measured={"float16": "the reference run's default (3.9x faster end to end); gate "
                             "(iii) must be run to license it on a given host"},
        notes="bfloat16 is not offered: it failed gate (iii) by ~4x on gemma-2, and this "
              "adapter has no measurement licensing it here.",
    )
    n_layers = N_LAYERS
    d_model = D_MODEL
    d_vocab = D_VOCAB

    def __init__(self, device=None, dtype=None, tc_chunk: int = 256):
        super().__init__(device=device, dtype=dtype)
        self.tc_chunk = int(tc_chunk)

    # ---------------------------------------------------------------- loading

    def tokenizer(self):
        if self._tok is None:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(MODEL, revision=MODEL_REV)
            assert tok.bos_token_id is None, \
                f"tokenizer now declares bos_token_id={tok.bos_token_id}; the adapter " \
                f"declares no BOS and the corpus would be mined off-distribution"
            self._tok = tok
        return self._tok

    def load_model(self, device, dtype):
        from transformer_lens import HookedTransformer

        try:
            m = HookedTransformer.from_pretrained_no_processing(
                MODEL, device=device, dtype=dtype, revision=MODEL_REV)
        except TypeError:
            m = HookedTransformer.from_pretrained_no_processing(
                MODEL, device=device, dtype=dtype)
        m.eval()
        assert m.cfg.n_layers == N_LAYERS, (m.cfg.n_layers, N_LAYERS)
        assert m.cfg.d_model == D_MODEL, (m.cfg.d_model, D_MODEL)
        assert m.cfg.positional_embedding_type == "rotary", m.cfg.positional_embedding_type
        # no sandwich norm: that is what makes the mlp module's output the write point
        assert not m.cfg.use_normalization_before_and_after, \
            "this model has a post-MLP norm; the write point would be hook_mlp_out instead"
        return m

    def load_dictionary(self, layer: int, device, dtype) -> ReluTranscoder:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file

        path = hf_hub_download(TC_REPO, TC_FILE.format(L=layer), revision=TC_REV)
        sd = load_file(path)
        tc = ReluTranscoder(layer, sd, device, dtype, chunk=self.tc_chunk)
        del sd
        return tc

    # ------------------------------------------------------------- primitives

    def embed(self, model, toks):
        return model.embed(toks)  # rotary-only: no positional embedding added

    def block(self, model, layer: int, resid):
        return model.blocks[layer](resid)

    def head(self, model, resid):
        """No logit softcap on Qwen3, so this is the whole head."""
        return model.unembed(model.ln_final(resid)).float()

    def tap(self, model, layer: int, replace_fn: Callable | None = None):
        """Pre-hook the MLP for its POST-GAIN input; post-hook it for the output.

        `pre` returns None (it never alters the input). With `replace_fn=None`
        the post hook also returns None, so the tap is completely read-only.
        """
        mlp = model.blocks[layer].mlp
        state: dict = {}

        def pre(mod, inputs):
            state["x"] = inputs[0]
            return None

        def post(mod, inputs, out):
            state["y"] = out
            if replace_fn is None:
                return None
            yhat = replace_fn(state["x"], out).to(out.dtype)
            state["yhat"] = yhat
            return yhat

        return state, (mlp.register_forward_pre_hook(pre),
                       mlp.register_forward_hook(post))


def qwen3_mwhanna(device=None, dtype=None, **kw) -> QwenMwhannaAdapter:
    return QwenMwhannaAdapter(device=device, dtype=dtype, **kw)
