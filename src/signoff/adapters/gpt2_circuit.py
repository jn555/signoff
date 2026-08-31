"""GPT-2-small as a CIRCUIT host: no dictionaries, a per-head tap, circuit mode.

PROVENANCE.  The model-side primitives (`embed`, `block`, `head`) are the ones
`gpt2_dunefsky.py` already proves against gate (i) — same loading, same
positional-embedding handling, same head.  What this adapter adds is `head_tap`:
the per-head write point circuit mode needs.

WHAT THE "ARTIFACT" IS HERE.  For a dictionary adapter the audited artifact is a
weight file with a repo and a revision.  For a circuit it is a CLAIM in a paper:
a keep-set of components said to carry a behaviour.  So `identity.dict_repo` is
the citation and `dict_revision` the arXiv version — the same question ("what,
exactly, is under audit?") with the same answer shape.  The keep-set itself
travels in the `ReplacementSpec` (see `circuit.CircuitSpec`), is digested into
the run tag, and lands in the gate fingerprint, so two circuits over the same
model can never be confused for one another.

--------------------------------------------------------------------- the taps
HEAD  `blocks.L.attn.hook_z` — (batch, pos, n_heads, d_head), the per-head
      attention output BEFORE W_O mixes the heads back into d_model.  Writing
      here knocks out one head and leaves its neighbours untouched; writing at
      `hook_result` or `hook_attn_out` cannot express that.
MLP   `blocks.L.hook_mlp_out` — inherited from the dictionary adapter's tap, so
      an MLP-ablating circuit uses the same seam a transcoder does.

---------------------------------------------------------------------- loading
`HookedTransformer.from_pretrained("gpt2-small")` WITH the default weight
processing, cast to float32.  This matches the released IOI code (EasyTransformer
with fold_ln / center_writing_weights / center_unembed on) and, more to the
point, all three transforms are logit-difference-preserving: fold_ln is exact,
center_unembed shifts every logit by one per-position constant, and
center_writing_weights removes a direction the following LayerNorm removes
anyway.  So the metric this case study reports is unaffected by the choice —
which is worth knowing, and is why it is written down rather than assumed.

------------------------------------------------------------------ tokenization
NO BOS.  The IOI release says so in its own source ("no end of texts, GPT-2
small wasn't trained this way"), and its prompts are raw sentences.  Declared
here rather than defaulted, because the same choice was wrong for gemma-scope.

------------------------------------------------------------------------ dtype
float32 throughout, on CPU or MPS.  With no 16-bit arithmetic anywhere, gate
(iii) is vacuous for this adapter and reports as such.
"""

from __future__ import annotations

from typing import Callable

import torch

from .base import (
    DtypePolicy,
    Identity,
    ModelAdapter,
    TapSpec,
    TokenizationSpec,
)

MODEL = "gpt2-small"
MODEL_HF_REPO = "openai-community/gpt2"
MODEL_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

N_LAYERS = 12
N_HEADS = 12
D_HEAD = 64
D_MODEL = 768
D_MLP = 3072
D_VOCAB = 50257

#: The claim under audit is a paper, not a weight file.
CLAIM_REPO = "arXiv:2211.00593 (Wang, Variengien, Conmy, Shlegeris, Steinhardt)"
CLAIM_REV = "v1 (2022-11-01); code at github.com/redwoodresearch/Easy-Transformer"


class Gpt2CircuitAdapter(ModelAdapter):
    """GPT-2-small with every attention head individually ablatable."""

    name = "gpt2-circuit"

    identity = Identity(
        release="gpt2-small+published-circuit",
        model_repo=MODEL_HF_REPO,
        model_revision=MODEL_REV,
        dict_repo=CLAIM_REPO,
        dict_revision=CLAIM_REV,
        notes="No dictionary: the audited artifact is a CIRCUIT CLAIM, carried in the "
              "ReplacementSpec as a CircuitSpec and digested into the run tag. "
              "`dict_repo`/`dict_revision` pin the claim the way a repo/revision pins "
              "a weight file.",
    )
    taps = TapSpec(
        input_hook="blocks.{L}.attn.hook_z",
        output_hook="blocks.{L}.hook_mlp_out",
        input_convention="per-head attention output z, (batch, pos, n_heads, d_head), "
                         "BEFORE W_O — the only point at which one head can be replaced "
                         "without touching its neighbours",
        output_convention="the MLP sublayer's output, which for GPT-2 IS the block's "
                          "additive contribution to the residual stream (no post-MLP norm)",
        notes="The head tap is a WRITE point for circuit mode and a READ point for "
              "ablation-value calibration; the MLP tap is the dictionary seam, reused so "
              "an MLP-ablating circuit needs no new machinery.",
    )
    tokenization = TokenizationSpec(
        bos_id=None,
        declared=True,
        exclude_position_0=True,
        notes="The IOI release prepends no BOS ('GPT-2 small wasn't trained this way', "
              "ioi_dataset.py); its prompts are raw sentences.",
    )
    dtype_policy = DtypePolicy(
        default="float32",
        allowed=("float32",),
        measured={"float32": "no 16-bit arithmetic anywhere; gate (iii) is vacuous "
                             "and reports as such"},
        notes="GPT-2-small in float32 is ~0.5 GB; there is no reason to take 16-bit "
              "error into a measurement this cheap.",
    )
    n_layers = N_LAYERS
    d_model = D_MODEL
    d_vocab = D_VOCAB
    n_heads = N_HEADS
    d_head = D_HEAD
    #: No dictionaries at all.  `available_layers()` returns [] and gate (ii)
    #: has nothing to measure — a circuit run WAIVES it, visibly, rather than
    #: being handed a number that means nothing.
    dictionary_layers: tuple[int, ...] = ()

    # ---------------------------------------------------------------- loading

    def tokenizer(self):
        """The model's tokenizer, with the DECLARED BOS convention enforced on it.

        TransformerLens hands back a tokenizer with `add_bos_token = True`, so a
        bare `tokenizer(text)` silently prepends <|endoftext|> — which for a
        length-aligned templated task means every prompt is one token longer
        than the grammar says, every per-position mean is shifted by one, and
        the END position the logit difference is read at is off by one.  Nothing
        downstream would look wrong.

        This adapter declares `bos_id=None`, so the declaration is enforced here
        rather than trusted, and verified against a known string.  (The
        dictionary adapter deliberately does NOT do this: experiment 01 mined
        its corpus through the tokenizer as TransformerLens hands it over, and
        its golden numbers are that corpus's.)
        """
        if self._tok is None:
            tok = self.model.tokenizer
            tok.add_bos_token = False
            probe = tok(" John")["input_ids"]
            if len(probe) != 1 or probe[0] == tok.bos_token_id:
                raise RuntimeError(
                    f"the tokenizer still prepends BOS (' John' -> {probe}); this adapter "
                    f"declares bos_id=None and every position-aligned metric depends on it")
            self._tok = tok
        return self._tok

    def load_model(self, device, dtype):
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained(MODEL, device=device)
        model.eval()
        model.to(dtype)
        # clause 7: the shape facts the head tap and every circuit keep-set assume
        assert model.cfg.n_layers == N_LAYERS, (model.cfg.n_layers, N_LAYERS)
        assert model.cfg.n_heads == N_HEADS, (model.cfg.n_heads, N_HEADS)
        assert model.cfg.d_head == D_HEAD, (model.cfg.d_head, D_HEAD)
        assert model.cfg.d_model == D_MODEL, (model.cfg.d_model, D_MODEL)
        assert model.cfg.d_mlp == D_MLP, (model.cfg.d_mlp, D_MLP)
        assert model.cfg.d_vocab == D_VOCAB, (model.cfg.d_vocab, D_VOCAB)
        assert model.cfg.positional_embedding_type == "standard", \
            model.cfg.positional_embedding_type
        assert not model.cfg.use_normalization_before_and_after, \
            "GPT-2 has no sandwich norm; hook_mlp_out would not be the write point"
        return model

    def load_dictionary(self, layer: int, device, dtype):
        raise NotImplementedError(
            "gpt2-circuit has NO dictionaries: the artifact under audit is a circuit "
            "claim, and out-of-circuit components are replaced by declared ablation "
            "values (see circuit.py), not by a learned reconstruction. Gate (ii) has "
            "nothing to measure here and must be WAIVED with a reason, not faked."
        )

    # ------------------------------------------------------------- primitives

    def embed(self, model, toks: torch.Tensor) -> torch.Tensor:
        """Learned absolute positional embeddings must be added here, or the
        layer-major pass is a different model and gate (i) fails."""
        return model.embed(toks) + model.pos_embed(toks, 0)

    def block(self, model, layer: int, resid: torch.Tensor) -> torch.Tensor:
        return model.blocks[layer](resid)

    def head(self, model, resid: torch.Tensor) -> torch.Tensor:
        return model.unembed(model.ln_final(resid)).float()

    def tap(self, model, layer: int, replace_fn: Callable | None = None):
        """The MLP seam, identical to the dictionary adapter's."""
        state: dict[str, torch.Tensor] = {}
        block = model.blocks[layer]

        def on_norm(mod, inputs, out):
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
            block.ln2.hook_normalized.register_forward_hook(on_norm),
            block.hook_mlp_out.register_forward_hook(on_mlp_out),
        )
        return state, handles

    def head_tap(self, model, layer: int, write_fn: Callable | None = None):
        """Read or overwrite per-head `z` at one layer.

        With `write_fn=None` the hook returns None and the tap is strictly
        read-only — that is the mode ablation-value calibration runs in, and
        the reason a clean forward under a calibration tap is still bit-exactly
        the model's own forward.
        """
        state: dict[str, torch.Tensor] = {}
        hook = model.blocks[layer].attn.hook_z

        def on_z(mod, inputs, out):
            state["z"] = out
            if write_fn is None:
                return None
            new = write_fn(out).to(out.dtype)
            state["z_written"] = new
            return new

        return state, (hook.register_forward_hook(on_z),)


def gpt2_circuit(device: str | None = None, dtype: str | None = None) -> Gpt2CircuitAdapter:
    return Gpt2CircuitAdapter(device=device, dtype=dtype)
