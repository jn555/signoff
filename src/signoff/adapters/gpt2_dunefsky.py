"""GPT-2-small + Dunefsky/Chlenski MLP transcoders.

PROVENANCE.  Extracted from experiments/01-divergence-witnesses/witness.py
(`Transcoder`, `load_transcoders`, `ReplacementModel`).  This is the adapter the
golden-number regression test pins: a fixed token set scored by this code must
reproduce experiment 01's own per-row numbers.

------------------------------------------------------------------- the taps
INPUT  `blocks.L.ln2.hook_normalized` — the transcoders were trained on the
       ln2-normalised MLP input of the PROCESSED TransformerLens model.
OUTPUT `blocks.L.hook_mlp_out` — GPT-2 has no post-MLP norm, so the MLP
       sublayer's output IS the block's additive contribution to the residual
       stream, and the two candidate write points coincide.  (They do NOT
       coincide on gemma-2, which is where the tap trap lives.)

---------------------------------------------------------------------- loading
`HookedTransformer.from_pretrained("gpt2-small")` — WITH the default weight
processing (fold_ln, center_writing_weights, center_unembed), and cast to
float32.  This is deliberate and load-bearing: these transcoders were trained
against the processed model, so `ln2.hook_normalized` is the gain-folded
normalised input they expect.  Switching to `from_pretrained_no_processing`
would change the tap's meaning and gate (ii) would read it.

------------------------------------------------------------------ tokenization
NO BOS.  exp-01 mined raw 64-token windows with no BOS prepended, and every
metric drops position 0.  That is declared here rather than defaulted, because
the same choice was WRONG for the BOS-trained gemma-scope suite (corpus NLL
7.005 -> 3.217 nats purely from prepending BOS).

------------------------------------------------------------------------ dtype
float32 throughout, on CPU or MPS.  With no 16-bit arithmetic anywhere, gate
(iii) is vacuous for this adapter and reports as such.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable, Sequence

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

TRANSCODER_REPO = "pchlenski/gpt2-transcoders"
TRANSCODER_REV = "4f3ca345572ce2d75018a60d10b4dca8bd899e99"
TRANSCODER_FILE = "final_sparse_autoencoder_gpt2-small_blocks.{L}.ln2.hook_normalized_24576.pt"

MODEL = "gpt2-small"
#: `gpt2` on the Hub; TransformerLens resolves the alias.
MODEL_HF_REPO = "openai-community/gpt2"
MODEL_REV = "607a30d783dfa663caf39e06633721c8d4cfcd7e"

N_LAYERS = 12
D_MODEL = 768
D_MLP = 3072
D_SAE = 24576
D_VOCAB = 50257

_STUB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_unpickle_stubs")


@dataclass
class Transcoder(Dictionary):
    """Dunefsky/Chlenski transcoder: mlp_in (ln2-normalised) -> mlp_out.

    Forward, verbatim from the release's own
    `transcoder_circuits/sae_training/sparse_autoencoder.py`:

        sae_in  = x - b_dec
        acts    = relu(sae_in @ W_enc + b_enc)
        sae_out = acts @ W_dec + b_dec_out      # b_dec_out because is_transcoder

    Note the INPUT PRE-BIAS (`x - b_dec`).  Neither the gemma-scope nor the
    mwhanna release has one (`apply_b_dec_to_input` is False for both), so it is
    exactly the kind of per-release detail that belongs in an adapter and
    nowhere else.
    """

    layer: int
    W_enc: torch.Tensor  # (d_in, d_sae)
    b_enc: torch.Tensor  # (d_sae,)
    W_dec: torch.Tensor  # (d_sae, d_out)
    b_dec: torch.Tensor  # (d_in,)  -- input pre-bias
    b_dec_out: torch.Tensor  # (d_out,)

    def to(self, device, dtype=torch.float32) -> "Transcoder":
        return Transcoder(
            self.layer,
            self.W_enc.to(device, dtype), self.b_enc.to(device, dtype),
            self.W_dec.to(device, dtype), self.b_dec.to(device, dtype),
            self.b_dec_out.to(device, dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """NOT `@torch.no_grad()` — deliberately, since the gradient oracle.

        Every MEASUREMENT caller reaches this from inside an outer `no_grad`
        (`PerLayerPlan.replaced_logits`, `capture_clean_run`, `measure_fvu` are
        all decorated), so dropping the decorator changes no scoring path's
        memory profile.  What it does change: a GCG-style miner needs the
        substituted forward to be differentiable end to end, and a `no_grad`
        decorator HERE severs the graph at the dictionary — the search would
        then be steered by the skip connections alone, silently.
        """
        acts = F.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        return acts @ self.W_dec + self.b_dec_out


class Gpt2DunefskyAdapter(ModelAdapter):
    """GPT-2-small with its 12 MLP sublayers replaceable by transcoders."""

    name = "gpt2-dunefsky"

    identity = Identity(
        release="gpt2-small+dunefsky-transcoders",
        model_repo=MODEL_HF_REPO,
        model_revision=MODEL_REV,
        dict_repo=TRANSCODER_REPO,
        dict_revision=TRANSCODER_REV,
        notes="Dunefsky et al. transcoders, redistributed by pchlenski; one per layer, "
              "d_sae 24576 (expansion 32). No variant-selection rule: the repo holds "
              "exactly one checkpoint per layer.",
    )
    taps = TapSpec(
        input_hook="blocks.{L}.ln2.hook_normalized",
        output_hook="blocks.{L}.hook_mlp_out",
        input_convention="ln2-normalised MLP input of the PROCESSED model (fold_ln applied, "
                         "so the LayerNorm gain is folded into W_in and hook_normalized is "
                         "the gain-free normalised residual)",
        output_convention="the MLP sublayer's output, which for GPT-2 IS the block's "
                          "additive contribution to the residual stream (no post-MLP norm)",
        notes="GPT-2 is the easy case: the two candidate write points coincide. On gemma-2 "
              "they do not, and the same hook NAME carries the opposite gain convention.",
    )
    tokenization = TokenizationSpec(
        bos_id=None,
        declared=True,
        exclude_position_0=True,
        notes="exp-01 mined raw windows with no BOS; position 0 is unconditioned and is "
              "dropped by every metric.",
    )
    dtype_policy = DtypePolicy(
        default="float32",
        allowed=("float32",),
        measured={"float32": "no 16-bit arithmetic anywhere; gate (iii) is vacuous "
                             "and reports as such"},
        notes="exp-01 ran float32 end to end on MPS/CPU. 16-bit is not offered rather "
              "than offered-and-ungated.",
    )
    n_layers = N_LAYERS
    d_model = D_MODEL
    d_vocab = D_VOCAB

    # ---------------------------------------------------------------- loading

    def tokenizer(self):
        if self._tok is None:
            self._tok = self.model.tokenizer
        return self._tok

    def load_model(self, device, dtype):
        from transformer_lens import HookedTransformer

        model = HookedTransformer.from_pretrained(MODEL, device=device)
        model.eval()
        model.to(dtype)
        # clause 7, model side: the shape facts the taps and the dictionaries assume
        assert model.cfg.n_layers == N_LAYERS, (model.cfg.n_layers, N_LAYERS)
        assert model.cfg.d_model == D_MODEL, (model.cfg.d_model, D_MODEL)
        assert model.cfg.d_mlp == D_MLP, (model.cfg.d_mlp, D_MLP)
        assert model.cfg.d_vocab == D_VOCAB, (model.cfg.d_vocab, D_VOCAB)
        assert model.cfg.positional_embedding_type == "standard", \
            model.cfg.positional_embedding_type
        assert not model.cfg.use_normalization_before_and_after, \
            "GPT-2 has no sandwich norm; hook_mlp_out would not be the write point"
        return model

    def load_dictionary(self, layer: int, device, dtype) -> Transcoder:
        """Load one transcoder, with the exp-01 assert block (clause 7).

        Every assertion here caught something once or guards something that
        would be silent: a transposed W_enc reads as a shape error, but a
        checkpoint trained against a DIFFERENT hook point would not — hence the
        hook_point / out_hook_point identity checks against this adapter's
        declared taps.
        """
        from huggingface_hub import hf_hub_download

        if _STUB_PATH not in sys.path:
            sys.path.insert(0, _STUB_PATH)  # unpickling stub, see _unpickle_stubs/
        path = hf_hub_download(
            TRANSCODER_REPO, TRANSCODER_FILE.format(L=layer), revision=TRANSCODER_REV
        )
        obj = torch.load(path, map_location="cpu", weights_only=False)
        cfg, sd = obj["cfg"], obj["state_dict"]
        assert cfg.is_transcoder is True, f"L{layer}: not a transcoder"
        assert cfg.hook_point == self.taps.input_hook.format(L=layer), cfg.hook_point
        assert cfg.out_hook_point == self.taps.output_hook.format(L=layer), cfg.out_hook_point
        assert cfg.d_in == D_MODEL and cfg.d_out == D_MODEL, (cfg.d_in, cfg.d_out)
        assert cfg.d_sae == D_SAE, cfg.d_sae
        assert tuple(sd["W_enc"].shape) == (D_MODEL, D_SAE), tuple(sd["W_enc"].shape)
        assert tuple(sd["W_dec"].shape) == (D_SAE, D_MODEL), tuple(sd["W_dec"].shape)
        assert tuple(sd["b_enc"].shape) == (D_SAE,)
        assert tuple(sd["b_dec"].shape) == (D_MODEL,)
        assert tuple(sd["b_dec_out"].shape) == (D_MODEL,)
        return Transcoder(
            layer, sd["W_enc"], sd["b_enc"], sd["W_dec"], sd["b_dec"], sd["b_dec_out"]
        ).to(device, dtype)

    def dictionary_meta(self, layer: int) -> dict:
        """Training metadata of one checkpoint, for the report's provenance block."""
        from huggingface_hub import hf_hub_download

        if _STUB_PATH not in sys.path:
            sys.path.insert(0, _STUB_PATH)
        path = hf_hub_download(
            TRANSCODER_REPO, TRANSCODER_FILE.format(L=layer), revision=TRANSCODER_REV
        )
        cfg = torch.load(path, map_location="cpu", weights_only=False)["cfg"]
        return dict(hook_point=cfg.hook_point, out_hook_point=cfg.out_hook_point,
                    d_in=cfg.d_in, d_sae=cfg.d_sae, d_out=cfg.d_out,
                    expansion_factor=cfg.expansion_factor,
                    l1_coefficient=cfg.l1_coefficient,
                    total_training_tokens=cfg.total_training_tokens,
                    run_name=cfg.run_name)

    # ------------------------------------------------------------- primitives

    def embed(self, model, toks: torch.Tensor) -> torch.Tensor:
        """GPT-2 uses LEARNED ABSOLUTE positional embeddings: they must be added
        here, or the layer-major pass is a different model and gate (i) fails."""
        return model.embed(toks) + model.pos_embed(toks, 0)

    def token_embedding_matrix(self, model) -> torch.Tensor:
        """`W_E`, (50257, 768).  GPT-2's embedding is additive in it, so the
        base class's `embed_from_onehot` is exact here — and `gradients.py`
        checks that against `embed()` before a search starts."""
        return model.W_E

    def block(self, model, layer: int, resid: torch.Tensor) -> torch.Tensor:
        return model.blocks[layer](resid)

    def head(self, model, resid: torch.Tensor) -> torch.Tensor:
        """Clause 5.  GPT-2 has no logit softcap, so this is the whole head."""
        return model.unembed(model.ln_final(resid)).float()

    def tap(self, model, layer: int, replace_fn: Callable | None = None):
        """Read the transcoder input at ln2.hook_normalized, the target at hook_mlp_out.

        With `replace_fn=None` both hooks return None and the tap is strictly
        read-only (the non-invasiveness rule that makes FVU measurable on the
        clean stream).  Structure follows gemma_artifact.py::tap; exp-01's
        `_hooks_for` did the same thing with TL's `run_with_hooks`.
        """
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

    # ------------------------------------------------------------------ misc

    def build_corpus(self, n_seqs: int, seq_len: int = 64, seed: int = 0):
        """Convenience: this adapter's tokenizer + the shared corpus miner."""
        from ..miners import build_corpus

        return build_corpus(self.tokenizer(), n_seqs, seq_len=seq_len, seed=seed,
                            bos_id=self.tokenization.bos_id)


def gpt2_dunefsky(device: str | None = None, dtype: str | None = None) -> Gpt2DunefskyAdapter:
    """Factory, mirroring the API sketch: `adapters.gpt2_dunefsky()`."""
    return Gpt2DunefskyAdapter(device=device, dtype=dtype)
