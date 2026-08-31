"""A tiny synthetic artifact.  NOT A MODEL — a fixture and a worked example.

Why this ships instead of living in the test directory:

  * It is the smallest complete statement of the adapter contract.  An adapter
    author can read it in one sitting and see every clause satisfied, without
    the weight-loading and release-archaeology that make the real adapters long.
  * It makes the whole pipeline — gates, scoring, mining, family stats, report,
    CLI — runnable on a CPU with no network and no weights, which is what keeps
    the refusal paths and the report format under CI.

It is deterministic, ~50k parameters, and means NOTHING about interpretability.
Any number it produces is about this fixture.  The registry lists it at tier
"test" and the reporter stamps reports from it as synthetic.
"""

from __future__ import annotations

from typing import Callable

import torch

from .base import (
    Dictionary,
    DtypePolicy,
    Identity,
    ModelAdapter,
    TapSpec,
    TokenizationSpec,
)

N_LAYERS = 4
D_MODEL = 16
D_HIDDEN = 32
D_VOCAB = 64
SEED = 0


class _Block:
    """resid -> resid + sublayer(norm(resid)).  The 'MLP' is `sublayer`."""

    def __init__(self, layer: int, g: torch.Generator):
        self.layer = layer
        self.W_in = torch.randn(D_MODEL, D_HIDDEN, generator=g) * 0.5
        self.W_out = torch.randn(D_HIDDEN, D_MODEL, generator=g) * 0.5
        self.tap_state: dict | None = None
        self.replace_fn: Callable | None = None

    def norm(self, resid):
        return resid / resid.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    def sublayer(self, x):
        return torch.tanh(x @ self.W_in) @ self.W_out

    def __call__(self, resid):
        x = self.norm(resid)
        y = self.sublayer(x)
        if self.tap_state is not None:
            self.tap_state["x"] = x
            self.tap_state["y"] = y
        if self.replace_fn is not None:
            y = self.replace_fn(x, y)
            if self.tap_state is not None:
                self.tap_state["yhat"] = y
        return resid + y


class _ToyModel:
    def __init__(self, dtype=torch.float32):
        g = torch.Generator().manual_seed(SEED)
        self.W_E = torch.randn(D_VOCAB, D_MODEL, generator=g) * 0.3
        self.W_pos = torch.randn(64, D_MODEL, generator=g) * 0.1
        self.W_U = torch.randn(D_MODEL, D_VOCAB, generator=g) * 0.3
        self.blocks = [_Block(L, g) for L in range(N_LAYERS)]
        self.cfg = type("cfg", (), dict(n_layers=N_LAYERS, d_model=D_MODEL, d_vocab=D_VOCAB))()

    def embed(self, toks):
        return self.W_E[toks] + self.W_pos[: toks.shape[1]].unsqueeze(0)

    def head(self, resid):
        return (resid / resid.norm(dim=-1, keepdim=True).clamp_min(1e-6)) @ self.W_U

    def __call__(self, toks, return_type="logits"):
        resid = self.embed(toks)
        for b in self.blocks:
            resid = b(resid)
        return self.head(resid)


class _ToyDictionary(Dictionary):
    """A deliberately imperfect reconstruction: the true sublayer plus a defect.

    `defect` is calibrated to be roughly the RELATIVE reconstruction error, so
    FVU comes out near `defect**2`: 0.35 is a healthy-looking suite, 1.5 or
    more produces the mis-tap signature (FVU ~ 1 everywhere) that gate (ii)
    must catch. The 1/sqrt(d_model) scaling is what makes the knob mean that.
    """

    def __init__(self, layer: int, block: _Block, defect: float = 0.35):
        g = torch.Generator().manual_seed(SEED * 1000 + layer)
        self.layer = layer
        self.block = block
        self.defect = float(defect)
        scale = defect / (D_MODEL ** 0.5)
        self.noise = torch.randn(D_MODEL, D_MODEL, generator=g) * scale
        self.bias = torch.randn(D_MODEL, generator=g) * scale * 0.1

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block.sublayer(x)
        return y + y @ self.noise + self.bias


class _Handle:
    def __init__(self, block: _Block):
        self.block = block

    def remove(self):
        self.block.tap_state = None
        self.block.replace_fn = None


class ToyAdapter(ModelAdapter):
    """The contract, in miniature.  See the module docstring — this is a fixture."""

    name = "toy"

    identity = Identity(
        release="synthetic-toy-artifact",
        model_repo="(synthetic)", model_revision="deterministic-seed-0",
        dict_repo="(synthetic)", dict_revision="deterministic-seed-0",
        notes="NOT A MODEL. A deterministic fixture for testing the pipeline, the gates "
              "and the report format without weights or a network.",
    )
    taps = TapSpec(
        input_hook="blocks.{L}.norm_out",
        output_hook="blocks.{L}.sublayer_out",
        input_convention="the normalised residual entering the sublayer",
        output_convention="the sublayer's additive contribution to the residual stream",
        notes="Synthetic: there is exactly one sensible tap and no convention trap.",
    )
    tokenization = TokenizationSpec(
        bos_id=None, declared=True, exclude_position_0=True,
        notes="No BOS; synthetic token ids are uniform over the toy vocabulary.",
    )
    dtype_policy = DtypePolicy(
        default="float32", allowed=("float32",),
        measured={"float32": "synthetic; gate (iii) is vacuous"},
    )
    n_layers = N_LAYERS
    d_model = D_MODEL
    d_vocab = D_VOCAB

    def __init__(self, device: str | None = None, dtype: str | None = None,
                 defect: float = 0.35, defect_overrides: dict[int, float] | None = None):
        super().__init__(device="cpu", dtype=dtype)   # synthetic: CPU only
        self.defect = defect
        self.defect_overrides = dict(defect_overrides or {})

    # ------------------------------------------------------------- contract

    def tokenizer(self):
        raise NotImplementedError(
            "the toy adapter has no tokenizer: it is a fixture, and its corpus is "
            "synthetic token ids (see `synthetic_corpus`)"
        )

    def load_model(self, device, dtype):
        return _ToyModel(dtype)

    def load_dictionary(self, layer: int, device, dtype) -> _ToyDictionary:
        # a mild depth gradient, so the per-site FVU table is not flat
        base = self.defect * (1.0 + 0.12 * layer)
        return _ToyDictionary(layer, self.model.blocks[layer],
                              defect=self.defect_overrides.get(layer, base))

    def embed(self, model, toks):
        return model.embed(toks)

    def block(self, model, layer: int, resid):
        return model.blocks[layer](resid)

    def head(self, model, resid):
        return model.head(resid).float()

    def tap(self, model, layer: int, replace_fn: Callable | None = None):
        b = model.blocks[layer]
        state: dict = {}
        b.tap_state = state
        b.replace_fn = replace_fn
        return state, (_Handle(b),)

    # ------------------------------------------------------------- fixtures

    def synthetic_corpus(self, n: int = 32, seq_len: int = 16, seed: int = 0,
                         n_probes: int = 8):
        """Tokens + corpus meta in the shape the runner and stats expect.

        A "probe family" is simulated by drawing its tokens from a restricted
        sub-vocabulary — genuinely different text statistics, so the family
        test has something real to find.
        """
        g = torch.Generator().manual_seed(seed)
        corpus = torch.randint(0, D_VOCAB, (n, seq_len), generator=g)
        probes = torch.randint(0, D_VOCAB // 8, (n_probes, seq_len), generator=g)
        toks = torch.cat([corpus, probes], 0) if n_probes else corpus
        meta = dict(
            model="toy", seq_len=seq_len, seed=seed, bos=False, bos_id=None,
            real_tokens_per_window=seq_len, n_corpus=n, n_probes=n_probes,
            synthetic=True,
            corpus=[dict(row=i, doc=1000 + i, offset=0, pile_set=None) for i in range(n)],
            probes=[dict(row=n + k, pid=f"toy:{k // 2}:{(k % 2) * 8}", source="toy",
                         doc=k // 2, offset=(k % 2) * 8, kw=["synthetic"], pile_set=None)
                    for k in range(n_probes)],
        )
        return toks, meta


def toy(**kwargs) -> ToyAdapter:
    return ToyAdapter(**kwargs)
