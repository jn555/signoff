"""Adapter registry.

Adapters are the only artifact-specific code in the package, and they are both
product and reference implementation: reading one should tell you everything
about how that artifact is wired, including the parts that are easy to get
wrong.  See `base.py` for the seven required contract clauses.

Imports are LAZY.  Constructing an adapter pulls in transformer_lens and
touches the HF cache; the CPU-only test subset must be able to import this
module without either.
"""

from __future__ import annotations

from typing import Any

from .base import (
    Dictionary,
    DtypePolicy,
    Identity,
    ModelAdapter,
    TapSpec,
    TokenizationSpec,
    pick_device,
)

#: name -> (module, factory, one-line description, tier)
#: tier: "ci"    — small enough for a CPU-only runner
#:       "local" — needs real RAM / gated weights; slow tests are marked for it
_REGISTRY: dict[str, tuple[str, str, str, str]] = {
    "gpt2-dunefsky": (
        "gpt2_dunefsky", "gpt2_dunefsky",
        "GPT-2-small + Dunefsky/Chlenski MLP transcoders (12 layers, d_sae 24576, float32)",
        "ci",
    ),
    "gemma-scope-2b": (
        "gemma_scope", "gemma_scope_2b",
        "gemma-2-2b + gemma-scope width-16k JumpReLU transcoders (26 layers, float16)",
        "local",
    ),
    "qwen3-mwhanna": (
        "qwen_mwhanna", "qwen3_mwhanna",
        "Qwen3-0.6B + mwhanna low-L0 ReLU transcoders (28 layers, float16)",
        "local",
    ),
    "llama32-clt-mntss": (
        "llama_clt", "llama32_clt_mntss",
        "Llama-3.2-1B + mntss CROSS-LAYER transcoder, the circuit-tracer artifact "
        "(16 layers x 32768 features, JumpReLU, no skip, float32)",
        "local",
    ),
    "toy": (
        "toy", "toy",
        "SYNTHETIC fixture — a deterministic 4-layer toy artifact. Not a model; it "
        "exists so the pipeline, the gates and the report run with no weights.",
        "test",
    ),
}

__all__ = [
    "ModelAdapter", "Identity", "TapSpec", "TokenizationSpec", "DtypePolicy",
    "Dictionary", "pick_device", "get", "available", "describe",
    "gpt2_dunefsky", "gemma_scope_2b", "qwen3_mwhanna", "llama32_clt_mntss", "toy",
]


def available() -> list[str]:
    return sorted(_REGISTRY)


def describe(name: str | None = None) -> Any:
    """One-line descriptions, without importing anything heavy."""
    if name is None:
        return {k: dict(description=v[2], tier=v[3]) for k, v in sorted(_REGISTRY.items())}
    if name not in _REGISTRY:
        raise KeyError(_unknown(name))
    mod, factory, desc, tier = _REGISTRY[name]
    return dict(name=name, description=desc, tier=tier,
                factory=f"{__name__}.{mod}:{factory}")


def _unknown(name: str) -> str:
    return f"unknown adapter {name!r}; available: {', '.join(available())}"


def get(name: str, **kwargs) -> ModelAdapter:
    """Construct an adapter by registry name."""
    if name not in _REGISTRY:
        raise KeyError(_unknown(name))
    import importlib

    mod, factory, _, _ = _REGISTRY[name]
    return getattr(importlib.import_module(f"{__name__}.{mod}"), factory)(**kwargs)


def __getattr__(name: str):
    """`adapters.gpt2_dunefsky()` — the API-sketch spelling, resolved lazily."""
    import importlib

    for mod, factory, _, _ in _REGISTRY.values():
        if name == factory:
            return getattr(importlib.import_module(f"{__name__}.{mod}"), factory)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
