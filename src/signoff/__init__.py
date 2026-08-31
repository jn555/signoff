"""A faithfulness falsifier for interpretability artifacts.

A replacement model (SAE / transcoder substitution) is an *explanation object*
that claims behavioural equivalence to a base model.  This package imports the
hardware equivalence-checking move: build the miter (base and replacement on
identical inputs), hunt for divergence, and report counterexamples rather than
summary statistics.

It is a falsifier, not a prover.  There is no proof engine and no completeness:
absence of witnesses at a search budget is NOT equivalence.  The strongest
positive verdict this package can emit is "no witness found under the declared
budget".  It never emits "faithful".

Status: pre-release (v0.1 development).  See README.md and STATUS.md.
"""

from __future__ import annotations

# The tool's name is derived, never written down.  `pyproject.toml`'s
# `project.name` and this package's directory name are the only two places a
# rename has to touch.
TOOL_NAME = __name__.split(".")[0]

__version__ = "0.1.0.dev0"

from .gates import (  # noqa: E402
    GATE_SPECS,
    GateFailure,
    GateReport,
    GateResult,
    GateSpec,
)
from .metrics import SeqMetrics, metrics_from_logits, per_position_kl  # noqa: E402
from .replacement import Replacement, ReplacementSpec  # noqa: E402

__all__ = [
    "TOOL_NAME",
    "__version__",
    "GATE_SPECS",
    "GateFailure",
    "GateReport",
    "GateResult",
    "GateSpec",
    "SeqMetrics",
    "metrics_from_logits",
    "per_position_kl",
    "Replacement",
    "ReplacementSpec",
    "adapters",
    "miners",
    "report",
    "runner",
    "stats",
]


def __getattr__(name: str):
    """Lazy submodule access.

    `runner`/`report`/`miners`/`adapters` pull in heavier optional
    dependencies (datasets, transformer_lens); importing `miter` must stay
    cheap enough for the CPU-only test subset.
    """
    if name in ("adapters", "miners", "report", "runner", "stats"):
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
