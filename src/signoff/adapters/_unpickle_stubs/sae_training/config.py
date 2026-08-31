"""Unpickling stub for Dunefsky/Chlenski transcoder checkpoints.

Those .pt files were saved with `torch.save` of an object graph containing
`sae_training.config.LanguageModelSAERunnerConfig`, so unpickling needs a class
of that import path to exist.  Nothing about the class matters: the checkpoint's
attribute dict is restored onto whatever instance we provide, and the adapter
reads the attributes it needs (hook_point, d_in, d_sae, ...) off it.

Vendored from experiments/01-divergence-witnesses/_stub/sae_training/config.py.
"""


class LanguageModelSAERunnerConfig:
    def __init__(self, *a, **k):
        pass
