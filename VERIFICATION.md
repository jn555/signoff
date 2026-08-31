# Weight-tier verification record

Primary record of the local (real-weight) gate-tier runs, witnessed by the
orchestrating session on 2026-08-31 (raw unittest output; host: M4 MacBook Air
16GB, mps, development venv, SIGNOFF_LOCAL=1, run sequentially — never co-resident):

- **gemma-2-2b + gemma-scope canonical 16k** (`TestGemmaGates`, 2 tests):
  `Ran 2 tests in 40.455s — OK` (includes gates i+ii two-layer smoke and
  `test_provenance_matches_the_live_listing`; log line
  "Loaded pretrained model google/gemma-2-2b into HookedTransformer").
- **Qwen3-0.6B + mwhanna lowl0** (`TestQwenGates`, 1 test):
  `Ran 1 test in 85.695s — OK` (log line
  "Loaded pretrained model Qwen/Qwen3-0.6B into HookedTransformer").
- **GPT-2 + Dunefsky**: weight-verified continuously in the CI tier (golden
  regression, ≤6.9e-5 nats vs committed exp-01 values).

STATUS.md §4 summarizes; this file is the underlying record.
