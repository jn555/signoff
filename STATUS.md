# STATUS — signoff v0.1 development

Build date 2026-08-31. Spec: `../drafts/tool-design.md`. Every module is an
**extraction** of code that already ran in `../experiments/01..03`; those
directories are immutable inputs and were not modified. Each module's docstring
names the source file(s) it came from.

Name: `signoff` (decided; the placeholder `miter` is gone). It lives in exactly
two places — `pyproject.toml:project.name` and `src/signoff/` — and nothing
inside the package hardcodes it (`signoff.TOOL_NAME` derives from `__name__`),
so a future rename is `git mv` plus two lines. License: MIT.

## Test results (honest)

**Trust model, in one line: the gates catch AUTHOR ERROR (mis-taps, dtype bugs,
registry drift, stale checkpoints), not tampering — measurement is delegated to
the adapter by design, so a report from a third-party adapter is that author's
claim mediated by this protocol, and should be read with the report's
"adapter-overridden measurement methods" line in hand.** See README "Trust
model".

```
python -m unittest discover -s tests
Ran 203 tests — OK (5 skipped)
```

- 3 skips are the gemma-scope / Qwen gate tests, which need 7.85 GB and 18.8 GB
  of local weights. They are skipped in the CPU suite but **have now been run**
  under `SIGNOFF_LOCAL=1` — see §4.
- 2 skips are the new Llama CLT gate tests, which are blocked on a gated model
  repo and 20.4 GB of dictionary — see §6. Its cross-layer forward and its
  cross-layer FVU ARE covered in the CPU suite, on the toy fixture.
- The suite runs in ~7 s on CPU with no network, including the golden regression.
- Executed with `../experiments/01-divergence-witnesses/.venv/bin/python`.
  That venv has no `pytest`, so tests are written on `unittest` and run under
  either runner; pytest markers (`slow`, `local_only`, `golden`) are applied
  when pytest is present.

**Golden regression — verified against real weights.** The committed 16-row
token set (`tests/golden/`) scored by the released code reproduces experiment
01's own per-row `d_mean` to **≤ 6.9e-5 nats** (tolerance declared at 5e-4; the
residual is CPU-vs-MPS float32 arithmetic, not a logic difference). Gate (i)
measures **exactly 0.0**. `flip` matches exactly.

**The corpus miner reproduces exp-01 bit-for-bit too.** An unrelated check
during the CLI acceptance run: `signoff audit --adapter gpt2-dunefsky --n 16`
mined a corpus **identical** (`(mined == committed).all()` is True) to the first
16 rows of `experiments/01-divergence-witnesses/corpus_tokens.pt`. So the
extraction reproduces exp-01's window SELECTION as well as its metrics — the
golden test would have caught a metric drift, and this catches a miner drift.

**CLI acceptance path — run and passing.** `signoff audit --adapter
gpt2-dunefsky --n 16 --search-seeds 1 --coverage` completes in ~30 s on CPU: all
gates pass, 16 rows scored, 4 search arms (constrained and unconstrained), a
gated markdown + JSON report with coverage cells. The full-scale
`--n 64`+ path is the same code with a larger `n`.

## Done, by priority

**1. Skeleton + Metrics + Stats — complete, tested.**
`metrics.py` (from `witness.py:metrics_from_logits`, `common03.py:per_position_kl`):
full-vocab float32 KL, vocab-aware chunking (verified numerically inert; for
GPT-2's vocabulary the chunk is exactly exp-01's 1024), position-0 exclusion
baked in. `stats.py`: the validity protocol as code. The four exp-02/03
behaviours the brief named are locked by dedicated tests —
pseudo-replication guard (three grouping levels; a single-text effect must
collapse from windowed to distinct_text), greedy nearest-NLL stratified pairing
(including a proof that the early break equals a full scan), NLL-residualised
tail (raw tail is the NLL readout, residual tail is the family, zero overlap),
and dual FVU (a synthetic case where the two estimators rank two sites
*oppositely* — the reason both always ship).

**2. Gates + refusing Runner — complete, tested.**
Eight gates as named `GateSpec` objects carrying id / description / **the
historical bug each caught** / diagnosis, with pure `check_*` verdict functions
(unit-testable with no weights). `GateReport.require()` raises `GateFailure`;
`Runner.report()` calls it first. Refusal is tested end-to-end: a mis-tapped
artifact writes **no** report file, the CLI exits **2**, and the quarantined
checkpoints (`gates.json`, `fvu.json`) survive for diagnosis. Waiving is
supported, explicit, and rendered in the report.

**2b. Trust-boundary work (post-critic).** A critic demonstrated three
gate-bypass paths. Two are the delegation property itself and cannot be closed;
the third was a bug and is closed.
- *Stamped, not closed:* an adapter that overrides a gate or the measurement
  under it self-reports. `adapters/base.py` now introspects the concrete class
  against the base class (`overridden_measurement_methods`, a module-level
  function so `contract()` cannot suppress it) and both report formats carry
  *adapter-overridden measurement methods: [...]* with a **[self-reported]**
  marker on each affected gate row. Empty for the reference adapters, except
  `gemma-scope-2b`'s declared `verify_provenance` override.
- *Closed:* **checkpoint binding**, the eighth gate. Gate verdicts are bound at
  measurement time to a hash over (adapter id, pinned revisions, dtype,
  replaced layer set, corpus size + signature, gate params) plus a digest of
  the verdicts. `report()` re-checks and refuses — standard refusal path, no
  file, exit 2 — on a mismatch, an unbound `gates.json`, or verdicts edited
  after the fact. The digest is **unkeyed by design** and the diagnosis says
  so: it catches a stale or hand-edited checkpoint, not an adversary.
- 22 tests in `tests/test_trust_boundary.py`, including the critic's own
  scratch-style bypass adapter (which reports, and is marked) and hand-edited /
  unbound / foreign-configuration `gates.json` (which are refused).

**3. GPT-2/Dunefsky adapter + golden regression — complete, verified.**
Full contract (all seven clauses). Gates (i) and (ii) pass on real weights;
per-row and subset-summary reproduction of exp-01 pinned in
`tests/test_golden_gpt2.py`. The unpickling stub for the release's checkpoints
is vendored into the package.

**4. gemma-scope and Qwen3 adapters — extracted, declaration-tested, and now
WEIGHT-VERIFIED (2026-08-31).** Both are complete extractions with pinned
revisions (recovered from the run cache), the gemma canonical-L0 freeze plus a
live re-derivation in `verify_provenance()`, the softcap-in-model-dtype head,
and the measured dtype table. CI tests assert the two declare **opposite** tap
and BOS conventions — the trap gate (ii) exists for.

Their `slow`/`local_only` real-weight gate tests have since been run under
`SIGNOFF_LOCAL=1` and passed:

- `gemma-scope-2b` — 2 tests, ~40 s: gates (i) and (ii) on a two-layer smoke
  against the real gemma-2-2b + gemma-scope weights, and `verify_provenance()`
  re-deriving the canonical-L0 selection from the **live** repository listing
  and matching the frozen table.
- `qwen3-mwhanna` — 1 test, ~86 s: gates (i) and (ii) on a two-layer smoke with
  real Qwen3-0.6B weights loaded.

They remain skipped in the CPU suite (7.85 GB / 18.8 GB of weights, gated Gemma
license), so the 3 skips above are not a coverage gap — they are a runner
constraint. No unverified adapter remains in the registry.

**5. Miners + Reporter + CLI — complete, tested end-to-end.**
Corpus sweep (`witness.build_corpus`, with BOS support), probe re-windowing with
the keyword detectors, and the λ-hinged greedy coordinate ascent with dual arms
and early stopping OFF by default. Reporter emits markdown + JSON from
checkpoints only, plus the (b,c,r,s,k) coverage emitter behind `--coverage`.
`signoff audit --adapter toy` runs the whole pipeline gated, on CPU, with no
network.

## 6. v0.1.1 — CLT adapter for the circuit-tracer stack. BUILT, weight-BLOCKED.

`adapters/llama_clt.py` (`llama32-clt-mntss`): **Llama-3.2-1B +
`mntss/clt-llama-3.2-1b-524k`**, the cross-layer transcoder the open
circuit-tracer stack ships. Full seven-clause contract; 29 tests in
`tests/test_llama_clt.py`. **The seam held — nothing in the runner, metrics,
gates or report changed.**

Investigated 2026-08-31 against the released source and the released checkpoint
headers, all cited in the adapter's provenance docstring. Findings that matter
beyond this adapter:

- **The repo MOVED.** `safety-research/circuit-tracer` now 301-redirects to
  **`decoderesearch/circuit-tracer`** (pinned at `6018ed8d`).
- **Both artifact families exist per model**, PLT *and* CLT, sharing the same
  hooks (`hook_resid_mid` -> `hook_mlp_out`). For Llama-3.2-1B the PLT
  (`mntss/transcoder-Llama-3.2-1B`) is ReLU **with** a skip connection
  (`W_skip` in its checkpoints); the CLT is JumpReLU with **no** skip.
- **A THIRD tap convention.** This artifact reads `hook_resid_mid` — the RAW,
  un-normalised residual stream. gemma-scope reads the PRE-gain normalised
  input and qwen3-mwhanna the POST-gain one. Three meanings, two hook names.
  A CI test asserts all three stay distinct.
- **Prefix-faithfulness.** `recon_W = b_dec_W + SUM_{L<=W} a_L W_dec_L[:,W-L,:]`,
  so a partial run reproduces the artifact **iff the layer set is a prefix
  `{0..k}`**. `run_tag` stamps `clt=prefix` or `clt=truncated`; a truncated set
  is allowed (localisation profiles need it) but never silently.
- **FVU is per WRITE layer, not per site.** A CLT gives a write layer one target
  and many read sites, so there is no per-`(read, write)` residual. Keys name
  the bundle (`"0..2->2"`); gate (ii) takes them unchanged.
- Four base measurement methods are overridden (`substitution_plan`,
  `measure_fvu`, `verify_provenance`, `run_tag`), so four gate rows print
  `[self-reported]`. Each override is argued in the docstring; `run_tag`'s
  buys two of those marks in exchange for not hiding the prefix/truncated
  distinction.

**Gates run so far.** `verify_provenance()` **PASSES against the live release**
(36 frozen entries: the `config.yaml` hook declaration plus the 32-file
inventory) — the CLT repo is ungated, so this is a real gate on the real
artifact, no weights. Gates (i)/(ii)/(iii) have **not** run: two blockers.

1. `meta-llama/Llama-3.2-1B` is **gated** (`gated: manual`); this host's HF
   token gets a `GatedRepoError 403` on even `config.json`. Access must be
   granted on HF before any model-touching gate can run.
2. The dictionary is **20.4 GB** (2.15 GB of encoders + 18.25 GB of decoders)
   against ~10 GB of free disk. The adapter is built for this: it slices
   individual decoder PLANES out of the safetensors files, so the smallest
   faithful cross-layer smoke — prefix `{0,1}` — is 0.67 GB resident in bf16
   (4.43 GB on disk) plus ~2.5 GB of model.

The cross-layer code is *not* untested for it: `CrossLayerPlan` and the
cross-layer `measure_fvu` run in the CPU suite against the toy fixture, where
a CLT with zeroed off-diagonal planes is checked to reduce **exactly** to
`PerLayerPlan`, a non-zero plane is checked to actually change the answer,
adding a read layer is checked not to move any write layer above it, and the
FVU the gate reports is checked to be the reconstruction the forward writes.

## Next milestone

Weight-verify `llama32-clt-mntss` — the only thing standing between it and a
full gate run is (a) HF access to `meta-llama/Llama-3.2-1B` and (b) ~7 GB of
free disk for the prefix-`{0,1}` smoke. Do those two and
`SIGNOFF_LOCAL=1 python -m unittest tests.test_llama_clt` runs gates (i) and
(ii) on the real artifact. A `mntss/transcoder-Llama-3.2-1B` PLT adapter (same
model, same hooks, ReLU **with** skip) would then share the model download and
give the CLT a per-layer control on the identical stack.

## Not built (deferred, with reasons)

- **`ErrorCalibrator` / null controls are implemented and unit-tested, but the
  Runner has no null-run stage.** `Replacement.null(...)` works and the forward
  is correct; wiring calibration-then-null into `audit()` (with the held-out
  calibration/eval disjointness assertion) is the missing piece.
- **`--waive GATE=reason` is not exposed on the CLI.** The API
  (`GateReport.waive`) exists and is tested; only the flag is missing.
- **Single-layer localisation profiles** are expressible (`Replacement(ad,
  layers=[L])`) but there is no stage that sweeps L and correlates FVU against
  behavioural damage (exp-02 study C / exp-03 study A).
- **`gate_fp32_replay` reloads the model on CPU** and is correct for GPT-2-scale
  artifacts; it has not been exercised at gemma scale, where the reference run
  needed the arms to be strictly non-co-resident and staged last.
- **No probe corpus ships.** Probe texts are referenced by (source, doc, offset)
  only; `mine_probes` reads a user-supplied local directory. This preserves the
  experiments' no-text-redistribution rule and leaves open decision #5 in the
  design doc genuinely open.
- **No GCG/gradient miner, no CEGAR, no attribution graphs, no SAE training, no
  UI, no hosted corpus, no multi-GPU** — all explicit v0.1 non-goals.
- **No CI workflow file.** The suite is CI-ready (CPU, ~7 s, no network) but no
  `.github/workflows` was added.

## Known rough edges

- `report.py` renders `independent_source` as `—` when probe ids do not use the
  `pile:` prefix (e.g. the toy fixture). Correct behaviour, but the prefix is a
  `family_test` argument that the Runner does not yet plumb from the adapter.
- `measure_fvu` buffers the clean sublayer outputs to compute exact global-mean
  denominators (following `run_b_gemma:stage_pass`). Memory is
  `fvu_n × T × d_model` per replaced layer — fine at smoke scale, and the reason
  `fvu_n` defaults to 16.
- `dual_fvu` applies `exclude_position_0` consistently to **both** estimators.
  The source computed the global estimator over all positions and the token
  median over positions ≥ 1; that asymmetry was a wart, and the deviation is
  documented in the function's docstring. Numbers can differ from
  `passBgb.json` in the last digits for this reason.
