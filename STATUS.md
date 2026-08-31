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
Ran 174 tests — OK (3 skipped)
```

- 3 skips are the gemma-scope / Qwen gate tests, which need 7.85 GB and 18.8 GB
  of local weights. They are skipped in the CPU suite but **have now been run**
  under `SIGNOFF_LOCAL=1` — see §4.
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

## Next milestone — v0.1.1 (promoted from v0.2 by demand review)

**CLT / attribution-graph adapter for the circuit-tracer stack** (the Gemma CLTs
used by the open circuit-tracer / Neuronpedia). This is the artifact generation
the growing user base actually produces, so it outranks everything else below.

**The seam is already in place** — this was designed in, not deferred:
- `SubstitutionPlan` owns the substituted forward; the adapter chooses it via
  `ModelAdapter.substitution_plan(spec)`. `PerLayerPlan` is v0.1's; a
  cross-layer plan needs no change to the runner, metrics, gates or report.
- A `Site` is `(read_layer, write_layer)`, not a layer index, and renders as
  `"3->7"`.
- Gate (ii) is keyed by **site**, not by layer, and accepts cross-layer ids.
- Tested in `tests/test_adapter_contract.py::TestCrossLayerSeam` with a real
  adapter subclass that overrides the plan.

Remaining work for that adapter is the artifact itself: CLT weight layout and
loading, the read-once/write-many forward, per-site FVU accumulation across
multiple write targets, and its own provenance freeze.

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
