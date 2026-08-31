# signoff

**Equivalence-checking discipline for interpretability claims.**

A replacement model — an SAE or transcoder substituted into a base model — is an
*explanation object* that claims behavioural equivalence. Today that claim is
validated by averages (mean KL, FVU) and hand-picked case studies. `signoff`
imports the hardware equivalence-checking move instead: build the miter (base and
replacement on identical inputs), **hunt for divergence**, and report
counterexamples rather than summary statistics.

It is a **falsifier, not a prover**. There is no proof engine and no
completeness argument: absence of witnesses at a search budget is *not*
equivalence. The strongest positive verdict it can emit is *"no witness found at
strength L3 under the declared budget"*. **It never emits "faithful"** — despite
the name, which describes the discipline it borrows, not a certificate it issues.

`signoff` is the reference implementation of the witness-mining protocol
described in the accompanying paper — the runnable companion to it — and its
gates and metrics are designed to be lifted into existing evaluation stacks
(SAELens, SAEBench) rather than to replace them.

> **Status: pre-release (v0.1 development).** The API is not stable. See
> [`STATUS.md`](STATUS.md) for what is built, what is verified, and what is not.

## Install

```bash
pip install -e ".[adapters]"     # core is torch-only; [adapters] adds transformer_lens etc.
```

## Quickstart

```python
from signoff import adapters, Replacement
from signoff.runner import Runner

ad  = adapters.get("gpt2-dunefsky")          # pins revisions; declares taps, BOS, dtype
rep = Replacement(ad, layers="all")          # or layers=[0, 11]; or Replacement.null(ad, "shuffled")
run = Runner(ad, rep, out="runs/gpt2")

run.mine(n=1500, seq_len=64)                 # deterministic seeded corpus windows
run.gates()                                  # (i) bit-exact clean pass  (ii) FVU tap check
                                             # (iii) fp32 replay — raises GateFailure, and
                                             #       everything after it refuses to run
rows = run.score()                           # d_mean / d_max / flip / nll, resumable
run.search(n_tail=10, n_median=10, lam=2.0)  # constrained + unconstrained arms
run.family()                                 # NLL-matched pairs at 3 grouping levels,
                                             # stratified low-NLL, residualised tail
run.report(formats=("markdown", "json"))     # refuses if any gate is failed or unrun
```

Or from the shell:

```bash
signoff audit --adapter gpt2-dunefsky --n 64      # the whole pipeline, gated
signoff gates --adapter gpt2-dunefsky --explain   # every gate and the bug it caught
signoff report --run runs/gpt2 --coverage         # (b,c,r,s,k) coverage cells
signoff adapters                                  # what is registered
```

Exit code **2** means a gate was not cleared and **no report was written**.

## The gates

A run that fails a gate refuses to emit a report. Numbers computed above a
failed gate are quarantined in the run directory, never rendered. Each gate
below earned its place by catching a real bug in the experiments this tool was
extracted from.

| Gate | What it checks | The bug it caught |
|---|---|---|
| **(i) base-vs-base** | A clean layer-major pass equals the model's own forward — KL 0 to dtype tolerance, all finite | gemma-2's logit softcap applied after an fp32 upcast — "more accurate" and wrong. 2-ulp logit error, base-vs-base KL 3.3e-3, which would have biased *every* measured divergence |
| **(ii) FVU sanity** | Per-site fraction-of-variance-unexplained is in a sane range after substitution | The tap trap: gemma-scope and the mwhanna Qwen suite use the **same hook name with opposite pre/post-gain conventions**. A mis-tap reads FVU ≈ 1 — the dictionaries look worthless when the wiring is wrong |
| **(iii) fp32 replay** | CPU float32 replay of n sequences agrees within 1e-2 nats, on **both** streams | bfloat16 injected ~4× the tolerance into every KL on gemma-2 (replaced KL max 3.09e-1 vs float16's 5.85e-4). The base stream carries no dictionaries, so its error is pure dtype — that column is the tell |
| **(iii′) paired-bound fallback** | When (iii) fails on absolutes: the fp16-vs-fp32 discrepancy of the *paired* difference, with a bootstrap CI | The gemma 26-layer run fails (iii) absolutely. Its family verdict survived only because the paired bound (0.037, CI [0.014, 0.061]) was **measured**, not assumed |
| **identity guard** | Every cached row carries a run tag and its own (doc, offset) | An fp16 run nearly reported bfloat16 divergences from a stale cache; a smoke corpus nearly resumed into a full run |
| **provenance freeze** | The live repo listing still selects the same dictionary variant per layer | The SAELens registry pointed at degenerate L0 = 5–15 gemma variants for **12 of 26 layers** |
| **BOS declaration** | The adapter declares a BOS convention and the miner enforces it | BOS-free eval of the BOS-trained gemma-scope suite moved corpus NLL by 3.8 nats and d_mean by 12%, and ~doubled gate-(iii) drift |
| **checkpoint binding** | The verdicts on hand are bound to the configuration that measured them, and still digest to what that run recorded | `report()` trusted whatever `gates.json` sat in the run directory. A checkpoint left by an earlier configuration, or one edited by hand from `fail` to `pass`, produced a clean report — the refusal was one text edit from decorative |

Suite health (tail ratio p99/median: 2.27 healthy GPT-2, 1.93 gemma-scope, 1.66
a saturated Qwen suite) is a **reported diagnostic, not a gate** — three points
is not a criterion, and the report says so.

## Trust model

**The gates catch author error. They are not tamper-proof.**

Measurement is delegated to the adapter *by design*. The adapter is the only
thing that knows how to run a particular artifact, so the gates in `gates.py`
are pure verdict functions over numbers that adapter code produced. That is what
makes the tool portable to a release this package has never seen — and it means
an adapter that overrides `gate_fvu` to return PASS gets a PASS. Two of the
three bypass paths a critic demonstrated against v0.1 are *this property*, not
bugs, and they cannot be closed without giving up adapters.

So what the gates actually buy you is protection against **the mistakes that
have really happened here**: a mis-tap that makes FVU ≈ 1, a dtype that injects
4× the tolerance into every KL, a registry that quietly points at degenerate
dictionaries, a stale cache from another run, a `gates.json` left behind by a
different configuration. Every gate in the table above earned its place by
catching one of those. None of them was an attack.

What the tool does instead of pretending:

- **It stamps the delegation.** Every report — markdown and JSON — carries
  *adapter-overridden measurement methods: [...]*, computed by introspecting the
  concrete adapter class against the base class. Any gate row whose measurement
  the adapter took over is rendered **[self-reported]**. The stamp is computed
  from the class by a module-level function, not read off the adapter, so an
  adapter cannot suppress it by overriding `contract()`. It is empty for the
  reference adapters, except `gemma-scope-2b`, which overrides
  `verify_provenance` to re-derive the canonical-L0 selection from the live
  listing — a declared extension point, and still a self-report, so it is
  listed.
- **It binds gate verdicts to the configuration that measured them.** A
  `gates.json` is hashed against the adapter identity, the pinned revisions, the
  dtype, the replaced layer set, the corpus size and signature, and the gate
  parameters, plus a digest of the verdicts. Emission re-checks it and refuses
  on a mismatch, an unbound file, or verdicts edited after the fact. This one
  *was* a real bug and is closed. The digest is **unkeyed**: it stops a stale or
  hand-edited checkpoint, not an adversary, who can recompute it.

**How to read a report you did not produce.** A report from a third-party
adapter is a claim by that adapter's author, mediated by this tool's protocol —
not an independent measurement. Check the *adapter-overridden measurement
methods* line first; if it is non-empty, the marked rows are the author's
numbers. The parts you can check yourself are the ones this tool does not
delegate: the pinned revisions, the declared taps and BOS convention, the
corpus signature, and the fact that the run refused or did not. Reproducing a
report means running the adapter yourself, which is why adapters are one file.

A signed-attestation or sandboxed-measurement design would change this. Neither
is in v0.1, and the tool does not imply either.

## What you get

- **Gate results, or a refused run.**
- **The divergence distribution** — d_mean / d_max / flip / nll with quantiles
  and tail ratios, not just a mean.
- **A witness corpus** — the top-K tail by (doc, offset) and family label.
  No text bodies are stored, logged or redistributed, anywhere.
- **Stratified family statistics** — NLL-matched paired tests at three grouping
  levels reported *together* (reporting the windowed level alone once overstated
  an effect by ~3×), plus a stratified low-NLL repeat and an NLL-residualised
  tail.
- **Optionally, a coverage report** over (b, c, r, s, k) cells with verdicts
  {validated, invalidated, open, waived}.
- **A trust stamp** — which base-class measurement methods this adapter
  overrode, and a **[self-reported]** marker on every gate row it measured
  itself. See [Trust model](#trust-model) for how to read one.

## Adapters

| Name | Artifact | Tier | Gates run against real weights |
|---|---|---|---|
| `gpt2-dunefsky` | GPT-2-small + Dunefsky/Chlenski MLP transcoders (12 layers, float32) | CPU CI | yes — plus a golden regression against experiment 01 |
| `gemma-scope-2b` | gemma-2-2b + gemma-scope width-16k JumpReLU (26 layers, float16) | local | yes — verified 2026-08-31 (gates (i)+(ii), live-listing provenance match) |
| `qwen3-mwhanna` | Qwen3-0.6B + mwhanna low-L0 ReLU transcoders (28 layers, float16) | local | yes — verified 2026-08-31 (gates (i)+(ii) on real Qwen3-0.6B weights) |
| `toy` | A synthetic fixture. **Not a model** — it exists so the pipeline runs with no weights | test | n/a — synthetic |

The two `local` adapters need real weights (7.85 GB of gemma-scope transcoders,
18.8 GB of Qwen ones, plus a gated Gemma license), so their gate tests are
skipped in CPU CI and run under `SIGNOFF_LOCAL=1`. See [`STATUS.md`](STATUS.md)
§4 for what those runs measured.

Adapters are both product and reference implementation: everything
artifact-specific lives in one file, including the parts that are easy to get
wrong. See `src/signoff/adapters/base.py` for the seven required contract
clauses — and its trust-boundary section — and `gemma_scope.py` for what an
adapter docstring owes its reader.

## Tests

```bash
python -m unittest discover -s tests     # CPU-only; ~7 s, no network
SIGNOFF_LOCAL=1 python -m unittest discover -s tests   # + the local-weight tiers
```

The gates double as the test suite backbone, the refusal paths are tested as
first-class behaviour, and a **golden-number regression** scores a committed,
fixed token set and requires it to reproduce experiment 01's own per-row numbers
within a declared tolerance. `tests/test_trust_boundary.py` holds the adversarial
side: a bypass adapter that overrides a gate to return PASS must still produce a
report marked **[self-reported]** on that row, and a hand-edited, unbound or
foreign-configuration `gates.json` must be refused.

## License

MIT. See [`LICENSE`](LICENSE).
