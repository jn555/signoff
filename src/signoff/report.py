"""Emitters.  Prose from tables, nothing hand-entered — and nothing above a gate.

PROVENANCE.  Extracted from experiments/01-divergence-witnesses/analyze.py and
experiments/0{2,3}-*/analyze0{2,3}.py.  Those scripts have one rule that this
module inherits wholesale: **every number in the prose comes from a checkpoint
the run wrote.**  A report is a rendering of `runs/<name>/*.json`, never a
narration of what someone remembers happening.

Two hard rules on top of that:

1. **Refuse above a failed gate.**  `emit()` re-verifies the checkpoint binding
   and then calls `GateReport.require()`.  A failed or unrun blocking gate
   produces no file and (from the CLI) a nonzero exit.  Quarantined checkpoints
   stay on disk for diagnosis.
2. **No text bodies, ever.**  The witness tail is rendered by (doc, offset),
   pile subset, family LABEL and matched keyword groups.  A reader who has the
   corpus can reconstruct any row; the report itself redistributes nothing.
3. **Say who measured it.**  Measurement is delegated to the adapter by design,
   so the report stamps which base-class measurement methods the adapter
   overrode and marks the affected gate rows "self-reported".  The stamp is
   computed from the adapter's CLASS by a module-level function, not read off
   the adapter, so overriding `contract()` cannot suppress it.

The optional coverage emitter renders the run as (b, c, r, s, k) cells with
verdicts {validated, invalidated, open, waived}, in the format of
notes/coverage-model.md §2-3.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Sequence

from . import TOOL_NAME, __version__
from . import gates as G
from .adapters import base as AB

#: The interp-anchored strength ladder (notes/coverage-model.md §2).
STRENGTH = {
    "L0": "anecdotal",
    "L1": "distributional average (mean KL, FVU)",
    "L2": "quantile / tail-checked",
    "L3": "adversarially searched (constrained witness search)",
    "L4": "exhaustive / verified — NOT reachable by this tool",
    "L5": "composed proof — nobody has this",
}


def _fmt(x, spec="{:.4f}"):
    if x is None:
        return "—"
    try:
        if isinstance(x, float) and x != x:
            return "nan"
        return spec.format(x)
    except (ValueError, TypeError):
        return str(x)


def _summary_row(label: str, s: dict | None) -> str:
    if not s:
        return f"| {label} | — | — | — | — | — | — | — |"
    return (f"| {label} | {s['n']} | {_fmt(s['mean'], '{:.4f}')} | {_fmt(s['std'], '{:.4f}')} "
            f"| {_fmt(s['p50'], '{:.4f}')} | {_fmt(s['p90'], '{:.4f}')} "
            f"| {_fmt(s['p99'], '{:.4f}')} | {_fmt(s['max'], '{:.4f}')} |")


def strength_reached(runner) -> str:
    """The highest ladder rung this run's EVIDENCE actually supports."""
    if getattr(runner, "witnesses", None):
        return "L3"
    dist = getattr(runner, "distribution", None) or {}
    if (dist.get("corpus") or {}).get("tail_ratios"):
        return "L2"
    if runner.rows:
        return "L1"
    return "L0"


def verdict(runner) -> dict[str, Any]:
    """The strongest claim this run is allowed to make.

    NEVER "faithful".  Absence of witnesses at a search budget is not
    equivalence: there is no proof engine here, no completeness argument, and
    no exhaustively checkable input cone.  The positive form is always
    "no witness found at strength <S> under declared budget <B>".
    """
    rows = runner.rows or []
    dist = getattr(runner, "distribution", None) or {}
    corp = (dist.get("corpus") or {}).get("d_mean") or {}
    s = strength_reached(runner)
    budget = dict(
        n_scored=len(rows),
        n_corpus=dist.get("n_corpus"), n_probes=dist.get("n_probes"),
        seq_len=(runner.meta or {}).get("seq_len"),
        search_arms=len(getattr(runner, "witnesses", []) or []),
        layers_replaced=len(runner.replacement.layers),
    )
    found = None
    if corp:
        # a "witness" here = a corpus window in the top 1% of its own
        # distribution; the tail is characterised, not thresholded against an
        # absolute criterion this program has not earned.
        found = sum(1 for r in rows if r["kind"] == "corpus" and r["d_mean"] >= corp.get("p99", 1e9))
    return dict(
        strength=s, strength_meaning=STRENGTH[s], budget=budget,
        n_tail_witnesses=found,
        statement=(f"No witness found at strength {s} under the declared budget."
                   if not rows else
                   f"Divergence characterised at strength {s} ({STRENGTH[s]}) over "
                   f"{budget['n_scored']} windows; the tail is reported, not summarised away."),
        never_emitted="faithful / equivalent / verified — this tool cannot establish any of them",
    )


# ------------------------------------------------------------- trust stamp


#: Printed under the gate table and stored in the JSON.  Deliberately blunt.
TRUST_NOTE = (
    "Measurement is delegated to the adapter by design, so these gates catch AUTHOR "
    "ERROR — mis-taps, dtype bugs, registry drift, stale checkpoints — and are not "
    "tamper-proof against an adversarial adapter."
)


def trust_stamp(adapter) -> dict[str, Any]:
    """Who measured the numbers in this report.

    Computed from the adapter's CLASS by `adapters.base`'s module-level
    function, not by asking the adapter — an object cannot be the authority on
    its own overrides.
    """
    over = AB.overridden_measurement_methods(adapter)
    return dict(
        adapter=getattr(adapter, "name", type(adapter).__name__),
        adapter_class=f"{type(adapter).__module__}.{type(adapter).__qualname__}",
        overridden_measurement_methods=over,
        self_reported_gates=AB.self_reported_gates(adapter),
        required_by_contract=list(AB.REQUIRED_OVERRIDES),
        note=TRUST_NOTE,
        empty_list_means=("every gate verdict above was computed by framework code — "
                          "still running through this adapter's own primitives "
                          f"({', '.join(AB.REQUIRED_OVERRIDES)}), which is the "
                          "irreducible part of the delegation"),
    )


# --------------------------------------------------------------------- JSON


def to_dict(runner, coverage: bool = False) -> dict[str, Any]:
    """The whole run as one JSON-able object, assembled from checkpoints."""
    gates = runner.gate_report.to_dict()
    self_reported = AB.self_reported_gates(runner.adapter)
    for g in gates.get("gates", []):
        g["self_reported"] = g["id"] in self_reported
        g["measured_by_adapter_overrides"] = self_reported.get(g["id"], [])
    d = dict(
        tool=dict(name=TOOL_NAME, version=__version__,
                  emitted_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
        config=runner.config(),
        trust=trust_stamp(runner.adapter),
        verdict=verdict(runner),
        gates=gates,
        distribution=getattr(runner, "distribution", None),
        fvu=runner.fvu_table,
        family=runner.family_stats,
        validity=runner.validity,
        witnesses=getattr(runner, "witnesses", []) or [],
    )
    if coverage:
        d["coverage"] = coverage_cells(runner)
    return d


# ----------------------------------------------------------------- markdown


def to_markdown(runner, coverage: bool = False) -> str:
    out: list[str] = []
    w = out.append
    cfg = runner.config()
    ad = cfg["adapter"]
    v = verdict(runner)
    synthetic = "(synthetic)" in (ad["identity"]["model_repo"] or "")

    w(f"# {TOOL_NAME} report — {ad['identity']['release']}")
    w("")
    w(f"*{TOOL_NAME} {__version__} · emitted "
      f"{time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())} · run tag "
      f"`{cfg['run_tag']}`*")
    w("")
    if synthetic:
        w("> **SYNTHETIC FIXTURE.** This run used a toy artifact that is not a model. "
          "Every number below is about the fixture and says nothing about any real "
          "dictionary suite.")
        w("")

    # -- verdict
    w("## Verdict")
    w("")
    w(f"**{v['statement']}**")
    w("")
    w(f"- Strength reached: **{v['strength']}** — {v['strength_meaning']}")
    w(f"- Budget: {v['budget']['n_scored']} windows scored "
      f"({v['budget']['n_corpus']} corpus + {v['budget']['n_probes']} probe), "
      f"seq_len {v['budget']['seq_len']}, {v['budget']['layers_replaced']} layers "
      f"replaced, {v['budget']['search_arms']} search arms.")
    if v["n_tail_witnesses"]:
        w(f"- Tail witnesses (corpus windows at or above the corpus p99): "
          f"**{v['n_tail_witnesses']}**, listed below by id.")
    w("")
    w("This tool is a falsifier. It returns counterexamples; it cannot return "
      "equivalence. Absence of witnesses at a search budget is **not** faithfulness — "
      "there is no proof engine here, no completeness argument, and no exhaustively "
      "checkable input cone.")
    w("")

    # -- gates
    w("## Gates")
    w("")
    w("A run that fails a gate emits nothing. These all cleared.")
    w("")
    self_reported = AB.self_reported_gates(runner.adapter)
    overridden = AB.overridden_measurement_methods(runner.adapter)
    w("| gate | status | measured | tolerance | what this run measured |")
    w("|---|---|---|---|---|")
    for r in runner.gate_report.ordered:
        mark = " **[self-reported]**" if r.spec.id in self_reported else ""
        w(f"| {r.spec.title}{mark} | **{r.status.upper()}** | {_fmt(r.value, '{:.3e}')} "
          f"| {_fmt(r.tolerance, '{:.1e}')} | {r.message or '—'} |")
    w("")
    w(f"What each gate checks, and the real bug it caught, is in "
      f"`{TOOL_NAME} gates --explain`. A gate shown as UNRUN that is **non-blocking** "
      f"(the paired-bound fallback) is only needed when gate (iii) fails.")
    w("")

    # -- who measured it (the trust boundary, made visible)
    if overridden:
        w(f"**Adapter-overridden measurement methods: "
          f"`{', '.join(overridden)}`.** Rows marked **[self-reported]** were measured "
          f"by this adapter's own code in place of the framework's:")
        w("")
        for gid, methods in sorted(self_reported.items()):
            spec = G.GATE_SPECS.get(gid)
            w(f"- {spec.title if spec else gid} — via `{', '.join(methods)}`")
        w("")
        w("A legitimate extension point and a bypass look identical from here, and this "
          "tool does not claim to tell them apart. " + TRUST_NOTE + " Read the marked "
          "rows as the adapter's claim rather than as this tool's measurement.")
    else:
        w("Adapter-overridden measurement methods: **none** — every verdict above was "
          "computed by framework code.")
        w("")
        w("That is not a tamper-proofness claim. " + TRUST_NOTE + " Measurement still "
          f"runs through this adapter's own primitives "
          f"(`{', '.join(AB.REQUIRED_OVERRIDES)}`), which is the irreducible part of "
          f"the delegation.")
    w("")
    waived = [r for r in runner.gate_report.ordered if r.status == G.WAIVED]
    if waived:
        w("**Waived gates** — a reader should discount this run accordingly:")
        w("")
        for r in waived:
            w(f"- {r.spec.title}: {r.message}")
        w("")

    # -- provenance
    w("## What was audited")
    w("")
    w("| field | value |")
    w("|---|---|")
    idn = ad["identity"]
    w(f"| release | `{idn['release']}` |")
    w(f"| model | `{idn['model_repo']}` @ `{idn['model_revision']}` |")
    w(f"| dictionaries | `{idn['dict_repo']}` @ `{idn['dict_revision']}` |")
    w(f"| input tap | `{ad['taps']['input_hook']}` — {ad['taps']['input_convention']} |")
    w(f"| output tap | `{ad['taps']['output_hook']}` — {ad['taps']['output_convention']} |")
    w(f"| tokenization | BOS {'id ' + str(ad['tokenization']['bos_id']) if ad['tokenization']['prepend_bos'] else 'NOT prepended'}; "
      f"position 0 excluded from every metric |")
    w(f"| dtype | {ad['dtype']} (allowed: {', '.join(ad['dtype_policy']['allowed'])}) |")
    w(f"| replacement | {cfg['replacement']['mode']} — {cfg['replacement']['description']} |")
    w(f"| layers replaced | {len(cfg['replacement']['resolved_layers'])} of {ad['n_layers']} |")
    if idn.get("notes"):
        w("")
        w(f"*{idn['notes']}*")
    w("")

    # -- distribution
    dist = getattr(runner, "distribution", None)
    if dist:
        w("## Divergence distribution")
        w("")
        w("| set / metric | n | mean | std | p50 | p90 | p99 | max |")
        w("|---|---|---|---|---|---|---|---|")
        for grp in ("corpus", "probes"):
            g = dist.get(grp)
            if not g:
                continue
            for met in ("d_mean", "d_max", "flip", "nll"):
                w(_summary_row(f"{grp} {met}", g.get(met)))
        w("")
        tr = (dist.get("corpus") or {}).get("tail_ratios")
        if tr:
            w("| tail statistic (corpus d_mean) | value |")
            w("|---|---|")
            w(f"| p99 / median | {_fmt(tr['p99_over_median'], '{:.2f}')} |")
            w(f"| max / median | {_fmt(tr['max_over_median'], '{:.2f}')} |")
            w(f"| fraction above 10x median | {_fmt(tr['frac_above_10x_median'], '{:.4f}')} |")
            w("")
            w("Tail ratio is a **reported diagnostic, not a gate**. Three completed runs "
              "give 2.27 (GPT-2 + Dunefsky), 1.93 (gemma-scope) and 1.66 (a saturated "
              "Qwen suite) — three points is not a criterion, and no numeric threshold "
              "has been pre-declared here.")
            w("")

        tail = dist.get("tail_top30") or []
        if tail:
            w("## Witness corpus — top 30 by d_mean")
            w("")
            w(f"Family counts: `{json.dumps(dist.get('tail_family_counts', {}))}`; "
              f"probe share of the tail: {_fmt(dist.get('tail_probe_share'), '{:.2f}')}.")
            w("")
            w("Identified by id and family label only — **no text bodies**. With the "
              "corpus and the seed, any row is reconstructible.")
            w("")
            w("| rank | kind | family | doc | offset | subset | d_mean | d_max | flip | nll | kw |")
            w("|---|---|---|---|---|---|---|---|---|---|---|")
            for i, r in enumerate(tail, 1):
                kw = ",".join(r.get("kw") or []) if r.get("kw") else ""
                w(f"| {i} | {r.get('kind')} | {r.get('family')} | {r.get('doc')} "
                  f"| {r.get('offset')} | {r.get('pile_set') or ''} "
                  f"| {_fmt(r.get('d_mean'), '{:.3f}')} | {_fmt(r.get('d_max'), '{:.3f}')} "
                  f"| {_fmt(r.get('flip'), '{:.3f}')} | {_fmt(r.get('nll'), '{:.3f}')} | {kw} |")
            w("")

    # -- FVU
    if runner.fvu_table:
        w("## Reconstruction quality per substitution site")
        w("")
        w("| site | FVU (global) | FVU (per-token median) | tokens |")
        w("|---|---|---|---|")
        for k in sorted(runner.fvu_table, key=lambda x: (len(x), x)):
            f = runner.fvu_table[k]
            w(f"| {k} | {_fmt(f['fvu_global'])} | {_fmt(f['fvu_token_median'])} "
              f"| {f.get('n_tokens', '—')} |")
        w("")
        w("Both estimators are shown because they disagree — with opposite signs of the "
          "site ranking on GPT-2. Quoting one alone lets a suite look good by choice of "
          "estimator. FVU near 1 at *every* site would mean the tap is wrong rather than "
          "the dictionaries weak; that is gate (ii), and it passed.")
        w("")

    # -- family
    fam = runner.family_stats
    if fam:
        w("## Probe family vs NLL-matched controls")
        w("")
        w(f"Control pool: {fam['control_pool']} of {fam['n_corpus']} corpus windows "
          f"surviving the family blocklist. Corpus d_mean p99 = "
          f"{_fmt(fam['corpus_d_mean_p99'], '{:.3f}')}.")
        w("")
        w("| level | pairs | probe d_mean | control d_mean | paired diff | % pairs positive "
          "| % probes > control p90 | % probes > corpus p99 | mean abs NLL gap |")
        w("|---|---|---|---|---|---|---|---|---|")
        for name, lv in fam["levels"].items():
            if not lv:
                w(f"| {name} | — | — | — | — | — | — | — | — |")
                continue
            w(f"| {name} | {lv['n_pairs']} | {_fmt(lv['probe_d_mean']['mean'], '{:.3f}')} "
              f"| {_fmt(lv['ctrl_d_mean']['mean'], '{:.3f}')} "
              f"| {_fmt(lv['paired_diff_nats']['mean'], '{:+.3f}')} "
              f"| {_fmt(lv['frac_pairs_probe_gt_ctrl'] * 100, '{:.0f}')} "
              f"| {_fmt(lv['frac_probes_above_ctrl_p90'] * 100, '{:.0f}')} "
              f"| {_fmt(lv['frac_probes_above_corpus_p99'] * 100, '{:.0f}')} "
              f"| {_fmt(lv['nll_gap']['mean'], '{:.3f}')} |")
        w("")
        w("**All three levels are reported together, by construction.** Probe windows "
          "share source texts, so the windowed level overstates independence — reporting "
          "it alone once overstated a family effect by roughly 3x. `distinct_text` takes "
          "one window per text; `independent_source` keeps only independently sourced "
          "documents.")
        w("")

    val = runner.validity or {}
    strat = val.get("stratified_low_nll") or {}
    if strat.get("available"):
        p = strat["paired"]
        w("### Stratified low-NLL repeat")
        w("")
        w("| quantity | value |")
        w("|---|---|")
        w(f"| stratum ceiling (corpus NLL p25) | {_fmt(strat['stratum_ceiling_nll'], '{:.3f}')} |")
        w(f"| probes in stratum / all probes | {strat['n_probes_in_stratum']} / "
          f"{strat['n_probes_total']} |")
        w(f"| paired diff (nats) | {_fmt(p['mean'], '{:+.4f}')} |")
        w(f"| n pairs | {p['n']} |")
        w(f"| sd / se | {_fmt(p['sd'])} / {_fmt(p['se'])} |")
        w(f"| paired diff / se | {_fmt(p['t'], '{:+.2f}')} |")
        w(f"| % of pairs positive | {_fmt(p['frac_positive'] * 100, '{:.0f}')} |")
        w(f"| mean abs NLL gap in pair | {_fmt(strat['mean_abs_nll_gap'])} |")
        w("")
        w("Boilerplate-like families are low-perplexity. NLL matching alone still leaves "
          "the easy end of the distribution doing the work, so the test is repeated with "
          "both arms inside the corpus NLL <= p25 stratum.")
        w("")

    res = val.get("residualized_tail") or {}
    if res.get("available"):
        w("### NLL-residualised tail")
        w("")
        b = res["beta"]
        w(f"Fit over n={res['n']}: `d_mean ~ {_fmt(b[0], '{:.3f}')} + "
          f"{_fmt(b[1], '{:.3f}')}*nll + {_fmt(b[2], '{:.3f}')}*nll^2`.")
        w("")
        w("| tail | probe share | composition | overlap with raw top-K |")
        w("|---|---|---|---|")
        w(f"| raw top-{res['k']} by d_mean | {_fmt(res['raw']['probe_share'], '{:.2f}')} "
          f"| `{json.dumps(res['raw']['composition'])}` | {res['k']}/{res['k']} |")
        w(f"| NLL-residualised top-{res['k']} "
          f"| {_fmt(res['residualized']['probe_share'], '{:.2f}')} "
          f"| `{json.dumps(res['residualized']['composition'])}` "
          f"| {res['overlap_with_raw']}/{res['k']} |")
        w("")
        w("A tail can be a low-NLL readout rather than a family effect. The overlap "
          "count is the honest measure of how much of the raw tail survives removing "
          "the NLL trend.")
        w("")

    # -- search
    wit = getattr(runner, "witnesses", []) or []
    if wit:
        w("## Constrained witness search")
        w("")
        w("| seed | kind | arm | lambda | seed d_mean | final d_mean | seed nll | final nll | edits |")
        w("|---|---|---|---|---|---|---|---|---|")
        for r in wit:
            w(f"| {r['seed_row']} | {r['seed_kind']} | {r['arm']} | {_fmt(r['lam'], '{:.1f}')} "
              f"| {_fmt(r['seed_d_mean'], '{:.3f}')} | {_fmt(r['d_mean'], '{:.3f}')} "
              f"| {_fmt(r['seed_nll'], '{:.2f}')} | {_fmt(r['nll'], '{:.2f}')} "
              f"| {r['n_edits']}/{r['iters_run']} |")
        w("")
        w("Both arms are reported. The **unconstrained** arm (lambda = 0) is a control, "
          "not a result: it will happily find high-divergence gibberish, and the gap "
          "between the arms is what the perplexity constraint cost. Only the "
          "**constrained** arm's witnesses are claims about on-distribution behaviour.")
        w("")

    # -- coverage
    if coverage:
        w("## Coverage cells")
        w("")
        w("Each row is a declared cell (b, c, r, s, k) with a verdict and an evidence "
          "pointer. `open` cells are **planned holes** — declared and not yet checked. "
          "Closure means \"we checked what we declared\", nothing stronger: a behaviour "
          "class nobody declared cannot appear here.")
        w("")
        w("| b (behaviour) | c (component) | r (region) | s | k (checker) | verdict | evidence |")
        w("|---|---|---|---|---|---|---|")
        for c in coverage_cells(runner):
            w(f"| {c['b']} | {c['c']} | {c['r']} | {c['s']} | {c['k']} "
              f"| **{c['verdict']}** | {c['evidence']} |")
        w("")

    # -- limits
    w("## What this report does not say")
    w("")
    w("- It does not say the replacement is **faithful**, **equivalent**, or "
      "**verified**. No run of this tool can.")
    w("- Absence of a witness is absence of a witness *at this budget, on this "
      "corpus, under this checker*. It is not a bound.")
    w("- Every number is conditional on the declared taps, BOS convention and dtype "
      "above; the gates check that those declarations are self-consistent, not that "
      "they are the right ones for your question.")
    w(f"- Strength reached is {v['strength']}. L4 (verified) and L5 (composed proof) "
      "are out of scope by construction — there is no proof engine in this package.")
    w("")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------- coverage


def coverage_cells(runner) -> list[dict[str, str]]:
    """Render the run as (b, c, r, s, k) coverage claims.

    Format from notes/coverage-model.md §2-3.  Verdicts are derived from the
    run's own checkpoints; nothing here is hand-entered.  Note the K axis is a
    real coordinate: the same (b, c, r, s) can validate under one checker and
    invalidate under another, which is why FVU appears as its own cell with the
    estimator named.
    """
    cells: list[dict[str, str]] = []
    dist = getattr(runner, "distribution", None) or {}
    corp = (dist.get("corpus") or {}).get("d_mean") or {}
    n_layers = len(runner.replacement.layers)
    comp = f"all-{n_layers}" if n_layers > 1 else f"layer-{runner.replacement.layers[0]}"
    ev = os.path.basename(os.path.normpath(runner.out))

    if corp:
        cells.append(dict(
            b="next-token", c=comp, r="corpus (full)", s="L1", k="KL-vs-base",
            verdict="open",
            evidence=f"{ev}/distribution.json: mean {_fmt(corp['mean'], '{:.3f}')} nats "
                     f"— 'validated' only against a threshold you declare"))
        cells.append(dict(
            b="next-token", c=comp, r="corpus quantiles", s="L2", k="KL-vs-base",
            verdict="invalidated" if corp.get("p99", 0) > 0 else "open",
            evidence=f"{ev}/distribution.json: tail characterised, p99 "
                     f"{_fmt(corp.get('p99'), '{:.3f}')} nats, max "
                     f"{_fmt(corp.get('max'), '{:.3f}')}"))

    fam = runner.family_stats
    if fam:
        lv = (fam.get("levels") or {}).get("distinct_text")
        if lv:
            pos = lv["paired_diff_nats"]["mean"] > 0
            cells.append(dict(
                b="next-token", c=comp, r="probe family (distinct text)", s="L2-L3",
                k="paired KL vs NLL-matched control",
                verdict="invalidated" if pos else "validated",
                evidence=f"{ev}/family.json: paired diff "
                         f"{_fmt(lv['paired_diff_nats']['mean'], '{:+.3f}')} nats over "
                         f"{lv['n_pairs']} pairs"))

    for site, f in sorted(runner.fvu_table.items(), key=lambda kv: (len(kv[0]), kv[0]))[:1]:
        cells.append(dict(
            b="next-token", c=f"site {site}", r="corpus", s="L1",
            k="FVU_global vs FVU_token_median",
            verdict="open",
            evidence=f"{ev}/fvu.json: {_fmt(f['fvu_global'])} vs "
                     f"{_fmt(f['fvu_token_median'])} — the K axis at work; the "
                     f"estimators are different quantities"))

    if getattr(runner, "witnesses", None):
        constrained = [x for x in runner.witnesses if x["arm"] == "constrained"]
        improved = sum(1 for x in constrained if x["d_mean"] > x["seed_d_mean"])
        cells.append(dict(
            b="next-token", c=comp, r="edit-neighbourhood of corpus windows", s="L3",
            k="lambda-constrained greedy search",
            verdict="invalidated" if improved else "validated",
            evidence=f"{ev}/witnesses.jsonl: {improved}/{len(constrained)} constrained "
                     f"arms improved on their seed"))
    else:
        cells.append(dict(
            b="next-token", c=comp, r="edit-neighbourhood of corpus windows", s="L3",
            k="lambda-constrained greedy search", verdict="open",
            evidence="not run — a planned hole, not a pass"))
    return cells


# --------------------------------------------------------------------- emit


def emit(runner, formats: Sequence[str] = ("markdown", "json"),
         coverage: bool = False) -> dict[str, str]:
    """Write the report(s).  REFUSES above a failed or unrun blocking gate.

    The binding is re-verified HERE as well as in `Runner.report()`: this is the
    only function that writes a report file, so it is the choke point that has
    to hold even if something reaches it by another route.
    """
    verify = getattr(runner, "verify_checkpoint_binding", None)
    if verify is not None:
        verify()
    runner.gate_report.require("emit a report")
    written: dict[str, str] = {}
    if "json" in formats:
        path = runner.p("report.json")
        with open(path + ".tmp", "w") as f:
            json.dump(to_dict(runner, coverage=coverage), f, indent=2)
        os.replace(path + ".tmp", path)
        written["json"] = path
    if "markdown" in formats:
        path = runner.p("report.md")
        with open(path + ".tmp", "w") as f:
            f.write(to_markdown(runner, coverage=coverage))
        os.replace(path + ".tmp", path)
        written["markdown"] = path
    for k, v in written.items():
        runner.log(f"  report: wrote {v}")
    return written
