"""The validity protocol, as code — so a run cannot skip it.

PROVENANCE.  Extracted from
  experiments/01-divergence-witnesses/witness.py            (summarize, quantiles)
  experiments/03-mechanism-and-validity/common03.py         (pearson, spearman)
  experiments/03-mechanism-and-validity/validity_common.py  (family, the greedy
                                                             NLL-matched pairing,
                                                             the grouping levels)
  experiments/03-mechanism-and-validity/analyze03.py        (study_bg6: stratified
                                                             low-NLL test, the
                                                             NLL-residualised tail)
  experiments/03-mechanism-and-validity/run_b_gemma.py      (stage_pass: dual FVU)

Why these particular checks exist — each is a correction that changed a
headline number in this program's own history:

* **Pseudo-replication guard.**  Probe windows share source texts, so windows
  are not independent observations.  Reporting only the windowed level
  overstated an exp-02 family effect roughly 3x; every family test here reports
  the windowed, distinct-text and independent-source levels TOGETHER, and the
  reporter renders all three or none.
* **NLL matching.**  Boilerplate is low-perplexity.  Without matching controls
  on base-model NLL, "this family diverges more" can be "low-NLL text diverges
  more".  Pairing is greedy nearest-NLL without replacement.
* **Stratified low-NLL repeat.**  Matching still leaves the easy end of the
  distribution doing the work; the test is repeated inside the corpus NLL <= p25
  stratum, where probe and control are both easy.
* **NLL-residualised tail.**  A top-K tail can be a low-NLL readout rather than
  a family effect.  d is regressed on a quadratic in NLL and the tail
  recomposed from the residuals; a family that only appears because it is
  low-NLL drops out.
* **Dual FVU.**  The global (summed) and per-token-median estimators disagree —
  with opposite signs — on GPT-2.  Reporting one alone lets an artifact look
  well- or badly-reconstructed by choice of estimator, so both are always
  emitted.

Nothing in this module reads, stores or logs a text body: rows are identified by
(source, doc, offset) and family LABELS only.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Iterable, Sequence

import torch

Row = dict[str, Any]

# --------------------------------------------------------------- descriptive


def quantiles(xs: Sequence[float], qs: Iterable[float] = (0.5, 0.9, 0.99)) -> dict[str, float]:
    """Index-rounded order statistics.  Verbatim from witness.py::quantiles.

    Deliberately NOT interpolated: the completed runs' pinned numbers use this
    estimator, and a switch to linear interpolation would silently move every
    p99 in the golden tests.
    """
    xs = list(xs)
    out: dict[str, float] = {}
    if not xs:
        return {f"p{int(q * 100)}": float("nan") for q in qs}
    t = torch.tensor(xs, dtype=torch.float32).sort().values
    for q in qs:
        idx = min(len(t) - 1, max(0, int(round(q * (len(t) - 1)))))
        out[f"p{int(q * 100)}"] = float(t[idx])
    return out


def summarize(xs: Sequence[float]) -> dict[str, float]:
    """n/mean/std/min/max + p50/p90/p99.  Verbatim from witness.py::summarize."""
    xs = list(xs)
    if not xs:
        return dict(n=0, mean=float("nan"), std=float("nan"), min=float("nan"),
                    max=float("nan"), p50=float("nan"), p90=float("nan"), p99=float("nan"))
    t = torch.tensor(xs, dtype=torch.float32)
    d = dict(
        n=len(xs),
        mean=float(t.mean()),
        std=float(t.std()) if len(xs) > 1 else 0.0,
        min=float(t.min()),
        max=float(t.max()),
    )
    d.update(quantiles(xs, (0.5, 0.9, 0.99)))
    return d


def pearson(a, b) -> float:
    """Verbatim from common03.py::pearson."""
    a = torch.as_tensor(a).reshape(-1).float()
    b = torch.as_tensor(b).reshape(-1).float()
    if a.numel() < 3:
        return float("nan")
    a = a - a.mean()
    b = b - b.mean()
    d = (a.norm() * b.norm()).item()
    return float("nan") if d == 0 else float((a @ b).item() / d)


def _rank(x) -> torch.Tensor:
    """Average ranks (ties get the mean rank).  Verbatim from common03.py::_rank."""
    x = torch.as_tensor(x).reshape(-1).float()
    n = x.numel()
    order = x.argsort()
    ranks = torch.empty(n, dtype=torch.float32)
    ranks[order] = torch.arange(n, dtype=torch.float32)
    xs = x[order]
    i = 0
    while i < n:  # tie-average
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + j) / 2.0
        i = j + 1
    return ranks


def spearman(a, b) -> float:
    """Verbatim from common03.py::spearman."""
    a = torch.as_tensor(a)
    if a.reshape(-1).numel() < 3:
        return float("nan")
    return pearson(_rank(a.cpu()), _rank(torch.as_tensor(b).cpu()))


def paired_summary(diffs: Sequence[float]) -> dict[str, float]:
    """mean / sd / se / t and the sign rate of a paired difference.

    From analyze03.py::study_bg6(a), which reports exactly these for the
    stratified test.  `t` is mean/SE — a descriptive standardised effect, not a
    p-value; this program does not report p-values off a single run.
    """
    d = list(diffs)
    if not d:
        return dict(n=0, mean=float("nan"), sd=float("nan"), se=float("nan"),
                    t=float("nan"), frac_positive=float("nan"))
    mu = sum(d) / len(d)
    sd = math.sqrt(sum((x - mu) ** 2 for x in d) / max(len(d) - 1, 1))
    se = sd / math.sqrt(len(d)) if len(d) else float("nan")
    return dict(
        n=len(d), mean=mu, sd=sd, se=se,
        t=(mu / se if se else float("nan")),
        frac_positive=sum(1 for x in d if x > 0) / len(d),
    )


def bootstrap_ci(
    xs: Sequence[float],
    stat: Callable[[Sequence[float]], float] = lambda v: sum(v) / len(v),
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI.  Used for the dtype-discrepancy bound of gate (iii').

    The gemma 26-layer run failed gate (iii) on absolutes; its family verdict
    survived only because the fp16-vs-fp32 PAIRED-difference discrepancy was
    measured with a CI (0.037, [0.014, 0.061]) instead of assumed.
    """
    v = list(xs)
    if len(v) < 2:
        return dict(n=len(v), point=(stat(v) if v else float("nan")),
                    lo=float("nan"), hi=float("nan"), alpha=alpha, n_boot=0)
    g = torch.Generator().manual_seed(int(seed))
    t = torch.tensor(v, dtype=torch.float64)
    boots = []
    for _ in range(int(n_boot)):
        idx = torch.randint(0, len(v), (len(v),), generator=g)
        boots.append(stat([float(x) for x in t[idx]]))
    b = torch.tensor(boots, dtype=torch.float64).sort().values
    lo_i = min(len(b) - 1, max(0, int(round((alpha / 2) * (len(b) - 1)))))
    hi_i = min(len(b) - 1, max(0, int(round((1 - alpha / 2) * (len(b) - 1)))))
    return dict(n=len(v), point=float(stat(v)), lo=float(b[lo_i]), hi=float(b[hi_i]),
                alpha=alpha, n_boot=int(n_boot))


# ------------------------------------------------------------------ dual FVU


def dual_fvu(num, den, exclude_position_0: bool = True) -> dict[str, float]:
    """Both FVU estimators.  From run_b_gemma.py::stage_pass.

    num, den : per-token numerator/denominator, (..., T) — see metrics.fvu_terms.

    fvu_global       = sum(num) / sum(den)   — variance-weighted; dominated by
                       high-variance tokens.
    fvu_token_median = median(num/den)       — the typical token.

    They are DIFFERENT quantities and they disagree in sign of the layer
    ranking on GPT-2, which is why both ship.  Quoting one alone is how a
    dictionary suite gets to look good by choice of estimator.

    NOTE ON THE SOURCE.  run_b_gemma computed the global estimator over all
    positions and the token median over positions >= 1.  That asymmetry is a
    wart, not a convention; here both obey `exclude_position_0` together.  Pass
    `exclude_position_0=False` to reproduce a whole-window number.
    """
    num = torch.as_tensor(num).float()
    den = torch.as_tensor(den).float()
    if num.shape != den.shape:
        raise ValueError(f"fvu term shape mismatch: {tuple(num.shape)} vs {tuple(den.shape)}")
    if exclude_position_0:
        if num.shape[-1] < 2:
            raise ValueError("cannot exclude position 0 from a window of width < 2")
        num, den = num[..., 1:], den[..., 1:]
    ratio = (num / den.clamp_min(1e-12)).reshape(-1)
    return dict(
        fvu_global=float(num.sum() / den.sum().clamp_min(1e-12)),
        fvu_token_median=float(ratio.median()),
        n_tokens=int(ratio.numel()),
    )


# ------------------------------------------------------------------- pairing


def greedy_nearest_nll_pairs(probes: Sequence[Row], pool: Sequence[Row]) -> list[Row]:
    """Match each probe to its nearest-NLL control, without replacement.

    Verbatim procedure from validity_common.py::family's inner `pair`: probes
    are consumed in ascending NLL, the pool is scanned in ascending NLL, and a
    control is used at most once.  The early break is exact on a sorted pool
    (once past the probe's NLL the gap only grows).

    Returns one dict per pair; every pair records `nll_gap` so a reader can see
    how good the matching actually was.
    """
    pool_sorted = sorted(pool, key=lambda r: r["nll"])
    used: set = set()
    pairs: list[Row] = []
    for pr in sorted(probes, key=lambda r: r["nll"]):
        best, bestd = None, None
        for r in pool_sorted:
            if r["row"] in used:
                continue
            d = abs(r["nll"] - pr["nll"])
            if bestd is None or d < bestd:
                best, bestd = r, d
            elif r["nll"] > pr["nll"] and d > bestd:
                break
        if best is None:
            continue
        used.add(best["row"])
        pid = pr.get("pid") or ""
        pairs.append(dict(
            pid=pr.get("pid"), source=str(pid).split(":")[0],
            doc=pr.get("doc"), offset=pr.get("offset"), probe_row=pr["row"],
            probe_d_mean=pr["d_mean"], probe_nll=pr["nll"],
            ctrl_row=best["row"], ctrl_d_mean=best["d_mean"],
            ctrl_nll=best["nll"], nll_gap=bestd,
        ))
    return pairs


def text_id(row: Row) -> str:
    """The SOURCE TEXT a probe window came from — `pile:5206`, `file:mit`.

    Windows of one text are not independent observations; this is the key the
    distinct-text grouping level deduplicates on.
    """
    return str(row.get("pid", "")).rsplit(":", 1)[0]


def grouping_levels(
    probes: Sequence[Row], independent_prefix: str = "pile"
) -> dict[str, list[Row]]:
    """The pseudo-replication guard: three nested levels of independence.

    windowed          every probe window.  Windows share source texts, so this
                      level OVERSTATES independence — it is reported, never
                      reported alone.
    distinct_text     one window per source text.
    independent_source
                      only windows from independently sourced documents (corpus
                      documents rather than the committed probe files).  Called
                      `pile_only` in exp-02/03.

    Reporting the three together is the exp-02 correction that cut a headline
    factor of ~3.
    """
    seen: set[str] = set()
    distinct: list[Row] = []
    for p in probes:
        tid = text_id(p)
        if tid not in seen:
            seen.add(tid)
            distinct.append(p)
    return {
        "windowed": list(probes),
        "distinct_text": distinct,
        "independent_source": [
            p for p in probes if str(p.get("pid", "")).startswith(f"{independent_prefix}:")
        ],
    }


def _level_stats(pairs: Sequence[Row], corpus_p99: float) -> Row:
    pd = [p["probe_d_mean"] for p in pairs]
    cd = [p["ctrl_d_mean"] for p in pairs]
    diff = [a - b for a, b in zip(pd, cd)]
    ctrl_p90 = quantiles(cd, (0.9,))["p90"]
    out = dict(
        n_pairs=len(pairs),
        probe_d_mean=summarize(pd), ctrl_d_mean=summarize(cd),
        paired_diff_nats=summarize(diff),
        paired=paired_summary(diff),
        nll_gap=summarize([p["nll_gap"] for p in pairs]),
        probe_nll=summarize([p["probe_nll"] for p in pairs]),
        ctrl_nll=summarize([p["ctrl_nll"] for p in pairs]),
        frac_pairs_probe_gt_ctrl=sum(1 for x in diff if x > 0) / len(diff),
        frac_probes_above_ctrl_p90=sum(1 for x in pd if x > ctrl_p90) / len(pd),
        ctrl_p90=ctrl_p90,
        frac_probes_above_corpus_p99=sum(1 for x in pd if x > corpus_p99) / len(pd),
        pairs=pairs,
    )
    return out


def family_test(
    rows: Sequence[Row],
    *,
    control_eligible: Callable[[Row], bool] | None = None,
    independent_prefix: str = "pile",
) -> Row:
    """Probe family vs NLL-matched corpus control, at all three grouping levels.

    Extracted from validity_common.py::family.  `control_eligible` is the
    blocklist that keeps the control pool disjoint from the probe family (in
    the completed runs: drop corpus windows whose text matches the license
    regex).  It is a PREDICATE ON ROWS so this module never handles text.

    Returns `None` when there is nothing to pair; the reporter renders that as
    MISSING rather than inventing a level.
    """
    corp = [r for r in rows if r.get("kind") == "corpus"]
    prob = [r for r in rows if r.get("kind") == "probe"]
    if not corp or not prob:
        return None
    elig = [r for r in corp if (control_eligible is None or control_eligible(r))]
    if not elig:
        return None
    corpus_p99 = quantiles([r["d_mean"] for r in corp], (0.99,))["p99"]
    out = dict(
        corpus_d_mean_p99=corpus_p99,
        control_pool=len(elig),
        n_corpus=len(corp),
        n_probes=len(prob),
        levels={},
    )
    for name, pr in grouping_levels(prob, independent_prefix).items():
        pairs = greedy_nearest_nll_pairs(pr, elig) if pr else []
        out["levels"][name] = _level_stats(pairs, corpus_p99) if pairs else None
    return out


def stratified_low_nll_test(rows: Sequence[Row], q: float = 0.25) -> Row:
    """Repeat the paired test inside the corpus NLL <= p_q stratum.

    Extracted from analyze03.py::study_bg6(a).  Boilerplate is low-NLL; if
    low-NLL windows simply diverge more, an NLL-matched paired difference can
    still be an NLL effect at the easy end.  Restricting BOTH arms to the easy
    stratum is the check.
    """
    corp = [r for r in rows if r.get("kind") == "corpus"]
    prob = [r for r in rows if r.get("kind") == "probe"]
    if not corp or not prob:
        return dict(available=False, reason="need both corpus and probe rows")
    nlls = sorted(r["nll"] for r in corp)
    ceiling = nlls[max(0, int(q * len(nlls)) - 1)]
    low_probes = [r for r in prob if r["nll"] <= ceiling]
    pool = [r for r in corp if r["nll"] <= ceiling]
    pairs = greedy_nearest_nll_pairs(low_probes, pool) if (low_probes and pool) else []
    diffs = [p["probe_d_mean"] - p["ctrl_d_mean"] for p in pairs]
    return dict(
        available=bool(pairs),
        quantile=q,
        stratum_ceiling_nll=ceiling,
        n_probes_in_stratum=len(low_probes),
        n_probes_total=len(prob),
        n_pool=len(pool),
        paired=paired_summary(diffs),
        mean_abs_nll_gap=(sum(p["nll_gap"] for p in pairs) / len(pairs)) if pairs else float("nan"),
    )


def polyfit_least_squares(xs: Sequence[float], ys: Sequence[float], degree: int = 2):
    """Normal-equation polynomial fit with Gauss-Jordan partial pivoting.

    Extracted from analyze03.py::study_bg6(b) (a 3x3 solve there, generalised to
    `degree` here).  Dependency-free on purpose: the residualisation must run in
    the CPU-only test subset, where numpy is optional.

    Returns the coefficient list [b0, b1, ...] or None if the design is singular.
    """
    n = len(xs)
    k = degree + 1
    if n < k:
        return None
    X = [[float(x) ** p for p in range(k)] for x in xs]
    XtX = [[sum(X[i][a] * X[i][b] for i in range(n)) for b in range(k)] for a in range(k)]
    Xty = [sum(X[i][a] * float(ys[i]) for i in range(n)) for a in range(k)]
    M = [XtX[i][:] + [Xty[i]] for i in range(k)]
    try:
        for c in range(k):
            pv = max(range(c, k), key=lambda r_: abs(M[r_][c]))
            M[c], M[pv] = M[pv], M[c]
            if abs(M[c][c]) < 1e-12:
                raise ZeroDivisionError
            for r_ in range(k):
                if r_ == c:
                    continue
                f = M[r_][c] / M[c][c]
                for j in range(c, k + 1):
                    M[r_][j] -= f * M[c][j]
        return [M[i][k] / M[i][i] for i in range(k)]
    except ZeroDivisionError:
        return None


def residualized_tail(rows: Sequence[Row], k: int = 30, degree: int = 2) -> Row:
    """Recompose the top-K tail from NLL-residualised divergence.

    Extracted from analyze03.py::study_bg6(b).  `d_mean` is regressed on a
    quadratic in base NLL; the tail is then taken on the residual.  A family
    that only reaches the tail because it is low-NLL drops out, and the
    `overlap_with_raw` count is the honest measure of how much of the raw tail
    was an NLL readout.

    Rows are annotated in place with `resid_d_mean` so the reporter can render
    either ordering from one pass.
    """
    rows = list(rows)
    if len(rows) < max(10, degree + 2):
        return dict(available=False, reason=f"need >= {max(10, degree + 2)} rows")
    xs = [r["nll"] for r in rows]
    ys = [r["d_mean"] for r in rows]
    beta = polyfit_least_squares(xs, ys, degree)
    if beta is None:
        return dict(available=False, reason="NLL design matrix is singular")
    for r, x in zip(rows, xs):
        r["resid_d_mean"] = r["d_mean"] - sum(b * (x ** p) for p, b in enumerate(beta))
    raw = sorted(rows, key=lambda r: -r["d_mean"])[:k]
    res = sorted(rows, key=lambda r: -r["resid_d_mean"])[:k]
    raw_rows = {r["row"] for r in raw}

    def composition(sel):
        c: dict[str, int] = {}
        for r in sel:
            fam = r.get("family") or r.get("kind") or "unlabelled"
            c[fam] = c.get(fam, 0) + 1
        return c

    def probe_share(sel):
        return sum(1 for r in sel if r.get("kind") == "probe") / max(len(sel), 1)

    return dict(
        available=True, k=k, degree=degree, n=len(rows), beta=beta,
        raw=dict(probe_share=probe_share(raw), composition=composition(raw),
                 rows=[r["row"] for r in raw]),
        residualized=dict(probe_share=probe_share(res), composition=composition(res),
                          rows=[r["row"] for r in res]),
        overlap_with_raw=len({r["row"] for r in res} & raw_rows),
    )


def tail_ratios(values: Sequence[float]) -> dict[str, float]:
    """Suite-health DIAGNOSTIC — reported, never gated in v0.1.

    p99/median was 2.27 on the healthy GPT-2 suite, 1.93 on gemma-scope and 1.66
    on the saturated Qwen suite.  Three points is not a criterion, and this tool
    does not pretend otherwise: the report prints the ratio and says explicitly
    that no threshold has been pre-declared.
    """
    s = summarize(values)
    med = max(s["p50"], 1e-12)
    return dict(
        p99_over_median=s["p99"] / med,
        max_over_median=s["max"] / med,
        frac_above_10x_median=(sum(1 for x in values if x > 10 * s["p50"]) / len(values))
        if values else float("nan"),
    )
