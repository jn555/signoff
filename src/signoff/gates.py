"""Gates: the checks a run must clear before it is allowed to say anything.

THE DESIGN PRINCIPLE, learned three separate times in the experiments this
package is extracted from: **a run that fails a gate refuses to emit a report.**
Numbers computed above a failed gate stay in the run directory as quarantined
checkpoints; the reporter will not render them, and the CLI exits nonzero.

Every gate is a named object carrying (a) what it checks, (b) the REAL BUG it
caught in this program's history, and (c) the diagnosis a failure should send
the user to.  The bug field is not decoration — it is why the gate is worth its
runtime, and it is printed on failure.

PROVENANCE.  Extracted from
  experiments/03-mechanism-and-validity/run_b_gemma.py     (stage_gates,
                                                            stage_gatecpu,
                                                            stage_fp32sub)
  experiments/03-mechanism-and-validity/validity_common.py (load_scored_rows —
                                                            the identity guard)
  experiments/03-mechanism-and-validity/gemma_artifact.py  (discover_canonical_l0
                                                            — provenance freeze)

All the `check_*` functions here are PURE: they take measured numbers and
return a verdict.  The measuring lives in the adapters and the runner, so the
whole gate layer is unit-testable on a CPU with no weights.

WHAT THESE GATES ARE FOR.  They catch AUTHOR ERROR — mis-taps, dtype bugs,
registry drift, stale caches, unbound checkpoints.  They are not tamper-proof:
measurement is delegated to the adapter by design (see `adapters/base.py`'s
trust-boundary section and the README's "Trust model"), so an adapter that
overrides a measurement method can hand a gate any number it likes.  The tool's
answer is to STAMP that delegation into the report rather than to pretend it
does not exist.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

Row = dict[str, Any]

PASS = "pass"
FAIL = "fail"
UNRUN = "unrun"
WAIVED = "waived"


@dataclass(frozen=True)
class GateSpec:
    """A named check.  Identity, purpose, provenance, and what a FAIL means."""

    id: str
    title: str
    checks: str
    bug: str
    diagnosis: str
    #: A blocking gate that is failed OR unrun stops report emission.  A
    #: non-blocking gate is a fallback that only matters when another failed.
    blocking: bool = True


GATE_SPECS: dict[str, GateSpec] = {
    "i-base-vs-base": GateSpec(
        id="i-base-vs-base",
        title="(i) base-vs-base",
        checks="A clean layer-major pass reproduces `model(toks)` exactly — KL == 0 "
               "to dtype tolerance, all values finite.",
        bug="gemma-2's logit softcap applied after an fp32 upcast: 'more accurate' and "
            "wrong. 2-ulp logit error, base-vs-base KL 3.3e-3 — it would have biased "
            "every measured divergence, because the base side came from the library's "
            "forward and only the replaced side from the hand-rolled head.",
        diagnosis="The block loop OR the resid->logits head disagrees with the reference "
                  "forward. To tell them apart, run the head alone on the reference "
                  "model's own final residual: if that reproduces the gap, the blocks are "
                  "fine and the head is at fault (op order or dtype around a softcap / "
                  "final norm). Non-finite values instead mean a 16-bit overflow.",
    ),
    "ii-fvu-sanity": GateSpec(
        id="ii-fvu-sanity",
        title="(ii) FVU sanity",
        checks="Per-SITE fraction-of-variance-unexplained is in a sane range after "
               "substitution — not ~1 everywhere. A site is a replaced layer, or a "
               "read->write pair for a cross-layer artifact.",
        bug="The tap trap. gemma-scope and the mwhanna Qwen suite use the SAME hook name "
            "with OPPOSITE pre/post-gain conventions, and gemma-2's sandwich norm puts the "
            "block's real additive contribution after a second norm. A mis-tap reads "
            "FVU ~ 1 — the dictionaries look worthless when the wiring is wrong.",
        diagnosis="FVU ~ 1 at EVERY site is a mis-tap, not a weak dictionary. Re-read the "
                  "adapter's declared taps against the release's own README: (a) is the "
                  "input pre-gain or post-gain normalised? (b) is the output the MLP "
                  "module's return value or the block's additive contribution to the "
                  "residual stream (post second norm)? A single bad layer is a weak "
                  "dictionary and is NOT a gate failure.",
    ),
    "iii-fp32-replay": GateSpec(
        id="iii-fp32-replay",
        title="(iii) fp32 replay",
        checks="A CPU float32 replay of n sequences reproduces the working-dtype run "
               "within 1e-2 nats, on BOTH the base and the replaced stream.",
        bug="bfloat16 injected ~4x the tolerance into every KL on gemma-2 (replaced KL max "
            "3.09e-1 vs float16's 5.85e-4). The base-stream column is the tell: it carries "
            "no transcoders, so its error is pure dtype. This forced the measured switch "
            "to float16 — precision, not range, was the binding constraint.",
        diagnosis="The working dtype is changing the measured divergence. Rerun with a "
                  "finer mantissa (float16 over bfloat16) or float32. If absolutes cannot "
                  "be cleared at any affordable dtype, gate (iii') downgrades the run to "
                  "paired-verdict-only claims with a MEASURED bound.",
    ),
    "iii-prime-paired-bound": GateSpec(
        id="iii-prime-paired-bound",
        title="(iii') paired-bound fallback",
        checks="When (iii) fails on absolutes: the working-dtype-vs-fp32 discrepancy of "
               "the PAIRED family difference, with a bootstrap CI, is below threshold.",
        bug="The gemma 26-layer run fails (iii) absolutely. Its family verdict survived "
            "only because the paired bound (0.037, CI [0.014, 0.061]) was MEASURED rather "
            "than assumed — a bias shared by both arms of a pair cancels, but only if "
            "someone checks by how much.",
        diagnosis="A paired verdict is not safe at this dtype either. The run cannot claim "
                  "the family difference; it needs a full fp32 arm or a smaller subset in "
                  "fp32 with the CI reported alongside.",
        blocking=False,
    ),
    "identity-guard": GateSpec(
        id="identity-guard",
        title="identity guard",
        checks="Every cached row carries a run tag and its own (doc, offset); both must "
               "match the corpus and the run being resumed.",
        bug="An fp16 run nearly reported bfloat16 divergences out of a stale cache, and a "
            "smoke corpus nearly resumed into a full run — row indices are reused across "
            "differently-mined corpora, so the numbers would have been silently mixed.",
        diagnosis="The cache is from a different corpus or a different run. Discard the "
                  "records file and rescore; never merge. If the tags differ only by "
                  "dtype or by replaced-layer set, those are different experiments.",
    ),
    "provenance-freeze": GateSpec(
        id="provenance-freeze",
        title="provenance freeze",
        checks="The live repository listing still selects the same dictionary variant per "
               "layer as the adapter's frozen expectation.",
        bug="The SAELens registry's entry for gemma-scope pointed at the FIRST (lowest-L0) "
            "variant per layer, which for 12 of 26 layers meant degenerate L0 = 5..15 "
            "near-empty dictionaries. The release's own canonical convention is the "
            "variant with average L0 nearest 100.",
        diagnosis="The dictionary repo's contents moved, or the selection rule changed. "
                  "Stop rather than silently measuring a different artifact: re-derive the "
                  "selection, diff it against the frozen table, and update the adapter "
                  "deliberately with a note.",
    ),
    "bos-declaration": GateSpec(
        id="bos-declaration",
        title="BOS declaration",
        checks="The adapter declares a BOS convention, and the mined corpus obeys it.",
        bug="BOS-free evaluation of the BOS-trained gemma-scope suite was badly "
            "off-distribution: corpus NLL 7.005 -> 3.217 nats/token purely from prepending "
            "BOS, d_mean shifted ~12%, and gate-(iii) drift roughly doubled.",
        diagnosis="The corpus was mined under a different BOS convention than the adapter "
                  "declares. Re-mine. If the artifact's own training protocol is unknown, "
                  "that is a provenance problem, not a default to guess at.",
    ),
    "ablation-provenance": GateSpec(
        id="ablation-provenance",
        title="ablation provenance",
        checks="A CIRCUIT-mode run's ablation values are stamped: a declared policy "
               "(mean / resample), a declared calibration set, and a digest over the "
               "actual values that still matches the values on hand.",
        bug="'Knocked out' is not a value. A circuit's faithfulness number is a function "
            "of what the ablated components were replaced BY, and Miller et al. "
            "(2407.08734) measured that choice moving faithfulness a lot — mean-ablation "
            "and resample-ablation are different checkers, not two spellings of one. A "
            "mean vector also comes from a calibration set that can drift (different "
            "templates, seed, size) with nothing in the output saying so.",
        diagnosis="Recompute the ablation values against the DECLARED calibration set and "
                  "re-stamp them. If the calibration set genuinely changed, the old "
                  "faithfulness numbers do not transfer — they were measured against a "
                  "different checker. Never hand-edit the digest.",
        # Non-blocking in the shared registry because it is N/A to a dictionary
        # artifact; `circuit.require_ablation_provenance()` enforces it for
        # every circuit run, raising the same GateFailure.
        blocking=False,
    ),
    "lrm-base-identity": GateSpec(
        id="lrm-base-identity",
        title="LRM ≡ base identity",
        checks="A LOCAL REPLACEMENT MODEL — transcoders everywhere, error nodes restored, "
               "attention patterns and normalisation scales frozen from the real forward — "
               "reproduces the base model's logits on its own prompt, to within the "
               "working dtype's own base-vs-base floor.",
        bug="Experiment 05 reported that the transcoder skeleton loses a multi-hop answer "
            "(p 0.41 -> 7e-6) and could only attribute the gap to {error nodes + frozen "
            "attention + frozen LN} JOINTLY, because it had never built the object "
            "attribution graphs are actually drawn on. Building it in experiment 06 made "
            "the decomposition possible — and made this identity the thing that says the "
            "object was built correctly. It is not a measurement: the LRM reproduces the "
            "model BY CONSTRUCTION, since `TC(x_clean) + (y_clean - TC(x_clean)) == "
            "y_clean` pins the stream to the clean trajectory at every layer. A non-zero "
            "value here means the construction is wrong, and every decomposition taken "
            "on top of it is a number about the wrong object.",
        diagnosis="The construction is wrong; do NOT read the decomposition. In order of "
                  "how often each was the cause: (a) the error nodes were formed or added "
                  "in the working dtype instead of float32, so each layer leaks an ulp and "
                  "the leak compounds with depth — the tell is a residual that grows "
                  "monotonically with the number of replaced layers; (b) the error node "
                  "was computed against a DIFFERENT input than the dictionary is fed at "
                  "the write site (clean cache built at one tap, substitution written at "
                  "another); (c) the cache is a different batch's — check the tokens "
                  "digest; (d) the frozen pattern or scale was written with the wrong "
                  "shape or head order, which a frozen-vs-clean control isolates in one "
                  "run. Note what this gate CANNOT see: the identity is achieved by the "
                  "error nodes alone, so a freeze that silently did nothing still passes. "
                  "Pair it with `check_freeze_efficacy`.",
        # Non-blocking in the shared registry for exactly `ablation-provenance`'s
        # reason: it is N/A to a run that restores nothing, and a permanently
        # UNRUN blocking gate would make `require()` meaningless for every
        # ordinary substitution run.  A run that CLAIMS to be a local
        # replacement model demands it with `replacement.require_lrm_identity()`,
        # which raises the same GateFailure.
        blocking=False,
    ),
    "freeze-efficacy": GateSpec(
        id="freeze-efficacy",
        title="freeze efficacy",
        checks="A run that claims to freeze attention or normalisation actually DIFFERS "
               "from the same substitution with those recomputed, and the freeze hooks "
               "fired the expected number of times.",
        bug="The positive control the LRM identity gate cannot be. Freezing is done by "
            "returning a cached tensor from a forward hook on a library's internal hook "
            "point; if the hook name moves, or the library stops routing that value "
            "through it, the hook still registers and still fires and simply has no "
            "effect. Every downstream number would then be labelled 'frozen' and be the "
            "recomputed run — and the LRM gate would still pass, because error nodes "
            "alone reproduce the base model.",
        diagnosis="The freeze is a no-op. Check that the hooked value is USED downstream "
                  "of the hook in the installed library version (in TransformerLens, "
                  "`hook_pattern`'s and `hook_scale`'s outputs are consumed by the very "
                  "next line — that is what makes them patchable), that the handles were "
                  "not removed before the forward, and that the fire counts match the "
                  "number of blocks the pass ran. Zero fires means the hook is on a module "
                  "the forward never reaches.",
        blocking=False,
    ),
    "checkpoint-binding": GateSpec(
        id="checkpoint-binding",
        title="checkpoint binding",
        checks="The gate verdicts on hand are BOUND to the configuration that measured "
               "them — a hash over the adapter identity, the pinned revisions, the "
               "dtype, the replaced layer set, the corpus size and signature, and the "
               "gate parameters — and the recorded verdicts still digest to what the "
               "binding says they did.",
        bug="`report()` re-attached to whatever `gates.json` sat in the run directory and "
            "trusted it. Two ways that bit: a gates.json left behind by an earlier "
            "configuration of the same directory licensed a report for numbers no gate "
            "in it had measured, and a file whose verdicts were edited by hand from "
            "'fail' to 'pass' produced a clean report. The refusal was one stale file, "
            "or one text edit, away from being decorative.",
        diagnosis="The gate checkpoint in this directory was written by a different "
                  "configuration, was edited after it was written, or was not written by "
                  "this tool at all. Re-run the gates; never hand-edit gates.json. If the "
                  "configuration genuinely changed (dtype, replaced layers, corpus size, "
                  "adapter revision), the old verdicts do not transfer — they were "
                  "measured on a different object. NOTE the limit: this binding catches a "
                  "stale or casually edited checkpoint, not an adversary, who can "
                  "recompute it. It is unkeyed by design.",
    ),
}


@dataclass
class GateResult:
    """The verdict of one gate, with the number that produced it."""

    spec: GateSpec
    status: str = UNRUN
    value: float | None = None
    tolerance: float | None = None
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def blocks_report(self) -> bool:
        """A blocking gate blocks unless it passed or was explicitly waived."""
        return self.spec.blocking and self.status not in (PASS, WAIVED)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            id=self.spec.id, title=self.spec.title, status=self.status,
            value=self.value, tolerance=self.tolerance, message=self.message,
            checks=self.spec.checks, bug=self.spec.bug,
            diagnosis=(self.spec.diagnosis if self.status == FAIL else None),
            blocking=self.spec.blocking, detail=self.detail,
        )

    def __str__(self) -> str:
        v = "" if self.value is None else f" value={self.value:.3e}"
        t = "" if self.tolerance is None else f" tol={self.tolerance:.1e}"
        return f"[{self.status.upper():5s}] {self.spec.title}{v}{t} — {self.message}"


class GateFailure(RuntimeError):
    """Raised when a run tries to proceed past, or emit a report above, a bad gate."""

    def __init__(self, results: Sequence[GateResult], action: str = "emit a report"):
        self.results = list(results)
        lines = [
            f"refusing to {action}: {len(self.results)} gate(s) not cleared.",
            "",
        ]
        for r in self.results:
            lines.append(str(r))
            if r.status == FAIL:
                lines.append(f"        why it exists: {r.spec.bug}")
                lines.append(f"        what to do:    {r.spec.diagnosis}")
            elif r.status == UNRUN:
                lines.append("        what to do:    run this gate; an unrun blocking gate "
                             "is treated exactly like a failed one.")
            lines.append("")
        lines.append("Numbers computed above a failed gate are quarantined in the run "
                     "directory and are not rendered.")
        super().__init__("\n".join(lines))


#: The binding gate's own verdict is excluded from the results digest: it is
#: recorded AFTER the binding is computed, and re-recorded at emission time.
BINDING_GATE = "checkpoint-binding"


@dataclass
class GateReport:
    """The gate results of one run, in declaration order."""

    results: dict[str, GateResult] = field(default_factory=dict)
    #: `{"config_hash": ..., "results_digest": ..., "fingerprint": {...}}`,
    #: written by the run that MEASURED these verdicts.  `None` means the
    #: verdicts are unbound — nothing ties them to a configuration.
    binding: dict[str, Any] | None = None
    #: True when these verdicts were read back from a `gates.json` rather than
    #: measured in this process.
    restored: bool = False

    def __post_init__(self):
        for gid, spec in GATE_SPECS.items():
            self.results.setdefault(gid, GateResult(spec))

    def record(self, result: GateResult) -> GateResult:
        self.results[result.id] = result
        return result

    # ------------------------------------------------------------- binding

    def results_digest(self) -> str:
        """A digest of the verdicts, excluding the binding gate's own.

        Unkeyed on purpose, and documented as such: it catches a checkpoint
        edited by hand after the fact, which is an author mistake.  An
        adversary recomputes it.  See the gate spec's diagnosis.
        """
        blob = json.dumps(
            [[r.spec.id, r.status, (None if r.value is None else repr(float(r.value)))]
             for r in self.ordered if r.spec.id != BINDING_GATE],
            sort_keys=True,
        )
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def bind(self, config_hash: str, fingerprint: Mapping[str, Any]) -> dict[str, Any]:
        """Tie the verdicts measured so far to the configuration that measured them."""
        self.binding = dict(config_hash=str(config_hash),
                            results_digest=self.results_digest(),
                            fingerprint=dict(fingerprint))
        self.restored = False
        return self.binding

    def waive(self, gate_id: str, reason: str) -> GateResult:
        """Explicitly waive a gate.  The waiver and its reason are RENDERED.

        Waiving is a documented, visible act — the report prints WAIVED and the
        reason next to the gate, so a reader can discount the run themselves.
        """
        r = self.results[gate_id]
        r.status = WAIVED
        r.message = f"waived: {reason}"
        if self.binding:
            # a waiver is an in-process, rendered decision, not an edit behind
            # the tool's back: re-digest so the binding still describes reality
            self.binding["results_digest"] = self.results_digest()
        return r

    @property
    def ordered(self) -> list[GateResult]:
        return [self.results[g] for g in GATE_SPECS if g in self.results]

    @property
    def blocking_failures(self) -> list[GateResult]:
        return [r for r in self.ordered if r.blocks_report]

    @property
    def ok(self) -> bool:
        return not self.blocking_failures

    def require(self, action: str = "emit a report") -> None:
        """Raise `GateFailure` unless every blocking gate passed or was waived."""
        if not self.ok:
            raise GateFailure(self.blocking_failures, action)

    def to_dict(self) -> dict[str, Any]:
        return dict(
            ok=self.ok,
            n_pass=sum(1 for r in self.ordered if r.status == PASS),
            n_fail=sum(1 for r in self.ordered if r.status == FAIL),
            n_unrun=sum(1 for r in self.ordered if r.status == UNRUN),
            n_waived=sum(1 for r in self.ordered if r.status == WAIVED),
            binding=(dict(self.binding) if self.binding else None),
            gates=[r.to_dict() for r in self.ordered],
        )

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GateReport":
        rep = cls()
        for g in d.get("gates", []):
            spec = GATE_SPECS.get(g["id"])
            if spec is None:
                continue
            rep.results[spec.id] = GateResult(
                spec, status=g.get("status", UNRUN), value=g.get("value"),
                tolerance=g.get("tolerance"), message=g.get("message", ""),
                detail=g.get("detail", {}) or {},
            )
        b = d.get("binding")
        rep.binding = dict(b) if isinstance(b, Mapping) else None
        rep.restored = True     # these verdicts were not measured in this process
        return rep


# ------------------------------------------------------------------- checks
# Pure verdict functions.  Measurement happens in the adapter / runner.


#: Per-dtype tolerance for gate (i).  From run_b_gemma.py::stage_gates.
BASE_VS_BASE_TOL = {"float32": 1e-6, "float16": 1e-4, "bfloat16": 1e-3}

#: Gate (iii) tolerance in nats.  From the SPEC of experiment 03.
FP32_REPLAY_TOL_NATS = 1e-2

#: Gate (iii') threshold on the paired-difference discrepancy, in nats.
PAIRED_BOUND_THRESHOLD = 0.05

#: FVU at or above this at EVERY layer means the tap is wrong.
MISTAP_FVU = 0.9


def check_base_vs_base(
    kl_max: float, *, dtype: str = "float32", tolerance: float | None = None,
    all_finite: bool = True, max_abs_logit_diff: float | None = None,
) -> GateResult:
    """Gate (i).  The clean layer-major pass must BE the model's own forward."""
    spec = GATE_SPECS["i-base-vs-base"]
    tol = BASE_VS_BASE_TOL.get(dtype, 1e-4) if tolerance is None else tolerance
    detail = dict(dtype=dtype, all_finite=bool(all_finite),
                  max_abs_logit_diff=max_abs_logit_diff)
    if not all_finite:
        return GateResult(spec, FAIL, kl_max, tol,
                          "non-finite values in the clean pass — a 16-bit overflow, not a "
                          "reconstruction problem", detail)
    ok = float(kl_max) < tol
    return GateResult(
        spec, PASS if ok else FAIL, float(kl_max), tol,
        ("layer-major clean pass reproduces the model forward"
         if ok else f"base-vs-base KL max {kl_max:.3e} >= tolerance {tol:.1e}"),
        detail,
    )


def check_fvu_sanity(
    fvu_by_site: Mapping[Any, float], *, mistap_fvu: float = MISTAP_FVU,
) -> GateResult:
    """Gate (ii).  FVU ~ 1 at every SITE means a mis-tap, not a bad dictionary.

    Keyed by substitution SITE, not by layer index: a per-layer plan has one
    site per replaced layer (`"0"`, `"1"`, ...), but a cross-layer plan (a CLT
    writing into several downstream layers from one read site) has sites like
    `"3->7"`.  The gate logic is identical either way — what it looks for is
    "reconstruction is worthless EVERYWHERE", which is the wiring signature.
    """
    spec = GATE_SPECS["ii-fvu-sanity"]
    vals = {str(k): float(v) for k, v in fvu_by_site.items()}
    if not vals:
        return GateResult(spec, UNRUN, None, mistap_fvu, "no per-site FVU measured")
    bad = {k: v for k, v in vals.items() if not (v == v) or v < 0}  # NaN or negative
    detail = dict(per_site=vals, mistap_fvu=mistap_fvu,
                  suspicious_sites=sorted(k for k, v in vals.items() if v >= mistap_fvu))
    if bad:
        return GateResult(spec, FAIL, None, mistap_fvu,
                          f"non-finite or negative FVU at sites {sorted(bad)}", detail)
    best = min(vals.values())
    worst = max(vals.values())
    detail["min"] = best
    detail["max"] = worst
    if best >= mistap_fvu:
        return GateResult(spec, FAIL, best, mistap_fvu,
                          f"FVU >= {mistap_fvu} at EVERY substitution site (best {best:.3f}, "
                          f"worst {worst:.3f}) — this is the mis-tap signature", detail)
    return GateResult(spec, PASS, worst, mistap_fvu,
                      f"per-site FVU in range (best {best:.3f}, worst {worst:.3f}); "
                      f"{len(detail['suspicious_sites'])} site(s) at/above "
                      f"{mistap_fvu} are weak dictionaries, not a wiring fault", detail)


def check_fp32_replay(
    replaced_kl_max: float, base_kl_max: float, *, n: int,
    tolerance: float = FP32_REPLAY_TOL_NATS, dtype: str = "float16",
) -> GateResult:
    """Gate (iii).  Both streams must survive a float32 replay."""
    spec = GATE_SPECS["iii-fp32-replay"]
    detail = dict(n=int(n), dtype=dtype, replaced_kl_max=float(replaced_kl_max),
                  base_kl_max=float(base_kl_max))
    ok = float(replaced_kl_max) < tolerance and float(base_kl_max) < tolerance
    worst = max(float(replaced_kl_max), float(base_kl_max))
    if ok:
        msg = f"float32 replay agrees within {tolerance:g} nats on both streams (n={n})"
    else:
        which = "base" if float(base_kl_max) >= tolerance else "replaced"
        msg = (f"{dtype} moves the {which} stream by {worst:.3e} nats >= {tolerance:g} "
               f"(n={n}); the base stream carries no dictionaries, so any error there is "
               f"pure dtype")
    return GateResult(spec, PASS if ok else FAIL, worst, tolerance, msg, detail)


def check_paired_bound(
    discrepancy: float, *, ci: Mapping[str, float] | None = None,
    threshold: float = PAIRED_BOUND_THRESHOLD,
) -> GateResult:
    """Gate (iii').  The fallback that keeps a PAIRED verdict alive after (iii) fails."""
    spec = GATE_SPECS["iii-prime-paired-bound"]
    detail = dict(ci=dict(ci) if ci else None, threshold=threshold)
    ok = float(discrepancy) < threshold
    hi = (ci or {}).get("hi")
    ci_txt = "" if hi is None else f" (CI upper {hi:.4f})"
    return GateResult(
        spec, PASS if ok else FAIL, float(discrepancy), threshold,
        (f"paired-difference discrepancy {discrepancy:.4f} nats{ci_txt} < {threshold} — "
         f"paired verdicts stand WITH THIS MEASURED BOUND; absolute divergences do not"
         if ok else
         f"paired-difference discrepancy {discrepancy:.4f} nats{ci_txt} >= {threshold} — "
         f"even the paired verdict is not safe at this dtype"),
        detail,
    )


def check_identity_guard(
    cached: Mapping[int, Row], row_meta: Mapping[int, Row], *, run_tag: str | None,
) -> GateResult:
    """Identity guard.  Extracted from validity_common.py::load_scored_rows.

    Two ways a cached row can be wrong, both of which have bitten:
      1. a smaller earlier run mined different windows, so row indices would be
         silently reused — caught by comparing (doc, offset);
      2. the earlier run used a different dtype or replaced a different layer
         set, so the scores are from another experiment — caught by `run_tag`.
    """
    spec = GATE_SPECS["identity-guard"]
    bad_corpus, bad_tag = [], []
    for k, d in cached.items():
        m = row_meta.get(k)
        if m is None or d.get("doc") != m.get("doc") or d.get("offset") != m.get("offset"):
            bad_corpus.append(k)
        elif run_tag is not None and d.get("run") != run_tag:
            bad_tag.append(k)
    detail = dict(n_cached=len(cached), n_bad_corpus=len(bad_corpus), n_bad_tag=len(bad_tag),
                  run_tag=run_tag,
                  foreign_tags=sorted({str(cached[k].get("run")) for k in bad_tag}),
                  bad_rows=sorted(bad_corpus + bad_tag)[:20])
    if bad_corpus:
        return GateResult(spec, FAIL, float(len(bad_corpus)), None,
                          f"{len(bad_corpus)}/{len(cached)} cached rows are from a DIFFERENT "
                          f"CORPUS (doc/offset mismatch)", detail)
    if bad_tag:
        return GateResult(spec, FAIL, float(len(bad_tag)), None,
                          f"{len(bad_tag)}/{len(cached)} cached rows are from a different run "
                          f"({', '.join(detail['foreign_tags'])} != {run_tag})", detail)
    return GateResult(spec, PASS, 0.0, None,
                      f"{len(cached)} cached rows all match this corpus and run tag", detail)


def check_provenance_freeze(
    expected: Mapping[Any, Any], observed: Mapping[Any, Any], *, what: str = "variant selection",
) -> GateResult:
    """Provenance freeze.  Extracted from gemma_artifact.py::discover_canonical_l0."""
    spec = GATE_SPECS["provenance-freeze"]
    exp = {str(k): v for k, v in expected.items()}
    obs = {str(k): v for k, v in observed.items()}
    diff = {k: (exp.get(k), obs.get(k)) for k in set(exp) | set(obs) if exp.get(k) != obs.get(k)}
    detail = dict(what=what, n_expected=len(exp), n_observed=len(obs),
                  differences={k: dict(expected=v[0], observed=v[1])
                               for k, v in sorted(diff.items())[:20]})
    if diff:
        return GateResult(spec, FAIL, float(len(diff)), None,
                          f"{len(diff)} layer(s) of the frozen {what} no longer match the "
                          f"live repository listing", detail)
    return GateResult(spec, PASS, 0.0, None,
                      f"frozen {what} matches the live listing ({len(exp)} entries)", detail)


def check_bos_declaration(
    *, declared_bos_id: int | None, corpus_uses_bos: bool,
    corpus_bos_id: int | None = None, declared: bool = True,
) -> GateResult:
    """BOS declaration.  The adapter must declare it; the miner must enforce it."""
    spec = GATE_SPECS["bos-declaration"]
    detail = dict(declared_bos_id=declared_bos_id, corpus_uses_bos=bool(corpus_uses_bos),
                  corpus_bos_id=corpus_bos_id)
    if not declared:
        return GateResult(spec, FAIL, None, None,
                          "the adapter does not declare a BOS convention; there is no "
                          "safe default to guess", detail)
    want = declared_bos_id is not None
    if want != bool(corpus_uses_bos):
        return GateResult(spec, FAIL, None, None,
                          f"adapter declares bos_id={declared_bos_id} but the corpus was "
                          f"mined with bos={bool(corpus_uses_bos)}", detail)
    if want and corpus_bos_id is not None and int(corpus_bos_id) != int(declared_bos_id):
        return GateResult(spec, FAIL, None, None,
                          f"corpus BOS token {corpus_bos_id} != declared {declared_bos_id}",
                          detail)
    return GateResult(spec, PASS, None, None,
                      (f"BOS declared (id {declared_bos_id}) and enforced by the miner"
                       if want else
                       "adapter declares NO BOS, and the corpus was mined without one"),
                      detail)


def check_checkpoint_binding(
    *, expected_hash: str, expected_digest: str,
    binding: Mapping[str, Any] | None, restored: bool = False,
    expected_fingerprint: Mapping[str, Any] | None = None,
) -> GateResult:
    """Checkpoint binding.  A restored `gates.json` is not evidence by itself.

    Three ways a gate checkpoint can fail to be about the run trying to use it:
      1. it carries no binding at all (hand-made, or from before binding);
      2. it is bound to a DIFFERENT configuration (dtype, replaced layers,
         corpus size, adapter revision) that happens to share a directory;
      3. its verdicts no longer digest to what the binding recorded — the file
         was edited after the run wrote it.

    The digest is unkeyed and the diagnosis says so: this catches a stale or
    hand-edited checkpoint, not an adversary.
    """
    spec = GATE_SPECS[BINDING_GATE]
    recorded_hash = (binding or {}).get("config_hash")
    recorded_digest = (binding or {}).get("results_digest")
    detail = dict(expected_hash=expected_hash, recorded_hash=recorded_hash,
                  expected_results_digest=expected_digest,
                  recorded_results_digest=recorded_digest, restored=bool(restored))
    if not binding or not recorded_hash:
        return GateResult(
            spec, FAIL, None, None,
            "these gate verdicts carry NO configuration binding — a gates.json that was "
            "hand-written, stripped, or produced before binding existed. Verdicts that "
            "cannot be tied to the configuration that measured them are not evidence "
            "about it", detail)
    if str(recorded_hash) != str(expected_hash):
        rec_fp = dict((binding or {}).get("fingerprint") or {})
        exp_fp = dict(expected_fingerprint or {})
        differing = sorted(k for k in set(rec_fp) | set(exp_fp)
                           if rec_fp.get(k) != exp_fp.get(k))
        detail["differing_fields"] = {
            k: dict(when_gated=rec_fp.get(k), now=exp_fp.get(k)) for k in differing[:20]}
        return GateResult(
            spec, FAIL, float(len(differing)) if differing else None, None,
            f"the gate verdicts were measured under a DIFFERENT configuration "
            f"({recorded_hash} != {expected_hash})"
            + (f"; differing: {', '.join(differing)}" if differing
               else "; the two fingerprints do not even name the same fields"),
            detail)
    if recorded_digest is not None and str(recorded_digest) != str(expected_digest):
        return GateResult(
            spec, FAIL, None, None,
            f"the gate verdicts do not match the digest the run recorded for them "
            f"({recorded_digest} != {expected_digest}) — gates.json was edited after it "
            f"was written", detail)
    return GateResult(
        spec, PASS, 0.0, None,
        f"gate verdicts are bound to this configuration ({expected_hash})"
        + (" and were restored from a checkpoint" if restored else ""), detail)


#: The gate ids of the restore/freeze pair, so callers name them once.
LRM_IDENTITY_GATE = "lrm-base-identity"
FREEZE_EFFICACY_GATE = "freeze-efficacy"

#: How much slack the LRM identity gets over the SAME dtype's base-vs-base floor.
#: The construction is exact in exact arithmetic, so the only honest reference
#: is the floor the dtype already imposes on reproducing the model at all
#: (gate (i)); this is a factor, not an absolute, because "fp16 tolerance" means
#: nothing until something says what fp16 costs on this model.
LRM_IDENTITY_FLOOR_FACTOR = 10.0


def check_lrm_base_identity(
    kl_max: float, *, max_abs_logit_diff: float | None = None,
    base_vs_base_kl_max: float | None = None, dtype: str = "float16",
    n_layers_replaced: int | None = None, all_finite: bool = True,
    identity_residual: float | None = None, tolerance: float | None = None,
) -> GateResult:
    """The local replacement model must BE the base model on its own prompt.

    `kl_max` is max per-position KL between the base model's own forward and the
    LRM's.  `base_vs_base_kl_max` is gate (i)'s floor measured on the SAME
    tokens: the LRM path runs through the hand-rolled layer-major forward and
    the hand-rolled head, so it inherits that floor and cannot be asked to beat
    it.  The tolerance is therefore `max(dtype floor, factor x measured floor)`
    unless a caller overrides it.

    `identity_residual` is the sharper, optional number: max |logit diff| between
    the LRM and the CLEAN LAYER-MAJOR pass, which shares every source of dtype
    error with it. Computed in float32 that residual is bitwise ZERO, so a
    non-zero value localises the bug to the construction rather than to the
    dtype — which is why it is reported separately instead of being folded in.
    """
    spec = GATE_SPECS[LRM_IDENTITY_GATE]
    floor = BASE_VS_BASE_TOL.get(dtype, 1e-4)
    if tolerance is None:
        tol = floor if base_vs_base_kl_max is None else max(
            floor, LRM_IDENTITY_FLOOR_FACTOR * float(base_vs_base_kl_max))
    else:
        tol = float(tolerance)
    detail = dict(dtype=dtype, all_finite=bool(all_finite),
                  max_abs_logit_diff=max_abs_logit_diff,
                  base_vs_base_kl_max=base_vs_base_kl_max,
                  dtype_floor=floor, identity_residual=identity_residual,
                  n_layers_replaced=n_layers_replaced)
    if not all_finite:
        return GateResult(spec, FAIL, kl_max, tol,
                          "non-finite logits from the local replacement model — an "
                          "overflow, not a construction error", detail)
    ok = float(kl_max) < tol
    where = ("" if n_layers_replaced is None
             else f" across {n_layers_replaced} replaced layers")
    if ok:
        msg = (f"the local replacement model reproduces the base model{where} "
               f"(KL max {kl_max:.3e} < {tol:.1e})")
        if identity_residual is not None:
            msg += (f"; identity residual vs the clean layer-major pass "
                    f"{identity_residual:.3e}")
        return GateResult(spec, PASS, float(kl_max), tol, msg, detail)
    return GateResult(
        spec, FAIL, float(kl_max), tol,
        f"the local replacement model does NOT reproduce the base model{where}: KL max "
        f"{kl_max:.3e} >= tolerance {tol:.1e}"
        + ("" if base_vs_base_kl_max is None
           else f" (the dtype's own base-vs-base floor here is {base_vs_base_kl_max:.3e})")
        + ". The construction is wrong; the decomposition above it is about a different "
          "object", detail)


def check_freeze_efficacy(
    frozen_vs_recomputed: float, *, hook_fires: Mapping[str, int] | None = None,
    expected_fires: Mapping[str, int] | None = None,
    min_difference: float = 0.0,
) -> GateResult:
    """The positive control: a frozen run must actually differ from a recomputed one.

    `frozen_vs_recomputed` is any non-negative divergence between the SAME
    substitution run with and without freezing — max |logit diff| is what the
    experiment uses.  Exactly zero means the freeze changed nothing, which for a
    substituted (already-diverged) stream can only mean the hooks did not bind.

    `hook_fires` / `expected_fires` catch the same failure one step earlier, and
    catch it even when a coincidence makes the difference non-zero.
    """
    spec = GATE_SPECS[FREEZE_EFFICACY_GATE]
    detail = dict(frozen_vs_recomputed=float(frozen_vs_recomputed),
                  hook_fires=(dict(hook_fires) if hook_fires else None),
                  expected_fires=(dict(expected_fires) if expected_fires else None))
    if hook_fires is not None and expected_fires is not None:
        wrong = {k: (hook_fires.get(k, 0), v) for k, v in expected_fires.items()
                 if hook_fires.get(k, 0) != v}
        if wrong:
            return GateResult(
                spec, FAIL, float(frozen_vs_recomputed), None,
                "freeze hooks did not fire as expected (site: got vs expected) "
                + ", ".join(f"{k}: {g} vs {e}" for k, (g, e) in sorted(wrong.items()))
                + " — the run is labelled frozen and is not", detail)
    if not (float(frozen_vs_recomputed) > float(min_difference)):
        return GateResult(
            spec, FAIL, float(frozen_vs_recomputed), min_difference,
            f"freezing changed nothing (frozen vs recomputed = "
            f"{float(frozen_vs_recomputed):.3e}): on a substituted stream the frozen and "
            f"recomputed values genuinely differ, so identical outputs mean the freeze "
            f"hooks are a no-op", detail)
    return GateResult(
        spec, PASS, float(frozen_vs_recomputed), min_difference,
        f"freezing measurably changes the substituted forward "
        f"(max |logit diff| vs recomputed {float(frozen_vs_recomputed):.3e})"
        + ("" if not hook_fires else
           f"; hooks fired {sum(hook_fires.values())} times across "
           f"{len(hook_fires)} sites"),
        detail)


def summary_table(report: GateReport) -> str:
    """One line per gate, for a terminal."""
    return "\n".join(str(r) for r in report.ordered)


def gate_catalogue() -> list[dict[str, str]]:
    """Every gate with the bug it caught — for `--explain` and the README table."""
    return [dict(id=s.id, title=s.title, checks=s.checks, bug=s.bug,
                 diagnosis=s.diagnosis, blocking=str(s.blocking))
            for s in GATE_SPECS.values()]


def iter_specs() -> Iterable[GateSpec]:
    return GATE_SPECS.values()
