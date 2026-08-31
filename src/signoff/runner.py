"""The staged, resumable pipeline — and the thing that refuses.

PROVENANCE.  Extracted from
  experiments/03-mechanism-and-validity/run_b_gemma.py     (the driver: stage
                                                            prefix semantics,
                                                            RUN_TAG, MINE_SIG,
                                                            checkpoint configs)
  experiments/03-mechanism-and-validity/common03.py        (atomic io, Tee)
  experiments/03-mechanism-and-validity/validity_common.py (score_rows: resume,
                                                            the identity guard)

STAGES: mine -> gates -> score -> search -> family -> report.  Every stage
checkpoints and resumes; assume any run can be killed.

THE REFUSAL.  `report()` calls `GateReport.require()` and raises `GateFailure`
if any blocking gate is failed or UNRUN.  Numbers computed above a failed gate
remain in the run directory as checkpoints — they are quarantined, not deleted,
because the diagnosis usually needs them — but nothing renders them.

IDENTITY.  Every scored row carries a `run` tag naming everything that changes
what a `d_mean` MEANS (artifact, dtype, replaced layers, mode, BOS), plus its
own (doc, offset).  A cache that disagrees with either is discarded, not merged.

CHECKPOINT BINDING.  Rows are not the only thing that can be stale: `gates.json`
is a file in a directory, and `report()` used to trust whatever it found there.
Gate verdicts are therefore BOUND, at the moment they are measured, to a hash
over the configuration that measured them (adapter identity and revisions,
dtype, replaced layer set, corpus size and signature, gate parameters) plus a
digest of the verdicts themselves.  Emission re-checks the binding and refuses
on a mismatch or an unbound file, through the same refusal path as any other
gate.  The digest is unkeyed: this catches a stale or hand-edited checkpoint —
author error — not an adversary.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Callable, Sequence

import torch

from . import __version__
from . import gates as G
from . import metrics as M
from . import miners as Mine
from . import stats as S
from .adapters import base as AB
from .replacement import Replacement

STAGES = ("mine", "gates", "score", "search", "family", "report")


# ------------------------------------------------------------------ atomic io
# From common03.py.  Write to .tmp and rename: a killed run leaves either the
# old checkpoint or the new one, never a half-written one.


def save_json(obj, path):
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(path + ".tmp", path)


def save_pt(obj, path):
    torch.save(obj, path + ".tmp")
    os.replace(path + ".tmp", path)


def load_json(path, default=None):
    if not os.path.exists(path):
        return default
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return default


class Tee:
    """Print and append to a log file.  From witness.py::Tee.

    Opens per line rather than holding a handle: log volume is a few hundred
    lines per run, and a run that is killed mid-stage should leave a complete
    log rather than a buffer.
    """

    def __init__(self, path: str | None = None, echo: bool = True):
        self.path = path
        self.echo = echo

    def __call__(self, *a):
        s = " ".join(str(x) for x in a)
        if self.echo:
            print(s, flush=True)
        if self.path:
            with open(self.path, "a") as f:
                f.write(s + "\n")

    def close(self):
        self.path = None


class Runner:
    """One audit of one (adapter, replacement) pair, in one output directory."""

    def __init__(self, adapter, replacement: Replacement | None = None, *,
                 out: str = "runs/run", seed: int = 0, batch: int = 8,
                 metrics_batch: int | None = None, echo: bool = True):
        self.adapter = adapter
        self.replacement = replacement or Replacement(adapter, layers="all")
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.seed = int(seed)
        self.batch = int(batch)
        # a 256k-vocab logit tensor is ~131 MB per 2 sequences in float32
        self.metrics_batch = int(metrics_batch or max(1, min(batch, 8)))
        self.log = Tee(self.p("run.log"), echo=echo)
        self.gate_report = G.GateReport()
        self.toks: torch.Tensor | None = None
        self.meta: dict[str, Any] | None = None
        self.rows: list[dict] | None = None
        self.fvu_table: dict[str, dict[str, float]] = {}
        self.witnesses: list[dict] = []
        self.family_stats: dict | None = None
        self.validity: dict | None = None
        self._restore()

    # ---------------------------------------------------------------- paths

    def p(self, *a) -> str:
        return os.path.join(self.out, *a)

    @property
    def run_tag(self) -> str:
        return self.adapter.run_tag(self.replacement)

    def _restore(self):
        """Reattach to an existing run directory."""
        g = load_json(self.p("gates.json"))
        if g:
            self.gate_report = G.GateReport.from_dict(g)
        meta = load_json(self.p("corpus_meta.json"))
        if meta and os.path.exists(self.p("corpus_tokens.pt")):
            self.meta = meta
            self.toks = torch.load(self.p("corpus_tokens.pt"))
        self.fvu_table = load_json(self.p("fvu.json"), {}) or {}
        self.family_stats = load_json(self.p("family.json"))
        self.validity = load_json(self.p("validity.json"))

    def config(self) -> dict[str, Any]:
        return dict(
            adapter=self.adapter.contract(),
            replacement=self.replacement.to_dict(),
            run_tag=self.run_tag, seed=self.seed, batch=self.batch,
            out=os.path.abspath(self.out),
        )

    # ------------------------------------------------------ gate binding

    #: Bumped when the fingerprint's FIELD SET changes, so an old checkpoint
    #: fails the binding gate loudly instead of hashing to a coincidence.
    BINDING_VERSION = "1"

    def config_fingerprint(self) -> dict[str, Any]:
        """Everything a gate verdict is ABOUT.

        A field belongs here iff changing it means the recorded verdicts were
        measured on a different object: the artifact and its pinned revisions,
        the dtype, the substituted layer set and mode, the corpus this was
        gated on, and the gate parameters the verdicts were judged against.
        Deliberately NOT here: the output directory, the device, the batch
        size, and anything else that moves numbers around without changing
        what was measured.
        """
        idn = self.adapter.identity
        ad = self.adapter
        return dict(
            binding_version=self.BINDING_VERSION,
            tool_version=__version__,
            adapter=ad.name,
            adapter_class=f"{type(ad).__module__}.{type(ad).__qualname__}",
            release=idn.release,
            model_repo=idn.model_repo, model_revision=idn.model_revision,
            dict_repo=idn.dict_repo, dict_revision=idn.dict_revision,
            dtype=ad.dtype_name,
            n_layers=int(ad.n_layers),
            replacement_mode=self.replacement.mode,
            replaced_layers=list(self.replacement.layers),
            replacement_seed=int(self.replacement.spec.seed),
            run_tag=self.run_tag,
            seed=int(self.seed),
            n=(int(self.toks.shape[0]) if self.toks is not None else None),
            seq_len=(int(self.toks.shape[1]) if self.toks is not None else None),
            corpus_signature=(self.meta or {}).get("signature"),
            bos=(self.meta or {}).get("bos"),
            gate_ids=list(G.GATE_SPECS),
            gate_params=dict(
                base_vs_base_tol=G.BASE_VS_BASE_TOL.get(ad.dtype_name),
                fp32_replay_tol_nats=ad.dtype_policy.fp32_replay_tolerance_nats,
                mistap_fvu=G.MISTAP_FVU,
                paired_bound_threshold=G.PAIRED_BOUND_THRESHOLD,
            ),
            # an adapter that takes over a measurement is a different measuring
            # instrument; verdicts do not carry across that change either
            adapter_overrides=AB.overridden_measurement_methods(ad),
        )

    def config_hash(self) -> str:
        blob = json.dumps(self.config_fingerprint(), sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def verify_checkpoint_binding(self) -> G.GateResult:
        """Re-check, at emission time, that these verdicts are about THIS run."""
        fp = self.config_fingerprint()
        return self.gate_report.record(G.check_checkpoint_binding(
            expected_hash=self.config_hash(),
            expected_digest=self.gate_report.results_digest(),
            binding=self.gate_report.binding, restored=self.gate_report.restored,
            expected_fingerprint=fp))

    # ----------------------------------------------------------- stage: mine

    def mine(self, n: int = 512, seq_len: int = 64, *, probe_dir: str | None = None,
             pile_docs: Sequence[int] = (), n_probes: int = 0, per_file: int = 10,
             fresh: bool = False, tokens: torch.Tensor | None = None,
             meta: dict | None = None):
        """Corpus windows (+ optional re-windowed probe family).

        `tokens`/`meta` inject a fixed corpus directly — that is how the golden
        test and the synthetic fixture skip the dataset entirely.
        """
        if tokens is not None:
            self.toks, self.meta = tokens, dict(meta or {})
        elif not fresh and self.toks is not None and \
                self.meta.get("n_corpus", 0) >= n and self.meta.get("seq_len") == seq_len:
            self.log(f"  mine: cached ({self.meta['n_corpus']} corpus + "
                     f"{self.meta.get('n_probes', 0)} probes)")
            return self.toks, self.meta
        else:
            tok = self.adapter.tokenizer()
            bos = self.adapter.tokenization.bos_id
            probes = Mine.mine_probes(
                tok, probe_dir=probe_dir, pile_docs=pile_docs, seq_len=seq_len,
                per_file=per_file, n_probes=n_probes, bos_id=bos) if n_probes else []
            probe_docs = {p["doc"] for p in probes if p["source"] == "pile"}
            ctoks, cmeta = Mine.build_corpus(tok, n + 300 if probe_docs else n, seq_len,
                                             seed=self.seed, bos_id=bos)
            if probe_docs:
                # the control pool must be disjoint from the probe texts
                keep = [k for k, m in enumerate(cmeta) if m["doc"] not in probe_docs][:n]
                ctoks, cmeta = ctoks[keep], [cmeta[k] for k in keep]
            real_len = seq_len - 1 if bos is not None else seq_len
            if probes:
                pt = torch.tensor([p["tokens"] for p in probes], dtype=torch.long)
                if bos is not None:
                    pt = torch.cat(
                        [torch.full((pt.shape[0], 1), int(bos), dtype=torch.long), pt], 1)
                toks_all = torch.cat([ctoks, pt])
            else:
                toks_all = ctoks
            self.toks = toks_all
            self.meta = dict(
                model=self.adapter.identity.model_repo, seq_len=seq_len, seed=self.seed,
                bos=bos is not None, bos_id=(int(bos) if bos is not None else None),
                real_tokens_per_window=real_len,
                n_corpus=len(cmeta), n_probes=len(probes),
                corpus=[dict(row=k, doc=m["doc"], offset=int(m["offset"]),
                             pile_set=m.get("pile_set")) for k, m in enumerate(cmeta)],
                probes=[dict(row=len(cmeta) + k, pid=p["pid"], source=p["source"],
                             doc=p["doc"], offset=p["offset"], kw=p["kw"],
                             pile_set=p.get("pile_set")) for k, p in enumerate(probes)],
            )
        self.meta.setdefault("n_corpus", int(self.toks.shape[0]))
        self.meta.setdefault("n_probes", 0)
        self.meta["signature"] = Mine.corpus_signature(self.meta)
        save_pt(self.toks, self.p("corpus_tokens.pt"))
        save_json(self.meta, self.p("corpus_meta.json"))
        self.log(f"  mine: {self.meta['n_corpus']} corpus + {self.meta['n_probes']} probe "
                 f"windows {tuple(self.toks.shape)}, bos={self.meta.get('bos')}")
        return self.toks, self.meta

    # ---------------------------------------------------------- stage: gates

    def gates(self, *, n: int = 4, fvu_n: int = 16, fp32_n: int = 0,
              strict: bool = True) -> G.GateReport:
        """Run every gate that can be run here, then (by default) STOP on failure.

        Order matters: the cheap structural gates first, so a mis-tap is found
        before anything expensive runs.  Gate (iii) is last and optional
        (`fp32_n > 0`) because its float32 arm cannot be co-resident with the
        working model.
        """
        if self.toks is None:
            raise RuntimeError("nothing mined yet: call mine() before gates()")
        rep = self.gate_report
        rep.record(self.adapter.verify_provenance())
        rep.record(self.adapter.gate_bos(self.meta or {}))
        rep.record(self.adapter.gate_base_vs_base(self.toks, n=n))
        verdict, table = self.adapter.gate_fvu(
            self.toks[: min(fvu_n, int(self.toks.shape[0]))],
            self.replacement.layers, batch=self.batch)
        rep.record(verdict)
        self.fvu_table = table
        save_json(table, self.p("fvu.json"))
        cached = self._load_cached_rows()
        rep.record(G.check_identity_guard(cached, self._row_meta(), run_tag=self.run_tag))
        if fp32_n > 0 or self.adapter.dtype_name == "float32":
            # float32 short-circuits inside the adapter: there is no 16-bit
            # error to replay, and it says so rather than reporting a number.
            rep.record(self.adapter.gate_fp32_replay(
                self.toks, self.replacement, n=max(fp32_n, 1)))
        else:
            self.log("  gate(iii) WARNING: a 16-bit dtype with fp32_n=0 — the float32 "
                     "spot-check is NOT running, so these numbers are ungated.")
        # bind LAST: the binding covers every verdict measured above, and the
        # binding gate's own verdict is excluded from the digest it seals
        rep.bind(self.config_hash(), self.config_fingerprint())
        self.verify_checkpoint_binding()      # records the binding gate's own verdict
        save_json(rep.to_dict(), self.p("gates.json"))
        for r in rep.ordered:
            self.log(f"  {r}")
        if strict:
            rep.require("continue past the gates")
        return rep

    # ---------------------------------------------------------- stage: score

    def _row_meta(self) -> dict[int, dict]:
        if not self.meta:
            return {}
        rm = {m["row"]: m for m in self.meta.get("corpus", [])}
        rm.update({m["row"]: m for m in self.meta.get("probes", [])})
        return rm

    def _load_cached_rows(self) -> dict[int, dict]:
        path = self.p("records.jsonl")
        if not os.path.exists(path):
            return {}
        done = {}
        with open(path) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                done[d["row"]] = d
        return done

    def score(self, *, fresh: bool = False) -> list[dict]:
        """Per-sequence divergence over the mined corpus, resumable per row.

        The identity guard runs FIRST: a cache from a different corpus or a
        different run tag is discarded and rescored, never merged.
        """
        if self.toks is None:
            raise RuntimeError("nothing mined yet: call mine() before score()")
        path = self.p("records.jsonl")
        row_meta = self._row_meta()
        done = {} if fresh else self._load_cached_rows()
        if done:
            guard = G.check_identity_guard(done, row_meta, run_tag=self.run_tag)
            self.gate_report.record(guard)
            if not guard.passed:
                self.log(f"  score: {guard.message} -> discarding the cache and rescoring")
                done = {}
        if fresh or not done:
            if os.path.exists(path):
                os.remove(path)  # truncate; never append onto a stale file
        model = self.adapter.model
        n = int(self.toks.shape[0])
        ts = time.time()
        with open(path, "a") as f:
            for i in range(0, n, self.metrics_batch):
                j = min(i + self.metrics_batch, n)
                if all(k in done for k in range(i, j)):
                    continue
                tk = self.toks[i:j].to(self.adapter.device)
                lb = self.adapter.base_logits(model, tk)
                lr = self.adapter.replaced_logits(model, tk, self.replacement)
                m = M.metrics_from_logits(lb, lr, tk)
                del lb, lr
                for k in range(i, j):
                    meta = row_meta.get(k, dict(row=k))
                    rec = dict(
                        row=k,
                        kind=("probe" if k >= self.meta.get("n_corpus", n) else "corpus"),
                        doc=meta.get("doc"), offset=meta.get("offset"),
                        pile_set=meta.get("pile_set"), pid=meta.get("pid"),
                        kw=meta.get("kw"),
                        d_mean=float(m.d_mean[k - i]), d_max=float(m.d_max[k - i]),
                        flip=float(m.flip[k - i]), nll=float(m.nll[k - i]),
                        run=self.run_tag,
                    )
                    f.write(json.dumps(rec) + "\n")
                    done[k] = rec
                f.flush()
        self.rows = [done[k] for k in sorted(done)]
        self.log(f"  score: {len(self.rows)} rows in {(time.time() - ts) / 60:.2f} min")
        self._summarize_distribution()
        return self.rows

    def _text_of(self, row: int) -> str | None:
        """Decoded REAL tokens of a window, or None if the adapter has no tokenizer.

        Used ONLY for family LABELS and the control blocklist.  No text body is
        ever written to a checkpoint, a log or a report.
        """
        try:
            tok = self.adapter.tokenizer()
        except Exception:
            return None
        r = self.toks[row]
        return tok.decode(r[1:] if (self.meta or {}).get("bos") else r)

    def _summarize_distribution(self) -> dict:
        rows = self.rows or []
        corp = [r for r in rows if r["kind"] == "corpus"]
        prob = [r for r in rows if r["kind"] == "probe"]
        tail = sorted(rows, key=lambda r: -r["d_mean"])[:30]
        fam_counts: dict[str, int] = {}
        for r in tail:
            t = self._text_of(r["row"])
            r["family"] = Mine.classify_witness(t) if t is not None else "unlabelled"
            fam_counts[r["family"]] = fam_counts.get(r["family"], 0) + 1
        dist = dict(
            artifact=self.adapter.identity.release, run_tag=self.run_tag, n=len(rows),
            n_corpus=len(corp), n_probes=len(prob),
            corpus={k: S.summarize([r[k] for r in corp])
                    for k in ("d_mean", "d_max", "flip", "nll")} if corp else None,
            probes={k: S.summarize([r[k] for r in prob])
                    for k in ("d_mean", "d_max", "flip", "nll")} if prob else None,
            tail_top30=[{k: v for k, v in r.items() if k != "tokens"} for r in tail],
            tail_family_counts=fam_counts,
            tail_probe_share=sum(1 for r in tail if r["kind"] == "probe") / max(len(tail), 1),
        )
        if corp:
            dist["corpus"]["tail_ratios"] = S.tail_ratios([r["d_mean"] for r in corp])
        save_json(dist, self.p("distribution.json"))
        self.distribution = dist
        if corp:
            c = dist["corpus"]["d_mean"]
            self.log(f"  [dist] corpus n={len(corp)} d_mean mean={c['mean']:.3f} "
                     f"p50={c['p50']:.3f} p99={c['p99']:.3f} max={c['max']:.3f} "
                     f"(p99/p50={dist['corpus']['tail_ratios']['p99_over_median']:.2f}) "
                     f"| top-30 families {fam_counts}")
        return dist

    # --------------------------------------------------------- stage: search

    def search(self, *, miner=None, n_tail: int = 5, n_median: int = 5,
               iters: int = 20, cands: int = 32, lam: float = 2.0,
               fresh: bool = False) -> list[dict]:
        """Constrained + unconstrained greedy arms from tail and median seeds."""
        if not self.rows:
            raise RuntimeError("nothing scored yet: call score() before search()")
        miner = miner or Mine.GreedyMiner(iters=iters, cands=cands, lam=lam, seed=self.seed)
        corp = [r for r in self.rows if r["kind"] == "corpus"] or self.rows
        nll_ceiling = S.quantiles([r["nll"] for r in corp], (0.99,))["p99"]
        self.log(f"  search: nll ceiling (corpus p99) = {nll_ceiling:.4f}")
        path = self.p("witnesses.jsonl")
        done = set()
        if os.path.exists(path) and not fresh:
            with open(path) as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        done.add((d["seed_row"], d["arm"]))
                    except Exception:
                        pass
        elif fresh and os.path.exists(path):
            os.remove(path)
        model = self.adapter.model

        def evaluate(batch: torch.Tensor):
            tk = batch.to(self.adapter.device)
            lb = self.adapter.base_logits(model, tk)
            lr = self.adapter.replaced_logits(model, tk, self.replacement)
            return lb, M.metrics_from_logits(lb, lr, tk)

        out: list[dict] = []
        with open(path, "a") as f:
            for idx, (kind, seed_rec) in enumerate(Mine.pick_seeds(corp, n_tail, n_median)):
                for arm, lam_override in Mine.ARMS:
                    if (seed_rec["row"], arm) in done:
                        continue
                    res = miner.search(
                        seed_tokens=self.toks[seed_rec["row"]], evaluate=evaluate,
                        d_vocab=self.adapter.d_vocab, nll_ceiling=nll_ceiling,
                        lam=lam_override, arm=arm, seed_row=seed_rec["row"],
                        seed_kind=kind, index=idx)
                    d = res.to_dict()
                    d["run"] = self.run_tag
                    f.write(json.dumps(d) + "\n")
                    f.flush()
                    out.append(d)
                    self.log(f"  seed {idx:2d} [{kind:6s}] {arm:13s}: d_mean "
                             f"{res.seed_d_mean:.3f} -> {res.d_mean:.3f} | nll "
                             f"{res.seed_nll:.2f} -> {res.nll:.2f} | edits "
                             f"{res.n_edits}/{res.iters_run} | {res.seconds:.0f}s")
        self.witnesses = out
        return out

    # --------------------------------------------------------- stage: family

    def family(self) -> dict | None:
        """The full validity protocol: three grouping levels, stratified, residualised."""
        if not self.rows:
            raise RuntimeError("nothing scored yet: call score() before family()")

        def eligible(r: dict) -> bool:
            t = self._text_of(r["row"])
            return True if t is None else not Mine.LICENSE_RE.search(t)

        fam = S.family_test(self.rows, control_eligible=eligible)
        self.family_stats = fam
        if fam:
            fam["run_tag"] = self.run_tag
            save_json(fam, self.p("family.json"))
            for name, lv in fam["levels"].items():
                if lv:
                    self.log(f"  [family:{name}] n={lv['n_pairs']} probe "
                             f"{lv['probe_d_mean']['mean']:.3f} vs control "
                             f"{lv['ctrl_d_mean']['mean']:.3f} nats (paired diff "
                             f"{lv['paired_diff_nats']['mean']:+.3f}, "
                             f"{lv['frac_pairs_probe_gt_ctrl'] * 100:.0f}% positive)")
        else:
            self.log("  family: nothing to pair (no probe family in this corpus)")
        self.validity = dict(
            stratified_low_nll=S.stratified_low_nll_test(self.rows),
            residualized_tail=S.residualized_tail(list(self.rows), k=30),
        )
        save_json(self.validity, self.p("validity.json"))
        return fam

    # --------------------------------------------------------- stage: report

    def report(self, formats: Sequence[str] = ("markdown", "json"),
               coverage: bool = False) -> dict[str, str]:
        """Emit — or REFUSE.  This is the contract the whole tool exists for."""
        from . import report as R

        self.verify_checkpoint_binding()
        self.gate_report.require("emit a report")
        return R.emit(self, formats=formats, coverage=coverage)

    # ------------------------------------------------------------- pipeline

    def audit(self, *, n: int = 64, seq_len: int = 64, fvu_n: int = 16,
              fp32_n: int = 0, search_seeds: int = 0, iters: int = 10,
              tokens=None, meta=None, formats=("markdown", "json"),
              coverage: bool = False) -> dict[str, str]:
        """mine -> gates -> score -> [search] -> family -> report, in order."""
        self.log(f"=== audit {self.run_tag} -> {self.out} ===")
        save_json(self.config(), self.p("config.json"))
        self.mine(n=n, seq_len=seq_len, tokens=tokens, meta=meta)
        self.gates(fvu_n=fvu_n, fp32_n=fp32_n)
        self.score()
        if search_seeds:
            self.search(n_tail=search_seeds, n_median=search_seeds, iters=iters)
        self.family()
        return self.report(formats=formats, coverage=coverage)
