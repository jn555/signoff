"""Command line.

    signoff audit    --adapter gpt2-dunefsky --n 64
    signoff gates    --adapter gpt2-dunefsky --n 8
    signoff search   --run runs/gpt2 --lam 2
    signoff report   --run runs/gpt2 --coverage
    signoff adapters
    signoff gates --explain

Every command mirrors a pipeline stage and is resumable: re-running `audit` on
an existing run directory reuses the corpus and the scored rows (subject to the
identity guard) instead of recomputing them.

EXIT CODES
    0  fine
    2  a gate was not cleared — NO REPORT WAS WRITTEN.  This is the core UX
       contract, so it is a distinct code a CI job can branch on.
    1  anything else
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from . import TOOL_NAME, __version__
from .gates import GATE_SPECS, GateFailure

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_GATE = 2


def _adapter(args):
    from . import adapters

    kw = {}
    if args.device:
        kw["device"] = args.device
    if args.dtype:
        kw["dtype"] = args.dtype
    return adapters.get(args.adapter, **kw)


def _runner(args, adapter=None):
    from .replacement import Replacement
    from .runner import Runner

    ad = adapter or _adapter(args)
    layers = "all"
    if getattr(args, "layers", None) and args.layers != "all":
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    rep = Replacement(ad, layers=layers, mode=getattr(args, "mode", "substitute"),
                      seed=args.seed)
    return Runner(ad, rep, out=args.run, seed=args.seed, batch=args.batch)


def _corpus_for(adapter, args):
    """Synthetic adapters carry their own corpus; real ones mine one."""
    if hasattr(adapter, "synthetic_corpus"):
        return adapter.synthetic_corpus(n=args.n, seq_len=args.seq_len,
                                        seed=args.seed, n_probes=args.n_probes)
    return None, None


# ------------------------------------------------------------------ commands


def cmd_adapters(args) -> int:
    from . import adapters

    info = adapters.describe()
    if args.json:
        print(json.dumps(info, indent=2))
        return EXIT_OK
    width = max(len(k) for k in info)
    for name, d in info.items():
        print(f"  {name:<{width}}  [{d['tier']}]  {d['description']}")
    print("\ntiers: ci = runs on a CPU runner · local = needs real RAM or gated "
          "weights · test = synthetic fixture, not a model")
    return EXIT_OK


def cmd_explain_gates(args) -> int:
    print(f"{TOOL_NAME} gates — each one exists because it caught something.\n")
    for spec in GATE_SPECS.values():
        print(f"## {spec.title}  ({spec.id}){'' if spec.blocking else '   [non-blocking]'}")
        print(f"   checks:    {spec.checks}")
        print(f"   caught:    {spec.bug}")
        print(f"   on FAIL:   {spec.diagnosis}\n")
    return EXIT_OK


def cmd_gates(args) -> int:
    if args.explain:
        return cmd_explain_gates(args)
    ad = _adapter(args)
    run = _runner(args, ad)
    toks, meta = _corpus_for(ad, args)
    run.mine(n=args.n, seq_len=args.seq_len, n_probes=args.n_probes,
             tokens=toks, meta=meta)
    run.gates(n=args.gate_n, fvu_n=args.fvu_n, fp32_n=args.fp32_n, strict=False)
    if not run.gate_report.ok:
        run.gate_report.require("continue")
    print("\nall gates cleared.")
    return EXIT_OK


def cmd_audit(args) -> int:
    ad = _adapter(args)
    run = _runner(args, ad)
    toks, meta = _corpus_for(ad, args)
    written = run.audit(
        n=args.n, seq_len=args.seq_len, fvu_n=args.fvu_n, fp32_n=args.fp32_n,
        search_seeds=args.search_seeds, iters=args.iters, tokens=toks, meta=meta,
        formats=tuple(args.format), coverage=args.coverage)
    print("\nwrote:")
    for k, v in written.items():
        print(f"  {k:<9} {v}")
    return EXIT_OK


def cmd_search(args) -> int:
    ad = _adapter(args)
    run = _runner(args, ad)
    if run.toks is None:
        print(f"{args.run}: no corpus checkpoint — run `{TOOL_NAME} audit` first",
              file=sys.stderr)
        return EXIT_ERROR
    run.gates(fvu_n=args.fvu_n, fp32_n=args.fp32_n, strict=True)
    run.rows = [v for _, v in sorted(run._load_cached_rows().items())] or run.score()
    run.search(n_tail=args.n_tail, n_median=args.n_median, iters=args.iters,
               cands=args.cands, lam=args.lam)
    return EXIT_OK


def cmd_report(args) -> int:
    ad = _adapter(args)
    run = _runner(args, ad)
    if run.toks is None:
        print(f"{args.run}: nothing to report on", file=sys.stderr)
        return EXIT_ERROR
    run.rows = [v for _, v in sorted(run._load_cached_rows().items())]
    if run.rows:
        run._summarize_distribution()
    import json as _json
    import os

    wp = run.p("witnesses.jsonl")
    if os.path.exists(wp):
        with open(wp) as f:
            run.witnesses = [_json.loads(line) for line in f if line.strip()]
    written = run.report(formats=tuple(args.format), coverage=args.coverage)
    for k, v in written.items():
        print(f"  {k:<9} {v}")
    return EXIT_OK


# -------------------------------------------------------------------- parser


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=TOOL_NAME,
        description="Falsify a replacement model's claim to behavioural equivalence: "
                    "build the miter, hunt for divergence, report counterexamples. "
                    "It never emits 'faithful'.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"exit 2 = a gate was not cleared and NO report was written.",
    )
    p.add_argument("--version", action="version", version=f"{TOOL_NAME} {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, with_corpus=True):
        sp.add_argument("--adapter", default="gpt2-dunefsky",
                        help="registry name; `%(prog)s adapters` lists them")
        sp.add_argument("--run", default=None, help="run directory (default runs/<adapter>)")
        sp.add_argument("--device", default=None)
        sp.add_argument("--dtype", default=None)
        sp.add_argument("--layers", default="all", help="'all' or a comma list")
        sp.add_argument("--mode", default="substitute",
                        help="substitute | null:shuffled | null:tied-shuffle | null:gaussian")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--batch", type=int, default=8)
        if with_corpus:
            sp.add_argument("-n", "--n", type=int, default=64, help="corpus windows")
            sp.add_argument("--seq-len", type=int, default=64)
            sp.add_argument("--n-probes", type=int, default=0)
        sp.add_argument("--fvu-n", type=int, default=16,
                        help="sequences used for the gate (ii) FVU measurement")
        sp.add_argument("--fp32-n", type=int, default=0,
                        help="sequences replayed in CPU float32 as gate (iii); 0 = skip "
                             "(a 16-bit run with 0 here is UNGATED and says so)")

    a = sub.add_parser("audit", help="the whole pipeline: mine, gate, score, family, report")
    common(a)
    a.add_argument("--search-seeds", type=int, default=0,
                   help="tail/median seeds per arm for the constrained search (0 = skip)")
    a.add_argument("--iters", type=int, default=10)
    a.add_argument("--format", nargs="+", default=["markdown", "json"],
                   choices=["markdown", "json"])
    a.add_argument("--coverage", action="store_true", help="emit the (b,c,r,s,k) cell table")
    a.set_defaults(func=cmd_audit)

    g = sub.add_parser("gates", help="run the gates and stop")
    common(g)
    g.add_argument("--gate-n", type=int, default=4, help="sequences for gate (i)")
    g.add_argument("--explain", action="store_true",
                   help="print every gate and the bug it caught, then exit")
    g.set_defaults(func=cmd_gates)

    s = sub.add_parser("search", help="constrained + unconstrained greedy witness search")
    common(s, with_corpus=False)
    s.add_argument("--n-tail", type=int, default=5)
    s.add_argument("--n-median", type=int, default=5)
    s.add_argument("--iters", type=int, default=20)
    s.add_argument("--cands", type=int, default=32)
    s.add_argument("--lam", type=float, default=2.0)
    s.set_defaults(func=cmd_search)

    r = sub.add_parser("report", help="re-emit from checkpoints (refuses above a bad gate)")
    common(r, with_corpus=False)
    r.add_argument("--format", nargs="+", default=["markdown", "json"],
                   choices=["markdown", "json"])
    r.add_argument("--coverage", action="store_true")
    r.set_defaults(func=cmd_report)

    ad = sub.add_parser("adapters", help="list the registered adapters")
    ad.add_argument("--json", action="store_true")
    ad.set_defaults(func=cmd_adapters)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "run", None) is None and hasattr(args, "adapter"):
        args.run = f"runs/{args.adapter}"
    try:
        return args.func(args)
    except GateFailure as e:
        print(f"\n{e}", file=sys.stderr)
        return EXIT_GATE
    except KeyboardInterrupt:
        print("\ninterrupted; checkpoints are on disk and the run is resumable",
              file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
