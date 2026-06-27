"""Command-line entry point to run studies and persist outputs.

Examples:
  python -m ai_structural_holes.cli graph
  python -m ai_structural_holes.cli power --baseline 0.2 --or 1.5
  python -m ai_structural_holes.cli study1 --mock --per-domain 1
  python -m ai_structural_holes.cli study4 --models openai/gpt-4o,deepseek/deepseek-chat

With no OPENROUTER_API_KEY (or --mock) the MockClient is used so everything runs
offline. Outputs go to outputs/<study>/.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

from .config import DEFAULT_MODELS, PATHS


def _save(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _models(arg: str):
    return [m.strip() for m in arg.split(",") if m.strip()]


def cmd_graph(args):
    from .causal.graph import describe_graph

    print(describe_graph())


def cmd_power(args):
    from .poweranalysis import n_for_logistic_or

    res = n_for_logistic_or(args.baseline, args.odds_ratio, args.power)
    print(
        f"baseline={res.baseline_rate}, OR={res.odds_ratio} -> "
        f"n/condition={res.n_per_condition} (power~{res.power:.3f})"
    )


def _common_run_kwargs(args):
    return dict(
        models=_models(args.models),
        per_domain=args.per_domain,
        seeds=tuple(range(args.seeds)),
        set_size=args.set_size,
        mock=True if args.mock else None,
        progress=args.progress,
        dry_run=getattr(args, "dry_run", False),
        price_in=getattr(args, "price_in", 2.0),
        price_out=getattr(args, "price_out", 6.0),
    )


def _maybe_dry(res) -> bool:
    """If `res` is a CallPlan (dry-run), print it and return True."""
    from .experiment.planning import CallPlan

    if isinstance(res, CallPlan):
        print(res.render())
        return True
    return False


def cmd_study1(args):
    from .studies import run_study1
    from .analysis.plots import ei_leverage_bar, ate_forest

    out = PATHS.output_dir / "study1"
    res = run_study1(**_common_run_kwargs(args))
    if _maybe_dry(res):
        return
    _save(res.frame, out / "trials.csv")
    _save(res.ate, out / "ate.csv")
    _save(res.ei, out / "ei_leverage.csv")
    ei_leverage_bar(res.ei, out / "ei_leverage.png", "Study 1: EI leverage")
    ate_forest(res.ate, out / "ate_forest.png", "Study 1: ATE by feature")
    print(res.ei[["factor", "EI_norm", "ATE"]].to_string(index=False))
    print(f"\nsaved -> {out}")


def cmd_study2(args):
    from .studies import run_study2
    from .analysis.plots import ei_leverage_bar

    out = PATHS.output_dir / "study2"
    res = run_study2(n_points=args.n_points, **_common_run_kwargs(args))
    if _maybe_dry(res):
        return
    _save(res.frame, out / "trials.csv")
    _save(res.coefficients, out / "coefficients.csv")
    _save(res.ei, out / "ei_leverage.csv")
    ei_leverage_bar(res.ei, out / "ei_leverage.png", "Study 2: EI leverage")
    print(res.coefficients.head(20).to_string(index=False))
    print(f"\nsaved -> {out}")


def cmd_study3(args):
    from .studies import run_study3

    out = PATHS.output_dir / "study3"
    res = run_study3(**_common_run_kwargs(args))
    if _maybe_dry(res):
        return
    _save(res.frame, out / "trials.csv")
    _save(res.ei_by_model, out / "ei_by_model.csv")
    if not res.ei_by_domain.empty:
        _save(res.ei_by_domain, out / "ei_by_domain.csv")
    print("consistency:", {k: res.consistency.get(k) for k in ("kendall_w", "mean_spearman")})
    print(f"\nsaved -> {out}")


def cmd_study4(args):
    from .studies import run_study4

    out = PATHS.output_dir / "study4"
    res = run_study4(**_common_run_kwargs(args))
    if _maybe_dry(res):
        return
    _save(res.frame, out / "trials.csv")
    _save(res.deception, out / "deception.csv")
    _save(res.delta_ei, out / "delta_ei.csv")
    print(res.deception.to_string(index=False))
    print(f"\nsaved -> {out}")


def build_parser():
    p = argparse.ArgumentParser(prog="ai_structural_holes")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("graph", help="print DAG + confounding sets")
    g.set_defaults(func=cmd_graph)

    pw = sub.add_parser("power", help="sample-size / power analysis")
    pw.add_argument("--baseline", type=float, default=0.2)
    pw.add_argument("--odds-ratio", dest="odds_ratio", type=float, default=1.5)
    pw.add_argument("--power", type=float, default=0.8)
    pw.set_defaults(func=cmd_power)

    def add_run_args(sp):
        sp.add_argument("--models", default=",".join(DEFAULT_MODELS[:2]))
        sp.add_argument("--per-domain", dest="per_domain", type=int, default=1)
        sp.add_argument("--seeds", type=int, default=1, help="number of seeds (0..n-1)")
        sp.add_argument("--set-size", dest="set_size", type=int, default=3)
        sp.add_argument("--mock", action="store_true")
        sp.add_argument("--no-progress", dest="progress", action="store_false",
                        default=True, help="关闭进度条(默认开启)")
        sp.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="只估算调用次数与费用，不真正调用 API")
        sp.add_argument("--price-in", dest="price_in", type=float, default=2.0,
                        help="输入价格假设 (USD / 1M tokens)")
        sp.add_argument("--price-out", dest="price_out", type=float, default=6.0,
                        help="输出价格假设 (USD / 1M tokens)")

    s1 = sub.add_parser("study1", help="single-feature paired intervention")
    add_run_args(s1)
    s1.set_defaults(func=cmd_study1)

    s2 = sub.add_parser("study2", help="fractional factorial")
    add_run_args(s2)
    s2.add_argument("--n-points", dest="n_points", type=int, default=16)
    s2.set_defaults(func=cmd_study2)

    s3 = sub.add_parser("study3", help="generalization across M/domain/prompt/R")
    add_run_args(s3)
    s3.set_defaults(func=cmd_study3)

    s4 = sub.add_parser("study4", help="reverse / adversarial")
    add_run_args(s4)
    s4.set_defaults(func=cmd_study4)

    return p


def main(argv=None):
    warnings.filterwarnings("ignore")
    PATHS.ensure()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
