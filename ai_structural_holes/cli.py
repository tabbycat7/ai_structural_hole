"""Command-line entry point to run studies and persist outputs.

Examples:
  python -m ai_structural_holes.cli graph
  python -m ai_structural_holes.cli power --baseline 0.2 --or 1.5
  python -m ai_structural_holes.cli study1 --mock --per-domain 1
  python -m ai_structural_holes.cli study4 --models openai/gpt-4o,deepseek/deepseek-chat

With no OPENROUTER_API_KEY (or --mock) the MockClient is used so everything runs
offline. Selection outputs go to outputs/<study>/<model_slug>/ (one subdirectory per
--models entry).
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Optional

from pathlib import Path

from .config import DEFAULT_MODELS, PATHS


def _save(df, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _write_review_sheet(
    audit_path: Path,
    out_path: Path,
    *,
    verification_path: Optional[Path] = None,
) -> Path:
    from .data.audit_sheet_review import write_audit_sheet_review

    ver = verification_path
    if ver is None:
        default_ver = audit_path.parent / "genuine_source_verification.csv"
        if default_ver.exists():
            ver = default_ver
    return write_audit_sheet_review(audit_path, out_path, verification_path=ver)


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
        gen_route=getattr(args, "gen_route", "template"),
        gen_model=getattr(args, "gen_model", None) or None,
        query_source=getattr(args, "query_source", "builtin"),
        distractor_route=getattr(args, "distractors", None) or None,
        concurrency=getattr(args, "concurrency", 1),
        use_variant_store=getattr(args, "use_variant_store", True),
        output_mode=getattr(args, "output_mode", "minimal"),
        progress_file=(
            Path(args.progress_file) if getattr(args, "progress_file", None) else None
        ),
        use_llm_cache=True if getattr(args, "llm_cache", False) else None,
    )


def _make_output_sink(args, study: str, models):
    if getattr(args, "no_incremental_output", False):
        return None
    from .study_output import StudyModelSink

    return StudyModelSink(
        study=study,
        models=models,
        output_root=PATHS.output_dir,
        refresh_every=getattr(args, "analysis_refresh_every", 100),
        refresh_sec=getattr(args, "analysis_refresh_sec", 300.0),
        resume=getattr(args, "resume", False),
    )


def _maybe_dry(res) -> bool:
    """If `res` is a CallPlan (dry-run), print it and return True."""
    from .experiment.planning import CallPlan

    if isinstance(res, CallPlan):
        print(res.render())
        return True
    return False


def cmd_watch_progress(args):
    """Poll the shared JSON progress file (for a second terminal)."""
    import time

    from .experiment.progress import DEFAULT_PROGRESS_FILE, format_progress_line, load_progress

    path = Path(args.file) if args.file else DEFAULT_PROGRESS_FILE
    interval = args.interval
    print(f"监视进度文件: {path}  (每 {interval}s 刷新, Ctrl-C 退出)\n")
    last_line = ""
    try:
        while True:
            data = load_progress(path)
            if data is None:
                line = f"(等待任务启动… 文件尚不存在: {path})"
            else:
                line = format_progress_line(data)
                if data.get("status") in ("done", "error"):
                    print(line)
                    print(f"\n任务已结束: status={data.get('status')}")
                    break
            if line != last_line:
                print(line, flush=True)
                last_line = line
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n已停止监视。")


def cmd_import_queries(args):
    """Import real user questions (+ their real passages) into the frozen pool."""
    from pathlib import Path as _Path

    from .data.query_pool import import_queries, pool_dir

    src = _Path(args.file)
    if not src.exists():
        raise SystemExit(f"找不到数据文件: {src}\n请先下载 DuReader 数据(见 docs/操作手册_宝宝级.md)。")
    domains = [d.strip() for d in args.domains.split(",") if d.strip()] or None
    stats = import_queries(
        src, per_domain=args.per_domain, domains=domains, seed=args.seed,
    )
    print(f"{'领域':<18}{'候选':>8}{'入选':>8}")
    for dom, s in stats.items():
        flag = "  (配额未满)" if s["imported"] < args.per_domain else ""
        print(f"{dom:<18}{s['candidates']:>8}{s['imported']:>8}{flag}")
    print(
        f"\n题库目录: {pool_dir()}"
        "\n请人工抽检问题与段落后将该目录提交进版本管理(冻结)。"
        "\n下一步: `gen-base --query-source pool` 为这些题目生成基线文章。"
    )


def cmd_gen_base(args):
    """Generate + freeze LLM baseline articles for the query pool."""
    from .data.base_articles import base_dir, generate_base_article, load_base_articles, save_base_article
    from .llm.client import get_client
    from .llm.parallel import map_concurrent
    from .studies.common import get_queries

    domains = [d.strip() for d in args.domains.split(",") if d.strip()] or None
    queries = get_queries(args.query_source, per_domain=args.per_domain, domains=domains)
    existing = load_base_articles(validated_only=False)
    client = get_client(mock=True if args.mock else None, use_llm_cache=True)
    progress = getattr(args, "progress", True)

    bar = None
    if progress:
        try:
            from tqdm import tqdm

            bar = tqdm(
                total=len(queries),
                desc="基线文章",
                unit="题",
                dynamic_ncols=True,
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            )
        except Exception:
            bar = None

    counter = {"ok": 0, "skip": 0, "fallback": 0}

    todo = []
    for q in queries:
        if q.id in existing and not args.force:
            counter["skip"] += 1
            if bar is not None:
                bar.set_postfix_str(f"skip {q.domain}  llm={counter['ok']} fb={counter['fallback']}")
                bar.update(1)
            else:
                print(f"[skip] {q.id} ({q.domain}) 已冻结，--force 可重新生成")
            continue
        todo.append(q)

    def _gen(q):
        return generate_base_article(client, q, args.model, max_attempts=args.max_attempts)

    def _on_result(_i, q, rec):
        path = save_base_article(rec)
        status = rec["generator"]
        if status == "llm":
            counter["ok"] += 1
        else:
            counter["fallback"] += 1
        if bar is not None:
            bar.set_postfix_str(
                f"{q.domain} [{status}] attempts={rec['attempts']} "
                f"chars={rec['n_chars']}  llm={counter['ok']} fb={counter['fallback']}"
            )
            bar.update(1)
        else:
            print(
                f"[{status}] {q.id} ({q.domain}) attempts={rec['attempts']} "
                f"chars={rec['n_chars']} -> {path}"
            )

    map_concurrent(_gen, todo, concurrency=args.concurrency, on_result=_on_result)

    n_ok, n_skip, n_fallback = counter["ok"], counter["skip"], counter["fallback"]
    if bar is not None:
        bar.close()

    print(
        f"\n完成: 新生成 {n_ok}, 模板回退 {n_fallback}, 跳过 {n_skip}。"
        f"\n基线文章目录: {base_dir()}"
        "\n请人工抽检正文后将该目录提交进版本管理(冻结)。"
    )


def cmd_gen_variants(args):
    """Repair + freeze all LLM variant articles (targets + distractors).

    Reuses already-frozen `llm` variants from `data/variant_articles/` for free
    and only calls the API for the ones that are missing (e.g. the old
    template-fallbacks). Every freshly generated variant is written back to the
    store, so a later `study1` run can reuse the whole material set and only pay
    for the selection trials. A manifest is exported for human spot-checking.
    """
    import pandas as pd

    from .data.base_articles import load_base_texts
    from .data.variant_articles import is_hit, load_variant_store, variant_path
    from .studies.common import assemble, get_queries, make_gen_client
    from .studies.design import ofat_pairs

    domains = [d.strip() for d in args.domains.split(",") if d.strip()] or None
    queries = get_queries(args.query_source, per_domain=args.per_domain, domains=domains)
    points = ofat_pairs()
    gen_model = args.model
    gen_client = make_gen_client("llm", True if args.mock else None, dry_run=False)

    base_texts = load_base_texts()

    if args.force:
        removed = 0
        for q in queries:
            path = variant_path(q.id)
            if path.exists():
                path.unlink()
                removed += 1
        print(f"[--force] 已清空 {removed} 个 query 的冻结变体，将全量重生成。")

    pre_store = load_variant_store()

    def _was_reused(article) -> bool:
        rec = pre_store.get(article.id)
        base_text = base_texts.get(article.query_id)
        return bool(rec and base_text and is_hit(rec, gen_model=gen_model, base_text=base_text))

    _, articles_by_id, _ = assemble(
        queries, points, set_size=args.set_size, route="llm",
        gen_client=gen_client, gen_model=gen_model,
        progress=args.progress, concurrency=args.concurrency,
        use_variant_store=True,
    )

    rows = []
    for art in articles_by_id.values():
        meta = art.meta or {}
        rows.append({
            "query_id": art.query_id,
            "article_id": art.id,
            "is_target": art.is_target,
            "role": meta.get("role", ""),
            "target_dim": meta.get("target_dim", ""),
            "generator": meta.get("generator", ""),
            "n_chars": art.n_chars,
            "source": "reused" if _was_reused(art) else "new",
        })
    manifest = pd.DataFrame(rows).sort_values(["query_id", "role", "target_dim"])

    out = PATHS.output_dir / "study1"
    _save(manifest, out / "variants_manifest.csv")

    n_total = len(manifest)
    n_reused = int((manifest["source"] == "reused").sum())
    n_new = n_total - n_reused
    n_fallback = int((manifest["generator"] == "template_fallback").sum())
    llm_chars = manifest.loc[manifest["generator"] == "llm", "n_chars"]
    if len(llm_chars):
        chars_desc = f"min={int(llm_chars.min())} median={int(llm_chars.median())} max={int(llm_chars.max())}"
    else:
        chars_desc = "n/a"
    print(
        f"\n变体总数 {n_total}: 复用 {n_reused} / 新生成 {n_new}"
        f"，仍为模板回退 {n_fallback}(应为 0，非 0 多为空响应)。"
        f"\nllm 变体字数: {chars_desc}"
        f"\n清单 -> {out / 'variants_manifest.csv'}"
        "\n请抽检字数/正文后，再运行 `study1 --gen-route llm` 复用这些冻结变体。"
    )


def cmd_regen_variants(args):
    """Scan the full variant store for truncated/short articles and regen them."""
    import pandas as pd

    from .data.variant_repair import repair_abnormal_variants
    from .studies.common import make_gen_client

    gen_client = make_gen_client("llm", True if args.mock else None, dry_run=False)
    summary = repair_abnormal_variants(
        client=gen_client,
        gen_model=args.model or None,
        min_chars=args.min_chars,
        min_base_ratio=args.min_base_ratio,
        max_attempts=args.max_attempts,
        concurrency=args.concurrency,
        progress=args.progress,
        dry_run=args.dry_run,
    )

    out = PATHS.output_dir / "study1"
    rows = summary.get("rows") or []
    if rows:
        df = pd.DataFrame(rows)
        tag = "regen_scan" if args.dry_run else "regen_report"
        _save(df, out / f"variants_{tag}.csv")

    print(
        f"\n扫描 {summary['scanned']} 篇变体，异常 {summary['abnormal']} 篇 "
        f"(min_chars={summary['min_chars']}, min_base_ratio={summary['min_base_ratio']})"
    )
    if args.dry_run:
        if rows:
            short = df.sort_values("n_chars").head(15)
            print("\n最短 15 篇预览:")
            print(short[["n_chars", "reason", "is_target", "query_id", "text_preview"]].to_string(index=False))
        print(f"\n(dry-run) 清单 -> {out / 'variants_regen_scan.csv'}")
        print("去掉 --dry-run 后将调用大模型重生成并写回 data/variant_articles/")
        return

    print(f"修复成功 {summary.get('fixed', 0)} 篇，仍异常 {summary.get('still_bad', 0)} 篇")
    if summary.get("still_bad"):
        bad = pd.DataFrame(rows)
        bad = bad[bad["status"] != "ok"]
        print(bad[["old_n_chars", "new_n_chars", "status", "query_id", "old_preview"]].head(10).to_string(index=False))
    print(f"\n报告 -> {out / 'variants_regen_report.csv'}")


def cmd_study1(args):
    import pandas as pd

    from .analysis.plots import ate_forest, ei_leverage_bar
    from .studies import run_study1
    from .study_output import models_to_save, persist_per_model, study1_tables

    out_base = PATHS.output_dir / "study1"
    models = _models(args.models)
    sink = _make_output_sink(args, "study1", models)
    if args.progress:
        from .experiment.progress import DEFAULT_PROGRESS_FILE

        pf = Path(args.progress_file) if args.progress_file else DEFAULT_PROGRESS_FILE
        print(f"跨终端进度: {pf}  (另开终端运行: python -m ai_structural_holes.cli watch-progress)")
    res = run_study1(output_sink=sink, **_common_run_kwargs(args))
    if _maybe_dry(res):
        return

    if sink is not None:
        saved = sink.finalize()
    else:
        def _save_model(sub: pd.DataFrame, out: Path) -> None:
            ate, ei = study1_tables(sub)
            _save(sub, out / "trials.csv")
            _save(ate, out / "ate.csv")
            _save(ei, out / "ei_leverage.csv")
            ei_leverage_bar(ei, out / "ei_leverage.png", f"Study 1: EI leverage ({sub['model'].iloc[0]})")
            ate_forest(ate, out / "ate_forest.png", f"Study 1: ATE by feature ({sub['model'].iloc[0]})")

        saved = persist_per_model("study1", res.frame, models, PATHS.output_dir, _save_model)
    if not saved:
        raise SystemExit("没有可保存的 trial 记录，请检查 --models 与运行结果。")

    for model in models_to_save(res.frame, models):
        sub = res.frame[res.frame["model"] == model]
        _, ei = study1_tables(sub)
        print(f"\n[{model}]")
        if ei.empty:
            parse_rate = float(sub["parse_ok"].mean()) if "parse_ok" in sub.columns else float("nan")
            print(
                f"(no EI estimates; parse_ok={parse_rate:.4f} — "
                "check model JSON output before interpreting results)"
            )
        else:
            print(ei[["factor", "EI_norm", "EI_share", "ATE"]].to_string(index=False))
    print(f"\nsaved -> {out_base}/<model>/  ({len(saved)} 个模型)")


def cmd_study1_viz(args):
    from .analysis.study1_viz import build_study1_viz

    study_dir = Path(args.study_dir) if args.study_dir else PATHS.output_dir / "study1"
    out_dir = Path(args.output_dir) if args.output_dir else study_dir / "viz"
    res = build_study1_viz(study_dir=study_dir, out_dir=out_dir)
    print(f"Study 1 可视化已生成 -> {res.html_path}")
    print(f"数据文件: {len(res.data_paths)}  静态图: {len(res.figure_paths)}")


def cmd_study2(args):
    import pandas as pd

    from .analysis.plots import ei_leverage_bar
    from .studies import run_study2
    from .study_output import models_to_save, persist_per_model, study2_tables

    out_base = PATHS.output_dir / "study2"
    models = _models(args.models)
    sink = _make_output_sink(args, "study2", models)
    res = run_study2(n_points=args.n_points, output_sink=sink, **_common_run_kwargs(args))
    if _maybe_dry(res):
        return

    if sink is not None:
        saved = sink.finalize()
    else:
        def _save_model(sub: pd.DataFrame, out: Path) -> None:
            coefs, ei = study2_tables(sub)
            _save(sub, out / "trials.csv")
            _save(coefs, out / "coefficients.csv")
            _save(ei, out / "ei_leverage.csv")
            ei_leverage_bar(ei, out / "ei_leverage.png", f"Study 2: EI leverage ({sub['model'].iloc[0]})")

        saved = persist_per_model("study2", res.frame, models, PATHS.output_dir, _save_model)
    if not saved:
        raise SystemExit("没有可保存的 trial 记录，请检查 --models 与运行结果。")

    for model in models_to_save(res.frame, models):
        coefs, _ = study2_tables(res.frame[res.frame["model"] == model])
        print(f"\n[{model}]")
        print(coefs.head(20).to_string(index=False))
    print(f"\nsaved -> {out_base}/<model>/  ({len(saved)} 个模型)")


def cmd_study3(args):
    import pandas as pd

    from .studies import run_study3
    from .study_output import models_to_save, persist_per_model, study3_tables

    out_base = PATHS.output_dir / "study3"
    models = _models(args.models)
    sink = _make_output_sink(args, "study3", models)
    res = run_study3(output_sink=sink, **_common_run_kwargs(args))
    if _maybe_dry(res):
        return

    if sink is not None:
        saved = sink.finalize()
    else:
        def _save_model(sub: pd.DataFrame, out: Path) -> None:
            ei, ei_by_domain = study3_tables(sub)
            _save(sub, out / "trials.csv")
            _save(ei, out / "ei_leverage.csv")
            if not ei_by_domain.empty:
                _save(ei_by_domain, out / "ei_by_domain.csv")

        saved = persist_per_model("study3", res.frame, models, PATHS.output_dir, _save_model)
    if not saved:
        raise SystemExit("没有可保存的 trial 记录，请检查 --models 与运行结果。")

    print("consistency (全模型汇总):", {
        k: res.consistency.get(k) for k in ("kendall_w", "mean_spearman")
    })
    print(f"\nsaved -> {out_base}/<model>/  ({len(saved)} 个模型)")


def cmd_study4(args):
    import pandas as pd

    from .studies import run_study4
    from .study_output import models_to_save, persist_per_model, study4_tables

    out_base = PATHS.output_dir / "study4"
    models = _models(args.models)
    sink = _make_output_sink(args, "study4", models)
    res = run_study4(output_sink=sink, **_common_run_kwargs(args))
    if _maybe_dry(res):
        return

    if sink is not None:
        saved = sink.finalize()
    else:
        def _save_model(sub: pd.DataFrame, out: Path) -> None:
            deception, delta_ei = study4_tables(sub)
            _save(sub, out / "trials.csv")
            _save(deception, out / "deception.csv")
            _save(delta_ei, out / "delta_ei.csv")

        saved = persist_per_model("study4", res.frame, models, PATHS.output_dir, _save_model)
    if not saved:
        raise SystemExit("没有可保存的 trial 记录，请检查 --models 与运行结果。")

    for model in models_to_save(res.frame, models):
        deception, _ = study4_tables(res.frame[res.frame["model"] == model])
        print(f"\n[{model}]")
        print(deception.to_string(index=False))
    print(f"\nsaved -> {out_base}/<model>/  ({len(saved)} 个模型)")


def cmd_study4_materials(args):
    from .studies import build_study4_materials

    out = PATHS.output_dir / "study4"
    audit = build_study4_materials(
        models=_models(args.models),
        per_domain=args.per_domain,
        set_size=args.set_size,
        mock=True if args.mock else None,
        progress=args.progress,
        gen_route=getattr(args, "gen_route", "llm"),
        gen_model=getattr(args, "gen_model", None) or None,
        query_source=getattr(args, "query_source", "builtin"),
        distractor_route=getattr(args, "distractors", None) or None,
        concurrency=getattr(args, "concurrency", 1),
        use_variant_store=getattr(args, "use_variant_store", True),
    )
    path = _save(audit, out / "audit_sheet.csv")
    _write_review_sheet(out / "audit_sheet.csv", out / "audit_sheet_review.csv")
    n = len(audit)
    if n:
        print(f"生成 {n} 篇目标材料(S1/S3 x none/genuine/fake):")
        print(audit.groupby(["dim", "variant"]).size().to_string())
        if "template_fallback" in set(audit.get("generator", [])):
            print("\n警告: 存在 template_fallback，请检查基线文章/生成配置后重跑。")
    print(
        f"\n审计表 -> {path}"
        f"\n便于核验的分组版 -> {out / 'audit_sheet_review.csv'}"
        "\n请人工逐条核验 verifiable/source_url_or_doi/verdict 后再运行 `study4`。"
    )


def cmd_audit_sheet_review(args):
    """Regroup audit_sheet.csv by query for manual review."""
    audit_path = Path(args.audit_sheet)
    if not audit_path.exists():
        raise SystemExit(f"找不到审计表: {audit_path}")

    out_path = Path(args.output) if args.output else audit_path.parent / "audit_sheet_review.csv"
    ver_path = Path(args.verification) if getattr(args, "verification", "") else None
    result = _write_review_sheet(audit_path, out_path, verification_path=ver_path)
    import pandas as pd

    df = pd.read_csv(result)
    n_groups = df["group_no"].nunique() if "group_no" in df.columns else 0
    print(f"已写出 {len(df)} 行，{n_groups} 题（每题 6 行: S1/S3 × none/genuine/fake）")
    print(f"-> {result}")


def cmd_build_corpus(args):
    """Aggregate + freeze the per-domain real-passage retrieval corpus (Study 5)."""
    from .retrieval.corpus import build_corpus, corpus_dir, load_corpus

    domains = [d.strip() for d in args.domains.split(",") if d.strip()] or None
    counts = build_corpus(domains=domains)

    print(f"{'领域':<18}{'文档数':>8}")
    for dom, n in counts.items():
        flag = "  (空; 先 import-queries)" if n == 0 else ""
        print(f"{dom:<18}{n:>8}{flag}")

    if args.embed:
        from .retrieval.retriever import HybridRetriever

        print("\n预计算并缓存 bge 语料向量(首次会下载模型)...")
        for dom, n in counts.items():
            if n:
                HybridRetriever(load_corpus(dom), mode="hybrid")
                print(f"  [{dom}] 向量已缓存")

    print(
        f"\n语料目录: {corpus_dir()}"
        "\n请人工抽检后提交进版本管理(冻结)。"
        "\n下一步: `study-rag` 复用 Study1 冻结目标文，在真实语料上跑检索+引用。"
    )


def cmd_study_rag(args):
    from .studies import run_study5
    from .study_output import models_to_save, study5_tables, study_model_dir

    out_base = PATHS.output_dir / "study_rag"
    models = _models(args.models)
    sink = _make_output_sink(args, "study_rag", models)

    if args.dry_run:
        from .studies.common import get_queries
        from .studies.design import ofat_pairs

        queries = get_queries(args.query_source, per_domain=args.per_domain)
        n_targets = len(ofat_pairs())
        n_gen = len(queries) * n_targets * args.top_k * len(models) * args.seeds
        print(
            f"(dry-run) 题目 {len(queries)} x 目标 {n_targets} x top_k {args.top_k} "
            f"x 模型 {len(models)} x seeds {args.seeds}\n"
            f"生成阶段约 {n_gen} 次模型调用(检索阶段本地免费; 目标文复用冻结变体，零生成)。"
        )
        return

    res = run_study5(
        models=models,
        per_domain=args.per_domain,
        seeds=tuple(range(args.seeds)),
        top_k=args.top_k,
        retriever=args.retriever,
        alpha=args.alpha,
        query_source=args.query_source,
        mock=True if args.mock else None,
        progress=args.progress,
        concurrency=args.concurrency,
        output_sink=sink,
        use_llm_cache=True if getattr(args, "llm_cache", False) else None,
    )

    # model-independent stage-1 outputs (shared)
    _save(res.reuse_manifest, out_base / "targets_manifest.csv")
    _save(res.retrieval_frame, out_base / "retrieval.csv")
    _save(res.ate_retrieved, out_base / "ate_retrieved.csv")

    src = res.reuse_manifest.get("source")
    if src is not None and len(src):
        n_reused = int((src == "reused").sum())
        n_fallback = int((src == "template_fallback").sum())
        print(f"目标文复用: {n_reused} reused / {n_fallback} template_fallback (应为 0)")

    if res.gen_frame.empty:
        print("\n没有生成阶段 trial(可能语料库为空或候选不足)，仅产出检索结果。")
        print(f"saved -> {out_base}/")
        return

    saved = []
    if sink is not None:
        saved = sink.finalize()
    elif not res.gen_frame.empty:
        for model in models_to_save(res.gen_frame, models):
            sub = res.gen_frame[res.gen_frame["model"] == model].copy()
            if sub.empty:
                continue
            ate_cite, ei, e2e = study5_tables(res.retrieval_frame, sub)
            out = study_model_dir("study_rag", model, PATHS.output_dir)
            _save(sub, out / "trials.csv")
            _save(ate_cite, out / "ate_cite.csv")
            _save(ei, out / "ei_leverage.csv")
            _save(e2e, out / "e2e.csv")
            saved.append(out)

    if not res.ate_retrieved.empty:
        print("\n[检索通道 ATE(retrieved)]")
        print(res.ate_retrieved[["factor", "ATE", "ci_low", "ci_high"]].to_string(index=False))
    print(f"\nsaved -> {out_base}/ (+ {len(saved)} 个模型子目录)")


def cmd_study6(args):
    from .studies import retrieval_by_model, run_study6
    from .study_output import models_to_save, study6_tables, study_model_dir

    out_base = PATHS.output_dir / "study6"
    models = _models(args.models)
    sink = _make_output_sink(args, "study6", models)

    if args.dry_run:
        from .studies.common import get_queries
        from .studies.design import ofat_pairs

        queries = get_queries(args.query_source, per_domain=args.per_domain)
        n_targets = len(ofat_pairs())
        n_rewrite = len(queries) * len(models)
        n_gen = len(queries) * n_targets * args.top_k * len(models) * args.seeds
        print(
            f"(dry-run) 题目 {len(queries)} x 目标 {n_targets} x top_k {args.top_k} "
            f"x 模型 {len(models)} x seeds {args.seeds}\n"
            f"改写阶段约 {n_rewrite} 次模型调用; 生成阶段约 {n_gen} 次模型调用"
            f"(检索阶段本地免费; 目标文复用冻结变体，零生成)。"
        )
        return

    res = run_study6(
        models=models,
        per_domain=args.per_domain,
        seeds=tuple(range(args.seeds)),
        top_k=args.top_k,
        retriever=args.retriever,
        alpha=args.alpha,
        n_queries=args.n_queries,
        fuse=args.fuse,
        query_source=args.query_source,
        mock=True if args.mock else None,
        progress=args.progress,
        concurrency=args.concurrency,
        output_sink=sink,
        use_llm_cache=True if getattr(args, "llm_cache", False) else None,
        resume=getattr(args, "resume", False),
    )

    # model-independent target reuse manifest (shared)
    _save(res.reuse_manifest, out_base / "targets_manifest.csv")

    src = res.reuse_manifest.get("source")
    if src is not None and len(src):
        n_reused = int((src == "reused").sum())
        n_fallback = int((src == "template_fallback").sum())
        print(f"目标文复用: {n_reused} reused / {n_fallback} template_fallback (应为 0)")

    # cross-model retrieval-channel comparison (the headline of Study 6)
    retr_by_model = retrieval_by_model(res.retrieval_frame)
    _save(retr_by_model, out_base / "retrieval_by_model.csv")

    # When resuming, a fully-completed run legitimately produces an empty
    # in-memory gen_frame (every trial was skipped). Fall back to the on-disk
    # trials.csv so we still finalize analysis instead of reporting "no trials".
    disk_has_trials = sink is not None and any(
        (study_model_dir("study6", m, PATHS.output_dir) / "trials.csv").exists()
        for m in models
    )
    if res.gen_frame.empty and not disk_has_trials:
        print("\n没有生成阶段 trial(可能语料库为空或候选不足)，仅产出检索与改写结果。")
        if sink is None:
            _save(res.retrieval_frame, out_base / "retrieval.csv")
            _save(res.rewrite_frame, out_base / "rewrites.csv")
        else:
            for model in models:
                retr_sub = res.retrieval_frame[res.retrieval_frame["model"] == model]
                if not retr_sub.empty:
                    out = study_model_dir("study6", model, PATHS.output_dir)
                    _save(retr_sub, out / "retrieval.csv")
        print(f"saved -> {out_base}/")
        return

    saved = []
    if sink is not None:
        # Re-save per-model retrieval.csv for every model that has trials on disk
        # (not just those with newly-run trials this pass).
        save_models = set(models_to_save(res.gen_frame, models))
        for model in models:
            out = study_model_dir("study6", model, PATHS.output_dir)
            if model not in save_models and not (out / "trials.csv").exists():
                continue
            retr_sub = res.retrieval_frame[res.retrieval_frame["model"] == model].copy()
            if not retr_sub.empty:
                _save(retr_sub, out / "retrieval.csv")
        saved = sink.finalize()
    else:
        for model in models_to_save(res.gen_frame, models):
            gen_sub = res.gen_frame[res.gen_frame["model"] == model].copy()
            retr_sub = res.retrieval_frame[res.retrieval_frame["model"] == model].copy()
            rw_sub = res.rewrite_frame[res.rewrite_frame["model"] == model].copy()
            if gen_sub.empty:
                continue
            ate_retrieved, ate_e2e, ei = study6_tables(retr_sub, gen_sub)
            out = study_model_dir("study6", model, PATHS.output_dir)
            _save(rw_sub, out / "rewrites.csv")
            _save(retr_sub, out / "retrieval.csv")
            _save(ate_retrieved, out / "ate_retrieved.csv")
            _save(gen_sub, out / "trials.csv")
            _save(ate_e2e, out / "ate_e2e.csv")
            _save(ei, out / "ei_leverage.csv")
            saved.append(out)

    if not retr_by_model.empty:
        print("\n[跨模型检索通道 ATE(retrieved) 概览]")
        cols = ["model", "factor", "ATE_retrieve", "recall_overall"]
        print(retr_by_model[cols].to_string(index=False))
    print(f"\nsaved -> {out_base}/ (+ {len(saved)} 个模型子目录)")


def cmd_audit_genuine(args):
    """Verify bibliographic sources in Study 4 genuine variants."""
    from .data.genuine_audit import summarize_verification, verify_genuine_audit_sheet

    audit_path = Path(args.audit_sheet)
    if not audit_path.exists():
        raise SystemExit(f"找不到审计表: {audit_path}\n请先运行 study4-materials。")

    import pandas as pd

    audit = pd.read_csv(audit_path)
    ver = verify_genuine_audit_sheet(audit)
    out_dir = Path(args.output_dir) if args.output_dir else PATHS.output_dir / "study4"
    out_dir.mkdir(parents=True, exist_ok=True)
    ver_path = _save(ver, out_dir / "genuine_source_verification.csv")
    summary = summarize_verification(ver)
    sum_path = _save(summary, out_dir / "genuine_source_verification_summary.csv")

    print(f"已核验 genuine 行: {len(ver)}")
    print("\n按状态汇总:")
    print(summary.to_string(index=False))
    n_reject = int((ver["recommended_verdict"] == "reject_regen").sum())
    n_manual = int(ver["recommended_verdict"].isin([
        "manual_review", "manual_number_check", "manual_expertise_review",
    ]).sum())
    print(
        f"\n建议打回重生成: {n_reject} 行"
        f"\n建议人工继续核对: {n_manual} 行"
        f"\n\n明细 -> {ver_path}"
        f"\n汇总 -> {sum_path}"
    )


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
        sp.add_argument("--gen-route", dest="gen_route", choices=["template", "llm"],
                        default="template",
                        help="文章生成路线: template=离线模板(默认), llm=基于冻结基线文章的大模型受控改写(需先 gen-base)")
        sp.add_argument("--gen-model", dest="gen_model", default=None,
                        help="改写用的模型 slug(默认取 --models 的第一个)")
        sp.add_argument("--query-source", dest="query_source",
                        choices=["builtin", "pool"], default="builtin",
                        help="题目来源: builtin=内置话题(默认), pool=冻结真实题库(需先 import-queries)")
        sp.add_argument("--distractors", choices=["template", "llm", "real"],
                        default=None,
                        help="陪跑文章来源: 默认跟随 --gen-route; real=题库中的真实网页段落")
        sp.add_argument("--concurrency", type=int, default=4,
                        help="并发调用大模型的数量(默认4;设1为串行;遇到限流可调小)")
        sp.add_argument("--no-variant-store", dest="use_variant_store",
                        action="store_false", default=True,
                        help="关闭变体文章库(默认开启): 关闭后不读写 data/variant_articles/，每次都重新改写")
        sp.add_argument("--output-mode", dest="output_mode",
                        choices=["minimal", "full"], default="minimal",
                        help="模型输出格式: minimal=仅 {\"choice\":\"A\"}(默认,更省token); "
                             "full=含 ranking/scores/reason(便于审计)")
        sp.add_argument("--progress-file", dest="progress_file", default=None,
                        help="跨终端进度 JSON 路径(默认 outputs/.run_progress.json)")
        sp.add_argument("--llm-cache", dest="llm_cache", action="store_true",
                        help="开启 LLM 响应磁盘缓存(.cache/llm/，Study 默认关闭)")
        sp.add_argument("--analysis-refresh-every", dest="analysis_refresh_every",
                        type=int, default=100,
                        help="运行中每 N 条 trial 重算分析 CSV(默认 100)")
        sp.add_argument("--analysis-refresh-sec", dest="analysis_refresh_sec",
                        type=float, default=300.0,
                        help="运行中至少每 M 秒重算分析 CSV(默认 300，与条数条件取 OR)")
        sp.add_argument("--no-incremental-output", dest="no_incremental_output",
                        action="store_true",
                        help="调试: 跑完后一次性写 CSV(仍默认关闭 LLM cache)")

    wp = sub.add_parser(
        "watch-progress",
        help="在另一个终端实时查看 study 运行进度(读取 .run_progress.json)",
    )
    wp.add_argument("--file", default="", help="进度文件路径(默认 outputs/.run_progress.json)")
    wp.add_argument("--interval", type=float, default=2.0, help="刷新间隔秒数(默认2)")
    wp.set_defaults(func=cmd_watch_progress)

    iq = sub.add_parser("import-queries",
                        help="从 DuReader 等数据集导入真实问题+真实段落，冻结为题库")
    iq.add_argument("--file", required=True,
                    help="数据文件路径(DuReader 2.0 json-lines 或含 question/passages 的 JSON)")
    iq.add_argument("--per-domain", dest="per_domain", type=int, default=50,
                    help="每领域导入的题目配额(默认 50;导入不花钱,可放心设大)")
    iq.add_argument("--domains", default="", help="逗号分隔的领域列表(默认全部)")
    iq.add_argument("--seed", type=int, default=0, help="抽样随机种子(可复现)")
    iq.set_defaults(func=cmd_import_queries)

    gb = sub.add_parser("gen-base", help="生成并冻结 LLM 基线文章(供 --gen-route llm 使用)")
    gb.add_argument("--model", default="deepseek/deepseek-chat",
                    help="用于写基线文章的模型 slug")
    gb.add_argument("--per-domain", dest="per_domain", type=int, default=2)
    gb.add_argument("--domains", default="", help="逗号分隔的领域列表(默认全部)")
    gb.add_argument("--query-source", dest="query_source",
                    choices=["builtin", "pool"], default="builtin",
                    help="为哪套题目生成基线文章: builtin=内置话题(默认), pool=冻结真实题库")
    gb.add_argument("--max-attempts", dest="max_attempts", type=int, default=5)
    gb.add_argument("--force", action="store_true", help="重新生成已冻结的基线文章")
    gb.add_argument("--mock", action="store_true")
    gb.add_argument("--concurrency", type=int, default=4,
                    help="并发调用大模型的数量(默认4;设1为串行;遇到限流可调小)")
    gb.add_argument("--no-progress", dest="progress", action="store_false",
                    default=True, help="关闭进度条(默认开启)")
    gb.set_defaults(func=cmd_gen_base)

    gv = sub.add_parser("gen-variants",
                        help="复用+重生成并冻结 Study1 的全部 LLM 变体文章(供之后 study1 复用)")
    gv.add_argument("--model", default="deepseek/deepseek-chat",
                    help="改写用的模型 slug(需与之后 study1 的 --gen-model 一致才能命中缓存)")
    gv.add_argument("--per-domain", dest="per_domain", type=int, default=50)
    gv.add_argument("--domains", default="", help="逗号分隔的领域列表(默认全部)")
    gv.add_argument("--set-size", dest="set_size", type=int, default=3)
    gv.add_argument("--query-source", dest="query_source",
                    choices=["builtin", "pool"], default="builtin",
                    help="题目来源: builtin=内置话题(默认), pool=冻结真实题库")
    gv.add_argument("--force", action="store_true",
                    help="忽略已冻结变体，全量重生成")
    gv.add_argument("--mock", action="store_true")
    gv.add_argument("--concurrency", type=int, default=4,
                    help="并发调用大模型的数量(默认4;设1为串行;遇到限流可调小)")
    gv.add_argument("--no-progress", dest="progress", action="store_false",
                    default=True, help="关闭进度条(默认开启)")
    gv.set_defaults(func=cmd_gen_variants)

    rv = sub.add_parser(
        "regen-variants",
        help="扫描全库异常变体(过短/截断)并用大模型重生成写回冻结库",
    )
    rv.add_argument("--model", default="",
                    help="改写模型(默认用每条记录里存的 gen_model)")
    rv.add_argument("--min-chars", dest="min_chars", type=int, default=50,
                    help="低于此字数视为异常并重生成(默认 50)")
    rv.add_argument("--min-base-ratio", dest="min_base_ratio", type=float, default=0.15,
                    help="低于基线文章该比例也视为截断异常(默认 0.15)")
    rv.add_argument("--max-attempts", dest="max_attempts", type=int, default=5)
    rv.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="只扫描列出异常，不调 API")
    rv.add_argument("--mock", action="store_true")
    rv.add_argument("--concurrency", type=int, default=8)
    rv.add_argument("--no-progress", dest="progress", action="store_false", default=True)
    rv.set_defaults(func=cmd_regen_variants)

    s1 = sub.add_parser("study1", help="single-feature paired intervention")
    add_run_args(s1)
    s1.set_defaults(func=cmd_study1)

    s1v = sub.add_parser(
        "study1-viz",
        help="generate the Study1 interactive constellation report and static figures",
    )
    s1v.add_argument(
        "--study-dir",
        default=str(PATHS.output_dir / "study1"),
        help="Study 1 output directory containing per-model subdirectories",
    )
    s1v.add_argument(
        "--output-dir",
        default="",
        help="visualization output directory (default: <study-dir>/viz)",
    )
    s1v.set_defaults(func=cmd_study1_viz)

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

    s4m = sub.add_parser(
        "study4-materials",
        help="只生成并冻结 Study4 的 S1/S3 真假材料并导出人工审计表(不跑选择实验)",
    )
    add_run_args(s4m)
    s4m.set_defaults(func=cmd_study4_materials)

    bc = sub.add_parser(
        "build-corpus",
        help="从冻结题库的真实段落聚合去重，按领域冻结 Study5 检索语料库",
    )
    bc.add_argument("--domains", default="", help="逗号分隔的领域列表(默认全部)")
    bc.add_argument("--embed", action="store_true",
                    help="顺便预计算并缓存 bge 语料向量(首次会下载模型)")
    bc.set_defaults(func=cmd_build_corpus)

    sr = sub.add_parser(
        "study-rag",
        help="Study5: 真实RAG检索环(检索通道+生成引用通道), 复用Study1冻结目标文",
    )
    add_run_args(sr)
    sr.add_argument("--top-k", dest="top_k", type=int, default=8,
                    help="检索返回并作为上下文的文档数(默认8)")
    sr.add_argument("--retriever", choices=["hybrid", "bm25", "dense"],
                    default="hybrid", help="检索器: hybrid=BM25+向量(默认), bm25, dense")
    sr.add_argument("--alpha", type=float, default=0.5,
                    help="混合权重: score=alpha*向量 + (1-alpha)*BM25(默认0.5)")
    # Study5 corpus/targets come from the frozen real pool, not builtin topics.
    sr.set_defaults(func=cmd_study_rag, query_source="pool")

    s6 = sub.add_parser(
        "study6",
        help="Study6: 查询改写驱动的RAG检索(检索因模型而异), 保留Study5固定检索作对照",
    )
    add_run_args(s6)
    s6.add_argument("--top-k", dest="top_k", type=int, default=8,
                    help="检索返回并作为上下文的文档数(默认8)")
    s6.add_argument("--retriever", choices=["hybrid", "bm25", "dense"],
                    default="hybrid", help="检索器: hybrid=BM25+向量(默认), bm25, dense")
    s6.add_argument("--alpha", type=float, default=0.5,
                    help="混合权重: score=alpha*向量 + (1-alpha)*BM25(默认0.5)")
    s6.add_argument("--n-queries", dest="n_queries", type=int, default=3,
                    help="每个问题让模型改写出的检索式上限(默认3)")
    s6.add_argument("--fuse", choices=["max", "mean"], default="max",
                    help="多检索式打分融合: max(默认)=逐文档取最大, mean=取平均")
    s6.add_argument("--resume", action="store_true",
                    help="断点续跑: 复用已保存的 rewrites.csv 与 trials.csv, 只补跑缺失的生成调用"
                         "(不加则重跑即从头, 清空旧 CSV)")
    # Study6 corpus/targets come from the frozen real pool, not builtin topics.
    s6.set_defaults(func=cmd_study6, query_source="pool")

    ag = sub.add_parser(
        "audit-sheet-review",
        help="将 audit_sheet 按题目分组排序，生成便于逐条核验的 audit_sheet_review.csv",
    )
    ag.add_argument(
        "--audit-sheet", dest="audit_sheet",
        default=str(PATHS.output_dir / "study4" / "audit_sheet.csv"),
    )
    ag.add_argument(
        "--output", default="",
        help="输出路径(默认 outputs/study4/audit_sheet_review.csv)",
    )
    ag.add_argument(
        "--verification", default="",
        help="可选: genuine_source_verification.csv(默认自动读取同目录)",
    )
    ag.set_defaults(func=cmd_audit_sheet_review)

    ag = sub.add_parser(
        "audit-genuine",
        help="逐条核验 Study4 audit_sheet 中 genuine 变体的来源真实性",
    )
    ag.add_argument(
        "--audit-sheet", dest="audit_sheet",
        default=str(PATHS.output_dir / "study4" / "audit_sheet.csv"),
        help="audit_sheet.csv 路径",
    )
    ag.add_argument(
        "--output-dir", dest="output_dir", default="",
        help="输出目录(默认 outputs/study4)",
    )
    ag.set_defaults(func=cmd_audit_genuine)

    return p


def main(argv=None):
    warnings.filterwarnings("ignore")
    PATHS.ensure()
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
