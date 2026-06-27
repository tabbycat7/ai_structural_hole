"""Study 4 - reverse / adversarial: genuine vs fabricated features.

For fakeable dimensions (S1 evidence, S3 expertise), build three target versions
per query: none (baseline), genuine (verifiable), fake (fabricated but
marker-bearing). Compares selection rates to quantify deception gain, the model's
discount on fakery, and a vulnerability index; also computes Delta-EI between the
fake and genuine routes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from ..analysis.metrics import deception_gain
from ..causal.ei import ei_from_do_table
from ..codebook import baseline_profile, get_dimension
from ..data.generation import make_article, make_queries
from ..data.schema import Article, CandidateSet
from ..experiment.runner import ExperimentRunner
from ..task.protocol import build_candidate_sets
from .common import make_distractor_pool

FAKEABLE_DIMS = ["S1", "S3"]


@dataclass
class Study4Result:
    frame: pd.DataFrame
    deception: pd.DataFrame
    delta_ei: pd.DataFrame


def _build(queries, fakeable_dims, set_size, seed):
    import random

    rng = random.Random(seed)
    q_by_id = {q.id: q for q in queries}
    a_by_id = {}
    sets = []
    for q in queries:
        distractors = make_distractor_pool(q, n=4, rng=rng)
        for d in distractors:
            a_by_id[d.id] = d
        for dim in fakeable_dims:
            top = get_dimension(dim).top_code()
            variants = {
                "none": (baseline_profile(), "genuine"),
                "genuine": ({**baseline_profile(), dim: top}, "genuine"),
                "fake": ({**baseline_profile(), dim: top}, "fake"),
            }
            for vname, (prof, auth) in variants.items():
                tgt = make_article(
                    q, prof, is_target=True, authenticity=auth,
                    suffix=f"{dim}-{vname}",
                    meta={"fakeable_dim": dim, "variant": vname},
                )
                a_by_id[tgt.id] = tgt
                sets.extend(
                    build_candidate_sets(
                        tgt, distractors, set_size=set_size,
                        counterbalance="all_positions", rng=rng,
                    )
                )
    return q_by_id, a_by_id, sets


def run_study4(
    models: Sequence[str],
    per_domain: int = 1,
    domains: Optional[Sequence[str]] = None,
    fakeable_dims: Sequence[str] = tuple(FAKEABLE_DIMS),
    prompt_styles: Sequence[str] = ("neutral", "critical_eval"),
    seeds: Sequence[int] = (0,),
    set_size: int = 3,
    mock: Optional[bool] = None,
    progress: bool = False,
    dry_run: bool = False,
    price_in: float = 2.0,
    price_out: float = 6.0,
):
    queries = make_queries(per_domain=per_domain, domains=domains)
    q_by_id, a_by_id, sets = _build(queries, list(fakeable_dims), set_size, seed=0)

    if dry_run:
        from ..experiment.planning import compute_plan

        return compute_plan(
            "study4", q_by_id, a_by_id, sets,
            n_models=len(list(models)), n_prompt_styles=len(list(prompt_styles)),
            n_seeds=len(list(seeds)), price_in=price_in, price_out=price_out,
        )

    runner = ExperimentRunner(
        models=models, prompt_styles=prompt_styles, seeds=seeds, mock=mock
    )
    frame = runner.run(q_by_id, a_by_id, sets, progress=progress, desc="Study4 反向对抗")

    # deception metrics per fakeable dimension
    dec_rows = []
    delta_rows = []
    for dim in fakeable_dims:
        sub = frame[frame.get("fakeable_dim", "") == dim]
        if sub.empty:
            continue
        dec_rows.append(deception_gain(sub, dim))

        # Delta-EI: EI of fake route vs genuine route (P(Y|do(dim)) per route)
        gen_df = sub[sub["variant"].isin(["none", "genuine"])]
        fake_df = sub[sub["variant"].isin(["none", "fake"])]

        def _ei_for(d):
            p0 = d[d["variant"] == "none"]["y"].mean()
            p1 = d[d["variant"] != "none"]["y"].mean()
            if pd.isna(p0) or pd.isna(p1):
                return float("nan")
            return ei_from_do_table({0: float(p0), 1: float(p1)}).ei

        ei_gen = _ei_for(gen_df)
        ei_fake = _ei_for(fake_df)
        delta_rows.append({
            "factor": dim,
            "EI_genuine": ei_gen,
            "EI_fake": ei_fake,
            "delta_EI": (ei_fake - ei_gen) if not (pd.isna(ei_fake) or pd.isna(ei_gen)) else float("nan"),
        })

    return Study4Result(
        frame=frame,
        deception=pd.DataFrame(dec_rows),
        delta_ei=pd.DataFrame(delta_rows),
    )
