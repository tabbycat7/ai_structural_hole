"""Pre-run estimation: how many API calls a study will make, and rough cost.

`compute_plan` reuses the *exact* assembled candidate sets a study built, so the
call count matches the real run precisely (the runner iterates
product(candidate_sets, models, prompt_styles, seeds)). Token/cost numbers are
rough estimates from sampled prompts and user-supplied prices; actual per-model
prices vary (check https://openrouter.ai/models).
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Sequence

from ..data.schema import Article, CandidateSet, Query
from ..task.prompts import build_messages

# Rough heuristics (transparent, adjustable).
TOKENS_PER_CHAR = 0.7          # mixed Chinese + markup
DEFAULT_OUTPUT_TOKENS_MINIMAL = 16
DEFAULT_OUTPUT_TOKENS_FULL = 150
DEFAULT_PRICE_IN = 2.0         # USD per 1M input tokens (blended guess)
DEFAULT_PRICE_OUT = 6.0        # USD per 1M output tokens (blended guess)


@dataclass
class CallPlan:
    study: str
    n_queries: int
    n_candidate_sets: int
    n_models: int
    n_prompt_styles: int
    n_seeds: int
    n_calls: int
    avg_input_tokens: int
    est_input_tokens_total: int
    est_output_tokens_total: int
    price_in: float
    price_out: float
    est_cost_usd: float

    def render(self) -> str:
        return (
            f"[DRY-RUN] {self.study}\n"
            f"  查询数 (queries)        : {self.n_queries}\n"
            f"  候选集数 (candidate sets): {self.n_candidate_sets}\n"
            f"  模型数 x 提示风格 x 重复 : {self.n_models} x {self.n_prompt_styles} x {self.n_seeds}\n"
            f"  => API 调用次数          : {self.n_calls}\n"
            f"  单次平均输入 tokens(估)  : ~{self.avg_input_tokens}\n"
            f"  总输入/输出 tokens(估)   : ~{self.est_input_tokens_total:,} / ~{self.est_output_tokens_total:,}\n"
            f"  价格假设 (in/out, $/1M)  : {self.price_in} / {self.price_out}\n"
            f"  预计费用(估, 未计缓存)   : ~${self.est_cost_usd:.2f} USD\n"
            f"  注: 实际价格随模型而异(见 openrouter.ai/models);缓存命中会更便宜。"
        )


def compute_plan(
    study: str,
    queries_by_id: Dict[str, Query],
    articles_by_id: Dict[str, Article],
    candidate_sets: Sequence[CandidateSet],
    n_models: int,
    n_prompt_styles: int,
    n_seeds: int,
    price_in: float = DEFAULT_PRICE_IN,
    price_out: float = DEFAULT_PRICE_OUT,
    sample: int = 24,
    seed: int = 0,
    output_mode: str = "minimal",
) -> CallPlan:
    n_calls = len(candidate_sets) * n_models * n_prompt_styles * n_seeds

    # estimate average input tokens from a sample of real prompts
    rng = random.Random(seed)
    sets = list(candidate_sets)
    rng.shuffle(sets)
    sample_sets = sets[: min(sample, len(sets))]
    char_counts: List[int] = []
    for cs in sample_sets:
        q = queries_by_id.get(cs.query_id)
        if q is None:
            continue
        texts = [articles_by_id[cid].text for cid in cs.ordered_ids]
        msgs, _ = build_messages(q.text, texts, "neutral", q.domain, output_mode=output_mode)
        char_counts.append(sum(len(m["content"]) for m in msgs))
    avg_chars = (sum(char_counts) / len(char_counts)) if char_counts else 0.0
    avg_input_tokens = int(avg_chars * TOKENS_PER_CHAR)

    out_per_call = (
        DEFAULT_OUTPUT_TOKENS_MINIMAL if output_mode == "minimal" else DEFAULT_OUTPUT_TOKENS_FULL
    )
    est_in = avg_input_tokens * n_calls
    est_out = out_per_call * n_calls
    cost = est_in / 1e6 * price_in + est_out / 1e6 * price_out

    n_queries = len(queries_by_id)
    return CallPlan(
        study=study,
        n_queries=n_queries,
        n_candidate_sets=len(candidate_sets),
        n_models=n_models,
        n_prompt_styles=n_prompt_styles,
        n_seeds=n_seeds,
        n_calls=n_calls,
        avg_input_tokens=avg_input_tokens,
        est_input_tokens_total=est_in,
        est_output_tokens_total=est_out,
        price_in=price_in,
        price_out=price_out,
        est_cost_usd=cost,
    )
