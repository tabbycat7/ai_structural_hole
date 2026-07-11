"""Bounded-concurrency map for parallelizing LLM calls.

`map_concurrent` runs `fn` over `items` using a thread pool (the OpenAI SDK is
synchronous and I/O-bound, so threads give full speedup). Results are always
returned in the *input* order, so downstream reassembly stays reproducible
regardless of the concurrency level. The optional `on_result` callback runs in
the calling thread as each item finishes, so it is safe to use for tqdm updates
and counter bookkeeping without extra locking.
"""
from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, TypeVar

T = TypeVar("T")
R = TypeVar("R")


def map_concurrent(
    fn: Callable[[T], R],
    items: Iterable[T],
    *,
    concurrency: int = 1,
    on_submit: Optional[Callable[[int, T], None]] = None,
    on_result: Optional[Callable[[int, T, R], None]] = None,
) -> List[R]:
    """Apply `fn` to each item; return results in input order.

    concurrency <= 1 runs sequentially (identical behaviour, no threads).
    `on_result(index, item, result)` is invoked from the calling thread as each
    result becomes available (order of invocation may differ under concurrency,
    but the returned list is always in input order).
    """
    seq: Sequence[T] = list(items)
    results: List[Optional[R]] = [None] * len(seq)

    if concurrency <= 1 or len(seq) <= 1:
        for i, item in enumerate(seq):
            res = fn(item)
            results[i] = res
            if on_result is not None:
                on_result(i, item, res)
        return results  # type: ignore[return-value]

    from concurrent.futures import ThreadPoolExecutor, as_completed

    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        fut_to_idx = {}
        for i, item in enumerate(seq):
            if on_submit is not None:
                on_submit(i, item)
            fut_to_idx[ex.submit(fn, item)] = i
        for fut in as_completed(fut_to_idx):
            i = fut_to_idx[fut]
            res = fut.result()
            results[i] = res
            if on_result is not None:
                on_result(i, seq[i], res)

    return results  # type: ignore[return-value]
