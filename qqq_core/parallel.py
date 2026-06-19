"""Small process-pool helpers for CPU-heavy research loops."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import os
from typing import Callable, Iterable, TypeVar


T = TypeVar("T")
R = TypeVar("R")

MAX_WORKERS_ENV_VAR = "QQQ_MAX_WORKERS"
PARALLEL_CHILD_ENV_VAR = "QQQ_PARALLEL_CHILD"


def _parse_workers(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return None
    return max(1, parsed)


def get_max_workers(item_count: int | None = None, default_cap: int = 8) -> int:
    """Return the requested process count, with safe defaults and child fallback."""

    if os.environ.get(PARALLEL_CHILD_ENV_VAR) == "1":
        return 1
    requested = _parse_workers(os.environ.get(MAX_WORKERS_ENV_VAR))
    workers = requested if requested is not None else min(default_cap, os.cpu_count() or 1)
    if item_count is not None:
        workers = min(workers, max(1, int(item_count)))
    return max(1, int(workers))


def should_parallelize(item_count: int, min_items: int = 2) -> bool:
    return int(item_count) >= int(min_items) and get_max_workers(item_count) > 1


def _mark_parallel_child() -> None:
    os.environ[PARALLEL_CHILD_ENV_VAR] = "1"


def ordered_parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int | None = None,
    label: str = "tasks",
) -> list[R]:
    """
    Execute ``func`` for every item, preserving input order.

    Falls back to serial execution when ``QQQ_MAX_WORKERS=1`` or there is only
    one item.  Exceptions propagate to the caller, matching normal loop behavior.
    """

    item_list = list(items)
    if not item_list:
        return []

    workers = max_workers or get_max_workers(len(item_list))
    workers = min(max(1, workers), len(item_list))
    if workers <= 1:
        print(f"[parallel] {label}: serial ({len(item_list)} tasks)")
        return [func(item) for item in item_list]

    print(f"[parallel] {label}: {len(item_list)} tasks with {workers} workers")
    results: list[R | None] = [None] * len(item_list)
    with ProcessPoolExecutor(max_workers=workers, initializer=_mark_parallel_child) as pool:
        future_to_idx = {
            pool.submit(func, item): idx
            for idx, item in enumerate(item_list)
        }
        done = 0
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            results[idx] = future.result()
            done += 1
            print(f"[parallel] {label}: {done}/{len(item_list)} done", flush=True)

    return [result for result in results]  # type: ignore[list-item]
