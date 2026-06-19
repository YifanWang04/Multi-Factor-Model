"""Pickle-backed cache for expensive workbook reads and derived DataFrames."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Callable, Iterable, TypeVar

from .paths import ProjectPaths


T = TypeVar("T")

CACHE_DIR_ENV_VAR = "QQQ_CACHE_DIR"
DISABLE_CACHE_ENV_VAR = "QQQ_DISABLE_CACHE"

_STATS = {"hits": 0, "misses": 0, "disabled": 0}


def cache_enabled() -> bool:
    return os.environ.get(DISABLE_CACHE_ENV_VAR, "").strip().lower() not in {"1", "true", "yes", "on"}


def get_cache_dir() -> Path:
    env_value = os.environ.get(CACHE_DIR_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser().resolve()
    return ProjectPaths.from_env().output_dir / "cache"


def _file_fingerprint(path: str | os.PathLike[str]) -> dict:
    p = Path(path).resolve()
    stat = p.stat()
    return {
        "path": str(p),
        "mtime_ns": stat.st_mtime_ns,
        "size": stat.st_size,
    }


def _context_payload() -> dict:
    keys = [
        "REBALANCE_OFFSET_DAYS",
        "REBALANCE_TICKER_UNIVERSE",
        "YFINANCE_TICKER_UNIVERSE",
        "QQQ_STRATEGY_PROFILE",
    ]
    return {key: os.environ.get(key) for key in keys}


def _cache_key(label: str, paths: Iterable[str | os.PathLike[str]], params: dict | None) -> str:
    payload = {
        "label": label,
        "files": [_file_fingerprint(path) for path in paths],
        "params": params or {},
        "context": _context_payload(),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def load_or_compute(
    label: str,
    paths: Iterable[str | os.PathLike[str]],
    params: dict | None,
    factory: Callable[[], T],
) -> T:
    """Return cached value when the source file fingerprints and params match."""

    if not cache_enabled():
        _STATS["disabled"] += 1
        return factory()

    cache_dir = get_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(label, paths, params)
    cache_path = cache_dir / f"{label}_{key}.pkl"

    if cache_path.is_file():
        try:
            with cache_path.open("rb") as fh:
                value = pickle.load(fh)
            _STATS["hits"] += 1
            return value
        except Exception:
            pass

    value = factory()
    _STATS["misses"] += 1
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pkl",
            delete=False,
            dir=cache_dir,
        ) as tmp:
            tmp_name = tmp.name
            pickle.dump(value, tmp, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp_name, cache_path)
        tmp_name = None
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.remove(tmp_name)
    return value


def cache_stats() -> dict[str, int]:
    return dict(_STATS)


def print_cache_summary(prefix: str = "[cache]") -> None:
    stats = cache_stats()
    print(
        f"{prefix} hits={stats['hits']} misses={stats['misses']} "
        f"disabled={stats['disabled']} dir={get_cache_dir()}"
    )
