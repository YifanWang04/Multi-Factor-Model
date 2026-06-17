"""Shared naming and parsing helpers for factor and strategy parameters."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Mapping


def build_factor_suffix(
    factor_indices: list[int] | tuple[int, ...] | None = None,
    default_indices: list[int] | tuple[int, ...] | None = None,
) -> str:
    """Build a stable factor suffix such as ``f95-101-62-65-32``."""

    indices = factor_indices if factor_indices is not None else default_indices
    if indices is None:
        return ""
    return "f" + "-".join(str(int(i)) for i in indices)


def composite_factors_path(
    base_dir: str | os.PathLike[str],
    factor_indices: list[int] | tuple[int, ...],
) -> str:
    """Return the composite-factor workbook path for ``factor_indices``.

    ``base_dir`` may be either a run directory or an existing
    ``composite_factor_reports`` directory. This preserves the historical
    rebalance-run layout while centralizing the naming rule.
    """

    base = Path(base_dir)
    name = f"composite_factors_{build_factor_suffix(factor_indices)}.xlsx"
    if base.name == "composite_factor_reports":
        return str(base / name)
    return str(base / "composite_factor_reports" / name)


def parse_strategy_param(param: str) -> tuple[str, int, int, int]:
    """Parse ``{weight_method}_{N}G_Top{R}_P{D}d`` strategy parameters."""

    match = re.match(r"^(.+)_(\d+)G_Top(\d+)_P(\d+)d$", param.strip())
    if not match:
        raise ValueError(
            f"策略参数格式错误: '{param}'，应为 "
            "{weight_method}_{N}G_Top{R}_P{D}d，"
            "例：max_return_10G_Top1_P10d"
        )
    return (
        match.group(1),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )


def strategy_param_from_params(params: Mapping[str, object]) -> str:
    """Build a strategy parameter string from parsed parameter fields."""

    weight_method = params.get("weight_method", "")
    group_num = params.get("group_num", "")
    target_rank = params.get("target_rank", "")
    rebalance_period = params.get("rebalance_period", "")
    if (
        weight_method != ""
        and group_num != ""
        and target_rank != ""
        and rebalance_period != ""
    ):
        return f"{weight_method}_{group_num}G_Top{target_rank}_P{rebalance_period}d"
    return ""


def safe_tag(value: object) -> str:
    """Convert ``value`` into a filesystem-friendly tag."""

    text = str(value).strip().replace(" ", "")
    return "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in text)
