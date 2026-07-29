"""Single source of truth for strategy profile selection.

This module owns the live/research profile identity used by strategy,
composite-factor, and rebalance-day entry points. It intentionally lives under
``qqq_config`` instead of ``config`` to avoid colliding with legacy
``analysis/single_factor/config.py`` imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import os
import re
from typing import Mapping

from qqq_config.ticker_universes import (
    NASDAQ_100_LAST_6_YEARS,
    ORIGINAL_108,
    ORIGINAL_108_PLUS_NASDAQ_100,
    ORIGINAL_143,
    TICKER_UNIVERSES,
)


@dataclass(frozen=True)
class StrategyProfile:
    """Core strategy configuration shared across entry points."""

    name: str
    factor_indices: tuple[int, ...]
    composite_sheet: str
    strategy_param: str
    ticker_universe: str
    # Exact yfinance download start date for rebalance-day runs.
    data_download_start_date: str | None = None
    description: str = ""
    max_weight: float = 0.4
    preserve_price_scale: bool = False
    price_scale_base_run_dir: str | None = None
    exit_policy: str = "fixed_rebalance"
    tp_base: float = 0.08
    sl_base: float = 0.05
    tp_sl_probability: float = 1.0

    def __post_init__(self) -> None:
        if self.data_download_start_date is None:
            return
        try:
            date.fromisoformat(self.data_download_start_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid data_download_start_date for profile {self.name!r}: "
                f"{self.data_download_start_date!r}; expected YYYY-MM-DD"
            ) from exc

    @property
    def ticker_symbols(self) -> tuple[str, ...]:
        try:
            return TICKER_UNIVERSES[self.ticker_universe]
        except KeyError as exc:
            available = ", ".join(sorted(TICKER_UNIVERSES))
            raise KeyError(
                f"Unknown ticker universe {self.ticker_universe!r} for profile "
                f"{self.name!r}. Available universes: {available}"
            ) from exc

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(f"alpha{i:03d}" for i in self.factor_indices)

    @property
    def factor_suffix(self) -> str:
        return "f" + "-".join(str(int(i)) for i in self.factor_indices)

    @property
    def rebalance_period(self) -> int:
        match = re.search(r"_P(\d+)d$", self.strategy_param.strip())
        if not match:
            raise ValueError(
                f"Invalid strategy_param for profile {self.name!r}: "
                f"{self.strategy_param!r}"
            )
        return int(match.group(1))


STRATEGY_PROFILES: Mapping[str, StrategyProfile] = {
    "Strategy1": StrategyProfile(
        name="Strategy1",
        factor_indices=(95, 101, 62, 65, 32),
        composite_sheet="ic_m3_N20",
        strategy_param="max_return_5G_Top1_P10d",
        ticker_universe="ORIGINAL_108",
        data_download_start_date="2023-01-01",
        max_weight=0.4,
        preserve_price_scale=True,
        price_scale_base_run_dir=None,
        exit_policy="fixed_rebalance",
        # tp_base=0.08,
        # sl_base=0.05,
        # tp_sl_probability=1.0,
        description="Strategy1 with fixed_rebalance profile.",
    ),    
    "Strategy11": StrategyProfile(
        name="Strategy11",
        factor_indices=(95, 101, 62, 65, 32),
        composite_sheet="ic_m3_N20",
        strategy_param="max_return_5G_Top1_P10d",
        ticker_universe="ORIGINAL_108",
        data_download_start_date="2023-01-01",
        max_weight=0.6,
        preserve_price_scale=True,
        price_scale_base_run_dir=r"D:\qqq\output\rebalance_runs\2026-06-24_155408_strategy11_offset0",
        exit_policy="fixed_rebalance",
        # tp_base=0.8,
        # sl_base=0.65,
        # tp_sl_probability=1.0,
        description="Strategy1 with 0.6 max_weight",
    ),
    ## 从2020年回测近六年的数据，使用纳斯达克100+初始的108支股票
    "Strategy12": StrategyProfile(
        name="Strategy12",
        factor_indices=(95, 101, 62, 65, 32),
        composite_sheet="ic_m3_N20",
        strategy_param="max_return_5G_Top1_P10d",
        ticker_universe="ORIGINAL_108_PLUS_NASDAQ_100",
        data_download_start_date="2020-01-01",
        max_weight=0.6,
        preserve_price_scale=False,
        price_scale_base_run_dir=None,
        exit_policy="fixed_rebalance",
        # tp_base=0.8,
        # sl_base=0.65,
        # tp_sl_probability=1.0,
        description="Strategy11 with NASQAQ100 tickers added to the universe.",
    ),
    "Strategy13": StrategyProfile(
        name="Strategy13",
        factor_indices=(95, 101, 62, 65, 32),
        composite_sheet="ic_m3_N20",
        strategy_param="max_return_5G_Top1_P10d",
        ticker_universe="ORIGINAL_108_PLUS_ROBOTICS",
        data_download_start_date="2023-01-01",
        max_weight=0.6,
        preserve_price_scale=True,
        price_scale_base_run_dir=r"D:\qqq\output\rebalance_runs\2026-06-24_155408_strategy11_offset0",
        exit_policy="fixed_rebalance",
        # tp_base=0.8,
        # sl_base=0.65,
        # tp_sl_probability=1.0,
        description="Strategy11 with ROBOTICS tickers added to the universe.",
    ),
    "Strategy2": StrategyProfile(
        name="Strategy2",
        factor_indices=(95, 24, 64, 65, 32),
        composite_sheet="ic_m3_N20",
        strategy_param="max_return_10G_Top1_P20d",
        ticker_universe="ORIGINAL_108",
        data_download_start_date="2023-01-01",
        max_weight=0.4,
        description="Strategy2 Legacy 2026-03/25 research profile.",
    ),
    "Strategy3": StrategyProfile(
        name="Strategy3",
        factor_indices=(95, 99, 27, 75, 19),
        composite_sheet="ic_m3_N20",
        strategy_param="max_return_5G_Top2_P20d",
        ticker_universe="ORIGINAL_143",
        data_download_start_date="2023-01-01",
        max_weight=0.4,
        description="Strategy3 June 2026 strategy profile.",
    ),
    "Strategy4": StrategyProfile(
        name="Strategy4",
        factor_indices=(95, 99, 27, 46, 19),
        composite_sheet="ic_m3_N10",
        strategy_param="max_return_5G_Top2_P20d",
        ticker_universe="ORIGINAL_143",
        data_download_start_date="2023-01-01",
        max_weight=0.4,
        description="Strategy4 June 2026 strategy profile.",
    ),
}


ACTIVE_STRATEGY_PROFILE = "Strategy11"
PROFILE_ENV_VAR = "QQQ_STRATEGY_PROFILE"


def get_strategy_profile(name: str | None = None) -> StrategyProfile:
    """Return the requested strategy profile, defaulting to the active profile."""

    profile_name = name or os.environ.get(PROFILE_ENV_VAR) or ACTIVE_STRATEGY_PROFILE
    try:
        return STRATEGY_PROFILES[profile_name]
    except KeyError as exc:
        available = ", ".join(sorted(STRATEGY_PROFILES))
        raise KeyError(
            f"Unknown strategy profile {profile_name!r}. Available profiles: {available}"
        ) from exc


def get_active_profile() -> StrategyProfile:
    """Return the active strategy profile, honoring QQQ_STRATEGY_PROFILE."""

    return get_strategy_profile()


def parse_factor_indices_csv(value: str) -> tuple[int, ...]:
    """Parse a comma-separated factor index list from environment/CLI input."""

    indices = tuple(int(x.strip()) for x in value.split(",") if x.strip())
    if not indices:
        raise ValueError("Factor index list is empty")
    return indices
