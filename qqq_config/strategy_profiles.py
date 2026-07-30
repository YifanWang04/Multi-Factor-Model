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
    # Optional fixed-week rebalance schedule. All three fields must be set
    # together. When unset, P{N}d keeps its strict N-trading-day meaning.
    rebalance_interval_weeks: int | None = None
    rebalance_weekday: int | None = None
    rebalance_week_anchor_date: str | None = None

    def __post_init__(self) -> None:
        if self.data_download_start_date is not None:
            try:
                date.fromisoformat(self.data_download_start_date)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid data_download_start_date for profile {self.name!r}: "
                    f"{self.data_download_start_date!r}; expected YYYY-MM-DD"
                ) from exc

        fixed_values = (
            self.rebalance_interval_weeks,
            self.rebalance_weekday,
            self.rebalance_week_anchor_date,
        )
        configured_count = sum(value is not None for value in fixed_values)
        if configured_count == 0:
            return
        if configured_count != len(fixed_values):
            raise ValueError(
                f"Profile {self.name!r} must set rebalance_interval_weeks, "
                "rebalance_weekday, and rebalance_week_anchor_date together"
            )

        if (
            not isinstance(self.rebalance_interval_weeks, int)
            or isinstance(self.rebalance_interval_weeks, bool)
            or self.rebalance_interval_weeks <= 0
        ):
            raise ValueError(
                f"Invalid rebalance_interval_weeks for profile {self.name!r}: "
                f"{self.rebalance_interval_weeks!r}; expected a positive integer"
            )
        if (
            not isinstance(self.rebalance_weekday, int)
            or isinstance(self.rebalance_weekday, bool)
            or not 1 <= self.rebalance_weekday <= 5
        ):
            raise ValueError(
                f"Invalid rebalance_weekday for profile {self.name!r}: "
                f"{self.rebalance_weekday!r}; expected one integer from 1 to 5"
            )

        try:
            week_anchor = date.fromisoformat(self.rebalance_week_anchor_date)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid rebalance_week_anchor_date for profile {self.name!r}: "
                f"{self.rebalance_week_anchor_date!r}; expected YYYY-MM-DD"
            ) from exc
        if week_anchor.isoweekday() != self.rebalance_weekday:
            raise ValueError(
                f"Profile {self.name!r} week anchor "
                f"{self.rebalance_week_anchor_date} is weekday "
                f"{week_anchor.isoweekday()}, not configured weekday "
                f"{self.rebalance_weekday}"
            )

        expected_period = self.rebalance_interval_weeks * 5
        if self.rebalance_period != expected_period:
            raise ValueError(
                f"Profile {self.name!r} uses P{self.rebalance_period}d but its "
                f"fixed-week schedule implies P{expected_period}d "
                f"({self.rebalance_interval_weeks} weeks x 5)"
            )

    @property
    def uses_fixed_week_rebalance(self) -> bool:
        return self.rebalance_interval_weeks is not None

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
    # 每两周的周五进行调仓
    "Strategy111": StrategyProfile(
        name="Strategy111",
        factor_indices=(95, 101, 62, 65, 32),
        composite_sheet="ic_m3_N20",
        strategy_param="max_return_5G_Top1_P10d",
        ticker_universe="ORIGINAL_108",
        data_download_start_date="2023-01-01",
        max_weight=0.6,
        preserve_price_scale=True,
        price_scale_base_run_dir=r"D:\qqq\output\rebalance_runs\2026-06-24_155408_strategy11_offset0",
        exit_policy="fixed_rebalance",
        rebalance_interval_weeks=2,
        rebalance_weekday=5,
        rebalance_week_anchor_date="2026-06-26",
        # tp_base=0.8,
        # sl_base=0.65,
        # tp_sl_probability=1.0,
        description="Strategy1 with 0.6 max_weight and biweekly Friday rebalancing.",
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
        factor_indices=(23, 60, 20, 10, 51),
        composite_sheet="ic_m1",
        strategy_param="max_return_5G_Top1_P5d",
        ticker_universe="ORIGINAL_108",
        data_download_start_date="2023-01-01",
        max_weight=0.6,
        preserve_price_scale=True,
        price_scale_base_run_dir=r"D:\qqq\output\rebalance_runs\2026-06-24_155408_strategy11_offset0",
        exit_policy="dynamic_tp_sl",
        tp_base=0.9,
        sl_base=0.65,
        tp_sl_probability=1.0,
        description="Strategy4",
    ),
}


ACTIVE_STRATEGY_PROFILE = "Strategy4"
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
