"""Single source of truth for strategy profile selection.

This module owns the live/research profile identity used by strategy,
composite-factor, and rebalance-day entry points. It intentionally lives under
``qqq_config`` instead of ``config`` to avoid colliding with legacy
``analysis/single_factor/config.py`` imports.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import re
from typing import Mapping


YFINANCE_TICKERS_US_108: tuple[str, ...] = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "TSLA", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "XOM", "LLY", "MRK", "ABBV", "PEP",
    "KO", "AVGO", "COST", "WMT", "BAC", "MCD", "CSCO", "ADBE", "CRM", "NFLX",
    "ORCL", "ACN", "TMO", "ABT", "CVX", "DHR", "TXN", "VZ", "NEE", "PM",
    "INTC", "QCOM", "HON", "IBM", "AMD", "LIN", "LOW", "GS", "MS", "UPS",
    "RTX", "SPGI", "CAT", "AMGN", "INTU", "DE", "ISRG", "MDT", "AXP", "BLK",
    "NOW", "LMT", "SCHW", "BA", "CB", "PLD", "BKNG", "CI", "TGT",
    "MO", "GE", "ADI", "GILD", "SYK", "EL", "ZTS", "USB", "PGR", "SO",
    "DUK", "CME", "APD", "BDX", "ITW", "EW", "CSX", "NSC", "CCJ", "SVM",
    "WPM", "PAAS", "TSM", "MU", "PLTR", "WDC", "STX", "VRT",
    "TER", "AEP", "TTMI", "RKLB", "ASTS", "SNDK", "RMBS", "ONDS", "HROW",
    "SANM", "ANET",
)

YFINANCE_TICKERS_US_143: tuple[str, ...] = (
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "BRK-B", "TSLA", "JPM", "JNJ",
    "V", "PG", "UNH", "HD", "MA", "XOM", "LLY", "MRK", "ABBV", "PEP",
    "KO", "AVGO", "COST", "WMT", "BAC", "MCD", "CSCO", "ADBE", "CRM", "NFLX",
    "ORCL", "ACN", "TMO", "ABT", "CVX", "DHR", "TXN", "VZ", "NEE", "PM",
    "INTC", "QCOM", "HON", "IBM", "AMD", "LIN", "LOW", "GS", "MS", "UPS",
    "RTX", "SPGI", "CAT", "AMGN", "INTU", "DE", "ISRG", "MDT", "AXP", "BLK",
    "NOW", "LMT", "SCHW", "BA", "CB", "PLD", "BKNG", "CI", "TGT",
    "MO", "GE", "ADI", "GILD", "SYK", "EL", "ZTS", "USB", "PGR", "SO",
    "DUK", "CME", "APD", "BDX", "ITW", "EW", "CSX", "NSC", "CCJ", "SVM",
    "WPM", "PAAS", "TSM", "MU", "PLTR", "WDC", "STX", "VRT",
    "TER", "AEP", "TTMI", "RKLB", "ASTS", "SNDK", "RMBS", "ONDS", "HROW",
    "SANM", "ANET",
    "AMAT", "LRCX", "CRDO", "ARM", "AAOI", "MRVL", "NBIS",
    "BN", "FN", "COHR", "FLY", "RDW", "GLW", "DELL",
    "HPE", "ALAB", "CIEN", "LITE", "MTSI", "ASML", "SNPS", "CDNS",
    "ETN", "GEV", "PWR", "CLS", "JBL", "FLEX", "FIX", "DDOG", "NET",
    "MDB", "PANW", "CRWD",
    "KLAC",  # June 12, 2026 1-for-10 split noted in research comments.
)

TICKER_UNIVERSES: Mapping[str, tuple[str, ...]] = {
    "US_108": YFINANCE_TICKERS_US_108,
    "US_143": YFINANCE_TICKERS_US_143,
}


@dataclass(frozen=True)
class StrategyProfile:
    """Core strategy configuration shared across entry points."""

    name: str
    factor_indices: tuple[int, ...]
    composite_sheet: str
    strategy_param: str
    ticker_universe: str
    description: str = ""

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
        ticker_universe="US_108",
        description="Strategy1 75 2.6_annual_return Legacy 2026-03/17 live profile.",
    ),
    "Strategy2": StrategyProfile(
        name="Strategy2",
        factor_indices=(95, 24, 64, 65, 32),
        composite_sheet="ic_m3_N20",
        strategy_param="max_return_10G_Top1_P20d",
        ticker_universe="US_108",
        description="Strategy2 Legacy 2026-03/25 research profile.",
    ),
    "Strategy3": StrategyProfile(
        name="Strategy3",
        factor_indices=(95, 99, 27, 75, 19),
        composite_sheet="ic_m3_N20",
        strategy_param="max_return_5G_Top2_P20d",
        ticker_universe="US_143",
        description="Strategy3 June 2026 strategy profile.",
    ),
    "Strategy4": StrategyProfile(
        name="Strategy4",
        factor_indices=(95, 99, 27, 46, 19),
        composite_sheet="ic_m3_N10",
        strategy_param="max_return_5G_Top2_P20d",
        ticker_universe="US_143",
        description="Strategy4 June 2026 strategy profile.",
    ),
    "Strategy5": StrategyProfile(
        name="Strategy5",
        factor_indices=(95, 99, 27, 19, 46),
        composite_sheet="ic_m3_N10",
        strategy_param="max_return_5G_Top2_P20d",
        ticker_universe="US_143",
        description="Strategy5 June 2026 strategy profile.",
    ),
}


ACTIVE_STRATEGY_PROFILE = "Strategy1"
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
