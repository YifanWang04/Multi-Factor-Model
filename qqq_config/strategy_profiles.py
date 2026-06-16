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


@dataclass(frozen=True)
class StrategyProfile:
    """Core strategy configuration shared across entry points."""

    name: str
    factor_indices: tuple[int, ...]
    composite_sheet: str
    strategy_param: str
    description: str = ""

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
        description="Strategy1 75 2.6_annual_return Legacy 2026-03/17 live profile.",
    ),
    "Strategy2": StrategyProfile(
        name="Strategy2",
        factor_indices=(95, 24, 64, 65, 32),
        composite_sheet="ic_m3_N20",
        strategy_param="max_return_10G_Top1_P20d",
        description="Strategy2 Legacy 2026-03/25 research profile.",
    ),
    "Strategy3": StrategyProfile(
        name="Strategy3",
        factor_indices=(95, 99, 27, 75, 19),
        composite_sheet="ic_m3_N20",
        strategy_param="max_return_5G_Top2_P20d",
        description="Strategy3 June 2026 strategy profile.",
    ),
    "Strategy4": StrategyProfile(
        name="Strategy4",
        factor_indices=(95, 99, 27, 46, 19),
        composite_sheet="ic_m3_N10",
        strategy_param="max_return_5G_Top2_P20d",
        description="Strategy4 June 2026 strategy profile.",
    ),
    "Strategy5": StrategyProfile(
        name="Strategy5",
        factor_indices=(95, 99, 27, 19, 46),
        composite_sheet="ic_m3_N10",
        strategy_param="max_return_5G_Top2_P20d",
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
