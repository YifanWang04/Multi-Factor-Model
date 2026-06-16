"""Project path and output-layout helpers.

This module is the single place for path rules that are shared across data,
factor, composite, strategy, and rebalance entry points. Generated factor
directories intentionally remain at their historical top-level locations, while
new reports are routed into a clearer ``output/`` layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path


PROJECT_ROOT_ENV_VAR = "QQQ_PROJECT_ROOT"
REBALANCE_OFFSET_ENV_VAR = "REBALANCE_OFFSET_DAYS"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_offset(default: int = 0) -> int:
    env_value = os.environ.get(REBALANCE_OFFSET_ENV_VAR)
    if env_value is not None:
        return int(env_value)
    return int(default)


def offset_dir_suffix(offset: int) -> str:
    return "" if int(offset) == 0 else f"_offset{int(offset)}d"


def price_filename(offset: int) -> str:
    if int(offset) == 0:
        return "us_top100_daily_2023_present.xlsx"
    return f"us_top100_daily_2023_present_offset{int(offset)}d.xlsx"


def _safe_profile(profile: str | None) -> str:
    if not profile:
        return "profile"
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(profile)).strip("_") or "profile"


@dataclass(frozen=True)
class ProjectPaths:
    """Resolved filesystem paths for one project root and offset."""

    root: Path
    offset: int = 0

    @classmethod
    def from_env(cls, offset: int | None = None) -> "ProjectPaths":
        root = Path(os.environ.get(PROJECT_ROOT_ENV_VAR, _repo_root())).resolve()
        return cls(root=root, offset=_resolve_offset() if offset is None else int(offset))

    @property
    def offset_suffix(self) -> str:
        return offset_dir_suffix(self.offset)

    @property
    def data_dir(self) -> Path:
        return self.root / "data"

    @property
    def price_filename(self) -> str:
        return price_filename(self.offset)

    @property
    def price_file(self) -> Path:
        return self.data_dir / self.price_filename

    @property
    def factor_raw_dir(self) -> Path:
        return self.root / f"factor_raw{self.offset_suffix}"

    @property
    def factor_processed_dir(self) -> Path:
        return self.root / f"factor_processed{self.offset_suffix}"

    @property
    def output_dir(self) -> Path:
        return self.root / "output"

    @property
    def research_dir(self) -> Path:
        return self.output_dir / "research"

    @property
    def strategy_dir(self) -> Path:
        return self.output_dir / "strategy"

    @property
    def rebalance_runs_dir(self) -> Path:
        return self.output_dir / "rebalance_runs"

    @property
    def research_single_factor_dir(self) -> Path:
        return self.research_dir / f"single_factor{self.offset_suffix}"

    @property
    def research_multi_factor_dir(self) -> Path:
        return self.research_dir / f"multi_factor{self.offset_suffix}"

    @property
    def research_composite_factor_dir(self) -> Path:
        return self.research_dir / f"composite_factor{self.offset_suffix}"

    @property
    def research_walk_forward_dir(self) -> Path:
        return self.research_dir / f"walk_forward{self.offset_suffix}"

    @property
    def strategy_backtest_dir(self) -> Path:
        return self.strategy_dir / f"backtest{self.offset_suffix}"

    @property
    def strategy_detailed_dir(self) -> Path:
        return self.strategy_dir / f"detailed{self.offset_suffix}"

    @property
    def strategy_review_dir(self) -> Path:
        return self.strategy_dir / f"review{self.offset_suffix}"

    @property
    def legacy_composite_factor_dir(self) -> Path:
        return self.output_dir / f"composite_factor_reports{self.offset_suffix}"

    @property
    def legacy_strategy_reports_dir(self) -> Path:
        return self.output_dir / f"strategy_reports{self.offset_suffix}"

    @property
    def legacy_single_factor_reports_dir(self) -> Path:
        return self.output_dir / f"single_factor_reports{self.offset_suffix}"

    @property
    def legacy_multi_factor_reports_dir(self) -> Path:
        return self.output_dir / f"multi_factor_reports{self.offset_suffix}"

    @property
    def legacy_walk_forward_reports_dir(self) -> Path:
        return self.output_dir / f"walk_forward_reports{self.offset_suffix}"

    def run_price_file(self, run_dir: str | os.PathLike[str] | None = None) -> Path:
        if run_dir:
            return Path(run_dir).resolve() / "data" / self.price_filename
        return self.price_file

    def run_factor_raw_dir(self, run_dir: str | os.PathLike[str] | None = None) -> Path:
        if run_dir:
            return Path(run_dir).resolve() / "factor_raw"
        return self.factor_raw_dir

    def run_factor_processed_dir(self, run_dir: str | os.PathLike[str] | None = None) -> Path:
        if run_dir:
            return Path(run_dir).resolve() / "factor_processed"
        return self.factor_processed_dir

    def run_composite_factor_dir(self, run_dir: str | os.PathLike[str] | None = None) -> Path:
        if run_dir:
            return Path(run_dir).resolve() / "composite_factor_reports"
        return self.research_composite_factor_dir

    def make_rebalance_run_dir(
        self,
        profile: str | None = None,
        now: datetime | None = None,
    ) -> Path:
        stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
        name = f"{stamp}_{_safe_profile(profile)}_offset{self.offset}"
        return self.rebalance_runs_dir / name

    def resolve_output_path(
        self,
        kind: str,
        profile: str | None = None,
        run_dir: str | os.PathLike[str] | None = None,
    ) -> Path:
        if kind == "price_file":
            return self.run_price_file(run_dir)
        if kind == "factor_raw_dir":
            return self.run_factor_raw_dir(run_dir)
        if kind == "factor_processed_dir":
            return self.run_factor_processed_dir(run_dir)
        if kind == "composite_factor_dir":
            return self.run_composite_factor_dir(run_dir)
        if kind == "single_factor_report_dir":
            return self.research_single_factor_dir
        if kind == "multi_factor_report_dir":
            return self.research_multi_factor_dir
        if kind == "walk_forward_report_dir":
            return self.research_walk_forward_dir
        if kind == "strategy_backtest_dir":
            return self.strategy_backtest_dir
        if kind == "strategy_detailed_dir":
            return self.strategy_detailed_dir
        if kind == "strategy_review_dir":
            return self.strategy_review_dir
        if kind == "rebalance_run_dir":
            return Path(run_dir).resolve() if run_dir else self.make_rebalance_run_dir(profile)
        if kind == "rebalance_report_file":
            base = Path(run_dir).resolve() if run_dir else self.make_rebalance_run_dir(profile)
            return base / "reports" / "rebalance_day_report.xlsx"
        raise ValueError(f"Unknown output path kind: {kind!r}")


def resolve_output_path(
    kind: str,
    profile: str | None = None,
    offset: int | None = None,
    run_dir: str | os.PathLike[str] | None = None,
) -> Path:
    """Resolve a generated output path using the standard project layout."""

    return ProjectPaths.from_env(offset=offset).resolve_output_path(
        kind=kind,
        profile=profile,
        run_dir=run_dir,
    )
