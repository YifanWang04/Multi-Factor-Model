"""Runtime context for one pipeline or report run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from .paths import ProjectPaths


PROFILE_ENV_VAR = "QQQ_STRATEGY_PROFILE"
RUN_DIR_ENV_VAR = "REBALANCE_RUN_DIR"


@dataclass(frozen=True)
class RunContext:
    """Resolved profile, offset, and run-directory context.

    This is the small Interface scripts should cross when they need to know
    where inputs and outputs live for one run. It keeps run-dir overrides and
    normal research output layout behind the same accessors.
    """

    paths: ProjectPaths
    profile: str
    run_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.run_dir is not None and not isinstance(self.run_dir, Path):
            object.__setattr__(self, "run_dir", Path(self.run_dir).resolve())

    @classmethod
    def from_env(
        cls,
        profile: str | None = None,
        run_dir: str | os.PathLike[str] | None = None,
    ) -> "RunContext":
        paths = ProjectPaths.from_env()
        if profile is None:
            try:
                from qqq_config.strategy_profiles import get_active_profile

                active_profile = os.environ.get(PROFILE_ENV_VAR) or get_active_profile().name
            except Exception:
                active_profile = os.environ.get(PROFILE_ENV_VAR) or "default"
        else:
            active_profile = profile

        env_run_dir = run_dir or os.environ.get(RUN_DIR_ENV_VAR)
        resolved_run_dir = Path(env_run_dir).resolve() if env_run_dir else None
        return cls(paths=paths, profile=active_profile, run_dir=resolved_run_dir)

    @property
    def offset(self) -> int:
        return self.paths.offset

    @property
    def root(self) -> Path:
        return self.paths.root

    @property
    def offset_suffix(self) -> str:
        return self.paths.offset_suffix

    @property
    def price_filename(self) -> str:
        return self.paths.price_filename

    @property
    def data_dir(self) -> Path:
        if self.run_dir:
            return self.run_dir / "data"
        return self.paths.data_dir

    @property
    def price_file(self) -> Path:
        return self.paths.run_price_file(self.run_dir)

    @property
    def factor_raw_dir(self) -> Path:
        return self.paths.run_factor_raw_dir(self.run_dir)

    @property
    def factor_processed_dir(self) -> Path:
        return self.paths.run_factor_processed_dir(self.run_dir)

    @property
    def composite_factor_dir(self) -> Path:
        return self.paths.run_composite_factor_dir(self.run_dir)

    @property
    def single_factor_report_dir(self) -> Path:
        return self.paths.research_single_factor_dir

    @property
    def multi_factor_report_dir(self) -> Path:
        return self.paths.research_multi_factor_dir

    @property
    def walk_forward_report_dir(self) -> Path:
        return self.paths.research_walk_forward_dir

    @property
    def strategy_backtest_dir(self) -> Path:
        return self.paths.strategy_backtest_dir

    @property
    def strategy_detailed_dir(self) -> Path:
        return self.paths.strategy_detailed_dir

    @property
    def strategy_review_dir(self) -> Path:
        return self.paths.strategy_review_dir

    @property
    def reports_dir(self) -> Path:
        if self.run_dir:
            return self.run_dir / "reports"
        return self.paths.output_dir

    @property
    def rebalance_report_file(self) -> Path:
        base = self.run_dir or self.paths.make_rebalance_run_dir(self.profile)
        return base / "reports" / "rebalance_day_report.xlsx"

    def resolve_output_path(self, kind: str) -> Path:
        """Resolve a standard output path for this context."""

        return self.paths.resolve_output_path(
            kind=kind,
            profile=self.profile,
            run_dir=self.run_dir,
        )
