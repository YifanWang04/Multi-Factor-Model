"""Runtime context for one pipeline or report run."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from .paths import ProjectPaths


@dataclass(frozen=True)
class RunContext:
    """Resolved profile, offset, and run-directory context."""

    paths: ProjectPaths
    profile: str
    run_dir: Path | None = None

    @classmethod
    def from_env(
        cls,
        profile: str | None = None,
        run_dir: str | os.PathLike[str] | None = None,
    ) -> "RunContext":
        paths = ProjectPaths.from_env()
        active_profile = profile or os.environ.get("QQQ_STRATEGY_PROFILE") or "default"
        env_run_dir = run_dir or os.environ.get("REBALANCE_RUN_DIR")
        resolved_run_dir = Path(env_run_dir).resolve() if env_run_dir else None
        return cls(paths=paths, profile=active_profile, run_dir=resolved_run_dir)

    @property
    def offset(self) -> int:
        return self.paths.offset

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
    def rebalance_report_file(self) -> Path:
        base = self.run_dir or self.paths.make_rebalance_run_dir(self.profile)
        return base / "reports" / "rebalance_day_report.xlsx"
