"""Pipeline orchestration seam for rebalance-day runs.

The implementation delegates to the current rebalance app internals so this
refactor keeps behavior stable. Callers should use this module instead of
importing private helpers from ``rebalance_app`` directly.
"""

from __future__ import annotations

from . import rebalance_app


def get_run_dir(run_dir_arg: str | None = None, skip_pipeline: bool = False) -> str:
    return rebalance_app._get_run_dir(run_dir_arg, skip_pipeline)


def run_pipeline_inline(run_dir: str, skip_pull: bool = False) -> None:
    rebalance_app._run_pipeline_inline(run_dir, skip_pull=skip_pull)


def run_pipeline_subprocess(run_dir: str, skip_pull: bool = False) -> None:
    rebalance_app._run_pipeline_subprocess(run_dir, skip_pull=skip_pull)


def sync_composite_factor_to_standard(run_dir: str, sheet: str) -> None:
    rebalance_app._sync_composite_factor_to_standard(run_dir, sheet)
