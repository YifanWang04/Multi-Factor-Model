"""Pipeline orchestration interface for rebalance-day runs.

This module is the seam between the rebalance-day app and the data/factor/
composite pipeline. The current implementation still delegates the low-level
work to ``rebalance_app`` internals to preserve behavior, but callers can now
use one stable Interface instead of coordinating private helpers themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
import os

from qqq_core.run_context import RunContext

from . import rebalance_app


@dataclass(frozen=True)
class RebalancePipelineOptions:
    """Options controlling one rebalance-day pipeline run."""

    skip_pipeline: bool = False
    skip_pull: bool = False
    inline_pipeline: bool = False
    run_dir_arg: str | None = None
    sync_sheet: str | None = None


@dataclass(frozen=True)
class RebalancePipelineResult:
    """Resolved outcome of the pipeline stage."""

    run_dir: str
    pipeline_ran: bool
    composite_synced: bool


def resolve_rebalance_run_dir(
    run_dir_arg: str | None = None,
    skip_pipeline: bool = False,
    context: RunContext | None = None,
) -> str:
    """Resolve and create the run directory for a rebalance workflow."""

    if run_dir_arg:
        run_dir = os.path.abspath(run_dir_arg)
    elif context is not None and context.run_dir is not None:
        run_dir = str(context.run_dir)
    elif context is not None:
        run_dir = str(context.paths.make_rebalance_run_dir(context.profile))
    else:
        run_dir = rebalance_app._get_run_dir(run_dir_arg, skip_pipeline)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def run_rebalance_pipeline(
    options: RebalancePipelineOptions,
    context: RunContext | None = None,
) -> RebalancePipelineResult:
    """Run the data/factor/composite pipeline and optional composite sync."""

    run_dir = resolve_rebalance_run_dir(
        options.run_dir_arg,
        options.skip_pipeline,
        context=context,
    )
    if options.skip_pipeline:
        return RebalancePipelineResult(
            run_dir=run_dir,
            pipeline_ran=False,
            composite_synced=False,
        )

    if options.inline_pipeline:
        rebalance_app._run_pipeline_inline(run_dir, skip_pull=options.skip_pull)
    else:
        rebalance_app._run_pipeline_subprocess(run_dir, skip_pull=options.skip_pull)

    composite_synced = False
    if options.sync_sheet:
        rebalance_app._sync_composite_factor_to_standard(run_dir, options.sync_sheet)
        composite_synced = True

    return RebalancePipelineResult(
        run_dir=run_dir,
        pipeline_ran=True,
        composite_synced=composite_synced,
    )


def get_run_dir(run_dir_arg: str | None = None, skip_pipeline: bool = False) -> str:
    return resolve_rebalance_run_dir(run_dir_arg, skip_pipeline)


def run_pipeline_inline(run_dir: str, skip_pull: bool = False) -> None:
    rebalance_app._run_pipeline_inline(run_dir, skip_pull=skip_pull)


def run_pipeline_subprocess(run_dir: str, skip_pull: bool = False) -> None:
    rebalance_app._run_pipeline_subprocess(run_dir, skip_pull=skip_pull)


def sync_composite_factor_to_standard(run_dir: str, sheet: str) -> None:
    rebalance_app._sync_composite_factor_to_standard(run_dir, sheet)
