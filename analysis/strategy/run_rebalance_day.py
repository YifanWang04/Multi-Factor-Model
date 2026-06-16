"""CLI wrapper for the rebalance-day workflow."""

from __future__ import annotations

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from analysis.strategy.rebalance.rebalance_app import (  # noqa: E402
    ACTIVE_STRATEGY_PROFILE,
    COMPOSITE_FACTOR_SHEET,
    SELECTED_FACTOR_INDICES,
    SELECTED_FACTOR_NAMES,
    STRATEGY_PARAM,
    main,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rebalance-day workflow and report")
    parser.add_argument("--skip-pipeline", action="store_true", help="reuse existing data")
    parser.add_argument("--skip-pull", action="store_true", help="skip yfinance pull during pipeline")
    parser.add_argument("--run-dir", type=str, default=None, help="existing or target run directory")
    parser.add_argument("--no-discord", action="store_true", help="disable Discord notification")
    parser.add_argument(
        "--inline",
        action="store_true",
        help="run pipeline in the current process; ignored with --skip-pipeline",
    )
    return parser.parse_args()


def cli() -> None:
    args = parse_args()
    main(
        skip_pipeline=args.skip_pipeline,
        skip_pull=args.skip_pull,
        run_dir_arg=args.run_dir,
        send_discord=not args.no_discord,
        inline_pipeline=args.inline,
    )


if __name__ == "__main__":
    cli()
