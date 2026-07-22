"""Preserve live rebalance price scale across yfinance refreshes.

Official rebalance runs need continuity: once a run has been generated, later
yfinance corporate-action rewrites should not silently rescale the historical
data used by that run.  This module keeps the previous run's workbook as the
canonical price scale and converts newly appended rows back to that scale when
a split-like stable ratio is detected.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from qqq_core.paths import ProjectPaths, price_filename


PRESERVE_PRICE_SCALE_ENV_VAR = "REBALANCE_PRESERVE_PRICE_SCALE"
PRICE_BASE_RUN_DIR_ENV_VAR = "REBALANCE_PRICE_BASE_RUN_DIR"
STRATEGY_PROFILE_ENV_VAR = "REBALANCE_STRATEGY_PROFILE"
MANIFEST_FILENAME = "price_snapshot_manifest.json"

PRICE_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Adj Close",
    "Adj Open",
    "Adj High",
    "Adj Low",
)
VOLUME_COLUMN = "Volume"

MIN_RATIO_SAMPLES = 50
MIN_PRICE_FACTOR_ABS_DIFFERENCE = 0.01
STABLE_RATIO_REL_TOLERANCE = 5e-3


@dataclass(frozen=True)
class PriceScaleAdjustment:
    ticker: str
    price_factor: float
    volume_factor: float
    overlap_rows: int
    new_rows_adjusted: int
    max_price_rel_deviation: float


@dataclass(frozen=True)
class PriceSnapshotResult:
    enabled: bool
    base_price_file: str | None
    base_run_dir: str | None
    output_price_file: str | None
    adjustments: list[PriceScaleAdjustment]
    tickers_frozen: int
    tickers_appended: int
    notes: list[str]


def env_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def preserve_price_scale_enabled() -> bool:
    return env_truthy(os.environ.get(PRESERVE_PRICE_SCALE_ENV_VAR))


def find_previous_price_file(
    current_run_dir: str | os.PathLike[str],
    profile_name: str,
    offset: int,
    base_run_dir: str | os.PathLike[str] | None = None,
) -> tuple[Path | None, Path | None]:
    override = str(base_run_dir) if base_run_dir else os.environ.get(PRICE_BASE_RUN_DIR_ENV_VAR)
    filename = price_filename(offset)
    if override:
        base_dir = Path(override).resolve()
        candidate = base_dir / "data" / filename
        return (candidate if candidate.is_file() else None), base_dir

    current = Path(current_run_dir).resolve()
    paths = ProjectPaths.from_env(offset=offset)
    suffix = f"_{_safe_profile(profile_name)}_offset{int(offset)}"
    candidates: list[Path] = []
    for run_dir in paths.rebalance_runs_dir.glob(f"*{suffix}"):
        run_dir = run_dir.resolve()
        if run_dir == current:
            continue
        name = run_dir.name.lower()
        if "hybrid" in name or "debug" in name or "test" in name:
            continue
        price_path = run_dir / "data" / filename
        report_path = run_dir / "reports" / "rebalance_day_report.xlsx"
        if price_path.is_file() and report_path.is_file():
            candidates.append(run_dir)

    if not candidates:
        return None, None
    previous = max(candidates, key=lambda p: p.stat().st_mtime_ns)
    return previous / "data" / filename, previous


def apply_preserved_price_scale(
    fresh_data: Mapping[str, pd.DataFrame],
    run_dir: str | os.PathLike[str],
    profile_name: str,
    offset: int,
    output_price_file: str | os.PathLike[str] | None = None,
    base_run_dir: str | os.PathLike[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], PriceSnapshotResult]:
    base_file, resolved_base_run_dir = find_previous_price_file(
        run_dir,
        profile_name,
        offset,
        base_run_dir=base_run_dir,
    )
    notes: list[str] = []
    if base_file is None:
        notes.append("No previous official run price workbook found; using fresh yfinance data.")
        result = PriceSnapshotResult(
            enabled=True,
            base_price_file=None,
            base_run_dir=None,
            output_price_file=str(output_price_file) if output_price_file else None,
            adjustments=[],
            tickers_frozen=0,
            tickers_appended=0,
            notes=notes,
        )
        return {k: v.copy() for k, v in fresh_data.items()}, result

    base_data = pd.read_excel(base_file, sheet_name=None)
    merged: dict[str, pd.DataFrame] = {}
    adjustments: list[PriceScaleAdjustment] = []
    tickers_frozen = 0
    tickers_appended = 0

    for ticker, fresh_df in fresh_data.items():
        fresh = _normalize_sheet(fresh_df)
        base_df = base_data.get(ticker)
        if base_df is None or fresh.empty or "Date" not in fresh.columns:
            merged[ticker] = fresh
            continue

        base = _normalize_sheet(base_df)
        if base.empty or "Date" not in base.columns:
            merged[ticker] = fresh
            continue

        cutoff = pd.to_datetime(base["Date"], errors="coerce").max()
        if pd.isna(cutoff):
            merged[ticker] = fresh
            continue

        old_rows = base[pd.to_datetime(base["Date"], errors="coerce") <= cutoff].copy()
        new_rows = fresh[pd.to_datetime(fresh["Date"], errors="coerce") > cutoff].copy()
        tickers_frozen += 1
        if not new_rows.empty:
            tickers_appended += 1

        adjustment = detect_price_scale_adjustment(base, fresh, ticker)
        if adjustment is not None and not new_rows.empty:
            new_rows = scale_rows_to_canonical(new_rows, adjustment)
            adjustment = PriceScaleAdjustment(
                ticker=adjustment.ticker,
                price_factor=adjustment.price_factor,
                volume_factor=adjustment.volume_factor,
                overlap_rows=adjustment.overlap_rows,
                new_rows_adjusted=len(new_rows),
                max_price_rel_deviation=adjustment.max_price_rel_deviation,
            )
            adjustments.append(adjustment)

        combined = pd.concat([old_rows, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=["Date"], keep="last")
        combined = combined.sort_values("Date").reset_index(drop=True)
        merged[ticker] = combined

    notes.append(
        f"Frozen history from {base_file}; appended fresh rows for {tickers_appended} tickers."
    )
    result = PriceSnapshotResult(
        enabled=True,
        base_price_file=str(base_file),
        base_run_dir=str(resolved_base_run_dir) if resolved_base_run_dir else None,
        output_price_file=str(output_price_file) if output_price_file else None,
        adjustments=adjustments,
        tickers_frozen=tickers_frozen,
        tickers_appended=tickers_appended,
        notes=notes,
    )
    return merged, result


def detect_price_scale_adjustment(
    base: pd.DataFrame,
    fresh: pd.DataFrame,
    ticker: str,
) -> PriceScaleAdjustment | None:
    base = _normalize_sheet(base)
    fresh = _normalize_sheet(fresh)
    if "Date" not in base.columns or "Date" not in fresh.columns:
        return None

    overlap = base.merge(fresh, on="Date", suffixes=("_base", "_fresh"))
    if overlap.empty:
        return None

    ratios: list[pd.Series] = []
    for col in PRICE_COLUMNS:
        base_col = f"{col}_base"
        fresh_col = f"{col}_fresh"
        if base_col not in overlap.columns or fresh_col not in overlap.columns:
            continue
        base_values = pd.to_numeric(overlap[base_col], errors="coerce")
        fresh_values = pd.to_numeric(overlap[fresh_col], errors="coerce")
        ratio = base_values.div(fresh_values.where(fresh_values.abs() > 1e-12))
        ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna()
        ratio = ratio[(ratio > 0) & np.isfinite(ratio)]
        if not ratio.empty:
            ratios.append(ratio)

    if not ratios:
        return None
    all_ratios = pd.concat(ratios, ignore_index=True)
    if len(all_ratios) < MIN_RATIO_SAMPLES:
        return None

    price_factor = float(all_ratios.median())
    if (
        not np.isfinite(price_factor)
        or abs(price_factor - 1.0) < MIN_PRICE_FACTOR_ABS_DIFFERENCE
    ):
        return None

    rel_dev = (all_ratios - price_factor).abs() / abs(price_factor)
    max_rel_dev = float(rel_dev.quantile(0.95))
    if max_rel_dev > STABLE_RATIO_REL_TOLERANCE:
        return None

    return PriceScaleAdjustment(
        ticker=ticker,
        price_factor=price_factor,
        volume_factor=1.0 / price_factor,
        overlap_rows=len(overlap),
        new_rows_adjusted=0,
        max_price_rel_deviation=max_rel_dev,
    )


def scale_rows_to_canonical(
    rows: pd.DataFrame,
    adjustment: PriceScaleAdjustment,
) -> pd.DataFrame:
    out = rows.copy()
    for col in PRICE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce") * adjustment.price_factor
    if VOLUME_COLUMN in out.columns:
        out[VOLUME_COLUMN] = pd.to_numeric(out[VOLUME_COLUMN], errors="coerce") * adjustment.volume_factor
    return out


def write_manifest(
    manifest_path: str | os.PathLike[str],
    result: PriceSnapshotResult,
) -> None:
    path = Path(manifest_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(result)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def manifest_path_for_run(run_dir: str | os.PathLike[str]) -> Path:
    return Path(run_dir) / "data" / MANIFEST_FILENAME


def load_manifest(path_or_run_dir: str | os.PathLike[str]) -> dict[str, Any] | None:
    path = Path(path_or_run_dir)
    if path.is_dir():
        path = manifest_path_for_run(path)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_adjustments_frame(manifest: Mapping[str, Any] | None) -> pd.DataFrame:
    columns = [
        "Ticker",
        "Price_Factor",
        "Volume_Factor",
        "Overlap_Rows",
        "New_Rows_Adjusted",
        "Max_Price_Rel_Deviation",
    ]
    if not manifest:
        return pd.DataFrame(columns=columns)

    rows = []
    for adjustment in manifest.get("adjustments") or []:
        rows.append(
            {
                "Ticker": adjustment.get("ticker", ""),
                "Price_Factor": adjustment.get("price_factor", ""),
                "Volume_Factor": adjustment.get("volume_factor", ""),
                "Overlap_Rows": adjustment.get("overlap_rows", ""),
                "New_Rows_Adjusted": adjustment.get("new_rows_adjusted", ""),
                "Max_Price_Rel_Deviation": adjustment.get("max_price_rel_deviation", ""),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def _normalize_sheet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" in out.columns:
        out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out


def _safe_profile(profile: str | None) -> str:
    if not profile:
        return "profile"
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(profile)).strip("_") or "profile"
