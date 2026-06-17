"""Excel workbook helpers shared across research and strategy modules."""

from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterator

import pandas as pd


def read_workbook_sheets(
    file_path: str | os.PathLike[str],
    sheet_filter: Callable[[str], bool] | None = None,
    **read_excel_kwargs,
) -> dict[str, pd.DataFrame]:
    """Read an Excel workbook into ``{sheet_name: DataFrame}`` with optional filtering."""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Excel file does not exist: {path}")
    sheets = pd.read_excel(path, sheet_name=None, **read_excel_kwargs)
    if sheet_filter is None:
        return sheets
    return {name: df for name, df in sheets.items() if sheet_filter(name)}


def require_sheet(file_path: str | os.PathLike[str], sheet_name: str) -> None:
    """Fail fast if ``sheet_name`` is absent from an Excel workbook."""

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Excel file does not exist: {path}")
    with pd.ExcelFile(path) as xl:
        if sheet_name not in xl.sheet_names:
            raise ValueError(
                f"Sheet {sheet_name!r} does not exist in {path.name}; "
                f"available sheets: {xl.sheet_names}"
            )


def read_sheet_with_datetime_index(
    file_path: str | os.PathLike[str],
    sheet_name: str | int,
    index_col: int | str = 0,
) -> pd.DataFrame:
    """Read one sheet and normalize its index to sorted ``DatetimeIndex``."""

    if not isinstance(sheet_name, int):
        require_sheet(file_path, sheet_name)
    df = pd.read_excel(file_path, sheet_name=sheet_name, index_col=index_col)
    df.index = pd.to_datetime(df.index)
    df = df.apply(pd.to_numeric, errors="coerce")
    df.sort_index(inplace=True)
    return df


def read_factor_sheet(
    file_path: str | os.PathLike[str],
    sheet_name: str | int = 0,
) -> pd.DataFrame:
    """Read a factor-like wide sheet with a normalized datetime index."""

    return read_sheet_with_datetime_index(file_path, sheet_name=sheet_name, index_col=0)


def read_factor_workbook(
    file_path: str | os.PathLike[str],
    sheet_filter: Callable[[str], bool] | None = None,
) -> dict[str, pd.DataFrame]:
    """Read every factor sheet and normalize each sheet to numeric time series."""

    sheets = read_workbook_sheets(file_path, sheet_filter=sheet_filter, index_col=0)
    out: dict[str, pd.DataFrame] = {}
    for name, df in sheets.items():
        tmp = df.copy()
        tmp.index = pd.to_datetime(tmp.index)
        tmp = tmp.apply(pd.to_numeric, errors="coerce")
        tmp.sort_index(inplace=True)
        out[name] = tmp
    return out


def read_price_workbook(
    file_path: str | os.PathLike[str],
    price_column: str = "Adj Close",
    sheet_filter: Callable[[str], bool] | None = None,
) -> pd.DataFrame:
    """Read a multi-sheet price workbook into a wide price DataFrame."""

    sheets = read_workbook_sheets(file_path, sheet_filter=sheet_filter)
    columns: dict[str, pd.Series] = {}
    for ticker, df in sheets.items():
        if "Date" not in df.columns or price_column not in df.columns:
            continue
        tmp = df.copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"])
        tmp = tmp.set_index("Date")
        columns[ticker] = tmp[price_column]
    if not columns:
        raise ValueError(f"No usable {price_column} data found in {file_path}")
    out = pd.concat(columns, axis=1)
    out = out.apply(pd.to_numeric, errors="coerce")
    out.sort_index(inplace=True)
    return out


@contextmanager
def atomic_excel_writer(
    output_path: str | os.PathLike[str],
    engine: str = "openpyxl",
    **writer_kwargs,
) -> Iterator[pd.ExcelWriter]:
    """Write an ``.xlsx`` atomically, replacing the destination only on success."""

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=target.suffix or ".xlsx",
            delete=False,
            dir=target.parent,
        ) as tmp:
            tmp_name = tmp.name
        with pd.ExcelWriter(tmp_name, engine=engine, **writer_kwargs) as writer:
            yield writer
        os.replace(tmp_name, target)
        tmp_name = None
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.remove(tmp_name)
