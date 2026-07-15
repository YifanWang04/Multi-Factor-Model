"""Named ticker universes shared by strategy and data configurations.

The Nasdaq-100 history below is a static research snapshot.  It is the union
of the constituents on 2026-07-15 and every constituent deleted from the
index during 2020-07-15 through 2026-07-15.  This captures securities that
left the index as well as securities that joined during the window.

Sources:
* Current constituents: https://www.nasdaq.com/market-activity/quotes/nasdaq-ndx-index
* Component changes: https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies

The static union is not a point-in-time membership table.  A historical
backtest that uses it unchanged will expose dates before a later constituent
actually joined the Nasdaq-100.
"""

from __future__ import annotations

from typing import Mapping


def _ordered_union(*universes: tuple[str, ...]) -> tuple[str, ...]:
    """Return a stable, duplicate-free union preserving first appearance."""

    return tuple(dict.fromkeys(ticker for universe in universes for ticker in universe))


ORIGINAL_108: tuple[str, ...] = (
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

ORIGINAL_143: tuple[str, ...] = (
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

NASDAQ_100_WINDOW_START = "2020-07-15"
NASDAQ_100_SNAPSHOT_AS_OF = "2026-07-15"

# Nasdaq-100 securities at the snapshot date.  Alphabet has two index
# securities, so the number of tickers can exceed the number of companies.
_NASDAQ_100_CURRENT_AS_OF_2026_07_15: tuple[str, ...] = (
    "ADBE", "AMD", "ABNB", "ALNY", "GOOGL", "GOOG", "AMZN", "AEP", "AMGN", "ADI",
    "AAPL", "AMAT", "APP", "ARM", "ASML", "ALAB", "ADSK", "ADP", "AXON", "BKR",
    "BKNG", "AVGO", "CDNS", "CTAS", "CSCO", "CCEP", "CMCSA", "CEG", "CPRT", "CRWV",
    "COST", "CRWD", "CSX", "DDOG", "DXCM", "FANG", "DASH", "EA", "EXC", "FAST",
    "FER", "FTNT", "GEHC", "GILD", "HONA", "HON", "IDXX", "INTC", "INTU", "ISRG",
    "KDP", "KLAC", "KHC", "LRCX", "LIN", "LITE", "MAR", "MRVL", "MELI", "META",
    "MCHP", "MU", "MSFT", "MSTR", "MDLZ", "MPWR", "MNST", "NBIS", "NFLX", "NVDA",
    "NXPI", "ORLY", "ODFL", "PCAR", "PLTR", "PANW", "PAYX", "PYPL", "PDD", "PEP",
    "QCOM", "REGN", "RKLB", "ROP", "ROST", "SNDK", "STX", "SHOP", "SPCX", "SBUX",
    "SNPS", "TMUS", "TTWO", "TER", "TSLA", "TXN", "TRI", "VRTX", "WMT", "WBD",
    "WDC", "WDAY", "XEL",
)

# Every security deleted during the six-year window, newest first.  Names
# still present in the snapshot are harmless because _ordered_union de-dupes.
_NASDAQ_100_DELETED_2020_07_15_TO_2026_07_15: tuple[str, ...] = (
    # 2026
    "CHTR", "CTSH", "INSM", "VRSK", "ZS", "CSGP", "TEAM", "AZN", "VSNT",
    # 2025
    "BIIB", "CDW", "GFS", "LULU", "ON", "TTD", "SOLS", "ANSS", "MDB",
    # 2024
    "ILMN", "MRNA", "SMCI", "DLTR", "WBA", "SIRI", "SPLK",
    # 2023
    "ALGN", "EBAY", "ENPH", "JD", "LCID", "ZM", "SGEN", "ATVI", "RIVN", "FI",
    # 2022
    "VRSN", "SWKS", "NTES", "BIDU", "MTCH", "DOCU", "OKTA", "XLNX", "PTON",
    # 2021
    "FOXA", "FOX", "CERN", "CHKP", "TCOM", "INCY", "MXIM", "ALXN",
    # 2020-07-15 through 2020-12-31
    "BMRN", "CTXS", "EXPE", "LBTYA", "LBTYK", "TTWO", "ULTA", "WDC", "NTAP", "CSGP",
)

## 来源文章美股投资网 AI机器人产业链和个股 https://mp.weixin.qq.com/s/lUDEERy0EW_ZtOSIxzxxdw 
_ROBOTICS: tuple[str, ...] = (
    "NVDA", "TSLA", "CCXI", "ON", "QCOM", "AMBA", "OUST", "ROK", "EMR",
    "PTC", "TER", "RRX", "NOVT", "VPG", "AMZN", "SYM", "APH", "TEL",
    "MPWR", "ADI", "TXN",
)

NASDAQ_100_LAST_6_YEARS: tuple[str, ...] = _ordered_union(
    _NASDAQ_100_CURRENT_AS_OF_2026_07_15,
    _NASDAQ_100_DELETED_2020_07_15_TO_2026_07_15,
)

ORIGINAL_108_PLUS_NASDAQ_100: tuple[str, ...] = _ordered_union(
    ORIGINAL_108,
    NASDAQ_100_LAST_6_YEARS,
)

ORIGINAL_108_PLUS_ROBOTICS: tuple[str, ...] = _ordered_union(
    ORIGINAL_108,
    _ROBOTICS,
)

TICKER_UNIVERSES: Mapping[str, tuple[str, ...]] = {
    "ORIGINAL_108": ORIGINAL_108,
    "ORIGINAL_143": ORIGINAL_143,
    "NASDAQ_100_LAST_6_YEARS": NASDAQ_100_LAST_6_YEARS,
    "ORIGINAL_108_PLUS_NASDAQ_100": ORIGINAL_108_PLUS_NASDAQ_100,
    "ORIGINAL_108_PLUS_ROBOTICS": ORIGINAL_108_PLUS_ROBOTICS,
}
