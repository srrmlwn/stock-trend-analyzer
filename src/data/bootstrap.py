"""Bootstrap module: downloads Kaggle and SimFin datasets and saves per-ticker pkl files.

Usage:
    python -m src.scheduler.main --bootstrap

This is a one-time operation that seeds the local OHLCV cache from bulk data sources
(Kaggle NYSE/NASDAQ datasets and SimFin bulk US share prices). After bootstrapping,
the daily Tiingo incremental fetch appends one bar per ticker per day.
"""

from __future__ import annotations

import logging
import pickle
import tempfile
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Expected OHLCV columns (case-normalised)
_OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Log progress every N tickers
_LOG_INTERVAL = 500

# Default cache directory for OHLCV pkl files
_DEFAULT_CACHE_DIR = ".cache/ohlcv"


def _get_csv_headers(csv_path: Path) -> list[str]:
    """Return the first-row column names of a CSV file without reading all rows.

    Args:
        csv_path: Path to the CSV file.

    Returns:
        List of column header strings (case-preserved).
    """
    try:
        header_df = pd.read_csv(csv_path, nrows=0)
        return list(header_df.columns)
    except Exception:
        return []


def _normalise_df(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    """Normalise a raw DataFrame into the canonical OHLCV format.

    Renames columns case-insensitively to Open/High/Low/Close/Volume, sets the
    DatetimeIndex named "Date" in ascending order, and drops all-NaN rows.

    The Date index/column is resolved first so it is not lost when the DataFrame
    is narrowed to OHLCV columns only.

    Args:
        df: Raw DataFrame from a data source.
        ticker: Ticker symbol (used only for log messages).

    Returns:
        Normalised DataFrame, or None if the result is empty after cleaning.
    """
    # Step 1: resolve the DatetimeIndex before selecting columns,
    # so a "Date" column is not accidentally dropped.
    if not isinstance(df.index, pd.DatetimeIndex):
        # Find a date column (case-insensitive)
        date_col = next((c for c in df.columns if str(c).lower() == "date"), None)
        if date_col is None:
            logger.debug("Bootstrap: no Date column/index for %s — skipping.", ticker)
            return None
        df = df.set_index(date_col)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    # Step 2: build a case-insensitive rename map for OHLCV columns
    rename: dict[str, str] = {}
    for col in df.columns:
        col_lower = str(col).lower()
        for target in _OHLCV_COLS:
            if col_lower == target.lower():
                rename[col] = target
                break

    df = df.rename(columns=rename)

    # Step 3: keep only the OHLCV columns that are present
    present = [c for c in _OHLCV_COLS if c in df.columns]
    if not present:
        logger.debug("Bootstrap: no OHLCV columns found for %s — skipping.", ticker)
        return None

    df = df[present].copy()
    df = df.dropna(how="all")

    if df.empty:
        return None

    df = df.sort_index()
    return df


def _write_pkl(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to a pickle file, creating parent directories as needed.

    Args:
        df: DataFrame to persist.
        path: Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(df, fh)  # noqa: S301


def _process_csv_dir(
    csv_dir: Path,
    cache_dir: Path,
    source_name: str,
) -> tuple[int, int]:
    """Parse per-ticker CSV files in csv_dir and write pkl files to cache_dir.

    Assumes each CSV is named ``<TICKER>.csv`` with columns:
    Date, Open, High, Low, Close, Volume (case-insensitive).

    Args:
        csv_dir: Directory containing the per-ticker CSV files.
        cache_dir: Root cache directory for output pkl files.
        source_name: Label used in log messages.

    Returns:
        Tuple of (written, skipped) counts.
    """
    csv_files = list(csv_dir.glob("*.csv"))
    total = len(csv_files)
    logger.info("Bootstrap [%s]: found %d CSV files.", source_name, total)

    written = 0
    skipped = 0

    for i, csv_path in enumerate(csv_files, start=1):
        ticker = csv_path.stem.upper()
        pkl_path = cache_dir / f"{ticker}.pkl"

        if pkl_path.exists():
            skipped += 1
            continue

        headers = _get_csv_headers(csv_path)
        # Detect if there is a date column (case-insensitive)
        date_cols = [h for h in headers if h.lower() == "date"]
        parse_dates: list[str] | bool = date_cols if date_cols else False

        try:
            raw = pd.read_csv(csv_path, parse_dates=parse_dates)
        except Exception as exc:
            logger.debug("Bootstrap [%s]: could not read %s — %s", source_name, csv_path, exc)
            skipped += 1
            continue

        df = _normalise_df(raw, ticker)
        if df is None:
            logger.debug("Bootstrap [%s]: empty result for %s — skipping.", source_name, ticker)
            skipped += 1
            continue

        _write_pkl(df, pkl_path)
        written += 1

        if i % _LOG_INTERVAL == 0:
            logger.info("Bootstrap [%s]: %d / %d tickers written.", source_name, written, total)

    return written, skipped


def _process_simfin_df(
    df: pd.DataFrame,
    cache_dir: Path,
) -> tuple[int, int]:
    """Split a SimFin MultiIndex DataFrame into per-ticker pkl files.

    SimFin returns a DataFrame with a (Ticker, Date) MultiIndex. This function
    splits it by ticker and writes each slice to the cache directory.

    Args:
        df: SimFin bulk DataFrame with (Ticker, Date) MultiIndex.
        cache_dir: Root cache directory for output pkl files.

    Returns:
        Tuple of (written, skipped) counts.
    """
    # Handle both MultiIndex (Ticker, Date) and flat index with Ticker column
    if isinstance(df.index, pd.MultiIndex):
        tickers: list[str] = [str(t) for t in df.index.get_level_values(0).unique().tolist()]
    elif "Ticker" in df.columns:
        tickers = [str(t) for t in df["Ticker"].unique().tolist()]
    else:
        logger.warning("Bootstrap [SimFin]: unexpected DataFrame structure — skipping.")
        return 0, 0

    total = len(tickers)
    logger.info("Bootstrap [SimFin]: found %d tickers.", total)

    written = 0
    skipped = 0

    for i, ticker in enumerate(tickers, start=1):
        pkl_path = cache_dir / f"{ticker.upper()}.pkl"

        if pkl_path.exists():
            skipped += 1
            continue

        try:
            if isinstance(df.index, pd.MultiIndex):
                ticker_df: pd.DataFrame = df.xs(ticker, level=0).copy()
            else:
                ticker_df = df[df["Ticker"] == ticker].drop(columns=["Ticker"]).copy()
        except (KeyError, TypeError) as exc:
            logger.debug("Bootstrap [SimFin]: could not extract %s — %s", ticker, exc)
            skipped += 1
            continue

        normalised = _normalise_df(ticker_df, ticker)
        if normalised is None:
            logger.debug("Bootstrap [SimFin]: empty result for %s — skipping.", ticker)
            skipped += 1
            continue

        _write_pkl(normalised, pkl_path)
        written += 1

        if i % _LOG_INTERVAL == 0:
            logger.info("Bootstrap [SimFin]: %d / %d tickers written.", written, total)

    return written, skipped


def _bootstrap_kaggle(cache_dir: Path, tmp_dir: Path) -> tuple[int, int]:
    """Download Kaggle NYSE and NASDAQ datasets and write pkl files.

    Gracefully skips if the ``kaggle`` package is not installed or
    ``~/.kaggle/kaggle.json`` is missing.

    Args:
        cache_dir: Root cache directory for output pkl files.
        tmp_dir: Temporary directory for downloaded files.

    Returns:
        Tuple of (written, skipped) counts across both datasets.
    """
    try:
        import kaggle  # noqa: F401
    except ImportError:
        logger.info(
            "Bootstrap: Kaggle download skipped — install the kaggle package and place "
            "your API key at ~/.kaggle/kaggle.json\n"
            "See: https://github.com/Kaggle/kaggle-api#api-credentials"
        )
        return 0, 0

    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if not kaggle_json.exists():
        logger.info(
            "Bootstrap: Kaggle download skipped — install the kaggle package and place "
            "your API key at ~/.kaggle/kaggle.json\n"
            "See: https://github.com/Kaggle/kaggle-api#api-credentials"
        )
        return 0, 0

    total_written = 0
    total_skipped = 0

    for dataset, label in [
        ("svaningelgem/nyse-daily-stock-prices", "Kaggle/NYSE"),
        ("svaningelgem/nasdaq-daily-stock-prices", "Kaggle/NASDAQ"),
    ]:
        dataset_dir = tmp_dir / label.replace("/", "_")
        dataset_dir.mkdir(parents=True, exist_ok=True)
        try:
            logger.info("Bootstrap [%s]: downloading dataset ...", label)
            kaggle.api.dataset_download_files(dataset, path=str(dataset_dir), unzip=True)
        except Exception as exc:
            logger.warning("Bootstrap [%s]: download failed — %s", label, exc)
            continue

        written, skipped = _process_csv_dir(dataset_dir, cache_dir, label)
        total_written += written
        total_skipped += skipped

    return total_written, total_skipped


def _bootstrap_simfin(cache_dir: Path, tmp_dir: Path) -> tuple[int, int]:
    """Download SimFin bulk US share prices and write pkl files.

    Gracefully skips if the ``simfin`` package is not installed.

    Args:
        cache_dir: Root cache directory for output pkl files.
        tmp_dir: Temporary directory for downloaded files.

    Returns:
        Tuple of (written, skipped) counts.
    """
    try:
        import simfin as sf  # noqa: F401
    except ImportError:
        logger.info(
            "Bootstrap: SimFin download skipped — install the simfin package "
            "(pip install simfin>=0.9.0) to include SimFin data."
        )
        return 0, 0

    try:
        logger.info("Bootstrap [SimFin]: downloading bulk US share prices ...")
        sf.set_api_key("free")
        sf.set_data_dir(str(tmp_dir))
        raw_df: pd.DataFrame = sf.load_shareprices(variant="daily", market="us")
    except Exception as exc:
        logger.warning("Bootstrap [SimFin]: download failed — %s", exc)
        return 0, 0

    return _process_simfin_df(raw_df, cache_dir)


def run_bootstrap(cache_dir: str = _DEFAULT_CACHE_DIR) -> None:
    """Run the full bootstrap pipeline: Kaggle + SimFin to local pkl store.

    Downloads Kaggle NYSE/NASDAQ datasets and SimFin bulk US share prices,
    parses them into per-ticker DataFrames, and saves each ticker as a
    pkl file in *cache_dir*. Existing pkl files are skipped (idempotent).

    Args:
        cache_dir: Directory for OHLCV pkl files (default: ``.cache/ohlcv``).
    """
    output_dir = Path(cache_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    total_skipped = 0

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        kaggle_written, kaggle_skipped = _bootstrap_kaggle(output_dir, tmp_dir)
        total_written += kaggle_written
        total_skipped += kaggle_skipped

        simfin_written, simfin_skipped = _bootstrap_simfin(output_dir, tmp_dir)
        total_written += simfin_written
        total_skipped += simfin_skipped

    logger.info(
        "Bootstrap complete: %d written, %d skipped (already exist)",
        total_written,
        total_skipped,
    )


__all__ = ["run_bootstrap"]
