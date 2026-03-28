"""Tests for src/data/fetcher.py — all network calls are mocked (offline)."""

import json
import os
import pickle
import time
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.data.fetcher import (
    _FUNDAMENTALS_CACHE_TTL_SECONDS,
    _get_last_stored_date,
    clear_cache,
    fetch_fundamentals,
    fetch_ohlcv,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _last_business_day(d: "date | None" = None) -> date:
    """Return *d* shifted back to the nearest business day (Mon-Fri).

    If *d* is None, uses today. Saturdays shift to Friday, Sundays to Friday.
    """
    if d is None:
        d = date.today()
    # weekday(): 0=Monday ... 4=Friday, 5=Saturday, 6=Sunday
    if d.weekday() == 5:  # Saturday
        return d - timedelta(days=1)
    if d.weekday() == 6:  # Sunday
        return d - timedelta(days=2)
    return d


def _make_ohlcv_df(n: int = 5, end: "date | None" = None) -> pd.DataFrame:
    """Return a minimal synthetic OHLCV DataFrame with *n* business-day rows.

    *end* is adjusted to the nearest business day so that all tests work
    regardless of whether they run on a weekday or weekend.
    """
    end = _last_business_day(end)
    dates = pd.date_range(end=str(end), periods=n, freq="D")
    rng = np.random.default_rng(0)
    close = 100.0 + rng.standard_normal(n).cumsum()
    return pd.DataFrame(
        {
            "Open": close * 0.99,
            "High": close * 1.01,
            "Low": close * 0.98,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


def _make_info_dict() -> dict:  # type: ignore[type-arg]
    return {
        "trailingPE": 28.5,
        "marketCap": 2_500_000_000_000,
        "averageVolume": 60_000_000,
        "currentPrice": 182.45,
    }


def _save_pkl(tmp_path: Path, ticker: str, df: pd.DataFrame) -> Path:
    """Write df as a pkl for ticker in tmp_path and return the path."""
    pkl = tmp_path / f"{ticker}.pkl"
    with pkl.open("wb") as fh:
        pickle.dump(df, fh)
    return pkl


# ---------------------------------------------------------------------------
# _get_last_stored_date
# ---------------------------------------------------------------------------


class TestGetLastStoredDate:
    """Unit tests for the _get_last_stored_date helper."""

    def test_returns_none_when_no_pkl(self, tmp_path: Path) -> None:
        result = _get_last_stored_date("AAPL", str(tmp_path))
        assert result is None

    def test_returns_correct_date_from_pkl(self, tmp_path: Path) -> None:
        last_bday = _last_business_day()
        df = _make_ohlcv_df(5, end=last_bday)
        _save_pkl(tmp_path, "AAPL", df)
        result = _get_last_stored_date("AAPL", str(tmp_path))
        assert result == last_bday

    def test_returns_none_for_empty_df(self, tmp_path: Path) -> None:
        empty_df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])
        _save_pkl(tmp_path, "AAPL", empty_df)
        result = _get_last_stored_date("AAPL", str(tmp_path))
        assert result is None

    def test_returns_none_for_corrupt_pkl(self, tmp_path: Path) -> None:
        pkl = tmp_path / "AAPL.pkl"
        pkl.write_bytes(b"not-a-pickle")
        result = _get_last_stored_date("AAPL", str(tmp_path))
        assert result is None

    def test_last_date_is_max_index(self, tmp_path: Path) -> None:
        dates = pd.to_datetime(["2024-01-01", "2024-03-15", "2024-02-10"])
        df = pd.DataFrame({"Close": [1.0, 2.0, 3.0]}, index=dates)
        _save_pkl(tmp_path, "MSFT", df)
        result = _get_last_stored_date("MSFT", str(tmp_path))
        assert result == date(2024, 3, 15)


# ---------------------------------------------------------------------------
# fetch_ohlcv — cache up-to-date (today already stored)
# ---------------------------------------------------------------------------


class TestFetchOhlcvCacheHit:
    """No network calls when today is already in the pkl.

    We patch date.today() in the fetcher to a fixed Monday so tests pass
    on any day of the week, including weekends.
    """

    _FAKE_TODAY = date(2025, 6, 2)  # Monday

    def test_cache_hit_returns_cached_df(self, tmp_path: Path) -> None:
        df = _make_ohlcv_df(5, end=self._FAKE_TODAY)
        _save_pkl(tmp_path, "AAPL", df)

        with patch("src.data.fetcher.date") as mock_date:
            mock_date.today.return_value = self._FAKE_TODAY
            with patch("src.data.fetcher.yf.download") as mock_dl:
                with patch("src.data.fetcher._fetch_tiingo_since") as mock_tiingo:
                    result = fetch_ohlcv(["AAPL"], cache_dir=str(tmp_path))

        mock_dl.assert_not_called()
        mock_tiingo.assert_not_called()
        assert "AAPL" in result

    def test_cache_hit_multiple_tickers(self, tmp_path: Path) -> None:
        for sym in ("AAPL", "MSFT"):
            df = _make_ohlcv_df(5, end=self._FAKE_TODAY)
            _save_pkl(tmp_path, sym, df)

        with patch("src.data.fetcher.date") as mock_date:
            mock_date.today.return_value = self._FAKE_TODAY
            with patch("src.data.fetcher.yf.download") as mock_dl:
                with patch("src.data.fetcher._fetch_tiingo_since") as mock_tiingo:
                    result = fetch_ohlcv(["AAPL", "MSFT"], cache_dir=str(tmp_path))

        mock_dl.assert_not_called()
        mock_tiingo.assert_not_called()
        assert set(result.keys()) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# fetch_ohlcv — no pkl (full yfinance download)
# ---------------------------------------------------------------------------


class TestFetchOhlcvNoPkl:
    """When no pkl exists, full yfinance batch download is triggered."""

    def test_no_pkl_triggers_batch_download(self, tmp_path: Path) -> None:
        df = _make_ohlcv_df(10)
        multi_index = pd.MultiIndex.from_tuples(
            [(col, "AAPL") for col in df.columns], names=["metric", "ticker"]
        )
        raw = pd.DataFrame(df.values, index=df.index, columns=multi_index)

        with patch("src.data.fetcher.yf.download", return_value=raw):
            result = fetch_ohlcv(["AAPL"], cache_dir=str(tmp_path))

        assert "AAPL" in result
        assert (tmp_path / "AAPL.pkl").exists()

    def test_no_pkl_creates_cache_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "ohlcv"

        with patch("src.data.fetcher.yf.download", return_value=pd.DataFrame()):
            fetch_ohlcv(["AAPL"], cache_dir=str(nested))

        assert nested.exists()

    def test_empty_download_skipped(self, tmp_path: Path) -> None:
        with patch("src.data.fetcher.yf.download", return_value=pd.DataFrame()):
            result = fetch_ohlcv(["INVALID"], cache_dir=str(tmp_path))

        assert "INVALID" not in result
        assert not (tmp_path / "INVALID.pkl").exists()


# ---------------------------------------------------------------------------
# fetch_ohlcv — stale pkl (incremental append)
# ---------------------------------------------------------------------------


class TestFetchOhlcvIncremental:
    """When pkl exists but is stale, incremental Tiingo/yfinance fetch runs.

    Use fixed dates (all weekdays) so tests are deterministic regardless of
    what day of the week the test suite runs.
    """

    # Fixed reference dates — all weekdays
    _OLD_END = date(2025, 6, 2)   # Monday — last date in existing pkl
    _NEW_DATE = date(2025, 6, 3)  # Tuesday — the new bar to append
    _FAKE_TODAY = date(2025, 6, 3)  # Tuesday — what "today" returns in the fetcher

    def test_stale_pkl_calls_tiingo_since(self, tmp_path: Path) -> None:
        old_df = _make_ohlcv_df(5, end=self._OLD_END)
        _save_pkl(tmp_path, "AAPL", old_df)

        new_bar = _make_ohlcv_df(1, end=self._NEW_DATE)

        with patch("src.data.fetcher.date") as mock_date:
            mock_date.today.return_value = self._FAKE_TODAY
            mock_date.fromisoformat = date.fromisoformat
            with patch(
                "src.data.fetcher._fetch_tiingo_since", return_value={"AAPL": new_bar}
            ) as mock_t:
                result = fetch_ohlcv(["AAPL"], cache_dir=str(tmp_path))

        mock_t.assert_called_once()
        assert "AAPL" in result
        assert len(result["AAPL"]) == len(old_df) + 1

    def test_stale_pkl_appends_and_saves(self, tmp_path: Path) -> None:
        old_df = _make_ohlcv_df(5, end=self._OLD_END)
        _save_pkl(tmp_path, "AAPL", old_df)

        new_bar = _make_ohlcv_df(1, end=self._NEW_DATE)

        with patch("src.data.fetcher.date") as mock_date:
            mock_date.today.return_value = self._FAKE_TODAY
            mock_date.fromisoformat = date.fromisoformat
            with patch("src.data.fetcher._fetch_tiingo_since", return_value={"AAPL": new_bar}):
                fetch_ohlcv(["AAPL"], cache_dir=str(tmp_path))

        with (tmp_path / "AAPL.pkl").open("rb") as fh:
            saved = pickle.load(fh)

        assert len(saved) == len(old_df) + 1
        assert pd.Timestamp(self._NEW_DATE) in saved.index

    def test_incremental_no_new_bars_returns_existing(self, tmp_path: Path) -> None:
        old_df = _make_ohlcv_df(5, end=self._OLD_END)
        _save_pkl(tmp_path, "AAPL", old_df)

        with patch("src.data.fetcher.date") as mock_date:
            mock_date.today.return_value = self._FAKE_TODAY
            mock_date.fromisoformat = date.fromisoformat
            with patch("src.data.fetcher._fetch_tiingo_since", return_value={}):
                result = fetch_ohlcv(["AAPL"], cache_dir=str(tmp_path))

        assert "AAPL" in result

    def test_incremental_no_duplicate_rows(self, tmp_path: Path) -> None:
        old_df = _make_ohlcv_df(5, end=self._OLD_END)
        _save_pkl(tmp_path, "AAPL", old_df)

        # Return old_end + new_date (overlap on old_end)
        overlap_df = _make_ohlcv_df(2, end=self._NEW_DATE)

        with patch("src.data.fetcher.date") as mock_date:
            mock_date.today.return_value = self._FAKE_TODAY
            mock_date.fromisoformat = date.fromisoformat
            with patch("src.data.fetcher._fetch_tiingo_since", return_value={"AAPL": overlap_df}):
                result = fetch_ohlcv(["AAPL"], cache_dir=str(tmp_path))

        assert result["AAPL"].index.is_unique


# ---------------------------------------------------------------------------
# _fetch_tiingo_since — Tiingo API key fallback
# ---------------------------------------------------------------------------


class TestFetchTiingoFallback:
    """_fetch_tiingo_since falls back to yfinance when TIINGO_API_KEY is absent."""

    def test_no_api_key_falls_back_to_yfinance(self, tmp_path: Path) -> None:
        from src.data.fetcher import _fetch_tiingo_since

        since = date.today() - timedelta(days=1)
        env = {k: v for k, v in os.environ.items() if k != "TIINGO_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch("src.data.fetcher._batch_download_since", return_value={}) as mock_yf:
                _fetch_tiingo_since(["AAPL"], since)

        mock_yf.assert_called_once_with(["AAPL"], since)

    def test_no_api_key_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        import logging

        from src.data.fetcher import _fetch_tiingo_since

        since = date.today() - timedelta(days=1)
        env = {k: v for k, v in os.environ.items() if k != "TIINGO_API_KEY"}
        with patch.dict(os.environ, env, clear=True):
            with patch("src.data.fetcher._batch_download_since", return_value={}):
                with caplog.at_level(logging.WARNING, logger="src.data.fetcher"):
                    _fetch_tiingo_since(["AAPL"], since)

        assert any("Tiingo API key not set" in r.message for r in caplog.records)

    def test_with_api_key_calls_tiingo_client(self) -> None:
        from src.data.fetcher import _fetch_tiingo_since

        since = date(2025, 6, 2)
        df = _make_ohlcv_df(1, end=date(2025, 6, 3))

        mock_client = MagicMock()
        mock_client.get_dataframe.return_value = df

        with patch.dict(os.environ, {"TIINGO_API_KEY": "fake-key"}):
            # create=True allows patching even if tiingo is not installed
            with patch("src.data.fetcher.TiingoClient", return_value=mock_client, create=True):
                with patch("src.data.fetcher._TIINGO_AVAILABLE", True):
                    with patch("src.data.fetcher.time.sleep"):
                        result = _fetch_tiingo_since(["AAPL"], since)

        mock_client.get_dataframe.assert_called_once()
        assert "AAPL" in result


# ---------------------------------------------------------------------------
# fetch_fundamentals — quarterly TTL
# ---------------------------------------------------------------------------


class TestFetchFundamentalsCacheHit:
    def test_cache_hit_returns_cached_data(self, tmp_path: Path) -> None:
        data = {
            "pe_ratio": 28.5,
            "market_cap": 2_500_000_000_000,
            "avg_volume": 60_000_000,
            "price": 182.45,
        }
        json_file = tmp_path / "AAPL.json"
        with json_file.open("w") as fh:
            json.dump(data, fh)

        with patch("src.data.fetcher.yf.Ticker") as mock_ticker_cls:
            result = fetch_fundamentals(["AAPL"], cache_dir=str(tmp_path))

        mock_ticker_cls.assert_not_called()
        assert result["AAPL"] == data

    def test_quarterly_ttl_fresh_at_30_days(self, tmp_path: Path) -> None:
        """A cache file 30 days old must still be a hit (within 90-day TTL)."""
        data = {"pe_ratio": 10.0, "market_cap": 1, "avg_volume": 1, "price": 1.0}
        json_file = tmp_path / "AAPL.json"
        with json_file.open("w") as fh:
            json.dump(data, fh)

        ts = time.time() - 30 * 24 * 60 * 60
        os.utime(json_file, (ts, ts))

        with patch("src.data.fetcher.yf.Ticker") as mock_ticker_cls:
            result = fetch_fundamentals(["AAPL"], cache_dir=str(tmp_path))

        mock_ticker_cls.assert_not_called()
        assert result["AAPL"]["pe_ratio"] == 10.0

    def test_stale_after_90_days(self, tmp_path: Path) -> None:
        stale_data = {"pe_ratio": 10.0, "market_cap": 1, "avg_volume": 1, "price": 1.0}
        json_file = tmp_path / "AAPL.json"
        with json_file.open("w") as fh:
            json.dump(stale_data, fh)

        ts = time.time() - 91 * 24 * 60 * 60
        os.utime(json_file, (ts, ts))

        fresh_info = _make_info_dict()
        mock_ticker_instance = MagicMock()
        type(mock_ticker_instance).info = property(lambda self: fresh_info)

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
            with patch("src.data.fetcher.time.sleep"):
                result = fetch_fundamentals(["AAPL"], cache_dir=str(tmp_path))

        assert result["AAPL"]["pe_ratio"] == 28.5

    def test_fundamentals_ttl_constant_is_90_days(self) -> None:
        assert _FUNDAMENTALS_CACHE_TTL_SECONDS == 90 * 24 * 60 * 60


# ---------------------------------------------------------------------------
# fetch_fundamentals — cache miss
# ---------------------------------------------------------------------------


class TestFetchFundamentalsCacheMiss:
    def test_cache_miss_fetches_and_saves(self, tmp_path: Path) -> None:
        info = _make_info_dict()

        mock_ticker_instance = MagicMock()
        type(mock_ticker_instance).info = property(lambda self: info)

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
            with patch("src.data.fetcher.time.sleep"):
                result = fetch_fundamentals(["AAPL"], cache_dir=str(tmp_path))

        assert result["AAPL"]["pe_ratio"] == 28.5
        assert result["AAPL"]["market_cap"] == 2_500_000_000_000
        assert result["AAPL"]["avg_volume"] == 60_000_000
        assert result["AAPL"]["price"] == 182.45

        json_file = tmp_path / "AAPL.json"
        assert json_file.exists()
        with json_file.open() as fh:
            cached = json.load(fh)
        assert cached == result["AAPL"]

    def test_missing_fields_use_none(self, tmp_path: Path) -> None:
        mock_ticker_instance = MagicMock()
        type(mock_ticker_instance).info = property(lambda self: {})

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
            with patch("src.data.fetcher.time.sleep"):
                result = fetch_fundamentals(["AAPL"], cache_dir=str(tmp_path))

        assert result["AAPL"] == {
            "pe_ratio": None,
            "market_cap": None,
            "avg_volume": None,
            "price": None,
        }


# ---------------------------------------------------------------------------
# fetch_fundamentals — retry
# ---------------------------------------------------------------------------


class TestFetchFundamentalsRetry:
    def test_retry_succeeds_on_third_attempt(self, tmp_path: Path) -> None:
        info = _make_info_dict()
        call_count = 0

        def info_property(self: object) -> dict:  # type: ignore[type-arg]
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("network error")
            return info

        mock_ticker_instance = MagicMock()
        type(mock_ticker_instance).info = property(info_property)

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
            with patch("src.data.fetcher.time.sleep"):
                result = fetch_fundamentals(["AAPL"], cache_dir=str(tmp_path))

        assert "AAPL" in result

    def test_retry_exhausted_skips_ticker(self, tmp_path: Path) -> None:
        mock_ticker_instance = MagicMock()
        type(mock_ticker_instance).info = property(
            lambda self: (_ for _ in ()).throw(ConnectionError("always fails"))
        )

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
            with patch("src.data.fetcher.time.sleep"):
                result = fetch_fundamentals(["AAPL"], cache_dir=str(tmp_path))

        assert "AAPL" not in result


# ---------------------------------------------------------------------------
# clear_cache
# ---------------------------------------------------------------------------


class TestClearCache:
    def test_clears_existing_cache_dir(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "mycache"
        cache_dir.mkdir()
        (cache_dir / "AAPL.pkl").write_bytes(b"data")
        (cache_dir / "MSFT.json").write_text("{}")

        clear_cache(str(cache_dir))

        assert not cache_dir.exists()

    def test_clear_nonexistent_dir_is_noop(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist"
        clear_cache(str(nonexistent))
        assert not nonexistent.exists()

    def test_clear_removes_nested_subdirs(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        sub = cache_dir / "ohlcv"
        sub.mkdir(parents=True)
        (sub / "AAPL.pkl").write_bytes(b"x")

        clear_cache(str(cache_dir))

        assert not cache_dir.exists()


# ---------------------------------------------------------------------------
# _batch_download_since — yfinance incremental fallback
# ---------------------------------------------------------------------------


class TestBatchDownloadSince:
    """Unit tests for the _batch_download_since helper (used as Tiingo fallback)."""

    def test_returns_data_multiindex(self) -> None:
        from src.data.fetcher import _batch_download_since

        since = date(2025, 1, 2)
        df = _make_ohlcv_df(3, end=date(2025, 1, 6))
        multi_index = pd.MultiIndex.from_tuples(
            [(col, "AAPL") for col in df.columns], names=["metric", "ticker"]
        )
        raw = pd.DataFrame(df.values, index=df.index, columns=multi_index)

        with patch("src.data.fetcher.yf.download", return_value=raw):
            result = _batch_download_since(["AAPL"], since)

        assert "AAPL" in result
        assert not result["AAPL"].empty

    def test_empty_download_returns_empty_dict(self) -> None:
        from src.data.fetcher import _batch_download_since

        since = date(2025, 1, 2)

        with patch("src.data.fetcher.yf.download", return_value=pd.DataFrame()):
            result = _batch_download_since(["AAPL"], since)

        assert result == {}

    def test_retry_on_network_failure(self) -> None:
        from src.data.fetcher import _batch_download_since

        since = date(2025, 1, 2)

        with patch("src.data.fetcher.yf.download", side_effect=ConnectionError("fail")):
            with patch("src.data.fetcher.time.sleep"):
                result = _batch_download_since(["AAPL"], since)

        assert result == {}

    def test_flat_columns_single_ticker(self) -> None:
        from src.data.fetcher import _batch_download_since

        since = date(2025, 1, 2)
        df = _make_ohlcv_df(3, end=date(2025, 1, 6))

        with patch("src.data.fetcher.yf.download", return_value=df):
            result = _batch_download_since(["AAPL"], since)

        assert "AAPL" in result
