"""Tests for src/data/fetcher.py — all yfinance calls are mocked (offline)."""

import json
import os
import pickle
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from src.data.fetcher import clear_cache, fetch_fundamentals, fetch_ohlcv


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ohlcv_df(n: int = 5) -> pd.DataFrame:
    """Return a minimal synthetic OHLCV DataFrame with *n* rows."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
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


def _make_info_dict() -> dict:
    return {
        "trailingPE": 28.5,
        "marketCap": 2_500_000_000_000,
        "averageVolume": 60_000_000,
        "currentPrice": 182.45,
    }


# ---------------------------------------------------------------------------
# fetch_ohlcv – cache hit
# ---------------------------------------------------------------------------

class TestFetchOhlcvCacheHit:
    """yfinance must NOT be called when a fresh cache file exists."""

    def test_cache_hit_returns_cached_df(self, tmp_path: Path) -> None:
        df = _make_ohlcv_df()
        pkl = tmp_path / "AAPL.pkl"
        with pkl.open("wb") as fh:
            pickle.dump(df, fh)
        # mtime is already "now", so cache is fresh

        with patch("src.data.fetcher.yf.Ticker") as mock_ticker:
            result = fetch_ohlcv(["AAPL"], cache_dir=str(tmp_path))

        mock_ticker.assert_not_called()
        assert "AAPL" in result
        pd.testing.assert_frame_equal(result["AAPL"], df)

    def test_cache_hit_multiple_tickers(self, tmp_path: Path) -> None:
        for sym in ("AAPL", "MSFT"):
            df = _make_ohlcv_df()
            with (tmp_path / f"{sym}.pkl").open("wb") as fh:
                pickle.dump(df, fh)

        with patch("src.data.fetcher.yf.Ticker") as mock_ticker:
            result = fetch_ohlcv(["AAPL", "MSFT"], cache_dir=str(tmp_path))

        mock_ticker.assert_not_called()
        assert set(result.keys()) == {"AAPL", "MSFT"}


# ---------------------------------------------------------------------------
# fetch_ohlcv – cache miss
# ---------------------------------------------------------------------------

class TestFetchOhlcvCacheMiss:
    """yfinance IS called when no cache file exists."""

    def test_cache_miss_fetches_and_saves(self, tmp_path: Path) -> None:
        df = _make_ohlcv_df()

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = df

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance) as mock_ticker_cls:
            result = fetch_ohlcv(["AAPL"], cache_dir=str(tmp_path))

        mock_ticker_cls.assert_called_once_with("AAPL")
        mock_ticker_instance.history.assert_called_once()
        assert "AAPL" in result
        pd.testing.assert_frame_equal(result["AAPL"], df)

        # Verify the pickle was written
        pkl = tmp_path / "AAPL.pkl"
        assert pkl.exists()
        with pkl.open("rb") as fh:
            cached_df = pickle.load(fh)
        pd.testing.assert_frame_equal(cached_df, df)

    def test_cache_miss_creates_cache_dir(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "ohlcv"
        df = _make_ohlcv_df()

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = df

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
            fetch_ohlcv(["AAPL"], cache_dir=str(nested))

        assert nested.exists()


# ---------------------------------------------------------------------------
# fetch_ohlcv – stale cache
# ---------------------------------------------------------------------------

class TestFetchOhlcvStaleCache:
    """A cache file older than 24 h must trigger a fresh yfinance fetch."""

    def test_stale_cache_refetches(self, tmp_path: Path) -> None:
        stale_df = _make_ohlcv_df(3)
        fresh_df = _make_ohlcv_df(10)

        pkl = tmp_path / "AAPL.pkl"
        with pkl.open("wb") as fh:
            pickle.dump(stale_df, fh)

        # Back-date mtime to 25 hours ago
        old_ts = time.time() - 25 * 3600
        os.utime(pkl, (old_ts, old_ts))

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = fresh_df

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
            result = fetch_ohlcv(["AAPL"], cache_dir=str(tmp_path))

        mock_ticker_instance.history.assert_called_once()
        pd.testing.assert_frame_equal(result["AAPL"], fresh_df)


# ---------------------------------------------------------------------------
# fetch_ohlcv – retry on network failure
# ---------------------------------------------------------------------------

class TestFetchOhlcvRetry:
    """Verify exponential-backoff retry: fail twice, succeed on third attempt."""

    def test_retry_succeeds_on_third_attempt(self, tmp_path: Path) -> None:
        df = _make_ohlcv_df()

        side_effects = [
            ConnectionError("network error"),
            ConnectionError("network error"),
            df,
        ]

        call_count = 0

        def fake_history(**kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            val = side_effects[call_count]
            call_count += 1
            if isinstance(val, Exception):
                raise val
            return val

        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = fake_history

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
            with patch("src.data.fetcher.time.sleep") as mock_sleep:
                result = fetch_ohlcv(["AAPL"], cache_dir=str(tmp_path))

        assert "AAPL" in result
        assert mock_ticker_instance.history.call_count == 3
        # Two sleeps: 2s after attempt 1, 4s after attempt 2
        assert mock_sleep.call_count == 2
        assert mock_sleep.call_args_list[0] == call(2.0)
        assert mock_sleep.call_args_list[1] == call(4.0)

    def test_retry_exhausted_skips_ticker(self, tmp_path: Path) -> None:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.side_effect = ConnectionError("always fails")

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
            with patch("src.data.fetcher.time.sleep"):
                result = fetch_ohlcv(["AAPL"], cache_dir=str(tmp_path))

        assert "AAPL" not in result
        assert mock_ticker_instance.history.call_count == 3


# ---------------------------------------------------------------------------
# fetch_ohlcv – invalid / empty ticker
# ---------------------------------------------------------------------------

class TestFetchOhlcvInvalidTicker:
    """Empty DataFrame returned by yfinance must be silently skipped."""

    def test_empty_df_skipped(self, tmp_path: Path) -> None:
        mock_ticker_instance = MagicMock()
        mock_ticker_instance.history.return_value = pd.DataFrame()

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
            result = fetch_ohlcv(["INVALID"], cache_dir=str(tmp_path))

        assert "INVALID" not in result
        # No pickle should have been created
        assert not (tmp_path / "INVALID.pkl").exists()

    def test_valid_and_invalid_ticker_mixed(self, tmp_path: Path) -> None:
        df = _make_ohlcv_df()

        def make_ticker(sym: str) -> MagicMock:
            instance = MagicMock()
            if sym == "AAPL":
                instance.history.return_value = df
            else:
                instance.history.return_value = pd.DataFrame()
            return instance

        with patch("src.data.fetcher.yf.Ticker", side_effect=make_ticker):
            result = fetch_ohlcv(["AAPL", "INVALID"], cache_dir=str(tmp_path))

        assert "AAPL" in result
        assert "INVALID" not in result


# ---------------------------------------------------------------------------
# fetch_fundamentals – cache hit
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


# ---------------------------------------------------------------------------
# fetch_fundamentals – cache miss
# ---------------------------------------------------------------------------

class TestFetchFundamentalsCacheMiss:
    def test_cache_miss_fetches_and_saves(self, tmp_path: Path) -> None:
        info = _make_info_dict()

        mock_ticker_instance = MagicMock()
        type(mock_ticker_instance).info = property(lambda self: info)

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
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
            result = fetch_fundamentals(["AAPL"], cache_dir=str(tmp_path))

        assert result["AAPL"] == {
            "pe_ratio": None,
            "market_cap": None,
            "avg_volume": None,
            "price": None,
        }


# ---------------------------------------------------------------------------
# fetch_fundamentals – stale cache
# ---------------------------------------------------------------------------

class TestFetchFundamentalsStaleCache:
    def test_stale_cache_refetches(self, tmp_path: Path) -> None:
        stale_data = {"pe_ratio": 10.0, "market_cap": 1, "avg_volume": 1, "price": 1.0}
        json_file = tmp_path / "AAPL.json"
        with json_file.open("w") as fh:
            json.dump(stale_data, fh)

        old_ts = time.time() - 25 * 3600
        os.utime(json_file, (old_ts, old_ts))

        fresh_info = _make_info_dict()
        mock_ticker_instance = MagicMock()
        type(mock_ticker_instance).info = property(lambda self: fresh_info)

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
            result = fetch_fundamentals(["AAPL"], cache_dir=str(tmp_path))

        assert result["AAPL"]["pe_ratio"] == 28.5


# ---------------------------------------------------------------------------
# fetch_fundamentals – retry on network failure
# ---------------------------------------------------------------------------

class TestFetchFundamentalsRetry:
    def test_retry_succeeds_on_third_attempt(self, tmp_path: Path) -> None:
        info = _make_info_dict()
        call_count = 0

        def info_property(self: object) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("network error")
            return info

        mock_ticker_instance = MagicMock()
        type(mock_ticker_instance).info = property(info_property)

        with patch("src.data.fetcher.yf.Ticker", return_value=mock_ticker_instance):
            with patch("src.data.fetcher.time.sleep") as mock_sleep:
                result = fetch_fundamentals(["AAPL"], cache_dir=str(tmp_path))

        assert "AAPL" in result
        assert mock_sleep.call_count == 2

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
        # Must not raise
        clear_cache(str(nonexistent))
        assert not nonexistent.exists()

    def test_clear_removes_nested_subdirs(self, tmp_path: Path) -> None:
        cache_dir = tmp_path / "cache"
        sub = cache_dir / "ohlcv"
        sub.mkdir(parents=True)
        (sub / "AAPL.pkl").write_bytes(b"x")

        clear_cache(str(cache_dir))

        assert not cache_dir.exists()
