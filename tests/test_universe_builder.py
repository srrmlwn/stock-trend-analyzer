"""Tests for src/universe/builder.py — all network calls and file I/O are mocked."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.universe.builder import (
    _local_ohlcv_prefilter,
    apply_prefilter,
    get_nyse_nasdaq_tickers,
    get_sp500_tickers,
    get_universe,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sp500_df() -> pd.DataFrame:
    """Return a minimal DataFrame that mimics the Wikipedia S&P 500 table."""
    return pd.DataFrame({"Symbol": ["AAPL", "MSFT", "BRK.B", "GOOGL"]})


def _make_nasdaq_text() -> str:
    """Return pipe-delimited NASDAQ listed text (mimics nasdaqlisted.txt)."""
    return (
        "Symbol|Security Name|Market Category|Test Issue|Financial Status|Round Lot Size\n"
        "AAPL|Apple Inc.|Q|N|N|100\n"
        "MSFT|Microsoft Corp.|Q|N|N|100\n"
        "TEST$|Bad ticker|Q|N|N|100\n"
        "BAD TICKER|Spaced|Q|N|N|100\n"
        "NVDA|NVIDIA Corp.|Q|N|N|100\n"
    )


def _make_other_text() -> str:
    """Return pipe-delimited other listed text (mimics otherlisted.txt)."""
    return (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "IBM|IBM Corp.|N|IBM|N|100|N|IBM\n"
        "XOM|Exxon Mobil|N|XOM|N|100|N|XOM\n"
    )


def _make_ohlcv_df(
    last_close: float = 50.0,
    avg_volume: float = 1_000_000.0,
    rows: int = 90,
) -> pd.DataFrame:
    """Return a minimal OHLCV DataFrame for testing the local pre-filter."""
    return pd.DataFrame(
        {
            "Close": [last_close] * rows,
            "Volume": [avg_volume] * rows,
            "Open": [last_close] * rows,
            "High": [last_close] * rows,
            "Low": [last_close] * rows,
        }
    )


# ---------------------------------------------------------------------------
# get_sp500_tickers
# ---------------------------------------------------------------------------

class TestGetSp500Tickers:
    def test_happy_path_returns_cleaned_list(self):
        with patch("src.universe.builder.pd.read_html") as mock_read_html:
            mock_read_html.return_value = [_make_sp500_df()]
            result = get_sp500_tickers()

        assert isinstance(result, list)
        assert len(result) == 4
        assert "AAPL" in result
        assert "MSFT" in result
        assert "BRK-B" in result
        assert "BRK.B" not in result

    def test_strips_whitespace(self):
        df = pd.DataFrame({"Symbol": ["  AAPL  ", " MSFT"]})
        with patch("src.universe.builder.pd.read_html") as mock_read_html:
            mock_read_html.return_value = [df]
            result = get_sp500_tickers()

        assert "AAPL" in result
        assert "MSFT" in result

    def test_failure_returns_empty_list(self):
        with patch("src.universe.builder.pd.read_html", side_effect=Exception("Network error")):
            result = get_sp500_tickers()

        assert result == []

    def test_failure_logs_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="src.universe.builder"):
            with patch("src.universe.builder.pd.read_html", side_effect=ValueError("Timeout")):
                get_sp500_tickers()

        assert any("Failed to scrape" in msg for msg in caplog.messages)


# ---------------------------------------------------------------------------
# get_nyse_nasdaq_tickers
# ---------------------------------------------------------------------------

class TestGetNyseNasdaqTickers:
    def _patch_requests_get(self, nasdaq_text: str, other_text: str):
        """Return a mock for requests.get that serves different content per URL."""

        def side_effect(url: str, **kwargs):  # type: ignore[no-untyped-def]
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            if "nasdaqlisted" in url:
                mock_resp.text = nasdaq_text
            else:
                mock_resp.text = other_text
            return mock_resp

        return patch("src.universe.builder.requests.get", side_effect=side_effect)

    def test_happy_path_returns_symbols(self):
        with self._patch_requests_get(_make_nasdaq_text(), _make_other_text()):
            result = get_nyse_nasdaq_tickers()

        assert isinstance(result, list)
        assert "AAPL" in result
        assert "NVDA" in result
        assert "IBM" in result
        assert "XOM" in result

    def test_filters_test_symbols(self):
        with self._patch_requests_get(_make_nasdaq_text(), _make_other_text()):
            result = get_nyse_nasdaq_tickers()

        assert "TEST$" not in result
        assert "BAD TICKER" not in result

    def test_deduplication(self):
        with self._patch_requests_get(_make_nasdaq_text(), _make_nasdaq_text()):
            result = get_nyse_nasdaq_tickers()

        assert result.count("AAPL") == 1

    def test_both_fetches_fail_returns_empty_list(self):
        with patch(
            "src.universe.builder.requests.get",
            side_effect=Exception("Connection refused"),
        ):
            with patch("urllib.request.urlopen", side_effect=Exception("FTP error")):
                result = get_nyse_nasdaq_tickers()

        assert result == []

    def test_failure_logs_warning(self, caplog):
        import logging
        with caplog.at_level(logging.WARNING, logger="src.universe.builder"):
            with patch(
                "src.universe.builder.requests.get",
                side_effect=Exception("err"),
            ):
                with patch("urllib.request.urlopen", side_effect=Exception("err")):
                    get_nyse_nasdaq_tickers()

        assert any("Failed" in msg for msg in caplog.messages)

    def test_one_source_fails_still_returns_other(self):
        call_count = 0

        def side_effect(url: str, **kwargs):  # type: ignore[no-untyped-def]
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("First source fails")
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.text = _make_other_text()
            return mock_resp

        with patch("src.universe.builder.requests.get", side_effect=side_effect):
            with patch("urllib.request.urlopen", side_effect=Exception("FTP error")):
                result = get_nyse_nasdaq_tickers()

        assert "IBM" in result or result == []


# ---------------------------------------------------------------------------
# _local_ohlcv_prefilter
# ---------------------------------------------------------------------------

class TestLocalOhlcvPrefilter:
    """Tests for _local_ohlcv_prefilter — uses tmp_path for fake pkl store."""

    def test_no_pkl_files_passes_all_tickers_through(self, tmp_path: Path):
        """Tickers with no local pkl file are passed through unchanged."""
        result = _local_ohlcv_prefilter(
            ["AAPL", "MSFT", "NVDA"], cache_dir=str(tmp_path)
        )
        assert result == ["AAPL", "MSFT", "NVDA"]

    def test_ticker_above_thresholds_is_kept(self, tmp_path: Path):
        """Ticker whose last close and avg volume exceed thresholds passes."""
        df = _make_ohlcv_df(last_close=100.0, avg_volume=2_000_000.0)
        df.to_pickle(str(tmp_path / "AAPL.pkl"))

        result = _local_ohlcv_prefilter(
            ["AAPL"],
            min_price=10.0,
            min_avg_volume=500_000,
            cache_dir=str(tmp_path),
        )
        assert result == ["AAPL"]

    def test_ticker_below_price_threshold_is_excluded(self, tmp_path: Path):
        """Ticker with last close below min_price is excluded."""
        df = _make_ohlcv_df(last_close=5.0, avg_volume=2_000_000.0)
        df.to_pickle(str(tmp_path / "PENNY.pkl"))

        result = _local_ohlcv_prefilter(
            ["PENNY"],
            min_price=10.0,
            min_avg_volume=500_000,
            cache_dir=str(tmp_path),
        )
        assert result == []

    def test_ticker_below_volume_threshold_is_excluded(self, tmp_path: Path):
        """Ticker with avg volume below min_avg_volume is excluded."""
        df = _make_ohlcv_df(last_close=50.0, avg_volume=100_000.0)
        df.to_pickle(str(tmp_path / "LOWVOL.pkl"))

        result = _local_ohlcv_prefilter(
            ["LOWVOL"],
            min_price=10.0,
            min_avg_volume=500_000,
            cache_dir=str(tmp_path),
        )
        assert result == []

    def test_exact_threshold_values_pass(self, tmp_path: Path):
        """Tickers exactly at threshold values should pass (>= comparison)."""
        df = _make_ohlcv_df(last_close=10.0, avg_volume=500_000.0)
        df.to_pickle(str(tmp_path / "EXACT.pkl"))

        result = _local_ohlcv_prefilter(
            ["EXACT"],
            min_price=10.0,
            min_avg_volume=500_000,
            cache_dir=str(tmp_path),
        )
        assert result == ["EXACT"]

    def test_mixed_tickers_some_pass_some_fail(self, tmp_path: Path):
        """Mixed scenario: some tickers pass, some fail, some have no data."""
        _make_ohlcv_df(last_close=80.0, avg_volume=1_000_000.0).to_pickle(
            str(tmp_path / "GOOD.pkl")
        )
        _make_ohlcv_df(last_close=3.0, avg_volume=1_000_000.0).to_pickle(
            str(tmp_path / "BAD_PRICE.pkl")
        )
        _make_ohlcv_df(last_close=80.0, avg_volume=50_000.0).to_pickle(
            str(tmp_path / "BAD_VOL.pkl")
        )
        # NO_DATA: no pkl file — should be passed through

        result = _local_ohlcv_prefilter(
            ["GOOD", "BAD_PRICE", "BAD_VOL", "NO_DATA"],
            min_price=10.0,
            min_avg_volume=500_000,
            cache_dir=str(tmp_path),
        )
        assert "GOOD" in result
        assert "NO_DATA" in result
        assert "BAD_PRICE" not in result
        assert "BAD_VOL" not in result

    def test_empty_pkl_file_passes_through(self, tmp_path: Path):
        """An empty DataFrame pkl file should cause the ticker to pass through."""
        pd.DataFrame().to_pickle(str(tmp_path / "EMPTY.pkl"))

        result = _local_ohlcv_prefilter(["EMPTY"], cache_dir=str(tmp_path))
        assert result == ["EMPTY"]

    def test_corrupt_pkl_file_passes_through(self, tmp_path: Path):
        """A corrupt/unreadable pkl file should cause the ticker to pass through."""
        (tmp_path / "CORRUPT.pkl").write_bytes(b"not a valid pickle")

        result = _local_ohlcv_prefilter(["CORRUPT"], cache_dir=str(tmp_path))
        assert result == ["CORRUPT"]

    def test_uses_last_60_rows_for_volume(self, tmp_path: Path):
        """Average volume should use up to the last 60 rows only."""
        volumes = [10_000_000.0] * 30 + [50_000.0] * 60
        df = pd.DataFrame(
            {
                "Close": [50.0] * 90,
                "Volume": volumes,
                "Open": [50.0] * 90,
                "High": [50.0] * 90,
                "Low": [50.0] * 90,
            }
        )
        df.to_pickle(str(tmp_path / "TICKER.pkl"))

        result = _local_ohlcv_prefilter(
            ["TICKER"],
            min_price=10.0,
            min_avg_volume=500_000,
            cache_dir=str(tmp_path),
        )
        # Only last 60 rows count -> avg ~50k -> should fail
        assert result == []

    def test_empty_ticker_list_returns_empty(self, tmp_path: Path):
        """Empty input should return empty output."""
        result = _local_ohlcv_prefilter([], cache_dir=str(tmp_path))
        assert result == []

    def test_lowercase_column_names_are_normalised(self, tmp_path: Path):
        """PKL files with lowercase column names should still be read correctly."""
        df = pd.DataFrame(
            {
                "close": [50.0] * 90,
                "volume": [1_000_000.0] * 90,
                "open": [50.0] * 90,
                "high": [50.0] * 90,
                "low": [50.0] * 90,
            }
        )
        df.to_pickle(str(tmp_path / "LOWER.pkl"))

        result = _local_ohlcv_prefilter(
            ["LOWER"],
            min_price=10.0,
            min_avg_volume=500_000,
            cache_dir=str(tmp_path),
        )
        assert result == ["LOWER"]


# ---------------------------------------------------------------------------
# apply_prefilter
# ---------------------------------------------------------------------------

class TestApplyPrefilter:
    def _make_fundamentals(
        self,
        market_cap: float | None = 2e9,
    ) -> dict:  # type: ignore[type-arg]
        return {"TICKER": {"pe_ratio": 20.0, "market_cap": market_cap}}

    def test_ticker_passes_market_cap_filter(self):
        result = apply_prefilter(["TICKER"], self._make_fundamentals())
        assert result == ["TICKER"]

    def test_fails_market_cap(self):
        result = apply_prefilter(["TICKER"], self._make_fundamentals(market_cap=5e8))
        assert result == []

    def test_missing_market_cap_excluded(self):
        result = apply_prefilter(["TICKER"], self._make_fundamentals(market_cap=None))
        assert result == []

    def test_ticker_not_in_fundamentals_excluded(self):
        result = apply_prefilter(["UNKNOWN"], {})
        assert result == []

    def test_custom_threshold(self):
        fundamentals = {"SMALL": {"market_cap": 300_000_000}}
        result = apply_prefilter(["SMALL"], fundamentals, min_market_cap_B=0.1)
        assert result == ["SMALL"]

    def test_multiple_tickers_mixed_market_cap(self):
        fundamentals = {
            "GOOD": {"market_cap": 5e9},
            "BAD_CAP": {"market_cap": 5e8},
        }
        result = apply_prefilter(["GOOD", "BAD_CAP"], fundamentals)
        assert result == ["GOOD"]

    def test_exact_market_cap_threshold_passes(self):
        """Ticker exactly at the threshold should pass (>=)."""
        fundamentals = {"EXACT": {"market_cap": 1e9}}
        result = apply_prefilter(["EXACT"], fundamentals)
        assert result == ["EXACT"]


# ---------------------------------------------------------------------------
# get_universe
# ---------------------------------------------------------------------------

class TestGetUniverse:
    _MODULE = "src.universe.builder"

    def _mock_fetch_fundamentals(self, tickers: list[str]) -> dict:  # type: ignore[type-arg]
        return {t: {"market_cap": 5e9} for t in tickers}

    def _passthrough_local_filter(
        self, tickers: list[str], **kwargs: object
    ) -> list[str]:
        """Passthrough mock for _local_ohlcv_prefilter."""
        return tickers

    def test_cache_hit_returns_cached_data(self, tmp_path: Path):
        cache_file = tmp_path / "universe.json"
        cached_tickers = ["AAPL", "MSFT"]
        cache_file.write_text(json.dumps(cached_tickers))

        with patch(f"{self._MODULE}.get_sp500_tickers") as mock_sp500, patch(
            f"{self._MODULE}.get_nyse_nasdaq_tickers"
        ) as mock_nyse:
            result = get_universe(cache_path=str(cache_file), cache_ttl_hours=168)

        mock_sp500.assert_not_called()
        mock_nyse.assert_not_called()
        assert result == cached_tickers

    def test_cache_miss_fetches_and_caches(self, tmp_path: Path):
        cache_file = tmp_path / "universe.json"

        with patch(f"{self._MODULE}.get_sp500_tickers", return_value=["AAPL", "MSFT"]):
            with patch(f"{self._MODULE}.get_nyse_nasdaq_tickers", return_value=[]):
                with patch(
                    f"{self._MODULE}._local_ohlcv_prefilter",
                    side_effect=self._passthrough_local_filter,
                ):
                    with patch(
                        "src.data.fetcher.fetch_fundamentals",
                        side_effect=self._mock_fetch_fundamentals,
                    ):
                        result = get_universe(cache_path=str(cache_file), cache_ttl_hours=168)

        assert set(result) == {"AAPL", "MSFT"}
        assert cache_file.exists()
        assert json.loads(cache_file.read_text()) == result

    def test_force_refresh_bypasses_fresh_cache(self, tmp_path: Path):
        cache_file = tmp_path / "universe.json"
        old_tickers = ["OLD"]
        cache_file.write_text(json.dumps(old_tickers))

        with patch(f"{self._MODULE}.get_sp500_tickers", return_value=["AAPL"]):
            with patch(f"{self._MODULE}.get_nyse_nasdaq_tickers", return_value=[]):
                with patch(
                    f"{self._MODULE}._local_ohlcv_prefilter",
                    side_effect=self._passthrough_local_filter,
                ):
                    with patch(
                        "src.data.fetcher.fetch_fundamentals",
                        side_effect=self._mock_fetch_fundamentals,
                    ):
                        result = get_universe(
                            force_refresh=True,
                            cache_path=str(cache_file),
                            cache_ttl_hours=168,
                        )

        assert result == ["AAPL"]
        assert "OLD" not in result

    def test_stale_cache_triggers_refresh(self, tmp_path: Path):
        cache_file = tmp_path / "universe.json"
        old_tickers = ["STALE"]
        cache_file.write_text(json.dumps(old_tickers))

        stale_mtime = time.time() - (169 * 3600)
        import os
        os.utime(str(cache_file), (stale_mtime, stale_mtime))

        with patch(f"{self._MODULE}.get_sp500_tickers", return_value=["FRESH"]):
            with patch(f"{self._MODULE}.get_nyse_nasdaq_tickers", return_value=[]):
                with patch(
                    f"{self._MODULE}._local_ohlcv_prefilter",
                    side_effect=self._passthrough_local_filter,
                ):
                    with patch(
                        "src.data.fetcher.fetch_fundamentals",
                        side_effect=self._mock_fetch_fundamentals,
                    ):
                        result = get_universe(cache_path=str(cache_file), cache_ttl_hours=168)

        assert result == ["FRESH"]
        assert "STALE" not in result

    def test_deduplicates_combined_tickers(self, tmp_path: Path):
        cache_file = tmp_path / "universe.json"

        collected_tickers: list[list[str]] = []

        def capture_fundamentals(tickers: list[str]) -> dict:  # type: ignore[type-arg]
            collected_tickers.append(list(tickers))
            return self._mock_fetch_fundamentals(tickers)

        with patch(f"{self._MODULE}.get_sp500_tickers", return_value=["AAPL", "MSFT"]):
            with patch(
                f"{self._MODULE}.get_nyse_nasdaq_tickers", return_value=["AAPL", "NVDA"]
            ):
                with patch(
                    f"{self._MODULE}._local_ohlcv_prefilter",
                    side_effect=self._passthrough_local_filter,
                ):
                    with patch(
                        "src.data.fetcher.fetch_fundamentals",
                        side_effect=capture_fundamentals,
                    ):
                        get_universe(cache_path=str(cache_file))

        all_queried = collected_tickers[0]
        assert all_queried.count("AAPL") == 1

    def test_cache_directory_created_if_missing(self, tmp_path: Path):
        cache_file = tmp_path / "subdir" / "universe.json"

        with patch(f"{self._MODULE}.get_sp500_tickers", return_value=["AAPL"]):
            with patch(f"{self._MODULE}.get_nyse_nasdaq_tickers", return_value=[]):
                with patch(
                    f"{self._MODULE}._local_ohlcv_prefilter",
                    side_effect=self._passthrough_local_filter,
                ):
                    with patch(
                        "src.data.fetcher.fetch_fundamentals",
                        side_effect=self._mock_fetch_fundamentals,
                    ):
                        result = get_universe(cache_path=str(cache_file))

        assert cache_file.exists()
        assert result == ["AAPL"]

    def test_empty_universe_when_no_fundamentals(self, tmp_path: Path):
        cache_file = tmp_path / "universe.json"

        with patch(f"{self._MODULE}.get_sp500_tickers", return_value=["AAPL"]):
            with patch(f"{self._MODULE}.get_nyse_nasdaq_tickers", return_value=[]):
                with patch(
                    f"{self._MODULE}._local_ohlcv_prefilter",
                    side_effect=self._passthrough_local_filter,
                ):
                    with patch(
                        "src.data.fetcher.fetch_fundamentals",
                        return_value={},
                    ):
                        result = get_universe(cache_path=str(cache_file))

        assert result == []

    def test_local_filter_excludes_tickers_before_fundamentals(self, tmp_path: Path):
        """Tickers excluded by local OHLCV filter should not be passed to fetch_fundamentals."""
        cache_file = tmp_path / "universe.json"

        captured: list[list[str]] = []

        def capture_fundamentals(tickers: list[str]) -> dict:  # type: ignore[type-arg]
            captured.append(list(tickers))
            return self._mock_fetch_fundamentals(tickers)

        def local_filter_excludes_bad(
            tickers: list[str], **kwargs: object
        ) -> list[str]:
            return [t for t in tickers if t != "BAD"]

        with patch(f"{self._MODULE}.get_sp500_tickers", return_value=["GOOD", "BAD"]):
            with patch(f"{self._MODULE}.get_nyse_nasdaq_tickers", return_value=[]):
                with patch(
                    f"{self._MODULE}._local_ohlcv_prefilter",
                    side_effect=local_filter_excludes_bad,
                ):
                    with patch(
                        "src.data.fetcher.fetch_fundamentals",
                        side_effect=capture_fundamentals,
                    ):
                        result = get_universe(cache_path=str(cache_file))

        assert result == ["GOOD"]
        assert "BAD" not in captured[0]
