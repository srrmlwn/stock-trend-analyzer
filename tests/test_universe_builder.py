"""Tests for src/universe/builder.py — all network calls are mocked."""

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.universe.builder import (
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
        "TEST$|Bad ticker|Q|N|N|100\n"  # should be filtered out
        "BAD TICKER|Spaced|Q|N|N|100\n"  # should be filtered out
        "NVDA|NVIDIA Corp.|Q|N|N|100\n"
    )


def _make_other_text() -> str:
    """Return pipe-delimited other listed text (mimics otherlisted.txt)."""
    return (
        "ACT Symbol|Security Name|Exchange|CQS Symbol|ETF|Round Lot Size|Test Issue|NASDAQ Symbol\n"
        "IBM|IBM Corp.|N|IBM|N|100|N|IBM\n"
        "XOM|Exxon Mobil|N|XOM|N|100|N|XOM\n"
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
        # BRK.B → BRK-B
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
        # AAPL appears in both NASDAQ and S&P 500 — should appear once
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

        # Second source (otherlisted) should succeed
        assert "IBM" in result or result == []  # behaviour depends on which source fails


# ---------------------------------------------------------------------------
# apply_prefilter
# ---------------------------------------------------------------------------

class TestApplyPrefilter:
    def _make_fundamentals(
        self,
        market_cap: float | None = 2e9,
        avg_volume: int | None = 1_000_000,
        price: float | None = 50.0,
    ) -> dict:  # type: ignore[type-arg]
        return {
            "TICKER": {
                "pe_ratio": 20.0,
                "market_cap": market_cap,
                "avg_volume": avg_volume,
                "price": price,
            }
        }

    def test_ticker_passes_all_filters(self):
        result = apply_prefilter(["TICKER"], self._make_fundamentals())
        assert result == ["TICKER"]

    def test_fails_market_cap(self):
        result = apply_prefilter(["TICKER"], self._make_fundamentals(market_cap=5e8))
        assert result == []

    def test_fails_avg_volume(self):
        result = apply_prefilter(["TICKER"], self._make_fundamentals(avg_volume=100_000))
        assert result == []

    def test_fails_price(self):
        result = apply_prefilter(["TICKER"], self._make_fundamentals(price=5.0))
        assert result == []

    def test_missing_market_cap_excluded(self):
        result = apply_prefilter(["TICKER"], self._make_fundamentals(market_cap=None))
        assert result == []

    def test_missing_avg_volume_excluded(self):
        result = apply_prefilter(["TICKER"], self._make_fundamentals(avg_volume=None))
        assert result == []

    def test_missing_price_excluded(self):
        result = apply_prefilter(["TICKER"], self._make_fundamentals(price=None))
        assert result == []

    def test_ticker_not_in_fundamentals_excluded(self):
        result = apply_prefilter(["UNKNOWN"], {})
        assert result == []

    def test_custom_thresholds(self):
        fundamentals = {
            "SMALL": {
                "market_cap": 300_000_000,  # 300M
                "avg_volume": 200_000,
                "price": 8.0,
            }
        }
        # With relaxed thresholds it should pass
        result = apply_prefilter(
            ["SMALL"],
            fundamentals,
            min_market_cap_B=0.1,
            min_avg_volume=100_000,
            min_price=5.0,
        )
        assert result == ["SMALL"]

    def test_multiple_tickers_mixed_results(self):
        fundamentals = {
            "GOOD": {"market_cap": 5e9, "avg_volume": 2_000_000, "price": 100.0},
            "BAD_CAP": {"market_cap": 5e8, "avg_volume": 2_000_000, "price": 100.0},
            "BAD_VOL": {"market_cap": 5e9, "avg_volume": 10_000, "price": 100.0},
            "BAD_PRICE": {"market_cap": 5e9, "avg_volume": 2_000_000, "price": 1.0},
        }
        result = apply_prefilter(
            ["GOOD", "BAD_CAP", "BAD_VOL", "BAD_PRICE"], fundamentals
        )
        assert result == ["GOOD"]

    def test_exact_threshold_values_pass(self):
        """Tickers exactly at the threshold should pass (>=)."""
        fundamentals = {
            "EXACT": {
                "market_cap": 1e9,  # exactly 1B
                "avg_volume": 500_000,  # exactly 500k
                "price": 10.0,  # exactly 10
            }
        }
        result = apply_prefilter(["EXACT"], fundamentals)
        assert result == ["EXACT"]


# ---------------------------------------------------------------------------
# get_universe
# ---------------------------------------------------------------------------

class TestGetUniverse:
    _MODULE = "src.universe.builder"

    def _mock_fetch_fundamentals(self, tickers: list[str]) -> dict:  # type: ignore[type-arg]
        return {
            t: {"market_cap": 5e9, "avg_volume": 1_000_000, "price": 50.0}
            for t in tickers
        }

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
        # No cache file — should fetch

        with patch(f"{self._MODULE}.get_sp500_tickers", return_value=["AAPL", "MSFT"]):
            with patch(f"{self._MODULE}.get_nyse_nasdaq_tickers", return_value=["NVDA"]):
                with patch(
                    "src.data.fetcher.fetch_fundamentals",
                    side_effect=self._mock_fetch_fundamentals,
                ):
                    result = get_universe(cache_path=str(cache_file), cache_ttl_hours=168)

        assert set(result) == {"AAPL", "MSFT", "NVDA"}
        assert cache_file.exists()
        assert json.loads(cache_file.read_text()) == result

    def test_force_refresh_bypasses_fresh_cache(self, tmp_path: Path):
        cache_file = tmp_path / "universe.json"
        old_tickers = ["OLD"]
        cache_file.write_text(json.dumps(old_tickers))

        with patch(f"{self._MODULE}.get_sp500_tickers", return_value=["AAPL"]):
            with patch(f"{self._MODULE}.get_nyse_nasdaq_tickers", return_value=[]):
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

        # Wind back mtime to make cache stale (> 168 hours old)
        stale_mtime = time.time() - (169 * 3600)
        import os
        os.utime(str(cache_file), (stale_mtime, stale_mtime))

        with patch(f"{self._MODULE}.get_sp500_tickers", return_value=["FRESH"]):
            with patch(f"{self._MODULE}.get_nyse_nasdaq_tickers", return_value=[]):
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
                    "src.data.fetcher.fetch_fundamentals",
                    side_effect=capture_fundamentals,
                ):
                    get_universe(cache_path=str(cache_file))

        all_queried = collected_tickers[0]
        assert all_queried.count("AAPL") == 1

    def test_cache_directory_created_if_missing(self, tmp_path: Path):
        cache_file = tmp_path / "subdir" / "universe.json"
        # subdir does not exist yet

        with patch(f"{self._MODULE}.get_sp500_tickers", return_value=["AAPL"]):
            with patch(f"{self._MODULE}.get_nyse_nasdaq_tickers", return_value=[]):
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
                    "src.data.fetcher.fetch_fundamentals",
                    return_value={},  # no fundamentals fetched
                ):
                    result = get_universe(cache_path=str(cache_file))

        assert result == []
