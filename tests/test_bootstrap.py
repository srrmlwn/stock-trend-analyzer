"""Tests for src/data/bootstrap.py."""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import pytest

from src.data.bootstrap import (
    _get_csv_headers,
    _normalise_df,
    _process_csv_dir,
    _process_simfin_df,
    _write_pkl,
    run_bootstrap,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv_df(n: int = 5) -> pd.DataFrame:
    """Return a small OHLCV DataFrame with a DatetimeIndex named 'Date'."""
    dates = pd.date_range("2020-01-01", periods=n, freq="D", name="Date")
    return pd.DataFrame(
        {
            "Open": [100.0] * n,
            "High": [105.0] * n,
            "Low": [95.0] * n,
            "Close": [102.0] * n,
            "Volume": [1_000_000] * n,
        },
        index=dates,
    )


# ---------------------------------------------------------------------------
# _normalise_df
# ---------------------------------------------------------------------------


class TestNormaliseDf:
    def test_canonical_columns_pass_through(self) -> None:
        df = _make_ohlcv_df()
        result = _normalise_df(df, "AAPL")
        assert result is not None
        assert list(result.columns) == ["Open", "High", "Low", "Close", "Volume"]
        assert result.index.name == "Date"
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_lowercase_column_names_are_renamed(self) -> None:
        df = _make_ohlcv_df()
        df.columns = [c.lower() for c in df.columns]  # type: ignore[assignment]
        result = _normalise_df(df, "AAPL")
        assert result is not None
        assert "Close" in result.columns

    def test_mixed_case_column_names(self) -> None:
        df = _make_ohlcv_df().reset_index()
        df = df.rename(
            columns={
                "Date": "date",
                "Open": "OPEN",
                "High": "high",
                "Low": "Low",
                "Close": "CLOSE",
                "Volume": "VoLuMe",
            }
        )
        result = _normalise_df(df, "TEST")
        assert result is not None
        assert set(result.columns) == {"Open", "High", "Low", "Close", "Volume"}

    def test_all_nan_rows_dropped(self) -> None:
        df = _make_ohlcv_df(3).astype(float)  # cast Volume to float so NaN assignment works
        df.iloc[:] = float("nan")
        result = _normalise_df(df, "AAPL")
        assert result is None

    def test_no_ohlcv_columns_returns_none(self) -> None:
        df = pd.DataFrame({"Foo": [1, 2, 3]})
        result = _normalise_df(df, "AAPL")
        assert result is None

    def test_no_date_column_or_index_returns_none(self) -> None:
        df = pd.DataFrame(
            {"Open": [1.0], "High": [2.0], "Low": [0.5], "Close": [1.5], "Volume": [1000]},
            index=[0],  # integer index, no Date
        )
        result = _normalise_df(df, "AAPL")
        assert result is None

    def test_date_column_becomes_index(self) -> None:
        df = _make_ohlcv_df().reset_index()  # Date becomes a column
        result = _normalise_df(df, "AAPL")
        assert result is not None
        assert isinstance(result.index, pd.DatetimeIndex)
        assert result.index.name == "Date"

    def test_output_is_sorted_ascending(self) -> None:
        df = _make_ohlcv_df(5)
        # Reverse the order
        df = df.iloc[::-1]
        result = _normalise_df(df, "AAPL")
        assert result is not None
        assert result.index.is_monotonic_increasing

    def test_partial_ohlcv_columns_accepted(self) -> None:
        """Normalise should succeed even when only some OHLCV columns are present."""
        df = _make_ohlcv_df()
        df = df[["Open", "Close"]].copy()
        result = _normalise_df(df, "AAPL")
        assert result is not None
        assert set(result.columns) == {"Open", "Close"}


# ---------------------------------------------------------------------------
# _write_pkl
# ---------------------------------------------------------------------------


class TestWritePkl:
    def test_round_trip(self, tmp_path: Path) -> None:
        df = _make_ohlcv_df()
        dest = tmp_path / "sub" / "AAPL.pkl"
        _write_pkl(df, dest)
        assert dest.exists()
        with dest.open("rb") as fh:
            loaded = pickle.load(fh)  # noqa: S301
        pd.testing.assert_frame_equal(df, loaded)

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        dest = tmp_path / "a" / "b" / "c" / "TEST.pkl"
        _write_pkl(_make_ohlcv_df(), dest)
        assert dest.exists()


# ---------------------------------------------------------------------------
# _get_csv_headers
# ---------------------------------------------------------------------------


class TestGetCsvHeaders:
    def test_returns_header_list(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "AAPL.csv"
        csv_path.write_text("Date,Open,High,Low,Close,Volume\n2020-01-01,100,105,95,102,1000000\n")
        headers = _get_csv_headers(csv_path)
        assert headers == ["Date", "Open", "High", "Low", "Close", "Volume"]

    def test_returns_empty_list_for_bad_file(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.csv"
        bad.write_bytes(b"\x00\x01\x02")  # binary garbage
        headers = _get_csv_headers(bad)
        assert isinstance(headers, list)


# ---------------------------------------------------------------------------
# _process_csv_dir
# ---------------------------------------------------------------------------


class TestProcessCsvDir:
    def _write_ticker_csv(self, directory: Path, ticker: str, n: int = 5) -> None:
        df = _make_ohlcv_df(n).reset_index()
        csv_path = directory / f"{ticker}.csv"
        df.to_csv(csv_path, index=False)

    def test_writes_pkl_for_each_ticker(self, tmp_path: Path) -> None:
        src = tmp_path / "csv"
        src.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()

        for ticker in ["AAPL", "MSFT", "GOOG"]:
            self._write_ticker_csv(src, ticker)

        written, skipped = _process_csv_dir(src, cache, "TestSource")
        assert written == 3
        assert skipped == 0
        for ticker in ["AAPL", "MSFT", "GOOG"]:
            assert (cache / f"{ticker}.pkl").exists()

    def test_skips_existing_pkl(self, tmp_path: Path) -> None:
        src = tmp_path / "csv"
        src.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()

        self._write_ticker_csv(src, "AAPL")
        # Pre-create pkl
        _write_pkl(_make_ohlcv_df(), cache / "AAPL.pkl")

        written, skipped = _process_csv_dir(src, cache, "TestSource")
        assert written == 0
        assert skipped == 1

    def test_idempotent_second_run(self, tmp_path: Path) -> None:
        src = tmp_path / "csv"
        src.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()

        for ticker in ["AAPL", "MSFT"]:
            self._write_ticker_csv(src, ticker)

        written1, skipped1 = _process_csv_dir(src, cache, "Run1")
        written2, skipped2 = _process_csv_dir(src, cache, "Run2")

        assert written1 == 2
        assert written2 == 0
        assert skipped2 == 2

    def test_empty_directory_returns_zeros(self, tmp_path: Path) -> None:
        src = tmp_path / "empty"
        src.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()
        written, skipped = _process_csv_dir(src, cache, "Empty")
        assert written == 0
        assert skipped == 0

    def test_ticker_name_uppercased(self, tmp_path: Path) -> None:
        """CSV named lowercase should produce an uppercase pkl."""
        src = tmp_path / "csv"
        src.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()

        df = _make_ohlcv_df().reset_index()
        (src / "aapl.csv").write_text(df.to_csv(index=False))

        _process_csv_dir(src, cache, "CaseTest")
        assert (cache / "AAPL.pkl").exists()

    def test_unreadable_csv_is_skipped(self, tmp_path: Path) -> None:
        src = tmp_path / "csv"
        src.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()

        (src / "BROKEN.csv").write_bytes(b"\x00\x01\x02")

        written, skipped = _process_csv_dir(src, cache, "BadCSV")
        assert written == 0
        assert skipped == 1


# ---------------------------------------------------------------------------
# _process_simfin_df
# ---------------------------------------------------------------------------


class TestProcessSimfinDf:
    def _make_simfin_multiindex_df(self, tickers: list[str], n: int = 5) -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            df = _make_ohlcv_df(n)
            df.index = pd.MultiIndex.from_arrays(
                [[ticker] * n, df.index],
                names=["Ticker", "Date"],
            )
            frames.append(df)
        return pd.concat(frames)

    def test_splits_multiindex_into_per_ticker_pkls(self, tmp_path: Path) -> None:
        tickers = ["AAPL", "MSFT"]
        df = self._make_simfin_multiindex_df(tickers)
        written, skipped = _process_simfin_df(df, tmp_path)
        assert written == 2
        assert skipped == 0
        for ticker in tickers:
            assert (tmp_path / f"{ticker}.pkl").exists()

    def test_skips_existing_pkl(self, tmp_path: Path) -> None:
        tickers = ["AAPL", "MSFT"]
        df = self._make_simfin_multiindex_df(tickers)
        # Pre-write one
        _write_pkl(_make_ohlcv_df(), tmp_path / "AAPL.pkl")
        written, skipped = _process_simfin_df(df, tmp_path)
        assert written == 1
        assert skipped == 1

    def test_flat_df_with_ticker_column(self, tmp_path: Path) -> None:
        n = 3
        dates = pd.date_range("2020-01-01", periods=n, freq="D", name="Date")
        df = pd.DataFrame(
            {
                "Ticker": ["AAPL"] * n,
                "Open": [100.0] * n,
                "High": [105.0] * n,
                "Low": [95.0] * n,
                "Close": [102.0] * n,
                "Volume": [1_000_000] * n,
                "Date": dates,
            }
        )
        written, skipped = _process_simfin_df(df, tmp_path)
        assert written == 1
        assert (tmp_path / "AAPL.pkl").exists()

    def test_unexpected_structure_returns_zeros(self, tmp_path: Path) -> None:
        df = pd.DataFrame({"Foo": [1, 2, 3]})
        written, skipped = _process_simfin_df(df, tmp_path)
        assert written == 0
        assert skipped == 0


# ---------------------------------------------------------------------------
# run_bootstrap — graceful skip when packages are absent
# ---------------------------------------------------------------------------


class TestRunBootstrapGracefulSkip:
    """run_bootstrap should complete without errors even when optional packages are missing."""

    def test_completes_without_kaggle_or_simfin(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """When neither kaggle nor simfin is importable, bootstrap logs and exits cleanly."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name in ("kaggle", "simfin"):
                raise ImportError(f"No module named '{name}'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        # Should not raise
        run_bootstrap(cache_dir=str(tmp_path / "ohlcv"))

    def test_kaggle_json_missing_skips_gracefully(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """If kaggle.json does not exist, Kaggle download is skipped without error."""
        import builtins

        real_import = builtins.__import__

        # Provide a dummy kaggle module so import succeeds
        import types

        dummy_kaggle = types.ModuleType("kaggle")
        dummy_kaggle.api = object()  # type: ignore[attr-defined]

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "kaggle":
                return dummy_kaggle
            if name == "simfin":
                raise ImportError("No module named 'simfin'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        # Point home to a tmp dir that has no .kaggle/kaggle.json
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        run_bootstrap(cache_dir=str(tmp_path / "ohlcv"))
        # No pkl files should have been written (no data downloaded)
        ohlcv_dir = tmp_path / "ohlcv"
        pkls = list(ohlcv_dir.glob("*.pkl")) if ohlcv_dir.exists() else []
        assert pkls == []


# ---------------------------------------------------------------------------
# pkl format correctness
# ---------------------------------------------------------------------------


class TestPklFormat:
    def test_saved_pkl_has_correct_format(self, tmp_path: Path) -> None:
        """CSV data saved as pkl should have DatetimeIndex named 'Date' and OHLCV columns."""
        src = tmp_path / "csv"
        src.mkdir()
        cache = tmp_path / "cache"
        cache.mkdir()

        df = _make_ohlcv_df(10).reset_index()
        (src / "AAPL.csv").write_text(df.to_csv(index=False))

        _process_csv_dir(src, cache, "FormatTest")

        pkl_path = cache / "AAPL.pkl"
        assert pkl_path.exists()

        with pkl_path.open("rb") as fh:
            loaded: pd.DataFrame = pickle.load(fh)  # noqa: S301

        assert isinstance(loaded.index, pd.DatetimeIndex)
        assert loaded.index.name == "Date"
        assert loaded.index.is_monotonic_increasing
        assert set(loaded.columns) == {"Open", "High", "Low", "Close", "Volume"}
        # No all-NaN rows
        assert not loaded.isnull().all(axis=1).any()

# ---------------------------------------------------------------------------
# Additional coverage: Kaggle actual download path and SimFin download path
# ---------------------------------------------------------------------------


class TestBootstrapKaggle:
    """Tests for the _bootstrap_kaggle function via run_bootstrap with mocked API."""

    def _make_csv_dir_with_ticker(self, directory: Path, ticker: str, n: int = 3) -> None:
        df = pd.DataFrame(
            {
                "Date": pd.date_range("2020-01-01", periods=n, freq="D"),
                "Open": [100.0] * n,
                "High": [105.0] * n,
                "Low": [95.0] * n,
                "Close": [102.0] * n,
                "Volume": [1_000_000] * n,
            }
        )
        (directory / f"{ticker}.csv").write_text(df.to_csv(index=False))

    def test_kaggle_download_success_writes_pkls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When Kaggle API is present and kaggle.json exists, pkls are written."""
        import builtins
        import types

        real_import = builtins.__import__

        # Create fake kaggle.json
        kaggle_dir = tmp_path / ".kaggle"
        kaggle_dir.mkdir()
        (kaggle_dir / "kaggle.json").write_text("{}")

        # The dataset dirs will be created during the download call
        ohlcv_dir = tmp_path / "ohlcv"

        def fake_dataset_download(dataset: str, path: str, unzip: bool = True) -> None:
            # Write one CSV per dataset call
            dest = Path(path)
            dest.mkdir(parents=True, exist_ok=True)
            if "nyse" in dataset:
                self._make_csv_dir_with_ticker(dest, "IBM")
            else:
                self._make_csv_dir_with_ticker(dest, "AMZN")

        dummy_api = types.SimpleNamespace(dataset_download_files=fake_dataset_download)
        dummy_kaggle = types.ModuleType("kaggle")
        dummy_kaggle.api = dummy_api  # type: ignore[attr-defined]

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "kaggle":
                return dummy_kaggle
            if name == "simfin":
                raise ImportError("No module named 'simfin'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        run_bootstrap(cache_dir=str(ohlcv_dir))

        assert (ohlcv_dir / "IBM.pkl").exists()
        assert (ohlcv_dir / "AMZN.pkl").exists()

    def test_kaggle_download_failure_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When the Kaggle download call raises, the source is skipped gracefully."""
        import builtins
        import types

        real_import = builtins.__import__

        kaggle_dir = tmp_path / ".kaggle"
        kaggle_dir.mkdir()
        (kaggle_dir / "kaggle.json").write_text("{}")

        def failing_download(dataset: str, path: str, unzip: bool = True) -> None:
            raise RuntimeError("Network error")

        dummy_api = types.SimpleNamespace(dataset_download_files=failing_download)
        dummy_kaggle = types.ModuleType("kaggle")
        dummy_kaggle.api = dummy_api  # type: ignore[attr-defined]

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "kaggle":
                return dummy_kaggle
            if name == "simfin":
                raise ImportError("No module named 'simfin'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)
        monkeypatch.setattr(Path, "home", staticmethod(lambda: tmp_path))

        ohlcv_dir = tmp_path / "ohlcv"
        run_bootstrap(cache_dir=str(ohlcv_dir))
        # No pkls should be written
        pkls = list(ohlcv_dir.glob("*.pkl")) if ohlcv_dir.exists() else []
        assert pkls == []


class TestBootstrapSimFin:
    """Tests for the _bootstrap_simfin function via run_bootstrap with mocked API."""

    def _make_simfin_multiindex_df(self, tickers: list[str], n: int = 5) -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            dates = pd.date_range("2020-01-01", periods=n, freq="D", name="Date")
            df = pd.DataFrame(
                {
                    "Open": [100.0] * n,
                    "High": [105.0] * n,
                    "Low": [95.0] * n,
                    "Close": [102.0] * n,
                    "Volume": [1_000_000] * n,
                },
                index=pd.MultiIndex.from_arrays(
                    [[ticker] * n, dates], names=["Ticker", "Date"]
                ),
            )
            frames.append(df)
        return pd.concat(frames)

    def test_simfin_download_success_writes_pkls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When SimFin API is present, pkls are written from returned DataFrame."""
        import builtins
        import types

        real_import = builtins.__import__

        simfin_df = self._make_simfin_multiindex_df(["AAPL", "TSLA"])

        dummy_sf = types.ModuleType("simfin")

        def fake_set_api_key(key: str) -> None:
            pass

        def fake_set_data_dir(path: str) -> None:
            pass

        def fake_load_shareprices(variant: str, market: str) -> pd.DataFrame:
            return simfin_df

        dummy_sf.set_api_key = fake_set_api_key  # type: ignore[attr-defined]
        dummy_sf.set_data_dir = fake_set_data_dir  # type: ignore[attr-defined]
        dummy_sf.load_shareprices = fake_load_shareprices  # type: ignore[attr-defined]

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "simfin":
                return dummy_sf
            if name == "kaggle":
                raise ImportError("No module named 'kaggle'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        ohlcv_dir = tmp_path / "ohlcv"
        run_bootstrap(cache_dir=str(ohlcv_dir))

        assert (ohlcv_dir / "AAPL.pkl").exists()
        assert (ohlcv_dir / "TSLA.pkl").exists()

    def test_simfin_download_failure_is_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When SimFin load raises, the source is skipped gracefully."""
        import builtins
        import types

        real_import = builtins.__import__

        dummy_sf = types.ModuleType("simfin")

        def fake_set_api_key(key: str) -> None:
            pass

        def fake_set_data_dir(path: str) -> None:
            pass

        def fake_load_shareprices(variant: str, market: str) -> pd.DataFrame:
            raise RuntimeError("SimFin API error")

        dummy_sf.set_api_key = fake_set_api_key  # type: ignore[attr-defined]
        dummy_sf.set_data_dir = fake_set_data_dir  # type: ignore[attr-defined]
        dummy_sf.load_shareprices = fake_load_shareprices  # type: ignore[attr-defined]

        def mock_import(name: str, *args: object, **kwargs: object) -> object:
            if name == "simfin":
                return dummy_sf
            if name == "kaggle":
                raise ImportError("No module named 'kaggle'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", mock_import)

        ohlcv_dir = tmp_path / "ohlcv"
        run_bootstrap(cache_dir=str(ohlcv_dir))
        pkls = list(ohlcv_dir.glob("*.pkl")) if ohlcv_dir.exists() else []
        assert pkls == []


class TestProcessSimfinDfEdgeCases:
    """Edge cases for _process_simfin_df not covered in the main suite."""

    def _make_simfin_multiindex_df(self, tickers: list[str], n: int = 3) -> pd.DataFrame:
        frames = []
        for ticker in tickers:
            dates = pd.date_range("2020-01-01", periods=n, freq="D", name="Date")
            df = pd.DataFrame(
                {
                    "Open": [100.0] * n,
                    "High": [105.0] * n,
                    "Low": [95.0] * n,
                    "Close": [102.0] * n,
                    "Volume": [1_000_000] * n,
                },
                index=pd.MultiIndex.from_arrays(
                    [[ticker] * n, dates], names=["Ticker", "Date"]
                ),
            )
            frames.append(df)
        return pd.concat(frames)

    def test_normalise_returns_none_is_skipped(self, tmp_path: Path) -> None:
        """Tickers whose normalised df is None (no OHLCV cols) are counted as skipped."""
        # Build a MultiIndex df with no recognisable OHLCV columns
        n = 3
        dates = pd.date_range("2020-01-01", periods=n, freq="D", name="Date")
        df = pd.DataFrame(
            {"Foo": [1, 2, 3]},
            index=pd.MultiIndex.from_arrays(
                [["ZZZZ"] * n, dates], names=["Ticker", "Date"]
            ),
        )
        written, skipped = _process_simfin_df(df, tmp_path)
        assert written == 0
        assert skipped == 1
