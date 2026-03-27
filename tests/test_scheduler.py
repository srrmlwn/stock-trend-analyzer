"""Tests for src/scheduler/main.py CLI entrypoint and scheduling logic."""

from __future__ import annotations

import sys
from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

import src.scheduler.main as scheduler_main
from src.scheduler.main import _run_backtest, _run_now, main


# ---------------------------------------------------------------------------
# _run_now tests
# ---------------------------------------------------------------------------


class TestRunNow:
    """Tests for the _run_now helper."""

    def test_run_now_calls_run_scan(self) -> None:
        """_run_now should call run_scan with the correct dry_run argument."""
        mock_signal = MagicMock()
        with patch("src.scheduler.main.run_scan", return_value=[mock_signal]) as mock_scan, \
             patch("src.scheduler.main.send_daily_report"):
            _run_now(dry_run=False)
            mock_scan.assert_called_once()
            _, kwargs = mock_scan.call_args
            assert kwargs.get("dry_run") is False or mock_scan.call_args[0][2] is False or True
            # Verify dry_run kwarg was False
            call_kwargs = mock_scan.call_args[1]
            assert call_kwargs.get("dry_run") is False

    def test_run_now_dry_run_skips_email(self) -> None:
        """When dry_run=True, send_daily_report should NOT be called."""
        with patch("src.scheduler.main.run_scan", return_value=[]) as mock_scan, \
             patch("src.scheduler.main.send_daily_report") as mock_send:
            _run_now(dry_run=True)
            mock_scan.assert_called_once()
            mock_send.assert_not_called()

    def test_run_now_non_dry_run_sends_email(self) -> None:
        """When dry_run=False, send_daily_report should be called once."""
        mock_signal = MagicMock()
        with patch("src.scheduler.main.run_scan", return_value=[mock_signal]), \
             patch("src.scheduler.main.send_daily_report") as mock_send:
            _run_now(dry_run=False)
            mock_send.assert_called_once()
            # Verify it was called with the signal list and a date
            args = mock_send.call_args[0]
            assert isinstance(args[1], date)

    def test_run_now_logs_signal_count(self, caplog: pytest.LogCaptureFixture) -> None:
        """_run_now should log the number of signals returned."""
        signals = [MagicMock(), MagicMock()]
        with patch("src.scheduler.main.run_scan", return_value=signals), \
             patch("src.scheduler.main.send_daily_report"):
            import logging
            with caplog.at_level(logging.INFO, logger="src.scheduler.main"):
                _run_now(dry_run=False)
            assert "2" in caplog.text


# ---------------------------------------------------------------------------
# _run_backtest tests
# ---------------------------------------------------------------------------


class TestRunBacktest:
    """Tests for the _run_backtest helper."""

    def test_run_backtest_prints_report(self) -> None:
        """_run_backtest should call run_backtest and print_backtest_report."""
        mock_result = MagicMock()
        mock_rules = [MagicMock()]
        with patch("src.scheduler.main.load_rules", return_value=mock_rules) as mock_load, \
             patch("src.scheduler.main.run_backtest", return_value=mock_result) as mock_bt, \
             patch("src.scheduler.main.print_backtest_report") as mock_report:
            _run_backtest(["AAPL", "MSFT"], start_date="2018-01-01", end_date="2023-12-31")
            mock_load.assert_called_once()
            mock_bt.assert_called_once_with(
                ["AAPL", "MSFT"],
                mock_rules,
                start_date="2018-01-01",
                end_date="2023-12-31",
            )
            mock_report.assert_called_once_with(mock_result)

    def test_run_backtest_default_dates(self) -> None:
        """_run_backtest should use default start_date and None end_date."""
        mock_result = MagicMock()
        with patch("src.scheduler.main.load_rules", return_value=[]), \
             patch("src.scheduler.main.run_backtest", return_value=mock_result) as mock_bt, \
             patch("src.scheduler.main.print_backtest_report"):
            _run_backtest(["TSLA"])
            call_kwargs = mock_bt.call_args[1]
            assert call_kwargs["start_date"] == "2015-01-01"
            assert call_kwargs["end_date"] is None


# ---------------------------------------------------------------------------
# main() CLI integration tests
# ---------------------------------------------------------------------------


class TestMainCLI:
    """Integration tests for the main() argument parser."""

    def test_main_run_now_flag(self) -> None:
        """main() with --run-now --dry-run should not raise any exception."""
        with patch.object(sys, "argv", ["main", "--run-now", "--dry-run"]), \
             patch("src.scheduler.main.run_scan", return_value=[]), \
             patch("src.scheduler.main.send_daily_report"):
            main()  # Should not raise

    def test_main_backtest_flag(self) -> None:
        """main() with --backtest --tickers should invoke _run_backtest."""
        mock_result = MagicMock()
        with patch.object(sys, "argv", ["main", "--backtest", "--tickers", "AAPL", "MSFT"]), \
             patch("src.scheduler.main.load_rules", return_value=[]), \
             patch("src.scheduler.main.run_backtest", return_value=mock_result) as mock_bt, \
             patch("src.scheduler.main.print_backtest_report"):
            main()
            mock_bt.assert_called_once()
            tickers_arg = mock_bt.call_args[0][0]
            assert "AAPL" in tickers_arg
            assert "MSFT" in tickers_arg

    def test_main_backtest_requires_tickers(self) -> None:
        """main() with --backtest but no --tickers should raise SystemExit."""
        with patch.object(sys, "argv", ["main", "--backtest"]):
            with pytest.raises(SystemExit):
                main()

    def test_main_run_now_no_dry_run_sends_email(self) -> None:
        """main() with --run-now (no --dry-run) should send email."""
        mock_signal = MagicMock()
        with patch.object(sys, "argv", ["main", "--run-now"]), \
             patch("src.scheduler.main.run_scan", return_value=[mock_signal]), \
             patch("src.scheduler.main.send_daily_report") as mock_send:
            main()
            mock_send.assert_called_once()

    def test_main_sets_log_level(self) -> None:
        """main() should configure logging at the specified level."""
        with patch.object(sys, "argv", ["main", "--run-now", "--log-level", "DEBUG"]), \
             patch("src.scheduler.main.run_scan", return_value=[]), \
             patch("src.scheduler.main.send_daily_report"), \
             patch("logging.basicConfig") as mock_log:
            main()
            mock_log.assert_called_once()
            call_kwargs = mock_log.call_args[1]
            import logging
            assert call_kwargs["level"] == logging.DEBUG

    def test_main_custom_config_paths(self) -> None:
        """main() should apply custom --config and --settings paths."""
        with patch.object(
            sys, "argv",
            ["main", "--run-now", "--config", "custom/rules.yaml", "--settings", "custom/settings.yaml"],
        ), \
             patch("src.scheduler.main.run_scan", return_value=[]) as mock_scan, \
             patch("src.scheduler.main.send_daily_report"):
            main()
            call_kwargs = mock_scan.call_args[1]
            assert call_kwargs["config_path"] == "custom/rules.yaml"
            assert call_kwargs["settings_path"] == "custom/settings.yaml"

    def test_main_no_args_exits(self) -> None:
        """main() with no arguments should raise SystemExit."""
        with patch.object(sys, "argv", ["main"]):
            with pytest.raises(SystemExit):
                main()


# ---------------------------------------------------------------------------
# _run_schedule tests
# ---------------------------------------------------------------------------


class TestRunSchedule:
    """Tests for the _run_schedule blocking scheduler."""

    def test_run_schedule_schedules_weekdays(self) -> None:
        """_run_schedule should register exactly 5 weekday jobs then loop."""
        import schedule as sched

        sched.clear()

        loop_counter = {"count": 0}

        def fake_sleep(_: int) -> None:
            loop_counter["count"] += 1
            if loop_counter["count"] >= 1:
                raise KeyboardInterrupt("stop loop")

        with patch("src.scheduler.main._load_settings", return_value={"scanner": {"run_time_et": "10:00"}}), \
             patch("time.sleep", side_effect=fake_sleep):
            try:
                scheduler_main._run_schedule()
            except KeyboardInterrupt:
                pass

        assert len(sched.jobs) == 5
        sched.clear()
