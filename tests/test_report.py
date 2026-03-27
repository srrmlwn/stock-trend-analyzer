"""Tests for src/report/builder.py and src/report/sender.py."""

import smtplib
from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from src.report.builder import _build_subject, build_report
from src.report.sender import send_daily_report, send_report
from src.rules.models import Signal

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_DATE = date(2025, 3, 27)

SAMPLE_SIGNAL = Signal(
    ticker="AAPL",
    action="BUY",
    rule_name="rsi_oversold",
    reason="RSI(14)=27.3 < 30",
    triggered_at=SAMPLE_DATE,
    price=182.45,
)

SELL_SIGNAL = Signal(
    ticker="TSLA",
    action="SELL",
    rule_name="macd_bearish",
    reason="MACD crossed below signal",
    triggered_at=SAMPLE_DATE,
    price=250.00,
)


# ---------------------------------------------------------------------------
# builder.py tests
# ---------------------------------------------------------------------------


def test_build_report_empty_signals() -> None:
    """build_report with an empty list returns a 'No signals today' HTML paragraph."""
    result = build_report([], SAMPLE_DATE)
    assert "No signals today" in result
    assert str(SAMPLE_DATE) in result


def test_build_report_with_signals() -> None:
    """build_report with signals produces an HTML table with ticker, action, rule, price."""
    result = build_report([SAMPLE_SIGNAL], SAMPLE_DATE)
    assert "AAPL" in result
    assert "BUY" in result
    assert "rsi_oversold" in result
    assert "$182.45" in result
    assert "<table" in result
    assert "<tr" in result


def test_build_report_buy_green() -> None:
    """BUY signal rows include the green background color code."""
    result = build_report([SAMPLE_SIGNAL], SAMPLE_DATE)
    assert "#d4edda" in result


def test_build_report_sell_red() -> None:
    """SELL signal rows include the red background color code."""
    result = build_report([SELL_SIGNAL], SAMPLE_DATE)
    assert "#f8d7da" in result


def test_build_subject() -> None:
    """_build_subject returns the correct formatted string."""
    signals = [SAMPLE_SIGNAL, SELL_SIGNAL]
    subject = _build_subject(signals, SAMPLE_DATE)
    assert "2025-03-27" in subject
    assert "2 signals" in subject
    # Check the em-dash separator is present
    assert "\u2014" in subject


# ---------------------------------------------------------------------------
# sender.py tests
# ---------------------------------------------------------------------------


def test_send_report_calls_smtp() -> None:
    """send_report creates an SMTP connection and calls login + sendmail."""
    mock_smtp_instance = MagicMock()
    with patch("smtplib.SMTP", return_value=mock_smtp_instance) as mock_smtp_cls:
        # Make the context manager work
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp_instance
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        send_report(
            html_body="<p>test</p>",
            subject="Test Subject",
            to_email="to@example.com",
            from_email="from@example.com",
            app_password="secret",
            smtp_host="smtp.gmail.com",
            smtp_port=587,
        )

    mock_smtp_instance.login.assert_called_once_with("from@example.com", "secret")
    mock_smtp_instance.sendmail.assert_called_once()
    call_args = mock_smtp_instance.sendmail.call_args[0]
    assert call_args[0] == "from@example.com"
    assert call_args[1] == "to@example.com"


def test_send_report_raises_on_failure() -> None:
    """If the SMTP session raises, send_report propagates the exception."""
    mock_smtp_instance = MagicMock()
    mock_smtp_instance.login.side_effect = smtplib.SMTPAuthenticationError(535, b"Bad credentials")

    with patch("smtplib.SMTP") as mock_smtp_cls:
        mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp_instance
        mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

        with pytest.raises(smtplib.SMTPAuthenticationError):
            send_report(
                html_body="<p>test</p>",
                subject="Test",
                to_email="to@example.com",
                from_email="from@example.com",
                app_password="bad_password",
            )


def test_send_daily_report_missing_env_raises() -> None:
    """send_daily_report raises ValueError when EMAIL_FROM is missing."""
    env_without_from = {
        "EMAIL_TO": "to@example.com",
        "EMAIL_APP_PASSWORD": "secret",
    }
    with patch.dict("os.environ", env_without_from, clear=True):
        with pytest.raises(ValueError, match="EMAIL_FROM/EMAIL_TO/EMAIL_APP_PASSWORD"):
            send_daily_report([SAMPLE_SIGNAL], SAMPLE_DATE)


def test_send_daily_report_uses_env_vars() -> None:
    """send_daily_report passes EMAIL_FROM/TO/APP_PASSWORD to send_report."""
    env_vars = {
        "EMAIL_FROM": "sender@gmail.com",
        "EMAIL_TO": "recipient@example.com",
        "EMAIL_APP_PASSWORD": "app_secret",
    }

    mock_smtp_instance = MagicMock()

    with patch.dict("os.environ", env_vars, clear=True):
        with patch("smtplib.SMTP") as mock_smtp_cls:
            mock_smtp_cls.return_value.__enter__ = lambda s: mock_smtp_instance
            mock_smtp_cls.return_value.__exit__ = MagicMock(return_value=False)

            send_daily_report([SAMPLE_SIGNAL], SAMPLE_DATE)

    mock_smtp_instance.login.assert_called_once_with("sender@gmail.com", "app_secret")
    sendmail_args = mock_smtp_instance.sendmail.call_args[0]
    assert sendmail_args[0] == "sender@gmail.com"
    assert sendmail_args[1] == "recipient@example.com"
