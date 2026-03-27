"""Sends HTML email reports via Gmail SMTP."""

import logging
import os
import smtplib
import ssl
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from src.report.builder import _build_subject, build_report
from src.rules.models import Signal

logger = logging.getLogger(__name__)

_SETTINGS_PATH = Path(__file__).resolve().parents[2] / "config" / "settings.yaml"


def send_report(
    html_body: str,
    subject: str,
    to_email: str,
    from_email: str,
    app_password: str,
    smtp_host: str = "smtp.gmail.com",
    smtp_port: int = 587,
) -> None:
    """Send an HTML email via Gmail SMTP using STARTTLS.

    Args:
        html_body: HTML content for the email body.
        subject: Email subject line.
        to_email: Recipient email address.
        from_email: Sender email address.
        app_password: Gmail app password for authentication.
        smtp_host: SMTP server hostname. Defaults to ``smtp.gmail.com``.
        smtp_port: SMTP server port. Defaults to ``587``.

    Raises:
        Exception: Re-raises any exception raised by the SMTP session.
    """
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = from_email
    msg["To"] = to_email
    msg.attach(MIMEText(html_body, "html"))

    logger.info("Connecting to SMTP %s:%s as %s.", smtp_host, smtp_port, from_email)
    with smtplib.SMTP(smtp_host, smtp_port) as smtp:
        smtp.ehlo()
        smtp.starttls(context=ssl.create_default_context())
        smtp.ehlo()
        smtp.login(from_email, app_password)
        smtp.sendmail(from_email, to_email, msg.as_string())
    logger.info("Email sent to %s with subject: %s", to_email, subject)


def send_daily_report(
    signals: list[Signal],
    scan_date: date,
) -> None:
    """Load email config from environment variables and send the daily report.

    Required environment variables:
        EMAIL_FROM: Sender Gmail address.
        EMAIL_TO: Recipient email address.
        EMAIL_APP_PASSWORD: Gmail app password.

    SMTP host and port are read from ``config/settings.yaml``.

    Args:
        signals: List of triggered trading signals.
        scan_date: The date of the scan.

    Raises:
        ValueError: If any required environment variable is missing.
        Exception: Re-raises any SMTP or network error from :func:`send_report`.
    """
    from_email = os.environ.get("EMAIL_FROM")
    to_email = os.environ.get("EMAIL_TO")
    app_password = os.environ.get("EMAIL_APP_PASSWORD")

    if not from_email or not to_email or not app_password:
        raise ValueError("EMAIL_FROM/EMAIL_TO/EMAIL_APP_PASSWORD must be set in .env")

    with open(_SETTINGS_PATH) as fh:
        settings = yaml.safe_load(fh)

    email_cfg = settings.get("email", {})
    smtp_host: str = email_cfg.get("smtp_host", "smtp.gmail.com")
    smtp_port: int = int(email_cfg.get("smtp_port", 587))

    html_body = build_report(signals, scan_date)
    subject = _build_subject(signals, scan_date)

    logger.debug("Sending daily report: subject=%r", subject)
    send_report(
        html_body=html_body,
        subject=subject,
        to_email=to_email,
        from_email=from_email,
        app_password=app_password,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
    )
