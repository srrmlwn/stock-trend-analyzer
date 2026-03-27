"""Formats trading signals into an HTML email body."""

import logging
from datetime import date

from src.rules.models import Signal

logger = logging.getLogger(__name__)

_BUY_COLOR = "#d4edda"
_SELL_COLOR = "#f8d7da"


def build_report(signals: list[Signal], scan_date: date) -> str:
    """Build HTML email body.

    Returns a simple HTML paragraph if signals is empty, otherwise an HTML
    table with one row per signal styled by action (BUY=green, SELL=red).

    Args:
        signals: List of triggered trading signals.
        scan_date: The date of the scan.

    Returns:
        An HTML string suitable for use as an email body.
    """
    if not signals:
        logger.info("No signals for %s — returning empty-report HTML.", scan_date)
        return f"<p>No signals today ({scan_date}).</p>"

    rows = "\n    ".join(_format_signal_row(s) for s in signals)
    html = (
        "<html><body>\n"
        f"<h2>Stock Signals &#8212; {scan_date}</h2>\n"
        '<table border="1" cellpadding="6" cellspacing="0" '
        'style="border-collapse:collapse;font-family:monospace">\n'
        "  <tr>\n"
        "    <th>Ticker</th><th>Action</th><th>Rule</th>"
        "<th>Reason</th><th>Price</th>\n"
        "  </tr>\n"
        f"    {rows}\n"
        "</table>\n"
        "</body></html>"
    )
    logger.info("Built report HTML for %d signal(s) on %s.", len(signals), scan_date)
    return html


def _format_signal_row(signal: Signal) -> str:
    """Return an HTML <tr> element for a single signal.

    BUY actions receive a green background; SELL actions receive red.

    Args:
        signal: The trading signal to format.

    Returns:
        An HTML ``<tr>`` string.
    """
    color = _BUY_COLOR if signal.action == "BUY" else _SELL_COLOR
    price_str = f"${signal.price:.2f}"
    return (
        f'<tr style="background:{color}">'
        f"<td>{signal.ticker}</td>"
        f"<td>{signal.action}</td>"
        f"<td>{signal.rule_name}</td>"
        f"<td>{signal.reason}</td>"
        f"<td>{price_str}</td>"
        "</tr>"
    )


def _build_subject(signals: list[Signal], scan_date: date) -> str:
    """Return the email subject string.

    Format: ``Stock Signals — {date} ({count} signals)``

    Args:
        signals: List of triggered trading signals.
        scan_date: The date of the scan.

    Returns:
        Formatted subject line string.
    """
    count = len(signals)
    return f"Stock Signals \u2014 {scan_date} ({count} signals)"
