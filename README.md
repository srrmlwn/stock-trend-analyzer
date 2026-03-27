# Stock Trend Analyzer

A self-hosted, fully customizable stock trend tracking system for personal use.
Built for **position trading** (weeks to months), running daily on end-of-day data.
No paid data services required.

## Features

- **Custom rules engine** — define buy/sell triggers in `config/rules.yaml`
- **Backtesting framework** — validate rules against 10 years of historical data
- **Daily email report** — actionable signal list each morning via Gmail SMTP

## Requirements

- Python 3.11+
- Gmail account with App Password enabled (for email reports)
- Alpha Vantage free API key (optional, for supplemental fundamental data)

## Installation

```bash
git clone https://github.com/srrmlwn/stock-trend-analyzer.git
cd stock-trend-analyzer

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

## Configuration

1. Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

2. Edit `config/rules.yaml` to define your buy/sell rules.

3. Edit `config/settings.yaml` to adjust universe filters and scan settings.

## Running the Scanner

```bash
# Dry run (prints signals to stdout, no email)
python -m src.scheduler.main --run-now --dry-run

# Full run (fetches data, evaluates rules, sends email)
python -m src.scheduler.main --run-now

# Run backtester on specific tickers
python -m src.scheduler.main --backtest --tickers AAPL MSFT NVDA

# Start scheduled mode (runs daily at 10am ET on weekdays)
python -m src.scheduler.main --schedule
```

## Running Tests

```bash
# All unit tests
pytest tests/

# With coverage report
pytest tests/ --cov=src --cov-report=term-missing

# Skip integration tests (no network required)
pytest tests/ -m "not integration"

# Integration tests only (requires network)
pytest tests/ -m integration
```

## Linting and Type Checking

```bash
ruff check src/
mypy src/
black src/ tests/
```

## Project Structure

```
stock-trend-analyzer/
├── config/
│   ├── rules.yaml          # Buy/sell rule definitions
│   └── settings.yaml       # Universe filters, position size, email config
├── src/
│   ├── universe/           # Stock universe builder and pre-filter
│   ├── data/               # Data fetching and caching (yfinance wrapper)
│   ├── indicators/         # Technical indicator calculations
│   ├── rules/              # Rule engine — parses YAML and evaluates conditions
│   ├── scanner/            # Daily scan orchestrator
│   ├── backtest/           # Backtesting engine (vectorbt wrapper)
│   ├── report/             # Email report builder and sender
│   └── scheduler/          # Job scheduler / entrypoint
├── tests/                  # One test file per src module
└── scripts/                # One-off utility scripts
```

## Data Sources

| Library | Used for |
|---|---|
| `yfinance` | OHLCV price history, basic fundamentals |
| `pandas-ta` | Technical indicator calculations |
| `vectorbt` | Backtesting simulation |
| `alpha_vantage` | Supplemental fundamental data (free tier) |

## License

MIT
