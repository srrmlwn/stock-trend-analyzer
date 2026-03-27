# CLAUDE.md — Agent Instructions

This file defines how every AI agent working on this project should behave.
Read this before writing a single line of code.

---

## Git workflow

- **Single developer** — no PRs, no feature branches required
- **Push directly to `main`** after each logical unit of work
- Commit frequently with descriptive messages; push at the end of each task or subtask
- Branch name convention is still `task/<id>-<description>` if you want an isolated branch,
  but merging to `main` and pushing is the final step — not opening a PR

---

## Your job

You are implementing one task from the TASKS section of the project context.
Your task has a clear scope, inputs, and outputs.
Do not work outside your task's scope. Do not refactor code that belongs to another task.

---

## Workflow — follow this exactly

### 1. Read first
- Read PROJECT.md to understand the full system
- Find your specific task in TASKS.md (or the TASKS section of the bootstrap file)
- Read any existing code in the modules your task touches
- Check `config/rules.yaml` for the config schema if your task involves rules or signals

### 2. Plan before coding
- Identify edge cases before writing logic
- If anything in the task spec is ambiguous, make the simplest reasonable assumption
  and document it in a comment

### 3. Write the code
- **Simple over clever** — if a junior developer can't read it in 30 seconds, simplify it
- **One function, one job** — keep functions small and focused
- **No dead code** — don't leave commented-out blocks or unused imports
- **Type hints on every function** — use Python type hints throughout
- **Docstrings on every public function** — one-line summary + param descriptions

### 4. Write the tests
- Every module gets a corresponding test file in `tests/`
- Test file naming: `tests/test_<module_name>.py`
- Cover the happy path, edge cases, and failure modes
- Use `pytest` — no other test framework
- Mock external API calls (yfinance, Alpha Vantage) — tests must run offline
- Minimum coverage expectation: **80% line coverage** on your module

### 5. Run and iterate

```bash
pytest tests/test_<your_module>.py -v
pytest tests/test_<your_module>.py --cov=<your_module> --cov-report=term-missing
ruff check src/
mypy src/
```

Fix every failure. Fix every lint error. Fix every type error.
Do not move to step 6 until all checks pass cleanly.

### 6. Self-review checklist
Before committing, verify:

- [ ] All tests pass (`pytest`)
- [ ] Coverage ≥ 80% on your module
- [ ] No lint errors (`ruff check`)
- [ ] No type errors (`mypy`)
- [ ] No hardcoded credentials, API keys, or file paths
- [ ] Config values live in `config/` or `.env`, not in code
- [ ] Functions have type hints and docstrings
- [ ] No unused imports or dead code
- [ ] README updated if you added a new dependency or changed how to run something

### 7. Commit and push to main

```bash
git add <files>
git commit -m "Task NN: <short description>"
git push origin main
```

---

## Code style

- **Python 3.11+**
- **Formatter:** `black` (line length 100)
- **Linter:** `ruff`
- **Type checker:** `mypy` (strict mode)
- **Test framework:** `pytest`

---

## Project structure

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
├── scripts/                # One-off utility scripts
├── .env.example
├── pyproject.toml
├── requirements.txt
├── CLAUDE.md
└── README.md
```

---

## What never goes in code

- API keys or credentials → use `.env`
- Email passwords → use app-specific Gmail password in `.env`
- Hardcoded ticker lists → use `config/settings.yaml`
- Hardcoded thresholds → use `config/rules.yaml`
- Magic numbers → use a named constant with a comment
