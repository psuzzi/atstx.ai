# x02-plan-v2 — Consistency Review

Review of x02-plan-v2 documents against the actual atstx repo state and agreed conventions.

## Issues to Fix

### 1. Python version
- **Plan says:** Python 3.11+
- **Repo uses:** Python 3.14
- **Action:** Update all docs to say 3.14

### 2. Project tooling
- **Plan says:** `requirements.txt` + `pip install`
- **Repo uses:** `uv` + `pyproject.toml`
- **Action:** Update Block 0 tasks and test commands. Replace `pip install -r requirements.txt` with `uv sync`. Replace `requirements.txt` references with `pyproject.toml`.

### 3. Source layout
- **Plan says:** Flat layout (`projectx_client.py`, `regime/`, `strategy/`, `execution/` at root)
- **Repo uses:** `src/atstx/` package layout with subpackages
- **Action:** Update all file paths in all docs. E.g. `regime/features.py` becomes `src/atstx/regime/features.py`. Update all import examples.

### 4. Configuration approach (RESOLVED)
- **Plan says:** `config.yaml` as central config
- **Repo has:** `.env` + `pydantic-settings`
- **Resolution:** Both are used. They're complementary:
  - `.env` + `settings.py` (BaseSettings) → secrets (API keys, credentials)
  - `config.yaml` + `config.py` (BaseModel) → strategy params (RRR, stops, HMM config, risk thresholds)
- **Action:** Update architecture to show both layers. Keep `config.yaml` schema from Architecture Section 3 but remove the `api.api_key` and `api.base_url` entries (those live in `.env`). Add `config.py` to the repo structure.

### 5. Repository structure (Section 10 of overall plan)
- **Plan shows:** Flat layout with `main.py`, `backtest.py` at root
- **Repo has:** `src/atstx/` package
- **Action:** Update to match actual structure. Entry points (`main.py`, `backtest.py`) could be `src/atstx/main.py` and `src/atstx/backtest.py`, or kept at root as thin wrappers. Decide.

### 6. Entry point scripts
- **Plan shows:** `main.py` and `backtest.py` at project root
- **Decision needed:** Keep as root-level scripts that import from `src/atstx/`, or make them part of the package? Root-level scripts are simpler for `uv run python main.py`. Package entry points via `[project.scripts]` are cleaner but add indirection.

## Internally Consistent (No Changes Needed)

- Signal logic symmetry (long in bull, short in bear, both in neutral)
- Filtered probabilities rule — consistent across all 6 docs
- RRR = 1.8 — used consistently, targets always `round(stop * rrr)`
- Block dependency chain (0→1→2→3[gate]→4→5→6)
- PLUG comments match between architecture Section 4 and future scope
- Slippage model (1 tick normal, 3 ticks crash)
- Theory doc (05) aligns with architecture and signal logic
- Crash regime logic (counter-trend long only, no shorting the crash)
- Neutral regime VWAP std dev bands logic

## Open Questions

1. **Entry point location** — root scripts vs package entry points? (see issue #6 above)
2. **Decision Point A** — 30-min vs 1-hour macro filter. Plan leaves this open. Recommend 30-min for more intraday context.
3. **Decision Point D** — Whether to implement neutral mean reversion in first session or stub as flat. Recommend implementing it but with a config flag to disable.
