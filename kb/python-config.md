# Configuration Conventions

Three-layer approach: secrets in `.env`, validated by `settings.py`; strategy parameters in `config.yaml`, validated by `config.py`.

## Layer 1 — `.env` (secrets and runtime values)

- Holds API keys, exchange credentials, account IDs, feature flags
- Always gitignored — never committed
- A committed `.env.example` documents required variables with dummy values
- Format: `KEY=value`, one per line, `#` comments

## Layer 2 — `src/atstx/settings.py` (secret validation)

- A `Settings(BaseSettings)` class using `pydantic-settings`
- Reads and validates `.env` at startup via `SettingsConfigDict(env_file=".env", extra="ignore")`
- Required fields (no default) fail loudly with `ValidationError` if missing
- Singleton access via `get_settings()` with `@lru_cache`
- Contains only secrets and environment-specific values

## Layer 3 — `config.yaml` + `src/atstx/config.py` (strategy parameters)

- `config.yaml` at project root — committed to git, trackable in history
- Contains strategy parameters, HMM config, risk thresholds, position sizing, stop distances, RRR, etc.
- `config.py` defines a Pydantic `BaseModel` (not `BaseSettings`) that validates the YAML
- Loaded once at startup via a `load_config()` function
- Everything typed, validated, and explicit

## Why Both

- `.env` + `settings.py` = secrets that differ per machine/environment
- `config.yaml` + `config.py` = strategy parameters you tune and want in git history
- They're complementary, not competing

## Startup Pattern

Both are instantiated once at startup and passed around. No global state, no magic.

```python
# In the entry point (main.py, backtest.py)
from atstx.settings import get_settings
from atstx.config import load_config

settings = get_settings()    # reads .env, validates secrets
config = load_config()       # reads config.yaml, validates strategy params

# Pass both to components that need them
```

## Rules

- `pyproject.toml` is for project metadata and dependencies only
- Code reads secrets via `get_settings()`, never `os.environ` directly
- Code reads strategy params via the config object, never hardcoded
- New secret: add field to `Settings`, add entry to `.env.example`
- New strategy param: add field to config model, add entry to `config.yaml`
