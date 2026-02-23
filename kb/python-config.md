# Configuration Conventions

Two-layer approach: secrets in `.env`, validation in `settings.py`.

## Layer 1 — `.env`

- Holds secrets and runtime values (API keys, credentials, feature flags)
- Always gitignored — never committed
- A committed `.env.example` documents required variables with dummy values
- Format: `KEY=value`, one per line, `#` comments

## Layer 2 — `src/atstx/settings.py`

- A `Settings(BaseSettings)` class using `pydantic-settings`
- Reads and validates `.env` at startup via `SettingsConfigDict(env_file=".env", extra="ignore")`
- Typed fields with defaults where appropriate
- Required fields (no default) fail loudly with `ValidationError` if missing
- Singleton access via `get_settings()` with `@lru_cache`
- This is the **single source of truth** for all runtime configuration

## Rules

- No separate JSON, YAML, or TOML config files for application settings
- `pyproject.toml` is for project metadata and dependencies only
- All code reads config through `get_settings()`, never directly from `os.environ`
- New config values: add field to `Settings`, add entry to `.env.example`

## Example

```python
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    projectx_api_key: str          # required — no default
    projectx_base_url: str = "https://api.projectx.com"
    debug: bool = False

@lru_cache
def get_settings() -> Settings:
    return Settings()
```
