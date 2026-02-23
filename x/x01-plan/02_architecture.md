# Topstep-X Automated Trading System — Architecture

> **Status:** Proposed — to be reviewed and confirmed with implementation agent before coding begins  
> **Decision points are marked with ⚠️ and must be resolved before or during implementation**

---

## 1. High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SESSION STARTUP                          │
│  1. Load config + credentials                               │
│  2. Authenticate with Project X API                         │
│  3. Fetch VIX → compute sentiment multiplier               │
│  4. Load pre-trained HMM model from disk                    │
│  5. Initialize risk manager (reset daily counters)          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MAIN LOOP (per bar)                      │
│                                                             │
│   Project X WebSocket / REST                                │
│         │                                                   │
│         ▼                                                   │
│   Bar Aggregator  ──→  5-min bars  ──→  Feature Extractor  │
│         │                                    │              │
│         │              ┌────────────────────▼──────┐       │
│         │              │   HMM Regime Detector     │       │
│         │              │   (GaussianHMM, 5 states) │       │
│         │              └────────────────────┬──────┘       │
│         │                                   │              │
│         ▼                                   ▼              │
│   1-min bars  ──→  Entry Trigger  ←─  Regime Label         │
│                          │                                  │
│                          ▼                                  │
│                   Macro Trend Filter                        │
│                   (30-min/1-hr EMA)                         │
│                          │                                  │
│                          ▼                                  │
│                   Calendar Blackout?                        │
│                          │                                  │
│                          ▼                                  │
│                   Risk Manager                              │
│                   (daily PnL check, position sizing,        │
│                    VIX multiplier, consecutive loss count)  │
│                          │                                  │
│                          ▼                                  │
│                   Order Manager                             │
│                   (place/cancel/track via Project X API)    │
│                          │                                  │
│                          ▼                                  │
│                   Trade Log (JSON append)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component Descriptions

### 2.1 `projectx_client.py` — API Wrapper

**Responsibility:** All communication with the Project X / Topstep X platform.

**Interfaces:**
- `authenticate(api_key) → session_token`
- `get_historical_bars(symbol, timeframe, start, end) → List[Bar]`
- `subscribe_bars(symbol, timeframe, callback)` — WebSocket stream
- `place_order(symbol, direction, size, order_type, stop, target) → order_id`
- `cancel_order(order_id)`
- `get_open_positions() → List[Position]`
- `get_account_info() → AccountInfo` (includes daily PnL, drawdown remaining)

> ⚠️ **Decision Point:** Project X API — confirm authentication method (JWT token, API key, OAuth). Confirm whether WebSocket is available for streaming bars or if polling REST is required. Confirm historical data availability and maximum lookback period.

> ⚠️ **Decision Point:** Does Project X provide 1-min and 5-min bars natively, or does the system need to aggregate from tick/second data?

---

### 2.2 `regime/features.py` — Feature Engineering

**Responsibility:** Convert a list of `Bar` objects into a numpy feature matrix for the HMM.

**Features computed per bar:**
- `return_pct` = `(close - prev_close) / prev_close`
- `volatility` = rolling 10-bar std of `return_pct`
- `volume_ratio` = `volume / rolling_20_bar_mean_volume`

**Interface:**
- `compute_features(bars: List[Bar], warmup_bars: int = 20) → np.ndarray`

**Notes:**
- First `warmup_bars` bars are dropped (rolling window needs them to stabilize)
- Returns shape `(n_bars - warmup, 3)`

> ⚠️ **Decision Point:** Whether to add additional features (bar range, VWAP distance, open-close ratio). Start with 3 features and expand only if regime separation is visually poor when plotted.

---

### 2.3 `regime/hmm_detector.py` — HMM Wrapper

**Responsibility:** Wrap `hmmlearn.GaussianHMM` for training and inference.

**Interface:**
- `train(features: np.ndarray, n_states: int = 5) → model`
- `save(model, path: str)`
- `load(path: str) → model`
- `predict_regime(model, recent_features: np.ndarray) → int` — returns current state label (0–4)
- `get_state_stats(model) → dict` — mean return and volatility per state (used for labeling states after training)

**State labeling (post-training):**
After training, inspect `model.means_` to assign human labels. The state with the highest mean return = Strong Bull, lowest = Crash, etc. This mapping is stored in `config.yaml`.

> ⚠️ **Decision Point:** How many recent bars to feed to `predict_regime` at runtime. Suggested: last 50 bars (enough sequence context without being too slow). Tune if regime switches are too laggy or too noisy.

> ⚠️ **Decision Point:** Covariance type for GaussianHMM. `"diag"` is recommended as a starting point (faster, less data-hungry than `"full"`).

---

### 2.4 `regime/train_hmm.py` — Offline Training Script

**Responsibility:** Standalone script to train the HMM and save the model.

**Inputs:**
- Historical 5-min bar CSV (from data/historical/)
- Config: n_states, feature params

**Outputs:**
- `models/hmm_model.pkl`
- Console output: state statistics, so the developer can manually label states

**Usage:**
```bash
python regime/train_hmm.py --data data/historical/MNQ_5min.csv --states 5
```

This script is run **once before the backtest and live trading**. It is also the script to re-run for periodic retraining (see Future Scope document).

---

### 2.5 `strategy/signal_generator.py` — Signal Logic

**Responsibility:** Given the current regime label, current 1-min bar, and VWAP value, return a `Signal` object.

**Interface:**
- `generate_signal(regime: int, bar_1min: Bar, vwap: float, macro_bias: str) → Signal`

**Signal object:**
```python
@dataclass
class Signal:
    direction: str      # "long", "short", "flat"
    size: int           # contracts (base size, before VIX multiplier)
    stop_ticks: int
    target_ticks: int
    reason: str         # for logging
```

**Logic summary per regime:**

| Regime | Trigger | Direction | Stop | Target |
|---|---|---|---|---|
| 0 (Strong Bull) | 1-min low ≤ VWAP, close > open | long | 10 ticks | 20 ticks |
| 1 (Bull) | 1-min low ≤ VWAP, close > open | long | 8 ticks | 12 ticks |
| 2 (Neutral) | — | flat | — | — |
| 3 (Bear) | 1-min high ≥ VWAP, close < open | short | 8 ticks | 12 ticks |
| 4 (Crash) | 1-min lower wick > 60% of bar range | long | 6 ticks | 15 ticks |

**Macro bias override:** If macro filter says "bear" and regime is 0 or 1, reduce to flat or size 0.

---

### 2.6 `strategy/filters.py` — Sentiment and Calendar Filters

**Responsibility:** Pre-trade checks that can block or reduce a signal.

**Interface:**
- `get_vix_multiplier() → float` — fetches VIX once at startup, returns 0.5 or 1.0
- `is_blackout_period(now: datetime) → bool` — returns True if within blackout window of high-impact event
- `get_macro_bias(bars_30min_or_1hr: List[Bar]) → str` — returns "bull", "bear", or "neutral" based on EMA

**Blackout times (hardcoded, configurable in config.yaml):**
- 08:30 ET — NFP, CPI, Jobless Claims
- 10:00 ET — ISM, Consumer Confidence
- 14:00 ET — FOMC
- 14:30 ET — Fed press conference

> ⚠️ **Decision Point:** EMA period for macro bias filter. Suggested: 20-period EMA on the 30-min or 1-hour bars. If price > EMA → bull bias, price < EMA → bear bias, within 0.1% of EMA → neutral.

---

### 2.7 `strategy/risk_manager.py` — Risk Manager

**Responsibility:** Position sizing, daily loss enforcement, consecutive loss tracking.

**Interface:**
- `check_can_trade() → bool` — returns False if daily limit hit or 3 consecutive losses
- `apply_size_multiplier(base_size: int, vix_multiplier: float) → int`
- `record_trade_result(pnl: float)`
- `get_daily_pnl() → float`
- `reset_daily()` — called at session start

**Rules:**
- If `daily_pnl <= -750` → no new trades, log reason
- If `consecutive_losses >= 3` → no new trades, log reason
- Final size = `max(1, floor(base_size * vix_multiplier))`

---

### 2.8 `execution/order_manager.py` — Order Manager

**Responsibility:** Place orders, track fill status, manage stops and targets.

**Interface:**
- `enter_trade(signal: Signal) → trade_id`
- `exit_trade(trade_id: str, reason: str)`
- `on_bar(bar: Bar)` — checks if stop or target has been hit for open trades
- `get_open_trades() → List[Trade]`

**Notes:**
- Uses bracket orders if Project X API supports them (stop + target in one order)
- Falls back to manual stop monitoring if not

> ⚠️ **Decision Point:** Confirm whether Project X API supports bracket (OCO) orders. If not, the order manager must monitor price manually and send separate stop/target orders.

---

### 2.9 `backtest.py` — Backtester

**Responsibility:** Replay historical bars through the complete signal pipeline and report PnL.

**Key design principle:** Uses the **same** `features.py`, `hmm_detector.py`, `signal_generator.py`, and `risk_manager.py` as live trading. Only `order_manager.py` is replaced with a simulated fill function.

**Simulated fill:**
```python
def simulate_fill(signal, next_bar, regime):
    slippage = 2 if regime == 4 else 1  # crash = 2 ticks, else 1
    if signal.direction == "long":
        fill_price = next_bar.open + slippage * TICK_SIZE
    elif signal.direction == "short":
        fill_price = next_bar.open - slippage * TICK_SIZE
    return fill_price
```

**Outputs:**
- Total PnL
- Max drawdown (in dollars)
- Win rate
- Average win / average loss ratio
- Trades per day
- Equity curve printed to console (or simple matplotlib chart)

---

## 3. Configuration (`config.yaml`)

```yaml
api:
  base_url: "https://..."          # Project X API base URL
  api_key: "${PROJECTX_API_KEY}"   # loaded from environment variable

trading:
  symbol: "MNQ"
  base_size: 1
  daily_loss_limit: 750
  max_consecutive_losses: 3

timeframes:
  regime: "5min"
  macro: "30min"          # DECISION POINT: 30min vs 1hour
  entry: "1min"

hmm:
  n_states: 5
  model_path: "models/hmm_model.pkl"
  predict_window: 50      # bars fed to predict at runtime
  # State label mapping — fill in after training by inspecting model.means_
  state_labels:
    0: "strong_bull"
    1: "bull"
    2: "neutral"
    3: "bear"
    4: "crash"

filters:
  blackout_minutes_before: 5
  blackout_minutes_after: 10
  high_impact_times_et:
    - "08:30"
    - "10:00"
    - "14:00"
    - "14:30"
  vix_fear_threshold: 30
  vix_elevated_threshold: 20

slippage:
  normal_ticks: 1
  crash_ticks: 2

macro_filter:
  ema_period: 20
```

---

## 4. Inter-Component Dependencies

```
projectx_client
    ↑
    └── used by: order_manager, data fetch scripts

features ← depends on: nothing (pure numpy)
    ↑
    └── used by: hmm_detector, backtest

hmm_detector ← depends on: features
    ↑
    └── used by: main loop, backtest, train_hmm script

signal_generator ← depends on: hmm_detector output, filters
    ↑
    └── used by: main loop, backtest

risk_manager ← depends on: nothing external (stateful)
    ↑
    └── used by: main loop, backtest

order_manager ← depends on: projectx_client, risk_manager
    ↑
    └── used by: main loop only (backtest uses simulate_fill instead)

filters ← depends on: yfinance (VIX), config
    ↑
    └── used by: signal_generator, main loop
```

---

## 5. Future Integration Points

These are **stub interfaces** built tonight that future components will connect to. See Future Scope document for details.

| Hook | Location | Future Component |
|---|---|---|
| `# PLUG: sentiment_score` | `signal_generator.py` line where signal strength is computed | News/NLP sentiment |
| `# PLUG: retrain_trigger` | `hmm_detector.py` after `predict_regime` | Drift detection → retraining |
| `# PLUG: rl_agent` | `signal_generator.py` return statement | RL agent replacing rule-based signals |
| `# PLUG: live_calendar` | `filters.py` blackout check | Live economic calendar API |
| `# PLUG: 7_state_upgrade` | `config.yaml` `n_states` field + `train_hmm.py` | 7-state model |
| `# PLUG: multi_instrument` | `main.py` main loop | Multi-symbol support |
