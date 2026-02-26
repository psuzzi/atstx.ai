# Topstep-X Automated Trading System — Architecture v2

> **Status:** Proposed — to be reviewed and confirmed with implementation agent before coding begins  
> **Decision points are marked with ⚠️ and must be resolved before or during implementation**  
> **Version:** v2 — adds symmetric signal logic, RRR parameter, filtered HMM probabilities, VWAP std dev bands

---

## 1. High-Level Data Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    SESSION STARTUP                          │
│  1. Load config + credentials from .env                     │
│  2. Authenticate with Project X API                         │
│  3. Fetch VIX → compute session size multiplier             │
│  4. Load pre-trained HMM model from disk                    │
│  5. Initialize RiskManager (reset daily counters)           │
│  6. Initialize running VWAP accumulator (resets at 09:30)   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    MAIN LOOP (per 1-min bar)                │
│                                                             │
│  Project X WebSocket / REST polling                         │
│         │                                                   │
│         ▼                                                   │
│  Bar Aggregator ──→ 5-min bars ──→ Feature Extractor        │
│         │                               │                   │
│         │            ┌──────────────────▼────────────┐     │
│         │            │  HMM Regime Detector           │     │
│         │            │  predict_proba() → [p0..p4]   │     │
│         │            │  argmax → current regime (0-4) │     │
│         │            └──────────────────┬─────────────┘    │
│         │                               │                   │
│         ▼                               ▼                   │
│  1-min bar + VWAP ──→  Signal Generator ←── Regime + Bias  │
│         │                    │                              │
│         │              Calendar Blackout?                   │
│         │                    │                              │
│         │              Risk Manager                         │
│         │              (daily PnL, consecutive losses,      │
│         │               VIX multiplier, regime multiplier)  │
│         │                    │                              │
│         │              Order Manager                        │
│         │              (place/track/cancel)                 │
│         │                    │                              │
│         ▼                    ▼                              │
│  order_manager.on_bar()    Trade Log (JSON append)          │
│  (stop/target monitoring)                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Component Descriptions

### 2.1 `projectx_client.py` — API Wrapper

**Responsibility:** All communication with the Project X / Topstep X platform.

**Core data structure:**
```python
@dataclass
class Bar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

@dataclass
class AccountInfo:
    daily_pnl: float
    total_drawdown_used: float
    open_positions: int
```

**Interfaces:**
- `authenticate(api_key: str) → str` — returns session token
- `get_historical_bars(symbol, timeframe, start, end) → List[Bar]`
- `subscribe_bars(symbol, timeframe, callback: Callable[[Bar], None])` — WebSocket stream
- `place_order(symbol, direction, size, order_type, stop_price, target_price) → str` — returns order_id
- `cancel_order(order_id: str)`
- `get_open_positions() → List[Position]`
- `get_account_info() → AccountInfo`

> ⚠️ **Decision Point 1:** Confirm Project X API authentication method (JWT token, API key header, OAuth2). Check API documentation before writing the auth method.

> ⚠️ **Decision Point 2:** Confirm whether Project X provides a WebSocket stream for bar data or if the system must poll REST. If REST only, implement a polling loop with configurable interval (default 10 seconds for 1-min bars).

> ⚠️ **Decision Point 3:** Confirm whether Project X provides native 1-min and 5-min OHLCV bars, or whether the system must aggregate from tick data. If tick aggregation is needed, add a `BarAggregator` class.

> ⚠️ **Decision Point 4:** If Project X historical data lookback is insufficient for HMM training (<30 days of 5-min bars), fall back to `yfinance` using `/NQ=F` as a proxy. HMM trains on market structure, so the proxy is acceptable. Switch to live Project X data once the trading session starts.

---

### 2.2 `regime/features.py` — Feature Engineering

**Responsibility:** Convert a list of `Bar` objects into a numpy feature matrix.

**Features:**

| Feature | Formula | Purpose |
|---|---|---|
| `return_pct` | `(close - prev_close) / prev_close` | Regime direction signal |
| `volatility` | Rolling 10-bar std of `return_pct` | Separates trending from choppy |
| `volume_ratio` | `volume / rolling_20_bar_mean(volume)` | Volume confirmation |

**Interface:**
```python
def compute_features(bars: List[Bar], warmup_bars: int = 20) -> np.ndarray:
    """Returns shape (n_bars - warmup, 3). Drops first warmup_bars."""
```

> ⚠️ **Decision Point 5:** If visualized state separation is poor after training, consider adding `bar_range_pct = (high - low) / close` as a 4th feature. Do not add preemptively — keep it simple and expand based on evidence.

---

### 2.3 `regime/hmm_detector.py` — HMM Wrapper

**Responsibility:** Train, save, load, and run inference on the GaussianHMM.

**Critical implementation note — filtered probabilities:**

For live trading, **always use `model.predict_proba(features)`** (the forward algorithm), which returns the probability distribution over states using only data up to the current bar. Do NOT use `model.predict()` (Viterbi) for live inference — Viterbi optimizes the entire path and can change past labels when new data arrives, making it unsuitable for real-time decisions.

```python
def predict_regime(self, recent_features: np.ndarray) -> tuple[int, np.ndarray]:
    """
    Returns (regime_id, probabilities).
    Uses predict_proba (filtered, forward algorithm) — not Viterbi.
    regime_id = argmax of last bar's probability vector.
    """
    proba = self.model.predict_proba(recent_features)
    last_bar_proba = proba[-1]  # probabilities for current bar only
    regime_id = int(np.argmax(last_bar_proba))
    return regime_id, last_bar_proba
```

The returned `probabilities` array can optionally be used to scale position size by regime confidence (future enhancement).

**Interface:**
- `train(features: np.ndarray, n_states: int = 5, covariance_type: str = "diag")`
- `save(path: str)`
- `load(path: str)`
- `predict_regime(recent_features: np.ndarray) -> tuple[int, np.ndarray]`
- `get_state_stats() -> dict` — mean return and volatility per state (for manual labeling)

> ⚠️ **Decision Point 6:** Covariance type. `"diag"` (diagonal) is recommended as the default — faster and less data-hungry than `"full"`. Switch to `"full"` only if states are hard to separate and more training data is available.

> ⚠️ **Decision Point 7:** Length of feature window fed to `predict_regime` at runtime. Recommended: last 50 bars (approximately 4 hours of 5-min data). If regime switches feel too slow/laggy, reduce. If too noisy, increase.

---

### 2.4 `regime/train_hmm.py` — Offline Training Script

**Responsibility:** Standalone script to train HMM and save model. Run once before backtesting and live trading. Re-run periodically for model refresh (see Future Scope).

```bash
python regime/train_hmm.py --data data/historical/MNQ_5min.csv --states 5
```

**Output:**
- `models/hmm_model.pkl`
- Console: per-state mean return and mean volatility → developer manually assigns labels in `config.yaml`

**State label assignment (post-training):**
Inspect `model.means_` after training. Sort states by mean return:
- Highest mean return → `strong_bull`
- Second highest → `bull`
- Near-zero return, lowest volatility → `neutral`
- Second lowest → `bear`
- Lowest mean return → `crash`

---

### 2.5 `strategy/signal_generator.py` — Signal Logic (Symmetric)

**Responsibility:** Given regime, current 1-min bar, VWAP, VWAP std dev, and macro bias — return a `Signal`.

**Data structures:**
```python
@dataclass
class Signal:
    direction: str        # "long", "short", "flat"
    size: int             # base contracts (before multipliers)
    stop_ticks: int       # stop distance from entry
    target_ticks: int     # = round(stop_ticks * rrr)
    reason: str           # for logging and debugging

@dataclass
class VWAPContext:
    vwap: float
    std_dev: float        # rolling intraday std of typical_price
    upper_band: float     # vwap + std_dev
    lower_band: float     # vwap - std_dev
```

**Full symmetric signal logic:**

```python
def generate_signal(regime, bar_1min, vwap_ctx, macro_bias, config) -> Signal:
    rrr = config.trading.rrr  # 1.8

    # --- LONG REGIMES ---
    if regime == STRONG_BULL:
        stop = config.stops.strong_bull  # 10 ticks
        if bar_1min.low <= vwap_ctx.vwap and bar_1min.close > bar_1min.open:
            if macro_bias != "bear":
                return Signal("long", 1, stop, round(stop * rrr), "strong_bull_pullback")

    elif regime == BULL:
        stop = config.stops.bull  # 8 ticks
        if bar_1min.low <= vwap_ctx.vwap and bar_1min.close > bar_1min.open:
            if macro_bias != "bear":
                return Signal("long", 1, stop, round(stop * rrr), "bull_pullback")

    # --- NEUTRAL — MEAN REVERSION (BOTH DIRECTIONS) ---
    elif regime == NEUTRAL:
        stop = config.stops.neutral  # 6 ticks
        min_range = config.trading.min_neutral_range_points  # 8 points default
        if vwap_ctx.std_dev * 2 / TICK_SIZE < min_range / TICK_SIZE:
            return Signal("flat", 0, 0, 0, "neutral_range_too_tight")

        if bar_1min.close < vwap_ctx.lower_band and bar_1min.close > bar_1min.open:
            # Price below lower band, bullish reversal bar
            return Signal("long", 1, stop, round(stop * rrr), "neutral_mean_rev_long")

        if bar_1min.close > vwap_ctx.upper_band and bar_1min.close < bar_1min.open:
            # Price above upper band, bearish reversal bar
            return Signal("short", 1, stop, round(stop * rrr), "neutral_mean_rev_short")

    # --- SHORT REGIMES ---
    elif regime == BEAR:
        stop = config.stops.bear  # 8 ticks
        if bar_1min.high >= vwap_ctx.vwap and bar_1min.close < bar_1min.open:
            if macro_bias != "bull":
                return Signal("short", 1, stop, round(stop * rrr), "bear_rally_fade")

    elif regime == CRASH:
        stop = config.stops.crash  # 6 ticks
        bar_range = bar_1min.high - bar_1min.low
        lower_wick = bar_1min.close - bar_1min.low  # for bullish bar
        if bar_range > 0 and (lower_wick / bar_range) > 0.6:
            # Exhaustion signal: buyers absorbing the crash
            return Signal("long", 1, stop, round(stop * rrr), "crash_exhaustion_bounce")

    # # PLUG: sentiment_score
    # Modify signal size or direction here based on external sentiment feed

    return Signal("flat", 0, 0, 0, "no_setup")
```

> **Note on neutral mean reversion target:** target is VWAP (not `stop * rrr` beyond entry). The `target_ticks` computed from RRR may exceed the actual VWAP distance. In practice: `target_ticks = min(round(stop * rrr), distance_to_vwap_in_ticks)`.

> ⚠️ **Decision Point 8:** Whether to allow neutral mean-reversion trades when macro bias is directional. Currently: no restriction. Alternative: only trade mean reversion when macro bias is "neutral". This reduces trades but improves quality.

---

### 2.6 `strategy/filters.py` — Filters

**Responsibility:** VIX sizing multiplier, calendar blackout, macro EMA trend bias.

**Interface:**
```python
def get_vix_multiplier() -> float
    # Returns 0.5 if VIX > 30, else 1.0

def is_blackout_period(now: datetime) -> bool
    # Returns True within blackout window of any high-impact event

def get_macro_bias(bars: List[Bar], ema_period: int = 20) -> str
    # Returns "bull", "bear", or "neutral" based on EMA of close prices
    # bull: last_close > ema * 1.001
    # bear: last_close < ema * 0.999
    # neutral: within 0.1% of ema

# PLUG: live_calendar — replace hardcoded HIGH_IMPACT_TIMES with API call
# PLUG: sentiment_score — add news sentiment multiplier here
```

---

### 2.7 `strategy/risk_manager.py` — Risk Manager

**Responsibility:** Position sizing, daily loss enforcement, consecutive loss tracking, per-regime size multipliers.

**Size multiplier logic:**
```python
REGIME_SIZE_MULTIPLIER = {
    "strong_bull": 1.0,
    "bull":        1.0,
    "neutral":     0.75,   # mean reversion is lower confidence
    "bear":        1.0,
    "crash":       0.5,    # crash bounces fail frequently
}

def compute_final_size(base_size, regime, vix_multiplier) -> int:
    regime_mult = REGIME_SIZE_MULTIPLIER[regime]
    return max(1, floor(base_size * regime_mult * vix_multiplier))
```

**Interface:**
- `check_can_trade() -> bool` — False if daily_pnl ≤ -750 or consecutive_losses ≥ 3
- `compute_final_size(base_size, regime, vix_multiplier) -> int`
- `record_trade_result(pnl: float)`
- `get_daily_pnl() -> float`
- `reset_daily()` — called at session open

---

### 2.8 `execution/order_manager.py` — Order Manager

**Responsibility:** Place, track, and manage orders. Two implementations: live and paper.

**Interface (shared by both):**
```python
class BaseOrderManager(ABC):
    def enter_trade(self, signal: Signal, entry_price: float) -> str
    def exit_trade(self, trade_id: str, reason: str)
    def on_bar(self, bar: Bar)  # checks stop/target hits
    def get_open_trades(self) -> List[Trade]
```

`PaperOrderManager` logs to `trades_paper.json` and simulates fills.  
`LiveOrderManager` calls `projectx_client.place_order(...)`.

Switch between them with one import line in `main.py`.

> ⚠️ **Decision Point 9:** Confirm whether Project X API supports bracket (OCO — One Cancels Other) orders with stop + target in a single API call. If yes, `enter_trade` places one bracket order. If no, the order manager must track stop/target manually via `on_bar()` and send separate cancel/market orders when a level is hit.

---

### 2.9 `backtest.py` — Backtester

**Key design principle:** Uses the **exact same** `features.py`, `hmm_detector.py`, `signal_generator.py`, `filters.py`, and `risk_manager.py` as live trading. Only the order execution is replaced with simulation.

**VWAP computation in backtest** (same formula as live):
```python
# Running VWAP — resets at each new session date
typical_price = (bar.high + bar.low + bar.close) / 3
cumulative_tp_vol += typical_price * bar.volume
cumulative_vol += bar.volume
vwap = cumulative_tp_vol / cumulative_vol
```

**Slippage model:**
```python
TICK_SIZE = 0.25   # MNQ
POINT_VALUE = 2.0  # MNQ: $2/point
COMMISSION_PER_SIDE = 0.50  # MNQ typical

def simulate_fill(signal, next_bar, regime_id):
    slippage_ticks = 3 if regime_id == 4 else 1
    slippage_pts = slippage_ticks * TICK_SIZE
    if signal.direction == "long":
        fill = next_bar.open + slippage_pts
    else:
        fill = next_bar.open - slippage_pts
    return fill
```

**Required output metrics:**
```
=== BACKTEST RESULTS ===
Period:            DD/MM/YYYY – DD/MM/YYYY
Total PnL:         $XXX.XX (net of slippage + commission)
Max Drawdown:      $XXX.XX
Win Rate:          XX.X%
Avg Win:           $XX.XX
Avg Loss:          $XX.XX
Win/Loss Ratio:    X.XX
Total Trades:      XXX
Trades/Day:        X.X
Days Simulated:    XX
Days hitting $750 limit: X (XX%)

PnL by Regime:
  Strong Bull:     $XXX  (XX trades)
  Bull:            $XXX  (XX trades)
  Neutral:         $XXX  (XX trades)
  Bear:            $XXX  (XX trades)
  Crash:           $XXX  (XX trades)
```

The **PnL by regime** breakdown is critical — it shows which regimes are profitable and which are not, so unprofitable regimes can be disabled or tuned independently.

---

## 3. Configuration (`config.yaml`) — Complete Schema

```yaml
api:
  base_url: "${PROJECTX_BASE_URL}"
  api_key: "${PROJECTX_API_KEY}"

trading:
  symbol: "MNQ"
  base_size: 1
  daily_loss_limit: 750
  max_consecutive_losses: 3
  rrr: 1.8                       # Risk/Reward Ratio — target = stop * rrr
  min_neutral_range_points: 8    # minimum VWAP std dev band width for neutral trades

timeframes:
  regime: "5min"
  macro: "30min"   # DECISION POINT A: 30min vs 1hour
  entry: "1min"

hmm:
  n_states: 5
  covariance_type: "diag"
  model_path: "models/hmm_model.pkl"
  predict_window: 50
  # Assigned manually after inspecting model.means_ post-training:
  state_labels:
    0: "strong_bull"
    1: "bull"
    2: "neutral"
    3: "bear"
    4: "crash"

stops:
  strong_bull: 10   # ticks
  bull: 8
  neutral: 6
  bear: 8
  crash: 6

regime_size_multipliers:
  strong_bull: 1.0
  bull: 1.0
  neutral: 0.75
  bear: 1.0
  crash: 0.5

vwap:
  std_dev_window: 20   # bars for rolling std of typical_price (neutral bands)
  session_open_et: "09:30"

filters:
  blackout_minutes_before: 5
  blackout_minutes_after: 10
  high_impact_times_et:
    - "08:30"
    - "10:00"
    - "14:00"
    - "14:30"
  vix_fear_threshold: 30
  macro_ema_period: 20
  macro_bias_threshold: 0.001   # 0.1% from EMA = neutral

slippage:
  normal_ticks: 1
  crash_ticks: 3

commission:
  per_side: 0.50   # USD per contract per side

instrument:
  tick_size: 0.25
  point_value: 2.0
```

---

## 4. Future Integration Points

| Hook | File + Location | Future Component |
|---|---|---|
| `# PLUG: sentiment_score` | `signal_generator.py`, after regime signal logic | News/NLP sentiment modifier |
| `# PLUG: live_calendar` | `filters.py`, `is_blackout_period()` | Live economic calendar API |
| `# PLUG: retrain_trigger` | `hmm_detector.py`, `predict_regime()` | Drift detection → auto-retraining |
| `# PLUG: rl_agent` | `signal_generator.py`, return statement | RL agent replacing rule-based signals |
| `# PLUG: 7_state_upgrade` | `config.yaml` `n_states` + `train_hmm.py` | 7-state model |
| `# PLUG: multi_instrument` | `main.py`, main loop | Multi-symbol support |
| `# PLUG: distributed_state` | `main.py`, initialization | Redis-backed shared state for multi-machine |
| `# PLUG: regime_confidence` | `signal_generator.py`, size calculation | Scale size by `max(proba)` confidence |
| `# PLUG: walk_forward` | `backtest.py`, main loop | Walk-forward optimization |
| `# PLUG: bracket_order` | `order_manager.py`, `enter_trade()` | OCO bracket if API supports it |
