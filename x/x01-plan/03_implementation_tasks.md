# Topstep-X Automated Trading System — Implementation Tasks

> **CRITICAL RULE FOR THE IMPLEMENTATION AGENT:**  
> Each block is **independently testable**. Do NOT start a new block until the test for the current block passes.  
> Each block can be given to the agent in isolation — do not assume context from previous blocks is loaded unless stated.  
> Blocks are ordered by dependency: earlier blocks are prerequisites for later ones.

---

## Block 0 — Environment Setup

**Estimated time:** 15 minutes  
**Depends on:** Nothing  
**Files touched:** `requirements.txt`, `.env.example`, `config.yaml`

### Tasks

1. Create the repository structure as defined in the Overall Plan document
2. Create `requirements.txt` with:
   ```
   hmmlearn>=0.3.0
   numpy>=1.24
   pandas>=2.0
   httpx>=0.25
   websockets>=11.0
   yfinance>=0.2.30
   pydantic-settings>=2.0
   joblib>=1.3
   scikit-learn>=1.3
   matplotlib>=3.7
   python-dotenv>=1.0
   pytz>=2023.3
   ```
3. Create `.env.example`:
   ```
   PROJECTX_API_KEY=your_key_here
   PROJECTX_BASE_URL=https://...
   ```
4. Create `config.yaml` with all fields from the Architecture document Section 3
5. Create a `Bar` dataclass in `projectx_client.py`:
   ```python
   @dataclass
   class Bar:
       timestamp: datetime
       open: float
       high: float
       low: float
       close: float
       volume: float
   ```

### ✅ Test to Pass

```bash
python -c "import hmmlearn; import yfinance; import httpx; print('ALL IMPORTS OK')"
```
Expected output: `ALL IMPORTS OK`  
No import errors of any kind.

---

## Block 1 — Project X API Authentication and Historical Data

**Estimated time:** 15 minutes  
**Depends on:** Block 0  
**Files touched:** `projectx_client.py`

### Tasks

1. Implement `authenticate(api_key) → session_token` in `projectx_client.py`
2. Implement `get_historical_bars(symbol, timeframe, start, end) → List[Bar]`
3. Fetch 60 days of 5-min bars for MNQ and save to `data/historical/MNQ_5min.csv`
4. Fetch 60 days of 30-min bars for MNQ and save to `data/historical/MNQ_30min.csv`
5. Fetch 60 days of 1-min bars for MNQ and save to `data/historical/MNQ_1min.csv`

> ⚠️ **Decision Point:** If Project X API does not provide historical data, use `yfinance` to download `/NQ=F` (NQ futures proxy) as a substitute for training and backtesting. This is acceptable since the HMM is learning market structure, not platform-specific data. Switch to Project X data once API access is confirmed.

> ⚠️ **Decision Point:** If Project X API historical lookback is less than 60 days, use whatever is available plus yfinance to extend the dataset.

### ✅ Test to Pass

```python
# test_block1.py
import pandas as pd
df = pd.read_csv("data/historical/MNQ_5min.csv")
assert len(df) > 1000, f"Expected >1000 bars, got {len(df)}"
assert all(c in df.columns for c in ["timestamp","open","high","low","close","volume"])
print(f"OK: {len(df)} 5-min bars loaded")
```

---

## Block 2 — HMM Training and Regime Labeling

**Estimated time:** 20 minutes  
**Depends on:** Block 1 (CSV file must exist)  
**Files touched:** `regime/features.py`, `regime/hmm_detector.py`, `regime/train_hmm.py`

### Tasks

1. Implement `compute_features(bars, warmup_bars=20)` in `regime/features.py`
   - Returns numpy array of shape `(n, 3)` with columns: `[return_pct, volatility, volume_ratio]`
2. Implement `HMMDetector` class in `regime/hmm_detector.py`:
   - `train(features, n_states=5)`
   - `save(path)`
   - `load(path)`
   - `predict_regime(recent_features)` — returns integer 0–4
   - `get_state_stats()` — returns dict with mean return and mean volatility per state
3. Implement `train_hmm.py` script that:
   - Loads CSV
   - Computes features
   - Trains HMM
   - Prints state statistics (mean return, mean volatility per state) so developer can assign labels
   - Saves model to `models/hmm_model.pkl`
4. Run the training script and **manually assign state labels** in `config.yaml` based on the printed statistics:
   - Highest mean return → state label `strong_bull`
   - Lowest mean return → state label `crash`
   - Near-zero return, low volatility → `neutral`
   - etc.

### ✅ Test to Pass

```python
# test_block2.py
from regime.hmm_detector import HMMDetector
from regime.features import compute_features
import pandas as pd, numpy as np

df = pd.read_csv("data/historical/MNQ_5min.csv")
bars = df.to_dict("records")
features = compute_features(bars)

detector = HMMDetector()
detector.load("models/hmm_model.pkl")

# predict on last 50 bars
regime = detector.predict_regime(features[-50:])
assert regime in [0, 1, 2, 3, 4], f"Invalid regime: {regime}"
print(f"OK: Current regime = {regime}")

stats = detector.get_state_stats()
print("State stats:", stats)
# Visually confirm: 5 distinct states with different mean returns
```

---

## Block 3 — Backtest

**Estimated time:** 30 minutes  
**Depends on:** Block 2 (model must be trained), Block 1 (CSV must exist)  
**Files touched:** `strategy/signal_generator.py`, `strategy/risk_manager.py`, `backtest.py`

### Tasks

1. Implement `Signal` dataclass in `strategy/signal_generator.py`
2. Implement `generate_signal(regime, bar_1min, vwap, macro_bias)` — see Architecture document Section 2.5 for full logic per regime
3. Implement `RiskManager` class in `strategy/risk_manager.py`:
   - `check_can_trade() → bool`
   - `apply_size_multiplier(base_size, vix_multiplier) → int`
   - `record_trade_result(pnl)`
   - `get_daily_pnl() → float`
   - `reset_daily()`
4. Implement `backtest.py`:
   - Load 5-min bars from CSV
   - Load 1-min bars from CSV
   - Load HMM model
   - For each 5-min bar: compute features → predict regime
   - For each corresponding 1-min bar within that 5-min period: call `generate_signal`
   - If signal is not flat: simulate fill with slippage (1 tick normal, 2 ticks in crash regime)
   - Track stop/target hit using subsequent 1-min bars
   - Record trade PnL
   - Print summary at end

5. Slippage model (implement exactly as follows):
   ```python
   TICK_SIZE = 0.25  # MNQ
   POINT_VALUE = 2.0  # MNQ: $2 per point

   def simulate_fill(signal, next_bar, regime_id):
       slippage_ticks = 2 if regime_id == 4 else 1
       slippage_pts = slippage_ticks * TICK_SIZE
       if signal.direction == "long":
           return next_bar.open + slippage_pts
       elif signal.direction == "short":
           return next_bar.open - slippage_pts
   ```

6. Print backtest summary:
   ```
   === BACKTEST RESULTS ===
   Total PnL:        $XXX.XX
   Max Drawdown:     $XXX.XX
   Win Rate:         XX.X%
   Avg Win:          $XX.XX
   Avg Loss:         $XX.XX
   Win/Loss Ratio:   X.XX
   Total Trades:     XXX
   Trades/Day:       X.X
   Days Simulated:   XX
   Sessions hitting daily limit: X
   ```

### ✅ Test to Pass

```bash
python backtest.py
```

Expected: Script runs without errors and prints the summary above with real numbers.  
Acceptance criteria:
- Max drawdown < $2,000 (the Topstep total limit)
- At least 30 trades in the period (strategy is active)
- Win rate > 35% (absolute minimum — below this the strategy is broken)
- No single simulated day hits the $750 daily loss limit more than 20% of days

> ⚠️ If acceptance criteria are NOT met, do not proceed to Block 4. Instead, inspect the regime plot and check that state labels in config.yaml are correctly assigned. Common issue: states are mislabeled and the signal logic is inverted.

---

## Block 4 — Sentiment Filters

**Estimated time:** 20 minutes  
**Depends on:** Block 3 (signal_generator must exist)  
**Files touched:** `strategy/filters.py`

### Tasks

1. Implement `get_vix_multiplier() → float` in `strategy/filters.py`:
   ```python
   import yfinance as yf

   def get_vix_multiplier() -> float:
       vix = yf.download("^VIX", period="2d", interval="1d")["Close"].iloc[-1]
       if vix > 30:
           return 0.5
       return 1.0
   ```

2. Implement `is_blackout_period(now: datetime) → bool`:
   - Read high-impact times from `config.yaml`
   - Return True if within `blackout_minutes_before` or `blackout_minutes_after` of any listed time
   - Times are in US Eastern timezone — ensure timezone conversion is correct

3. Implement `get_macro_bias(bars: List[Bar], ema_period: int = 20) → str`:
   - Compute EMA of close prices over the last `ema_period` bars
   - If last close > EMA * 1.001 → return "bull"
   - If last close < EMA * 0.999 → return "bear"
   - Otherwise → return "neutral"

4. Integrate filters into `signal_generator.py`:
   - Before returning any non-flat signal, call `is_blackout_period` — return flat if True
   - Pass `macro_bias` into regime signal logic (documented in Architecture Section 2.5)
   - `# PLUG: sentiment_score` comment where additional sentiment inputs will be added later

### ✅ Test to Pass

```python
# test_block4.py
from strategy.filters import get_vix_multiplier, is_blackout_period, get_macro_bias
from datetime import datetime
import pytz

# VIX multiplier
mult = get_vix_multiplier()
assert mult in [0.5, 1.0], f"Unexpected multiplier: {mult}"
print(f"OK: VIX multiplier = {mult}")

# Blackout — test known blackout time
et = pytz.timezone("America/New_York")
blackout_time = datetime.now(et).replace(hour=8, minute=30, second=0)
assert is_blackout_period(blackout_time) == True, "08:30 should be blackout"

# Blackout — test safe time
safe_time = datetime.now(et).replace(hour=11, minute=0, second=0)
assert is_blackout_period(safe_time) == False, "11:00 should not be blackout"

print("OK: All filter tests passed")
```

---

## Block 5 — Live Order Manager

**Estimated time:** 15 minutes  
**Depends on:** Block 1 (Project X client), Block 4  
**Files touched:** `execution/order_manager.py`

### Tasks

1. Implement `OrderManager` class:
   - `enter_trade(signal: Signal) → str` (returns trade_id)
     - Calls `projectx_client.place_order(...)` with stop and target
     - Logs entry to JSON trade log
   - `exit_trade(trade_id: str, reason: str)`
     - Cancels open stop/target orders
     - Places market close order
   - `on_bar(bar: Bar)`
     - For each open trade, check if stop or target price has been crossed
     - Call `exit_trade` if so
   - `get_open_trades() → List[Trade]`

2. Add `# PLUG: bracket_order` comment if Project X does not support OCO — mark exactly where bracket support will be added

> ⚠️ **Decision Point:** If Project X API does not support live trading in the test/paper environment tonight, implement a `PaperOrderManager` class that mirrors `OrderManager` exactly but logs orders to a file instead of sending them. The live `OrderManager` can be swapped in later by changing one import line in `main.py`.

### ✅ Test to Pass

```python
# test_block5.py — paper mode test
from execution.order_manager import PaperOrderManager
from strategy.signal_generator import Signal

manager = PaperOrderManager()
signal = Signal(direction="long", size=1, stop_ticks=10, target_ticks=20, reason="test")
trade_id = manager.enter_trade(signal)
assert trade_id is not None
print(f"OK: Paper trade entered, id={trade_id}")

manager.exit_trade(trade_id, "test_exit")
assert len(manager.get_open_trades()) == 0
print("OK: Trade exited cleanly")
```

---

## Block 6 — Main Loop Integration

**Estimated time:** 15 minutes  
**Depends on:** All previous blocks  
**Files touched:** `main.py`

### Tasks

1. Implement `main.py` that:
   - Loads config
   - Authenticates with Project X
   - Fetches VIX multiplier once
   - Loads HMM model
   - Subscribes to 5-min and 1-min bar streams (or polls if WebSocket unavailable)
   - On each 1-min bar:
     - Check `risk_manager.check_can_trade()`
     - Check `is_blackout_period(now)`
     - Compute macro bias from recent 30-min bars
     - Get current regime from HMM (using last 50 5-min bars)
     - Call `generate_signal(regime, bar, vwap, macro_bias)`
     - If signal is not flat: `risk_manager.apply_size_multiplier` → `order_manager.enter_trade`
     - Call `order_manager.on_bar(bar)` to manage open positions
   - Logs all activity with timestamps

2. Add graceful shutdown on Ctrl+C — close all open positions before exit

3. VWAP computation: maintain a running VWAP that resets at session open (09:30 ET)
   ```python
   # Running VWAP
   vwap = cumulative_sum(typical_price * volume) / cumulative_sum(volume)
   # where typical_price = (high + low + close) / 3
   # Reset at 09:30 ET each day
   ```

### ✅ Test to Pass

Run in paper/dry-run mode for 5 minutes during market hours:

```bash
python main.py --paper
```

Expected:
- No crashes or import errors
- Console shows bar data being received
- Console shows regime being detected every 5 minutes
- Console shows "signal=flat" or a trade signal with all fields populated
- If a signal fires: paper order logged to `trades_paper.json`
- Daily PnL counter visible in logs

---

## Summary: Block Dependency Chain

```
Block 0 (env)
    └─→ Block 1 (API + data)
            └─→ Block 2 (HMM train)
                    └─→ Block 3 (backtest) ← GATE: must pass before continuing
                            └─→ Block 4 (filters)
                                    └─→ Block 5 (order manager)
                                            └─→ Block 6 (main loop)
```

**If Block 3 fails its acceptance criteria → stop and debug the HMM regime labeling before proceeding. Do not write live execution code with a broken strategy.**
