# Topstep-X Automated Trading System — Implementation Tasks v2

> **CRITICAL RULE FOR THE IMPLEMENTATION AGENT:**  
> Each block is **independently testable**. Do NOT start a new block until the test for the current block passes.  
> Blocks can be handed to the agent in isolation — do not assume context from previous blocks is in memory.  
> Blocks are ordered by hard dependency: earlier blocks are prerequisites for later ones.  
> **Version:** v2 — updated with symmetric long/short, RRR parameter, VWAP std dev bands, filtered HMM probabilities

---

## Block 0 — Environment Setup

**Estimated time:** 15 minutes  
**Depends on:** Nothing  
**Files:** `requirements.txt`, `.env.example`, `config.yaml`, `projectx_client.py` (Bar dataclass only)

### Tasks

1. Create full repository structure as defined in Overall Plan v2 Section 10
2. Create `requirements.txt`:
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
4. Create `config.yaml` with the **complete schema** from Architecture v2 Section 3
5. Create `projectx_client.py` with `Bar` and `AccountInfo` dataclasses only (no API calls yet)

### ✅ Test to Pass

```bash
pip install -r requirements.txt
python -c "import hmmlearn; import yfinance; import httpx; import numpy; print('ALL IMPORTS OK')"
python -c "from projectx_client import Bar; print('Bar dataclass OK')"
```
Expected: no errors.

---

## Block 1 — Historical Data

**Estimated time:** 15 minutes  
**Depends on:** Block 0  
**Files:** `projectx_client.py` (full implementation)

### Tasks

1. Implement `authenticate(api_key) → str` in `projectx_client.py`
2. Implement `get_historical_bars(symbol, timeframe, start, end) → List[Bar]`
3. Fetch and save to CSV:
   - `data/historical/MNQ_1min.csv` — 60 days
   - `data/historical/MNQ_5min.csv` — 60 days
   - `data/historical/MNQ_30min.csv` — 60 days (for macro filter)

> ⚠️ **Decision Point 1:** If Project X API is unavailable or has insufficient lookback, use `yfinance` with `/NQ=F` as fallback:
> ```python
> import yfinance as yf
> df = yf.download("/NQ=F", period="60d", interval="5m")
> ```
> Document which data source was used in a comment at the top of the download script.

### ✅ Test to Pass

```python
# test_block1.py
import pandas as pd

for tf, min_bars in [("1min", 5000), ("5min", 1000), ("30min", 200)]:
    df = pd.read_csv(f"data/historical/MNQ_{tf}.csv")
    assert len(df) >= min_bars, f"{tf}: expected >={min_bars} bars, got {len(df)}"
    assert all(c in df.columns for c in ["timestamp","open","high","low","close","volume"])
    print(f"OK: MNQ_{tf}.csv — {len(df)} bars")
```

---

## Block 2 — HMM Training and Regime Labeling

**Estimated time:** 25 minutes  
**Depends on:** Block 1 (CSV files must exist)  
**Files:** `regime/features.py`, `regime/hmm_detector.py`, `regime/train_hmm.py`

### Tasks

1. Implement `compute_features(bars, warmup_bars=20) → np.ndarray` in `regime/features.py`
   - Returns shape `(n, 3)`: `[return_pct, volatility, volume_ratio]`
   - Drop first `warmup_bars` rows

2. Implement `HMMDetector` class in `regime/hmm_detector.py`:
   ```python
   class HMMDetector:
       def train(self, features: np.ndarray, n_states: int = 5,
                 covariance_type: str = "diag") -> None
       def save(self, path: str) -> None
       def load(self, path: str) -> None
       def predict_regime(self, recent_features: np.ndarray) -> tuple[int, np.ndarray]:
           """Uses predict_proba (filtered, forward algorithm) — NOT Viterbi."""
           # proba = self.model.predict_proba(recent_features)
           # last_bar = proba[-1]
           # return int(np.argmax(last_bar)), last_bar
       def get_state_stats(self) -> dict:
           """Returns mean return and mean volatility per state for manual labeling."""
           # PLUG: retrain_trigger — add log-likelihood monitoring here
   ```

3. Implement `train_hmm.py` script:
   - Load CSV, compute features, train, print state stats, save model
   - Output format for state stats (so developer can assign labels):
     ```
     State 0: mean_return=+0.0023, mean_volatility=0.0008 → likely: strong_bull
     State 1: mean_return=+0.0008, mean_volatility=0.0006 → likely: bull
     State 2: mean_return=-0.0001, mean_volatility=0.0004 → likely: neutral
     State 3: mean_return=-0.0009, mean_volatility=0.0007 → likely: bear
     State 4: mean_return=-0.0031, mean_volatility=0.0015 → likely: crash
     ```
   - **After seeing actual output:** update `state_labels` in `config.yaml` accordingly

4. Run training:
   ```bash
   python regime/train_hmm.py --data data/historical/MNQ_5min.csv --states 5
   ```

### ✅ Test to Pass

```python
# test_block2.py
from regime.hmm_detector import HMMDetector
from regime.features import compute_features
import pandas as pd, numpy as np

df = pd.read_csv("data/historical/MNQ_5min.csv")
bars = df.to_dict("records")
features = compute_features(bars)

assert features.shape[1] == 3, "Expected 3 features"
assert len(features) > 0, "No features computed"

detector = HMMDetector()
detector.load("models/hmm_model.pkl")

regime_id, proba = detector.predict_regime(features[-50:])
assert regime_id in [0, 1, 2, 3, 4], f"Invalid regime: {regime_id}"
assert len(proba) == 5, "Expected 5 probabilities"
assert abs(sum(proba) - 1.0) < 1e-6, "Probabilities must sum to 1"

print(f"OK: regime={regime_id}, confidence={max(proba):.2%}")
print(f"Full proba: {proba}")
# Visually verify: one state has high probability, others low — model is decisive
```

**Additional manual check:** plot the regime labels on the price chart and verify they visually correspond to the expected market conditions. If states look random or uninterpretable, re-examine the feature computation and potentially adjust `n_states`.

---

## Block 3 — Backtest (HARD GATE)

**Estimated time:** 30 minutes  
**Depends on:** Block 2 (model must exist), Block 1 (CSVs must exist)  
**Files:** `strategy/signal_generator.py`, `strategy/risk_manager.py`, `backtest.py`

### Tasks

1. Implement `Signal` and `VWAPContext` dataclasses in `strategy/signal_generator.py`

2. Implement `compute_vwap_context(bars_1min_today: List[Bar], std_window: int = 20) → VWAPContext`:
   ```python
   # Running VWAP — call with all bars since 09:30 ET today
   # typical_price = (high + low + close) / 3
   # vwap = sum(tp * volume) / sum(volume)
   # std_dev = rolling std of typical_price over last std_window bars
   # upper_band = vwap + std_dev
   # lower_band = vwap - std_dev
   ```

3. Implement `generate_signal(regime, bar_1min, vwap_ctx, macro_bias, config) → Signal`
   - Full symmetric logic as defined in Architecture v2 Section 2.5
   - Read `rrr` from config: `target_ticks = round(stop_ticks * config.trading.rrr)`
   - Leave `# PLUG: sentiment_score` comment at the correct location

4. Implement `RiskManager` in `strategy/risk_manager.py`

5. Implement `backtest.py`:
   - Load 5-min and 1-min CSVs
   - Group 1-min bars by day; reset VWAP at 09:30 ET each day
   - For each 5-min bar: compute features → `predict_regime` (filtered probabilities)
   - For each 1-min bar in the same period: call `generate_signal`
   - Simulate fill: `next_bar.open ± (slippage_ticks * TICK_SIZE)`
   - Track stop/target hit using subsequent 1-min bars
   - Run full `RiskManager` logic (respect daily limit, consecutive losses)
   - Also compute and apply `get_macro_bias` from 30-min bars (use stub returning "neutral" if not yet implemented)
   - Print full summary including **PnL by regime** breakdown

6. Add commission to every closed trade: `pnl -= 2 * COMMISSION_PER_SIDE` (round trip)

### ✅ Test to Pass

```bash
python backtest.py
```

**Acceptance criteria (all must pass before proceeding):**

| Metric | Minimum Threshold | Notes |
|---|---|---|
| Script runs without error | Required | No exceptions |
| Total trades | ≥ 30 | Strategy must be active |
| Max drawdown | < $2,000 | Hard Topstep limit |
| Sessions hitting daily limit | < 30% of days | More means risk is too high |
| Win rate | > 35% | Below this = strategy is broken |
| At least one regime shows positive PnL | Required | Confirms signal logic works |

> ⚠️ **If acceptance criteria are NOT met:** do not proceed to Block 4. Instead:
> 1. Check that `state_labels` in `config.yaml` are correctly assigned
> 2. Plot regime labels on the price chart to verify they make visual sense
> 3. Common issue: long/short logic inverted because states are mislabeled
> 4. Adjust neutral regime parameters (`min_neutral_range_points`, `std_dev_window`) if neutral trades are the problem
> 5. Check that filtered probabilities are being used (not Viterbi)

---

## Block 4 — Filters and Macro Bias

**Estimated time:** 20 minutes  
**Depends on:** Block 3 (signal_generator must exist)  
**Files:** `strategy/filters.py`

### Tasks

1. Implement `get_vix_multiplier() → float`:
   ```python
   import yfinance as yf
   def get_vix_multiplier() -> float:
       vix = yf.download("^VIX", period="2d", interval="1d")["Close"].iloc[-1]
       return 0.5 if float(vix) > config.filters.vix_fear_threshold else 1.0
   ```

2. Implement `is_blackout_period(now: datetime) → bool`:
   - Read times from `config.yaml`
   - All times are US Eastern — use `pytz` for timezone handling
   - `# PLUG: live_calendar` comment where the API call will replace the hardcoded list

3. Implement `get_macro_bias(bars: List[Bar], ema_period: int = 20) → str`:
   - EMA of close prices
   - Returns "bull" / "bear" / "neutral" based on threshold from config
   - Used on 30-min bars (passed in from main loop)

4. Re-run backtest with macro filter now active (replace the "neutral" stub from Block 3):
   ```bash
   python backtest.py
   ```
   Observe if win rate or drawdown improves. Note results.

### ✅ Test to Pass

```python
# test_block4.py
from strategy.filters import get_vix_multiplier, is_blackout_period, get_macro_bias
from datetime import datetime
import pytz, pandas as pd

mult = get_vix_multiplier()
assert mult in [0.5, 1.0], f"Unexpected: {mult}"
print(f"OK: VIX multiplier = {mult}")

et = pytz.timezone("America/New_York")

blackout = datetime.now(et).replace(hour=8, minute=30, second=0, microsecond=0)
assert is_blackout_period(blackout) == True

safe = datetime.now(et).replace(hour=11, minute=0, second=0, microsecond=0)
assert is_blackout_period(safe) == False

print("OK: Blackout tests passed")

df = pd.read_csv("data/historical/MNQ_30min.csv")
bars = df.to_dict("records")
bias = get_macro_bias(bars[-30:])
assert bias in ["bull", "bear", "neutral"]
print(f"OK: macro_bias = {bias}")
```

---

## Block 5 — Order Manager

**Estimated time:** 15 minutes  
**Depends on:** Block 1 (projectx_client), Block 4  
**Files:** `execution/order_manager.py`

### Tasks

1. Define `Trade` dataclass:
   ```python
   @dataclass
   class Trade:
       trade_id: str
       direction: str
       entry_price: float
       stop_price: float
       target_price: float
       size: int
       entry_time: datetime
       regime: str
       status: str  # "open", "closed_stop", "closed_target", "closed_manual"
       pnl: float = 0.0
   ```

2. Implement `PaperOrderManager(BaseOrderManager)`:
   - `enter_trade(signal, entry_price) → str` — creates Trade, appends to `trades_paper.json`
   - `exit_trade(trade_id, reason)` — marks closed, computes PnL, updates JSON
   - `on_bar(bar)` — checks stop/target hit for each open trade
   - `# PLUG: bracket_order` comment in `enter_trade`

3. Implement `LiveOrderManager(BaseOrderManager)`:
   - Same interface but calls `projectx_client.place_order(...)`

4. In `main.py`, select manager via config:
   ```python
   if config.mode == "paper":
       order_manager = PaperOrderManager()
   else:
       order_manager = LiveOrderManager(client)
   ```

> ⚠️ **Decision Point 9:** If Project X does not support bracket orders, `LiveOrderManager.enter_trade()` must track stop/target manually via `on_bar()` and send a cancel + market order when a level is crossed. Implement this fallback if needed.

### ✅ Test to Pass

```python
# test_block5.py
from execution.order_manager import PaperOrderManager
from strategy.signal_generator import Signal
import os, json

if os.path.exists("trades_paper.json"):
    os.remove("trades_paper.json")

mgr = PaperOrderManager()
signal = Signal("long", 1, 8, 14, "test")
trade_id = mgr.enter_trade(signal, entry_price=21000.00)
assert trade_id is not None

trades = mgr.get_open_trades()
assert len(trades) == 1

mgr.exit_trade(trade_id, "test_exit")
assert len(mgr.get_open_trades()) == 0

with open("trades_paper.json") as f:
    log = json.load(f)
assert len(log) == 1
assert log[0]["status"] in ["closed_stop", "closed_target", "closed_manual"]

print(f"OK: Paper trade cycle complete, PnL={log[0]['pnl']}")
```

---

## Block 6 — Main Loop Integration

**Estimated time:** 15 minutes  
**Depends on:** All previous blocks  
**Files:** `main.py`

### Tasks

1. Implement `main.py`:
   - Parse `--paper` / `--live` flag
   - Session startup: auth, VIX fetch, model load, risk manager reset, VWAP accumulator init
   - Subscribe to or poll 1-min and 5-min bar streams
   - On each 1-min bar:
     - Update running VWAP; reset if new session date
     - Refresh 5-min regime (using last 50 5-min bars via `predict_regime`)
     - Get macro bias from last 20 30-min bars
     - Call `is_blackout_period(now)` — skip if True
     - Call `risk_manager.check_can_trade()` — skip if False
     - Call `generate_signal(regime, bar, vwap_ctx, macro_bias, config)`
     - If not flat: `compute_final_size(...)` → `order_manager.enter_trade(...)`
     - Call `order_manager.on_bar(bar)` — manage open positions
   - Graceful Ctrl+C: close all open positions before exit
   - Log every bar, every signal, every trade to console with timestamp

2. Leave `# PLUG: multi_instrument` comment at the main loop entry point

### ✅ Test to Pass

Run paper mode for 5 minutes during market hours (or simulate with replayed bars if market is closed):

```bash
python main.py --paper
```

**Expected output (every bar):**
```
2025-XX-XX 09:35:15 | Bar: O=21000.00 H=21015.00 L=20995.00 C=21010.00 V=1234
2025-XX-XX 09:35:15 | Regime: bull (1) | Confidence: 78.3% | MacroBias: bull
2025-XX-XX 09:35:15 | VWAP: 21005.50 | Upper: 21022.10 | Lower: 20988.90
2025-XX-XX 09:35:15 | Signal: long | Stop: 8tk | Target: 14tk | Reason: bull_pullback
2025-XX-XX 09:35:15 | DailyPnL: $0.00 | ConsecLosses: 0 | CanTrade: True
```

Acceptance: no crashes, regime detected every 5 min, signals generated or flat with valid reason, paper trades logged if signals fire.

---

## Block Order and Gates

```
Block 0 (env + dataclasses)
    └─→ Block 1 (historical data)
            └─→ Block 2 (HMM training)
                    └─→ Block 3 (backtest) ◄── HARD GATE
                                                Do not pass unless all
                                                acceptance criteria met
                            └─→ Block 4 (filters + macro bias)
                                    └─→ Block 5 (order manager)
                                            └─→ Block 6 (main loop)
```

**If Block 3 fails:** stop. Fix regime labeling. Re-run backtest. Do not write live code with a broken strategy.
