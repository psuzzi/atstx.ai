# Topstep-X Automated Trading System — Overall Plan

> **Session:** Initial implementation session  
> **Goal:** Have a working, backtested, live-executable trading system by end of session  
> **Target platform:** Topstep X (prop firm), accessed via Project X APIs  
> **Language:** Python 3.11+

---

## 1. Objective

Build an automated trading system that:

1. Connects to the **Project X API** to receive market data and place orders on the **Topstep X** platform
2. Uses a **Hidden Markov Model (HMM)** to classify the current market into one of **5 regime states**
3. Applies **deterministic signal rules per regime** to generate trade signals (no RL tonight)
4. Enforces **strict risk management** aligned with Topstep X account limits
5. Can be run locally on one or more machines from a **git repository**
6. Is designed so that future components (sentiment, retraining, RL) can be **plugged in cleanly**

---

## 2. Platform Constraints (Topstep X)

These are hard constraints that the system must respect at all times:

| Constraint | Value | Notes |
|---|---|---|
| Total max drawdown | $2,000 | Account-level, enforced by Topstep |
| Daily loss limit | ~$750 | Self-imposed conservative limit |
| Max consecutive losses before stop | 3 | Coded into risk manager |
| Starting instrument | MNQ (Micro NQ) | 1 point = $2, 1 tick = $0.50 |
| Starting position size | 1 contract | Scale up only after proven performance |

The system must check daily PnL at every trade entry and refuse to open new positions if the daily limit is approaching.

---

## 3. Market Analysis Approach

### 3.1 Timeframe Stack

Three timeframes are used together:

| Timeframe | Role | How Used |
|---|---|---|
| **30-min or 1-hour** | Macro trend bias | Simple EMA filter — are we in a bull or bear day? Sets directional bias. |
| **5-min** | Regime detection | HMM runs on 5-min bars. This is the primary signal timeframe. |
| **1-min** | Entry timing | Once regime and bias agree, 1-min bar pattern triggers the actual entry. |

> **Decision Point:** 30-min vs 1-hour for the macro filter. Both are valid. 30-min gives more bars for intraday context (4–6 bars in early session vs 2–3 for 1-hour). 1-hour is simpler and less ambiguous. **To be decided with the implementation agent.**

### 3.2 Regime States (5-State HMM)

The HMM is trained on 5-min bars and classifies the market into 5 states:

| State ID | Label | Market Character | Trading Bias |
|---|---|---|---|
| 0 | Strong Bull | High positive returns, volume confirming, shallow pullbacks | Full size long |
| 1 | Bull | Positive drift, moderate volume, normal retracements | Normal size long |
| 2 | Neutral | Near-zero drift, low volume, price oscillating around VWAP | Flat or mean-reversion only |
| 3 | Bear | Negative drift, volume rising on down moves | Normal size short |
| 4 | Crash / Strong Bear | Fast waterfall, volume spike, wide spread | Wait for reversal — reduced size long on bounce |

> **Decision Point:** Whether to expand to 7 states in the future. The 7-state version adds "Soft Bull" between Strong Bull and Bull, and "Soft Bear" between Bear and Crash. This requires significantly more training data (months vs weeks) to be stable. Upgrade path is documented in the Future Scope document.

### 3.3 HMM Features

The HMM is trained on a feature vector computed from each 5-min bar:

- **Return:** `(close - prev_close) / prev_close`
- **Volatility:** rolling 10-bar standard deviation of returns
- **Volume ratio:** `current_volume / rolling_20_bar_avg_volume`

> **Decision Point:** Whether to add additional features such as bar range (high-low) or VWAP distance. Start simple with these 3 features and expand if regime separation is poor.

---

## 4. Signal Logic Per Regime

### States 0 & 1 — Bull / Strong Bull
- **Entry trigger:** 1-min bar where `low <= VWAP` and `close > open` (bullish bar touching VWAP)
- **Stop:** 8–10 ticks below entry
- **Target:** State 0: 20 ticks (2:1), State 1: 12 ticks (1.5:1)
- **Condition:** only if macro timeframe trend bias is neutral or bullish

### State 2 — Neutral
- **No directional trades**
- Optional: fade extreme deviations from VWAP (>0.5 ATR), very small size
- Safest choice: stay flat entirely

### States 3 & 4 — Bear / Crash
- State 3 mirrors State 1 but short: sell rallies to VWAP
- State 4 (Crash): **do not short the crash itself** — wait for a 1-min bar with a long lower wick (>60% of bar range), indicating exhaustion. Enter long the bounce. Stop below wick low. Target 15 ticks.

---

## 5. Sentiment / Macro Filters (Implemented Tonight)

### VIX Filter
- Fetch VIX once at session open via `yfinance`
- `VIX > 30` → `fearful` → position size multiplier = 0.5
- `VIX 20–30` → `elevated` → multiplier = 1.0
- `VIX < 20` → `calm` → multiplier = 1.0

### Economic Calendar Blackout
- Hardcoded high-impact US times (Eastern):
  - 08:30 — CPI, NFP, Jobless Claims
  - 10:00 — ISM, Consumer Confidence
  - 14:00 — FOMC decision
  - 14:30 — Fed press conference
- No new trades within 5 minutes before or 10 minutes after these times
- Configurable in `config.yaml`

---

## 6. Backtesting Requirements

Backtesting is **mandatory before live trading**. The backtest must:

- Replay historical 5-min bars through the exact same HMM + signal code used in live trading
- Model **slippage**: +1 tick on long entries, -1 tick on short entries (2-3 ticks in Crash regime)
- Track: total PnL, max drawdown, win rate, avg win/loss ratio, trades per day
- Flag any backtest scenario where the daily loss limit could be hit in a single bad session
- Be runnable entirely **offline** (CSV + pre-trained model, no API required)

---

## 7. Decisions Made in This Session

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Best ecosystem for hmmlearn, fast iteration |
| HMM library | `hmmlearn` GaussianHMM | Well-documented, easy serialization |
| Number of states | 5 | Balance between granularity and data requirements |
| Entry timeframe | 1-min | Fine enough for timing without excessive noise |
| Regime timeframe | 5-min | Standard for intraday futures |
| Macro filter | 30-min or 1-hour EMA (TBD) | Prevents trading against dominant trend |
| Slippage model | 1 tick normal, 2-3 ticks crash | Conservative, realistic for MNQ |
| Starting instrument | MNQ | Low risk per tick, good for initial testing |
| Starting size | 1 contract | Preserves capital while system is proven |
| Sentiment tonight | VIX only | News NLP too complex for tonight |
| Calendar tonight | Hardcoded times | Live API deferred to future |
| ML for entry | Deterministic rules | RL deferred; rules are understandable and debuggable |

---

## 8. Repository Structure (Target)

```
topstep-trader/
├── README.md
├── config.yaml                  # credentials, params (secrets gitignored)
├── requirements.txt
├── main.py                      # live trading entry point
├── backtest.py                  # backtesting entry point
├── projectx_client.py           # Project X API wrapper
├── regime/
│   ├── features.py              # feature computation from bars
│   ├── hmm_detector.py          # HMM model wrapper (train + predict)
│   └── train_hmm.py             # offline training script
├── strategy/
│   ├── signal_generator.py      # regime + bias → signal
│   ├── risk_manager.py          # position sizing, daily loss check
│   └── filters.py               # VIX filter, calendar blackout
├── execution/
│   └── order_manager.py         # place, track, cancel orders
├── models/
│   └── hmm_model.pkl            # serialized trained model (gitignored)
└── data/
    └── historical/              # CSVs for training/backtesting
```

---

## 9. Session Timeline (Target 2 Hours)

| Time | Block | Deliverable |
|---|---|---|
| 15 min | Block 0 | Environment setup, API auth verified |
| 15 min | Block 1 | Historical data pulled and saved to CSV |
| 20 min | Block 2 | HMM trained, model saved, regimes inspectable |
| 30 min | Block 3 | Backtest running, PnL/drawdown results visible |
| 15 min | Block 4 | VIX filter + calendar blackout integrated |
| 15 min | Block 5 | Live order manager wired, paper trade confirmed |
| 10 min | Block 6 | main.py integration, end-to-end smoke test |
