# Topstep-X Automated Trading System — Overall Plan v2

> **Session:** Initial implementation session  
> **Goal:** Have a working, backtested, live-executable trading system by end of session  
> **Target platform:** Topstep X (prop firm), accessed via Project X APIs  
> **Language:** Python 3.11+  
> **Version:** v2 — updated with symmetric long/short strategy, RRR parameter, neutral mean-reversion logic, and theoretical grounding

---

## 1. Objective

Build an automated trading system that:

1. Connects to the **Project X API** to receive market data and place orders on the **Topstep X** platform
2. Uses a **Hidden Markov Model (HMM)** to classify the current market into one of **5 regime states**
3. Exploits the **full symmetry of MNQ futures**: buying in bull regimes, selling in bear regimes, and trading mean reversion in neutral
4. Applies **deterministic signal rules per regime** driven by a single configurable **Risk/Reward Ratio (RRR = 1.8)**
5. Enforces **strict risk management** aligned with Topstep X account limits
6. Can be run locally on one or more machines from a **git repository**
7. Is designed so that future components (sentiment, retraining, RL) can be **plugged in cleanly**

---

## 2. Theoretical Foundation

### 2.1 Hidden Markov Models for Regime Detection

The HMM is well-established in academic literature for market regime identification. Key properties that make it suitable here:

- **Markets exhibit regime-switching behavior**: returns, volatility, and serial correlation shift abruptly due to macroeconomic, regulatory, or behavioral changes. Standard time-series models that assume stationarity break down across these transitions. The HMM explicitly models this switching.
- **GaussianHMM** (from `hmmlearn`) fits a multivariate Gaussian emission distribution per state. This captures the different return/volatility characteristics of each regime.
- **The Baum-Welch algorithm** (Expectation-Maximization) trains the model by learning both transition probabilities and emission parameters from unlabeled data — no manual labeling required during training.
- **Research confirms** (LSEG, MDPI 2020, QuantStart) that HMM outperforms k-means and GMM for out-of-sample regime identification on futures data.
- **Critical live trading consideration**: use **filtered probabilities** (forward algorithm, `predict_proba`) for regime inference at runtime, not the Viterbi path (`predict`). Viterbi optimizes the entire historical path and can retroactively change past labels when new data arrives — it is suitable for post-hoc analysis but not for live decisions. Filtered probabilities use only data up to the current bar and are stable.
- **Periodic retraining is required**: the literature explicitly confirms that transition probabilities and emission distributions are non-stationary. A model trained once will degrade as market structure evolves.

### 2.2 VWAP as the Core Execution Benchmark

Volume Weighted Average Price (VWAP) is the theoretically and practically preferred intraday reference for equity index futures:

- **Institutional benchmark**: large institutions execute against VWAP to minimize market impact, creating a gravitational pull on price throughout the session. This is a structural, persistent effect, not a retail indicator.
- **Two distinct strategy modes** depending on regime:
  - **Trend-following** (bull/bear regimes): VWAP acts as dynamic support/resistance. Price pulling back to VWAP in a trend is a high-probability continuation entry.
  - **Mean reversion** (neutral regime): price stretched beyond VWAP ± N standard deviations tends to revert. Standard deviation bands anchored to VWAP define entry zones.
- **Research support**: Zarattini & Aziz (SSRN 2023) demonstrate VWAP-based strategies on NQ instruments yield significantly superior risk-adjusted returns vs buy-and-hold, including through bear markets.
- VWAP **resets at session open** (09:30 ET) and accumulates volume throughout the session. It is most reliable after the first 30 minutes when sufficient volume has accumulated.

### 2.3 Futures Symmetry

MNQ futures can be bought (long) or sold short with equal ease, margin, and execution. The strategy fully exploits this:
- Bull regimes → long bias, buy pullbacks to VWAP
- Bear regimes → short bias, sell rallies to VWAP
- Neutral regime → both directions, fade VWAP standard deviation extremes
- Crash regime → counter-trend long only on exhaustion signal (asymmetric — see Section 4)

---

## 3. Platform Constraints (Topstep X)

These are hard constraints that the system must respect at all times:

| Constraint | Value | Notes |
|---|---|---|
| Total max drawdown | $2,000 | Account-level, enforced by Topstep |
| Daily loss limit | $750 | Self-imposed conservative limit |
| Max consecutive losses before stop | 3 | Coded into risk manager |
| Starting instrument | MNQ (Micro NQ) | 1 point = $2, 1 tick = $0.50 |
| Starting position size | 1 contract | Scale up only after proven performance |

The system must check daily PnL at every trade entry and refuse to open new positions if the daily limit is approaching.

---

## 4. Market Analysis Approach

### 4.1 Timeframe Stack

| Timeframe | Role | How Used |
|---|---|---|
| **30-min** (or 1-hour — see Decision Point) | Macro trend bias | EMA filter — bull day / bear day / neutral. Sets directional bias. |
| **5-min** | Regime detection | HMM runs on 5-min bars. Primary signal timeframe. |
| **1-min** | Entry timing | Once regime and bias agree, 1-min bar pattern triggers the actual entry. |

> **Decision Point A:** 30-min vs 1-hour for the macro filter. 30-min gives 4–6 bars in the early session (vs 2–3 for 1-hour), providing better intraday context. 1-hour is simpler and less ambiguous. Both are valid. **To be confirmed with the implementation agent.**

### 4.2 Regime States (5-State HMM)

| State ID | Label | Market Character | Trading Mode |
|---|---|---|---|
| 0 | Strong Bull | High positive returns, confirming volume, shallow pullbacks | Long — full rules |
| 1 | Bull | Positive drift, moderate volume, normal retracements | Long — standard rules |
| 2 | Neutral | Near-zero drift, low volume, price oscillating around VWAP | Mean reversion — both directions |
| 3 | Bear | Negative drift, volume rising on down moves | Short — standard rules |
| 4 | Crash / Strong Bear | Fast waterfall, volume spike, wide spread | Counter-trend long on exhaustion only |

### 4.3 HMM Features (5-min bars)

| Feature | Formula | Notes |
|---|---|---|
| `return_pct` | `(close - prev_close) / prev_close` | Primary regime signal |
| `volatility` | Rolling 10-bar std of `return_pct` | Distinguishes trending from choppy |
| `volume_ratio` | `volume / rolling_20_bar_mean(volume)` | Volume confirmation |

> **Decision Point B:** Whether to add bar range `(high - low)` or VWAP distance as a 4th feature. Start with 3. Expand only if the trained model shows poor state separation when visualized.

---

## 5. Risk/Reward Framework

### 5.1 The RRR Parameter

All trades in all regimes use a single configurable **Risk/Reward Ratio**:

```yaml
# config.yaml
trading:
  rrr: 1.8   # Risk/Reward Ratio — target = stop_distance * rrr
```

This means:
- Stop distance is set per regime (based on typical noise level)
- Target distance is always `stop_ticks * rrr`
- If stop = 10 ticks → target = 18 ticks
- If stop = 6 ticks → target = 10.8 ticks (rounds to 11)

This single parameter controls the entire system's reward profile and can be tuned during backtesting without touching any signal logic.

### 5.2 Stop Distances Per Regime

| State | Label | Stop Distance | Target (× 1.8 RRR) | Rationale |
|---|---|---|---|---|
| 0 | Strong Bull | 10 ticks | 18 ticks | Wider stop: stronger trend, deeper pullbacks acceptable |
| 1 | Bull | 8 ticks | 14 ticks (rounds to 15) | Standard trending stop |
| 2 | Neutral | 6 ticks | 11 ticks | Tight stop: if it doesn't revert quickly, invalidated |
| 3 | Bear | 8 ticks | 14 ticks | Mirror of State 1 |
| 4 | Crash | 6 ticks | 11 ticks | Very tight: crash bounces fail fast; small size |

> **Note on Neutral stop:** Mean reversion trades have a lower theoretical win rate than trend-following. The tight stop and lower position size multiplier (0.75x) compensate. If neutral mean reversion performs poorly in backtest, the stop distance is the first parameter to tune.

---

## 6. Signal Logic Per Regime (Symmetric)

### States 0 & 1 — Bull / Strong Bull (Long Only)
- **Entry trigger:** 1-min bar where `low ≤ VWAP` and `close > open` (bullish bar touching or below VWAP)
- **Condition:** macro bias is neutral or bullish
- **Direction:** long
- **Stop:** below entry by stop_ticks
- **Target:** above entry by `stop_ticks * rrr`

### State 2 — Neutral (Both Directions — Mean Reversion)
- **Range definition:** VWAP ± 1 standard deviation (computed as rolling intraday std of `typical_price`)
- **Long entry:** 1-min close below `VWAP - 1_std` with a bullish reversal bar (close > open), targeting VWAP
- **Short entry:** 1-min close above `VWAP + 1_std` with a bearish reversal bar (close < open), targeting VWAP
- **Minimum range width check:** if `2 * std_dev < min_neutral_range_points` (configurable, default 8 points), stay flat — range too tight for viable R/R after slippage
- **Position size multiplier:** 0.75x (mean reversion is inherently lower confidence than trend)
- **Exit:** target = VWAP (not VWAP ± other side), stop = entry ± stop_ticks

> **Decision Point C:** Standard deviation period for VWAP bands in neutral regime. Suggested: rolling 20-bar std of `(high + low + close) / 3`. This adapts to intraday volatility. Alternatively, use a fixed ATR-based band. **Confirm with implementation agent.**

> **Decision Point D:** Whether to trade neutral mean reversion at all in the first session. It is the most complex regime to implement correctly and the most dangerous if misconfigured. Acceptable to stub it as `flat` initially and enable after backtest validation.

### States 3 & 4 — Bear / Crash (Short or Counter-Trend Long)

**State 3 (Bear) — Mirror of State 1:**
- **Entry trigger:** 1-min bar where `high ≥ VWAP` and `close < open` (bearish bar touching or above VWAP)
- **Condition:** macro bias is neutral or bearish
- **Direction:** short

**State 4 (Crash):**
- **Do NOT short the crash itself** — fills are poor, spread widens, slippage is severe
- **Wait for exhaustion:** 1-min bar where lower wick > 60% of total bar range (strong buying rejecting lower prices)
- **Direction:** long (counter-trend bounce)
- **Size multiplier:** 0.5x — crash bounces fail frequently
- **Slippage model:** 2–3 ticks (not 1)

---

## 7. Sentiment / Macro Filters (Implemented Tonight)

### VIX Filter
- Fetch VIX once at session open via `yfinance`
- `VIX > 30` → position size multiplier = 0.5
- `VIX ≤ 30` → multiplier = 1.0

### Economic Calendar Blackout
- Hardcoded high-impact US times (Eastern): 08:30, 10:00, 14:00, 14:30
- No new trades within 5 minutes before or 10 minutes after
- Fully configurable in `config.yaml`

---

## 8. Backtesting Requirements

Backtesting is mandatory before live trading. The backtest must:

- Replay historical bars through the exact same HMM + signal code as live
- Use **filtered probabilities** for regime inference (matching live behavior)
- Model slippage: 1 tick normal, 2–3 ticks in Crash regime
- Apply full commission model: ~$0.50/contract/side for MNQ
- Track: total PnL, max drawdown, win rate, avg win/loss, trades/day, PnL per regime
- Flag any day where the $750 daily limit was breached
- Be runnable entirely offline (CSV + pre-trained model)

---

## 9. All Decisions Made in This Session

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Best ecosystem for hmmlearn, fast iteration |
| HMM library | `hmmlearn` GaussianHMM | Academically validated, simple API |
| HMM inference method | Filtered probabilities (`predict_proba`) | Stable for live use; Viterbi is retrospective only |
| Number of states | 5 | Balance between granularity and data requirements |
| Regime timeframe | 5-min | Standard for intraday futures; not noisy like 1-min |
| Entry timeframe | 1-min | Fine-grained timing within 5-min regime signal |
| Macro filter | 30-min or 1-hour EMA (TBD — Decision Point A) | Prevents trading against dominant trend |
| Trading symmetry | Full — long in bull, short in bear, both in neutral | MNQ futures allow symmetric long/short |
| Neutral regime strategy | VWAP mean reversion with std dev bands | Theoretically sound; academically supported |
| Risk/Reward Ratio | 1.8 (configurable parameter `rrr` in config.yaml) | Single parameter controlling all targets |
| Slippage model | 1 tick normal, 2–3 ticks crash | Conservative, realistic for MNQ |
| Starting instrument | MNQ | $2/point, low capital risk per trade |
| Starting size | 1 contract | Preserves capital while system is proven |
| Sentiment tonight | VIX filter only | News NLP too complex for tonight |
| Calendar tonight | Hardcoded times | Live API deferred to future scope |
| ML for entry | Deterministic rules | RL deferred; rules are understandable and debuggable |

---

## 10. Repository Structure

```
topstep-trader/
├── README.md
├── config.yaml                  # all params (secrets gitignored via .env)
├── requirements.txt
├── main.py                      # live trading entry point
├── backtest.py                  # backtesting entry point
├── projectx_client.py           # Project X API wrapper
├── regime/
│   ├── features.py              # feature computation from bars
│   ├── hmm_detector.py          # HMM wrapper (train, filtered predict, save/load)
│   └── train_hmm.py             # offline training script
├── strategy/
│   ├── signal_generator.py      # regime + bias → Signal (symmetric)
│   ├── risk_manager.py          # sizing, daily loss, consecutive loss tracking
│   └── filters.py               # VIX, calendar blackout, macro EMA bias
├── execution/
│   └── order_manager.py         # place/track/cancel orders (+ PaperOrderManager)
├── models/
│   └── hmm_model.pkl            # serialized model (gitignored)
└── data/
    └── historical/              # CSVs for training/backtesting
```
