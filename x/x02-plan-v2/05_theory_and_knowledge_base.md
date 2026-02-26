# Topstep-X Trading System — Theory & Knowledge Base

> **Purpose:** A living document explaining the theory behind this trading system.  
> Updated incrementally as the system is built, tested, and refined.  
> Each section includes: the concept, why it matters for this system, and practical implications for the code.
>
> **How to use this document:**  
> - Add a new section when you learn something important during implementation  
> - Add a `## Decision Log` entry when you make a design choice so future-you remembers why  
> - Add a `## Test Results` entry when a backtest or live session teaches you something  
> - Mark sections with `[VALIDATED]` when the concept has been confirmed by actual results

---

## Table of Contents

1. [Market Regimes — The Core Idea](#1-market-regimes)
2. [Hidden Markov Models](#2-hidden-markov-models)
3. [Viterbi vs Filtered Probabilities — A Critical Distinction](#3-viterbi-vs-filtered-probabilities)
4. [VWAP — The Institutional Benchmark](#4-vwap)
5. [The 5-State Regime Framework](#5-the-5-state-regime-framework)
6. [Symmetric Futures Trading](#6-symmetric-futures-trading)
7. [Risk/Reward Ratio as a System Parameter](#7-riskreward-ratio)
8. [The Three-Timeframe Stack](#8-three-timeframe-stack)
9. [Slippage and Commission — Why They Matter More Than You Think](#9-slippage-and-commission)
10. [Decision Log](#10-decision-log)
11. [Test Results & Observations](#11-test-results--observations)

---

## 1. Market Regimes

### What is a Market Regime?

Financial markets do not behave the same way at all times. They alternate between distinct behavioral states — called **regimes** — that have different statistical properties:

- In a **trending regime**, each bar tends to move in the same direction as the previous one. Volatility is moderate, volume is rising, and pullbacks are shallow before continuation.
- In a **volatile or crash regime**, moves are large and fast, volume spikes, and the normal relationships between price and volume break down.
- In a **neutral or ranging regime**, price oscillates around a central value with no clear direction. Volume is low and moves are mean-reverting.

These differences are not cosmetic. A strategy that works in a trending regime (buy pullbacks, hold for continuation) will lose money in a ranging regime, and vice versa. **The single most important insight of this system is that you should not use the same strategy in all market conditions.** You detect the regime first, then apply the appropriate strategy.

### Why Regimes Change

Regimes shift due to:
- **Macro events:** interest rate decisions, economic data releases, geopolitical shocks
- **Institutional behavior:** large players accumulating or distributing positions over days or weeks
- **Market microstructure:** changes in liquidity, options expiry, futures rollover dates
- **Sentiment shifts:** fear/greed cycles, momentum chasing, panic selling

The key property for modeling purposes is that **regimes tend to persist.** Once the market enters a trend, it usually stays trending for some time before switching. This persistence is what makes the HMM a good model — it is designed to capture exactly this "sticky" behavior.

### Practical Implication for This System

The HMM is trained on 5-min bars and classifies the current market into one of 5 states. The signal generator then applies a different trading strategy per state. This means:

- Bad market conditions (wrong regime for a strategy) are detected and avoided
- Good market conditions are identified and exploited with the appropriate edge
- The same code runs in all conditions — only the regime label changes the behavior

---

## 2. Hidden Markov Models

### What is a Markov Model?

A **Markov chain** is a system that moves between states where the probability of the next state depends only on the current state — not on the full history. This is called the **Markov property** or "memorylessness."

Example: if the market is currently in a bull regime, the probability of being in a bull regime next period depends only on being in a bull regime now — not on how it got there. The transition probability matrix captures this:

```
From \ To   | Strong Bull | Bull  | Neutral | Bear  | Crash
------------|-------------|-------|---------|-------|-------
Strong Bull |    0.70     | 0.20  |  0.08   | 0.02  | 0.00
Bull        |    0.15     | 0.65  |  0.15   | 0.04  | 0.01
Neutral     |    0.05     | 0.20  |  0.50   | 0.20  | 0.05
Bear        |    0.00     | 0.03  |  0.15   | 0.65  | 0.17
Crash       |    0.00     | 0.10  |  0.20   | 0.40  | 0.30
```

*(These are illustrative values — the actual probabilities are learned from data by the Baum-Welch algorithm)*

Notice that the crash regime has a high probability of staying in crash or transitioning to bear — it rarely jumps directly to bull. This makes intuitive sense and is a property the model learns automatically from historical data.

### What Makes It "Hidden"?

In a regular Markov chain, you can observe which state you are in. In a **Hidden Markov Model**, the states themselves are not directly observable. You only observe signals that are *caused by* the hidden states.

In our case:
- **Hidden states:** the true market regime (bull, bear, neutral, crash, etc.)
- **Observable signals:** returns, volatility, volume ratio (the features computed from price bars)

You cannot directly see "the market is in bull regime." You see price movements. The HMM learns the relationship between the hidden states and the observable signals during training, then uses that relationship to infer the most likely hidden state given new observations.

### How Training Works — Baum-Welch (Expectation-Maximization)

The model is trained using the **Baum-Welch algorithm**, a special case of Expectation-Maximization (EM):

1. **E-step:** Given the current model parameters, compute the probability of being in each state at each point in the training data
2. **M-step:** Update the model parameters (transition probabilities, emission means and variances) to maximize the likelihood of the observed data
3. Repeat until convergence

Crucially, **no labels are provided during training.** You do not tell the model "this period was bull, this was bear." The model discovers the states from the data alone. After training, you inspect what it learned (the mean return and volatility of each state) and manually assign human-readable labels. The state with the highest mean return becomes "strong bull," the one with the lowest becomes "crash," etc.

### GaussianHMM

This system uses `GaussianHMM` from the `hmmlearn` library. Each hidden state emits observations drawn from a multivariate Gaussian distribution. This means each regime has a characteristic mean vector `[mean_return, mean_volatility, mean_volume_ratio]` and a covariance structure.

The `"diag"` covariance type (recommended as starting point) assumes the features are independent within each state, which reduces the number of parameters and avoids overfitting with limited data.

### Key Limitation: Non-Stationarity

The HMM assumes that the statistical properties of each regime are constant over time (stationary). In reality, what "bull market" looks like in 2024 is different from 2020 or 2008. This means the model degrades over time and requires **periodic retraining** on recent data. This is addressed in the Future Scope document.

---

## 3. Viterbi vs Filtered Probabilities — A Critical Distinction

> **Why this section exists:** This is one of the most common sources of invisible performance inflation in backtests using HMM-based strategies. Getting this wrong makes the backtest look much better than the system will actually perform in live trading.

### The Football Analogy

Imagine you are watching a football match live. At minute 60, you need to decide whether the home team is "dominating" or "struggling" — and you can only use what you have seen up to minute 60.

Now imagine instead you are editing the match highlights **after the final whistle**. You know the score was 3-0. Looking back, you can now say "they were clearly dominating from minute 20" — because you have the full picture.

These are two completely different situations. **Live trading is the first one.**

### Viterbi — The Post-Game Editor

The **Viterbi algorithm** finds the single most probable sequence of hidden states that explains the entire observed sequence — from the first bar to the last.

To compute the label for bar 500, it uses information from bars 501, 502, 503... all the way to the end of the sequence. **It looks into the future.**

Two critical problems for live trading:

**Problem 1 — Future data does not exist at the time of decision.** When you are making a trade at bar 500, you do not have bars 501 onward. Viterbi cannot run until the session ends.

**Problem 2 — Labels change retroactively.** If you run Viterbi on bars 1–500 and get a regime label, then bar 501 arrives and you run it again on bars 1–501 — **the label for bar 500 can change.** The new information caused the algorithm to revise its assessment of what happened earlier.

In a backtest that uses Viterbi, this retroactive revision happens silently. Every bar is labeled with the benefit of hindsight. The backtest looks like the system had perfect regime knowledge, when in reality it had none.

**In a backtest, Viterbi is look-ahead bias.** The backtest uses future data to label the past, then tests a strategy on that labeled data. The resulting performance numbers are unrealistic and cannot be reproduced in live trading.

Viterbi **is** useful for one thing: post-hoc visualization and analysis. After a session ends, you can use Viterbi to get the cleanest, most interpretable picture of what regimes occurred. But never for real-time decisions.

### Filtered Probabilities — The Live Commentator

The **forward algorithm** (used by `model.predict_proba()` in hmmlearn) works differently. For each bar it computes:

*"Given everything I have seen up to and including this bar, what is the probability I am currently in each regime?"*

It only uses past and present data. When bar 501 arrives, it updates the estimate for bar 501 — it does **not** change the estimate for bar 500. Each decision is based only on information available at that moment.

The output for a given bar is a probability vector:
```python
[0.05, 0.72, 0.18, 0.04, 0.01]
# [strong_bull, bull, neutral, bear, crash]
```

You take `argmax` to get the current regime (bull, at 72% confidence). This is stable, causal, and consistent with what you would actually know at the time of the decision.

### The Practical Difference

| Property | Viterbi | Filtered Probabilities |
|---|---|---|
| Uses future data? | Yes | No |
| Labels stable after new bar? | No — can change retroactively | Yes — only current bar updates |
| Suitable for live trading? | No | Yes |
| Suitable for backtest? | Only if you understand the bias | Yes — this is the correct choice |
| Suitable for post-hoc analysis? | Yes — cleanest labels | Less clean but still useful |
| In hmmlearn | `model.predict(X)` | `model.predict_proba(X)` |

### The Rule for This System

**Always use `model.predict_proba(X)` for any decision that affects trading.** This includes both the live trading loop and the backtest. The goal of the backtest is to simulate what the system would have done in real time, so it must use the same information constraints as the live system.

Only use `model.predict(X)` (Viterbi) for visualization, debugging, and retrospective analysis — never as the basis for a trade signal.

### Why This Matters for Backtest Integrity

Using filtered probabilities in the backtest means:
- The backtest simulates genuine real-time uncertainty about the current regime
- If the system is profitable in the backtest, the edge is real — not an artifact of knowing the regime labels perfectly
- The live performance will be closer to the backtest performance

This is the correct foundation for trusting the backtest results enough to deploy real capital.

---

## 4. VWAP

### What is VWAP?

**Volume Weighted Average Price (VWAP)** is the average price at which a security has traded throughout the session, weighted by the volume at each price level:

```
VWAP = Σ(typical_price × volume) / Σ(volume)

where typical_price = (high + low + close) / 3
```

It resets at the start of each trading session (09:30 ET for US equity index futures) and accumulates throughout the day. The longer into the session, the more stable it becomes because more volume has been incorporated.

### Why VWAP Matters — The Institutional Reason

Large institutional investors (pension funds, asset managers, banks) use VWAP as a **benchmark for execution quality**. If they need to buy 10,000 contracts, they want to prove to their clients that they got a fair price. Buying at or below the day's VWAP means they got a better-than-average price.

This creates a structural, persistent gravitational effect: institutions are constantly executing orders relative to VWAP, which means price tends to gravitate toward VWAP throughout the session. This is not a retail indicator pattern — it is driven by the mechanics of how large capital is deployed.

### Two Strategy Modes Using VWAP

**Trend-following mode (bull and bear regimes):**  
VWAP acts as dynamic support (in bull regime) or resistance (in bear regime). When price pulls back to VWAP in a trending session, it represents a "fair value" reentry point for institutions still executing their directional orders. These pullbacks tend to continue in the direction of the trend.

*Entry rule: in bull regime, buy when 1-min bar touches VWAP from above and shows a bullish close (close > open).*

**Mean reversion mode (neutral regime):**  
When price stretches significantly above or below VWAP without volume support, it has moved away from institutional fair value. The gravitational pull back toward VWAP creates a mean reversion opportunity.

*Entry rule: in neutral regime, sell when price closes above VWAP + 1 standard deviation (overbought relative to VWAP); buy when price closes below VWAP - 1 standard deviation (oversold relative to VWAP).*

### VWAP Standard Deviation Bands

The standard deviation bands are computed from the intraday distribution of `typical_price`:

```python
# Running computation — updates with each new bar
typical_price = (bar.high + bar.low + bar.close) / 3
std_dev = rolling_std(typical_prices_today, window=20)
upper_band = vwap + std_dev
lower_band = vwap - std_dev
```

These bands automatically adapt to the day's volatility — on a high-volatility day the bands are wider, on a quiet day they are tighter. This is more robust than using a fixed point distance from VWAP.

### Important Practical Notes

- VWAP is unreliable in the first 15–30 minutes of the session — not enough volume has accumulated to be meaningful. Consider not trading mean reversion before 10:00 ET.
- In strongly trending sessions, mean reversion against VWAP frequently fails. This is exactly why the neutral regime gate exists — only trade mean reversion when the HMM says the market is ranging, not trending.
- Avoid VWAP-based entries around high-impact news events — the sudden volume spike distorts VWAP and makes the bands meaningless for a short period.

### Research Support

Zarattini & Aziz (SSRN, 2023) tested a simple VWAP-based long/short strategy on NQ and QQQ instruments from 2018 to 2023. The strategy — going long when price is above VWAP, short when below — produced a 671% return net of commissions on a $25,000 account with a maximum drawdown of only 9.4% and a Sharpe ratio of 2.1, compared to a 126% return and 37% drawdown for buy-and-hold over the same period. This demonstrates that VWAP is a genuinely powerful intraday benchmark, not just a popular indicator.

---

## 5. The 5-State Regime Framework

### Why 5 States

The choice of 5 states is a practical compromise between expressiveness and data requirements:

- **3 states** (bull/neutral/bear) is the most common choice in academic literature. It converges with relatively little data (30 days of 5-min bars) and is easy to interpret. However, it collapses the important distinction between "bull" and "strong bull" — which leads to the same strategy being applied in conditions that warrant different sizing and targets.

- **7 states** (adding "soft bull," "soft bear") would provide the finest granularity and the most tailored per-state strategies. However, it requires several months of training data to reliably separate adjacent states, and risks states collapsing into each other or flickering rapidly.

- **5 states** captures the most practically important distinctions — particularly separating the crash regime (where the short side has poor fills and the strategy is counter-trend long) from the bear regime (where normal short entries work well) — without requiring excessive data.

### The Crash State — A Special Case

The crash state is intentionally treated differently from the bear state. In a crash:

- **Do not short.** Spreads widen, fills are poor, and the move is often so extended that by the time a short entry fires, the risk of a snap reversal is high.
- **Wait for exhaustion.** A 1-min bar with a lower wick greater than 60% of the total bar range indicates buyers are absorbing the selling — this is the exhaustion signal.
- **Buy the bounce with a tight stop.** The counter-trend long trade after a crash exhaustion is one of the best risk/reward setups in intraday futures trading. The lower wick low is the invalidation level, providing a natural and tight stop.

This is why having the crash state separate from bear is worth the additional model complexity.

### State Labeling After Training

The HMM does not know which state is "bull" and which is "bear." It only learns that some states have certain statistical properties. After training, you inspect `model.means_` — the mean feature vector for each state — and assign labels based on mean return:

```
Highest mean return  →  strong_bull (State X)
Next highest         →  bull        (State X)
Near-zero return     →  neutral     (State X)
Negative return      →  bear        (State X)
Most negative return →  crash       (State X)
```

The actual state ID numbers (0, 1, 2, 3, 4) are arbitrary — they are assigned by the training algorithm and can vary between training runs. The labels are stored in `config.yaml` and map the arbitrary IDs to human-readable names. **This mapping must be verified after every retraining.**

---

## 6. Symmetric Futures Trading

### Why Futures Are Different From Stocks

When trading stocks, going short requires borrowing shares, paying a borrow fee, and managing margin requirements differently from long positions. Many retail brokers restrict or do not offer shorting at all.

Futures contracts are symmetric by design. To go short, you simply sell a contract. The mechanics, margin, and execution are identical to going long. There is no borrow fee and no restriction on shorting.

For MNQ (Micro E-mini NASDAQ futures):
- **Long 1 contract:** you profit $2 for every 1-point increase in NQ
- **Short 1 contract:** you profit $2 for every 1-point decrease in NQ

This symmetry means the strategy should be fully symmetric. Applying only long entries would leave half the available trading opportunities on the table — specifically, all the bear and crash regime setups.

### The Neutral Regime — Both Directions

In the neutral regime, the market oscillates around VWAP in both directions. Both long and short entries are valid, using the VWAP standard deviation bands as the trigger:

- Price stretches **above** VWAP + 1 std → short entry (fade the overbought move, target VWAP)
- Price stretches **below** VWAP - 1 std → long entry (fade the oversold move, target VWAP)

This makes the neutral regime the most active regime in terms of trade frequency, but also the most dangerous — a ranging market can break into a trend at any moment, turning a mean reversion trade into a losing trend-fighting position. This is managed by the tight 6-tick stop and 0.75x position size multiplier.

---

## 7. Risk/Reward Ratio

### The RRR Parameter

All trade targets in this system are computed from a single configurable parameter:

```yaml
trading:
  rrr: 1.8   # Risk/Reward Ratio
```

```python
target_ticks = round(stop_ticks * rrr)
```

A RRR of 1.8 means: for every 1 tick risked, you aim to make 1.8 ticks.

### Why 1.8

The choice of 1.8 is intentional:

- It is above 1.5, which is a common minimum threshold for viable intraday strategies after slippage and commission
- It is below 2.0, which means targets are achievable in typical intraday moves without requiring exceptional continuation
- It provides a viable win rate requirement: at RRR = 1.8, you need a win rate above approximately 36% to be profitable (`1 / (1 + RRR) = 1 / 2.8 ≈ 35.7%`)

### The Win Rate Requirement

The break-even win rate for any given RRR is:

```
break_even_win_rate = 1 / (1 + RRR)

RRR = 1.0  →  50.0% win rate needed
RRR = 1.5  →  40.0% win rate needed
RRR = 1.8  →  35.7% win rate needed
RRR = 2.0  →  33.3% win rate needed
RRR = 3.0  →  25.0% win rate needed
```

A win rate above 36% is a relatively low bar for a regime-filtered strategy with defined entry rules. However, after slippage and commission, the effective RRR is slightly lower than 1.8 — so the true break-even win rate is slightly above 36%. This is why the backtest acceptance criteria requires win rate > 35%.

### Why One RRR for All Regimes

Using a single RRR parameter makes the system easier to tune and understand. To change the reward profile of the entire system, you change one number. The stop distances per regime (which reflect the noise level of each state) do the work of differentiating between states — the RRR then uniformly scales the target from whatever stop distance is appropriate for that state.

---

## 8. Three-Timeframe Stack

### The Principle

Top-down analysis: assess market context at a higher timeframe first, then drill down to the trading timeframe, then to the entry timeframe. Only take trades where multiple timeframes agree.

### The Three Levels

| Timeframe | Role | Tool |
|---|---|---|
| 30-min (or 1-hour) | Macro trend bias — "what kind of day is this?" | EMA: price above → bull day, below → bear day |
| 5-min | Regime detection — "what is the market doing right now?" | HMM 5-state classifier |
| 1-min | Entry timing — "is this the right moment within the regime?" | Bar pattern: pullback to VWAP with confirming close |

### Why Not Just Use 5-Min

Without the 30-min macro filter, the system might go long in a bull 5-min regime during a strongly bearish day — catching a temporary counter-trend bounce in the middle of a major downtrend. These trades have lower probability of success and typically hit their stop before reaching their target.

The macro filter acts as a directional gate: in bull regimes, it confirms the 5-min signal; in bear regimes during a bear day, it confirms the short; in neutral sessions, it allows both directions.

### Why Not Just Use 1-Min

Using 1-min bars for regime detection produces too many false regime switches. A 10-point dip on a 1-min bar looks like a "crash" signal but is often just a normal intraday pullback. The HMM needs enough bars with consistent statistical properties to identify a genuine regime — 5-min bars provide this without being too slow to react to intraday changes.

---

## 9. Slippage and Commission

### Why This Section Exists

Slippage and commission are the most commonly underestimated costs in backtesting. A strategy that looks marginally profitable before costs will lose money live. Getting this right in the backtest is critical for having realistic expectations.

### MNQ Cost Structure

```
Tick size:              0.25 points
Point value:            $2.00
Tick value:             $0.50

Typical commission:     ~$0.50 per contract per side
Round-trip commission:  ~$1.00 per trade

Normal slippage:        1 tick = $0.50 per side
Round-trip slippage:    ~$1.00 per trade (normal regime)
                        ~$2.00–$3.00 per trade (crash regime — wide spreads)

Total round-trip cost (normal):  ~$2.00 per trade
Total round-trip cost (crash):   ~$4.00–$5.00 per trade
```

### How Slippage Is Modeled in the Backtest

```python
# Normal regimes: 1 tick slippage
fill_long  = next_bar.open + (1 * TICK_SIZE)   # pay a bit more than open
fill_short = next_bar.open - (1 * TICK_SIZE)   # receive a bit less than open

# Crash regime: 3 ticks slippage (wide spread, fast market)
fill_long  = next_bar.open + (3 * TICK_SIZE)
```

This models the fact that market orders fill at worse prices than the theoretical open, because the bid-ask spread and order flow work against you.

### The Practical Impact

On a 10-point stop (20 ticks), the $2.00 round-trip cost represents 4 ticks — 20% of the stop distance. This is not negligible. A strategy that generates 2-tick wins on average will lose money after costs even with a 60% win rate.

The RRR of 1.8 with 8-tick stops generates 14-tick targets = $7.00 gross per winning trade. After $2.00 costs the net is $5.00. On a losing trade, the $4.00 stop loss becomes $6.00 after costs. The effective post-cost RRR is approximately 0.83 — which is why the 36% win rate minimum matters so much.

---

## 10. Decision Log

*Add entries here when you make a design choice. Include the date, the options considered, and the reason for the choice.*

---

### [Initial Session] — Filtered Probabilities vs Viterbi

**Decision:** Use `model.predict_proba()` (forward algorithm / filtered probabilities) for all live and backtest regime inference. Never use `model.predict()` (Viterbi) for trading decisions.

**Options considered:**
- Viterbi (`model.predict(X)`) — finds globally optimal state sequence for the full history
- Filtered probabilities (`model.predict_proba(X)`) — estimates current state using only past and present data

**Reason:** Viterbi uses future data relative to any given bar, making it unsuitable for real-time use and creating look-ahead bias in backtests. Filtered probabilities are causal — they only use information available at the time of the decision — which ensures backtest results are honest and transferable to live performance. Viterbi is retained for post-hoc visualization only.

---

### [Initial Session] — 5 States vs 3 vs 7

**Decision:** Start with 5 states.

**Options considered:**
- 3 states: simpler, less data needed, academically most common
- 5 states: adds important separation between bull/strong_bull and bear/crash
- 7 states: maximum granularity, requires months of data

**Reason:** The crash/bear distinction alone justifies 5 over 3 — the trading logic in a crash (counter-trend long on exhaustion) is fundamentally different from a bear (short rallies). The soft_bull/soft_bear distinction of 7 states adds value but requires data not available tonight. Upgrade path documented in Future Scope.

---

### [Initial Session] — RRR = 1.8

**Decision:** Use RRR = 1.8 as the starting configurable parameter.

**Reason:** Provides a realistic target (not too ambitious to be unreachable, not too conservative to be unprofitable after costs). Sets break-even win rate at ~36%, which is achievable with a regime-filtered strategy. Single parameter controls the entire reward profile and is easy to tune during backtesting.

---

### [Initial Session] — VWAP Standard Deviation Bands for Neutral

**Decision:** Use VWAP ± 1 rolling standard deviation of typical_price for neutral regime entry zones.

**Options considered:**
- Fixed ATR-based offset from VWAP
- VWAP ± 1 standard deviation (chosen)
- VWAP ± 2 standard deviations (fewer trades, higher win rate per trade)

**Reason:** Standard deviation bands automatically adapt to intraday volatility — wider on volatile days, tighter on quiet days. This is more robust than a fixed ATR offset. Starting at ±1 std rather than ±2 generates more trades, which is important for the initial backtest having enough data to evaluate the strategy. Can tighten to ±2 if win rate on neutral trades is unsatisfactory.

---

## 11. Test Results & Observations

*Add entries here after each backtest or live session. Include the date, configuration, key metrics, and what you learned.*

---

*[No entries yet — add results after first backtest run]*

**Template for a new entry:**

```
### [Date] — [Description]

**Configuration:**
- HMM states: 5
- Training data: MNQ 5-min, [start date] to [end date]
- RRR: 1.8
- Macro filter: 30-min / 1-hour
- Neutral mean reversion: enabled / disabled

**Results:**
- Total PnL: $XXX
- Max drawdown: $XXX
- Win rate: XX%
- Trades total: XXX
- Trades/day: X.X

**PnL by regime:**
- Strong Bull: $XXX (XX trades)
- Bull: $XXX (XX trades)
- Neutral: $XXX (XX trades)
- Bear: $XXX (XX trades)
- Crash: $XXX (XX trades)

**Observations:**
- [What worked, what didn't, what to try next]
- [Did the regime labels make visual sense on the chart?]
- [Were there specific market conditions that caused unexpected losses?]

**Changes made based on this session:**
- [Parameter adjustments, logic changes, new decision log entries]
```
