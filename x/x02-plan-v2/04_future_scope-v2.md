# Topstep-X Automated Trading System — Future Scope v2

> ⚠️ **THESE TOPICS ARE OUT OF SCOPE FOR THE INITIAL IMPLEMENTATION SESSION.**  
>  
> The initial session must be implemented such that every item on this list can be added later **without restructuring the codebase.** Each section identifies exactly which file and which `# PLUG:` comment to connect to. The implementation agent must leave all `# PLUG:` comments at the correct locations.  
> **Version:** v2 — adds regime confidence scaling, symmetric strategy extensions, anchored VWAP

---

## 1. HMM Periodic Retraining

**Why it is needed:** Academic literature explicitly confirms that HMM transition probabilities and emission distributions are non-stationary. A model trained once will degrade silently as volatility regimes, correlation structures, and macro conditions evolve. The model will still produce labels, but they will become increasingly wrong.

**Recommended approach — two triggers:**

**Scheduled retraining:** Every 4 weeks, retrain on a rolling 90-day window of 5-min bars. 90 days provides enough sequences for stable convergence on a 5-state model while discarding stale data from a different macro environment.

**Drift-triggered retraining:** Monitor the log-likelihood of incoming feature vectors under the current model. If the 5-day rolling mean of `model.score(recent_features)` drops more than 20% below the training baseline, trigger an emergency retrain. This catches abrupt regime shifts (e.g., a volatility explosion after a macro event) that the scheduled retrain would miss.

**Retraining pipeline:**
1. Fetch latest N days of data
2. Retrain on rolling window
3. Compare new model vs old model on held-out 10-day validation period
4. Swap model file only if new model scores better on validation
5. Reassign state labels based on new `model.means_` — automate this with the sorted mean-return heuristic

**Integration point:**
```python
# In regime/hmm_detector.py, predict_regime() method
# PLUG: retrain_trigger
# After computing log_likelihood = self.model.score(recent_features),
# push to a DriftMonitor object. If monitor.is_drifting():
#     call RetrainPipeline.run_async()
```

---

## 2. News Sentiment Integration

**Why it is needed:** The HMM detects price/volume patterns but is structurally blind to cause. A crash regime during a Fed announcement has different recovery characteristics than a random sell-off — duration, bounce magnitude, and re-entry timing all differ. Sentiment can inform both signal confidence and position sizing.

**Recommended sources:** Benzinga Pro API or Alpaca News API — both provide pre-scored sentiment (positive/negative/neutral) per headline. No NLP pipeline needed on the system side.

**Implementation approach:**
- Fetch headlines every 5 minutes during the session
- Aggregate headline sentiment scores into a single `[-1.0, +1.0]` session sentiment value
- Use as a position size modifier, not a direction signal (the HMM handles direction)

**Integration point:**
```python
# In strategy/signal_generator.py, generate_signal() method
# PLUG: sentiment_score
# sentiment_score: float in [-1.0, 1.0]
# Suggested usage:
#   if sentiment_score < -0.5 and regime in [STRONG_BULL, BULL]:
#       signal.size = max(1, signal.size - 1)  # reduce long size in negative news
#   if sentiment_score > 0.5 and regime in [CRASH, BEAR]:
#       signal.size = max(1, signal.size - 1)  # reduce short/bounce size in positive news
```

Also:
```python
# In strategy/filters.py, get_vix_multiplier() area
# PLUG: sentiment_score
# Add: news_multiplier = sentiment_client.get_session_multiplier()
# Final multiplier = vix_multiplier * news_multiplier
```

**Estimated implementation time:** Half a day

---

## 3. Live Economic Calendar API

**Why it is needed:** The hardcoded blackout times miss one-off events (emergency Fed meetings, surprise CPI revisions, geopolitical announcements) and require manual updates as schedules change.

**Recommended sources:**
- Trading Economics API (paid, ~$30/mo) — comprehensive, reliable
- Alpha Vantage Economic Calendar (free tier) — limited but functional
- Forex Factory API (scraping approach, free) — less reliable

**Integration point:**
```python
# In strategy/filters.py, is_blackout_period() method
# PLUG: live_calendar
# Replace:
#   HIGH_IMPACT_TIMES = config.filters.high_impact_times_et
# With:
#   events = calendar_client.get_todays_high_impact_events()
#   HIGH_IMPACT_TIMES = [e.time_et for e in events]
```

**Estimated implementation time:** 2–3 hours

---

## 4. Reinforcement Learning Agent

**Why it is needed:** The current signal logic is deterministic rules per regime. An RL agent can optimize entry timing, exit decisions, and position sizing within each regime using actual PnL as the training reward, adapting to patterns that are too subtle or too dynamic to encode as rules.

**Prerequisite:** The backtest loop must first be wrapped as a Gymnasium (OpenAI Gym) environment. The Block 3 backtest structure already makes this straightforward since the signal generation step is the only component that needs to be swappable.

**State space:**
```python
state = [
    regime_id,           # 0-4
    regime_confidence,   # max(proba), 0-1
    vwap_distance_std,   # (price - vwap) / std_dev
    bar_momentum,        # close - open, normalized
    volume_ratio,        # current vs rolling avg
    time_of_day_norm,    # 0=open, 1=close
    daily_pnl_pct,       # daily PnL as % of daily limit
]
```

**Action space:** `{flat, long_small, long_full, short_small, short_full}`

**Reward:** realized PnL per trade, penalized by proximity to daily loss limit

**Recommended framework:** Stable-Baselines3 with PPO

**Integration point:**
```python
# In strategy/signal_generator.py, generate_signal() return statement
# PLUG: rl_agent
# Replace:
#   return deterministic_signal(...)
# With:
#   if config.use_rl_agent:
#       return rl_agent.act(build_state_vector(regime, bar, vwap_ctx, risk_manager))
#   else:
#       return deterministic_signal(...)
```

**Estimated implementation time:** 1–2 days for a working prototype; weeks for a well-tuned, deployed agent

---

## 5. Regime Confidence-Based Position Sizing

**Why it is needed:** The filtered probability output `predict_proba` already provides a confidence measure (e.g., regime=bull at 92% confidence vs 54% confidence). Scaling position size by confidence is a theoretically sound improvement that requires minimal code.

**Implementation:**
```python
# In strategy/risk_manager.py, compute_final_size()
# PLUG: regime_confidence
# Add confidence parameter:
# def compute_final_size(base_size, regime, vix_multiplier, regime_confidence) -> int:
#     confidence_mult = 0.5 if regime_confidence < 0.6 else 1.0
#     return max(1, floor(base_size * regime_mult * vix_multiplier * confidence_mult))
```

**Estimated implementation time:** 1 hour

---

## 6. Anchored VWAP

**Why it is needed:** The standard session VWAP resets at 09:30 ET every day, which loses context from significant intraday events. Anchored VWAP starts the calculation from a user-defined event (e.g., a major news release, a key swing high/low, the open after a gap). For the crash regime specifically, anchoring VWAP to the start of the crash move provides a more meaningful reversion target.

**Integration point:**
```python
# In strategy/signal_generator.py, CRASH regime logic
# PLUG: anchored_vwap
# Replace:
#   vwap_ctx.vwap as target
# With:
#   crash_anchored_vwap = vwap_accumulator.get_anchored(crash_start_time)
#   use crash_anchored_vwap as the reversion target for the bounce trade
```

**Estimated implementation time:** 3–4 hours

---

## 7. 7-State HMM Upgrade

**Why it is needed:** The 5-state model merges "Soft Bull" with "Bull" and "Soft Bear" with "Bear." If backtesting reveals that these merged states have meaningfully different optimal parameters, splitting them will improve edge.

**7-state spectrum:**

| State | Label |
|---|---|
| 0 | Strong Bull |
| 1 | Bull |
| 2 | Soft Bull |
| 3 | Neutral |
| 4 | Soft Bear |
| 5 | Bear |
| 6 | Crash |

**Data requirement:** At least 3–6 months of 5-min bars for stable 7-state convergence. Do not attempt with less — states will collapse or flicker.

**Integration point:**
```yaml
# In config.yaml
# PLUG: 7_state_upgrade
# Change n_states: 5 → n_states: 7
# Re-run train_hmm.py
# Add state_labels entries for states 5 and 6
# Add entries to stops, regime_size_multipliers, and REGIME_ACTION in signal_generator.py
```

---

## 8. Walk-Forward Optimization for Backtesting

**Why it is needed:** The current backtest trains the HMM on the full dataset and tests on the same data (in-sample). This overstates performance. Walk-forward testing trains on a window, tests on the next unseen period, rolls forward, and averages results — giving a realistic out-of-sample performance estimate.

**Implementation approach:**
- 70/30 train/test split as baseline
- Rolling walk-forward: 60-day train window, 10-day test window, step 10 days at a time
- Report mean and standard deviation of performance metrics across windows (variance matters — a strategy with good mean but high variance is not reliable)

**Integration point:**
```python
# In backtest.py, main function
# PLUG: walk_forward
# Wrap current backtest in:
# def walk_forward_backtest(data, train_window=60, test_window=10):
#     for train_df, test_df in rolling_windows(data, train_window, test_window):
#         model = train_hmm(train_df)
#         results = run_backtest(test_df, model)
#         yield results
```

**Estimated implementation time:** 4–6 hours

---

## 9. Multi-Instrument Support

**Why it is needed:** Once the system is proven on MNQ, it can be expanded to MES, M2K, MGC, or crypto perpetual futures. Each instrument needs its own HMM model, tick size config, and risk parameters. Risk must still be tracked at the account level across all instruments.

**Integration point:**
```python
# In main.py, session startup
# PLUG: multi_instrument
# Replace single-instrument loop with:
# instruments = config.trading.instruments  # list of symbol configs
# for inst in instruments:
#     threads.start(run_instrument_loop, inst, shared_risk_manager)
# Note: RiskManager must be thread-safe and track total account PnL across all instruments
```

---

## 10. Cloud Deployment / Multi-Machine Operation

**Why it is needed:** Running from a single local machine creates a single point of failure. For reliability and the ability to run from home, office, or VPS simultaneously (with only one machine trading at a time), shared state management is required.

**Key challenges:**
- Only one machine should place orders at a time (prevent duplicate trades)
- Trade log must be accessible from all machines
- HMM model must be synchronized after retraining

**Recommended approach:**
- Lightweight Redis instance for shared state (daily PnL, open positions, current regime)
- Redis distributed lock ensures only one machine is active trader at a time
- Model sync via S3 or equivalent: upload after retraining, all machines download at startup

**Integration point:**
```python
# In main.py, initialization
# PLUG: distributed_state
# Replace:
#   risk_manager = RiskManager(config)
# With:
#   risk_manager = RedisRiskManager(config, redis_url=config.redis_url)
# And add distributed lock:
#   with redis_lock("trading_active", timeout=60):
#       run_main_loop()
```

---

## 11. Performance Monitoring and Alerting

**Why it is needed:** Without monitoring, silent failures go undetected (model drift, API disconnections, strategy breakdown). Alerts allow timely intervention before losses accumulate.

**Minimum viable monitoring:**
- Daily summary (email or Telegram): PnL, trades, win rate, regime distribution
- Alert conditions: daily loss > 50% of limit, win rate below 30% over rolling 10 days, no trades in 2 hours during market hours, API connection error

**Integration point:**
```python
# In main.py, end-of-session cleanup and error handlers
# PLUG: monitoring
# monitor.send_daily_summary(trades_today, daily_pnl, regime_distribution)
# monitor.send_alert(message) on error conditions
```

---

## Priority and Sequencing

| Feature | Priority | Estimated Time | When to Implement |
|---|---|---|---|
| HMM Periodic Retraining | High | 4–8 hours | Within 2 weeks of first live trade |
| Regime Confidence Sizing | High | 1 hour | After first backtest validation |
| Performance Monitoring | High | 2–3 hours | Before going full-size live |
| Walk-Forward Optimization | High | 4–6 hours | Before scaling position size |
| Live Economic Calendar | Medium | 2–3 hours | Second session |
| News Sentiment | Medium | 4–6 hours | After retraining is working |
| Anchored VWAP | Medium | 3–4 hours | When crash regime is tuned |
| RL Agent | Low | 2–3 days | Only after strategy is profitable |
| 7-State HMM | Low | 2 hours + data | Only with 3+ months of data |
| Multi-Instrument | Low | 4–8 hours | After single-instrument is proven |
| Cloud / Multi-Machine | Low | 1–2 days | Only if running from multiple locations |
