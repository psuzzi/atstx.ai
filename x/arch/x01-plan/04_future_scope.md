# Topstep-X Automated Trading System — Future Scope

> ⚠️ **THESE TOPICS ARE OUT OF SCOPE FOR THE INITIAL IMPLEMENTATION SESSION.**  
>  
> The initial session must be implemented such that **every item on this list can be added later without restructuring the codebase.** Each section identifies exactly where in the existing code the new component connects. The implementation agent must leave `# PLUG:` comments at each integration point.

---

## 1. HMM Periodic Retraining

**Why this is needed:** Market regime characteristics change over time. A model trained on 2024 data will gradually misclassify 2026 regimes as volatility levels, correlations, and macro conditions shift. Without retraining, the model will degrade silently — it will still produce regime labels but they will become increasingly wrong.

**Recommended approach:**

Two retraining triggers should be implemented:

- **Scheduled retraining:** Retrain the model every 4 weeks using a rolling window of the most recent 90 days of 5-min bars. The 90-day window ensures enough data for stable HMM convergence while discarding stale data.
- **Drift-triggered retraining:** Monitor the log-likelihood of incoming bar features under the current model. If the rolling 5-day average log-likelihood drops below a threshold (e.g., 20% below the training baseline), trigger an emergency retrain.

**Integration point:**
```python
# In regime/hmm_detector.py, predict_regime() method
# PLUG: retrain_trigger
# After predicting, compute log_likelihood = model.score(recent_features)
# Feed to drift monitor. If drift detected, call retrain pipeline.
```

**Script to implement:**
- `regime/retrain_pipeline.py` — fetches latest N days of data, retrains, validates new model against old on held-out data, swaps only if new model is better

---

## 2. News Sentiment Integration

**Why this is needed:** The HMM sees price and volume patterns but is blind to cause. A crash regime during a Fed announcement behaves differently from a random selloff — recovery time, continuation probability, and magnitude differ significantly. News sentiment can inform position sizing and regime confidence.

**Recommended data source:** Benzinga Pro API or Alpaca News API — both provide pre-scored sentiment (positive/negative/neutral) per headline, so no NLP pipeline is needed on the system side.

**Integration point:**
```python
# In strategy/signal_generator.py, generate_signal() method
# PLUG: sentiment_score
# sentiment_score: float in range [-1.0, 1.0]
# Usage: if sentiment_score < -0.5 and regime in [0, 1]: reduce size or skip long
# Usage: if sentiment_score > 0.5 and regime in [3, 4]: reduce size or skip short
```

**Estimated implementation time:** Half a day (API integration + calibration)

---

## 3. Live Economic Calendar API

**Why this is needed:** The initial implementation uses hardcoded blackout times. These miss one-off events (emergency Fed meetings, geopolitical announcements, earnings surprises) and require manual updates when schedules change.

**Recommended approach:** Integrate with a free or low-cost economic calendar API:
- Forex Factory API (scrape, free)
- Trading Economics API (paid, $30/mo)
- Alpha Vantage Economic Calendar (free tier)

**Integration point:**
```python
# In strategy/filters.py, is_blackout_period() method
# PLUG: live_calendar
# Replace hardcoded HIGH_IMPACT_TIMES list with:
# events = calendar_client.get_todays_events(impact="high")
# Return True if any event is within blackout window
```

---

## 4. Reinforcement Learning Agent

**Why this is needed:** The current signal logic is deterministic rules per regime. An RL agent can learn to optimize entry timing, position sizing, and exit decisions within each regime using actual reward (PnL) as the training signal.

**Recommended approach:**
- Use the existing backtest infrastructure as the RL training environment
- State: `[regime, vwap_distance, bar_momentum, time_of_day, daily_pnl_pct]`
- Action: `{flat, long_small, long_full, short_small, short_full}`
- Reward: realized PnL of the trade, penalized by drawdown
- Framework: Stable-Baselines3 PPO or A2C (well-documented, Python-native)

**Key requirement:** The backtest loop must be wrapped as a Gym environment before RL training begins. The backtest in Block 3 is already structured to support this — the signal generation step is the only component that needs to be swappable.

**Integration point:**
```python
# In strategy/signal_generator.py, generate_signal() return statement
# PLUG: rl_agent
# Replace:
#   return deterministic_signal(regime, bar, vwap, macro_bias)
# With:
#   return rl_agent.act(state_vector)
# where state_vector is computed from the same inputs
```

**Estimated implementation time:** 1–2 days for a working prototype; weeks for a well-tuned agent

---

## 5. 7-State HMM Upgrade

**Why this is needed:** The initial 5-state model merges "Soft Bull" with "Bull" and "Soft Bear" with "Bear". If backtesting reveals that these merged states have meaningfully different optimal signal logic (different stop distances, different targets, different hold times), splitting them will improve performance.

**The 7-state spectrum (when implemented):**

| State | Label |
|---|---|
| 0 | Strong Bull |
| 1 | Bull |
| 2 | Soft Bull |
| 3 | Neutral |
| 4 | Soft Bear |
| 5 | Bear |
| 6 | Crash |

**Data requirement:** At least 3–6 months of 5-min bars for stable 7-state convergence. Do not attempt with less data — states will collapse or flicker.

**Integration point:**
```yaml
# In config.yaml
# PLUG: 7_state_upgrade
# Change:
#   n_states: 5
# To:
#   n_states: 7
# Then re-run train_hmm.py and reassign state labels
# Update REGIME_ACTION mapping in signal_generator.py for the 2 new states
```

---

## 6. Multi-Instrument Support

**Why this is needed:** Once the system is proven on MNQ, it can be expanded to other instruments (MES, M2K, MGC, or crypto futures). Each instrument has different tick sizes, margin requirements, and regime characteristics.

**Integration point:**
```python
# In main.py
# PLUG: multi_instrument
# Replace single symbol loop with:
# for symbol in config.instruments:
#     run_symbol_loop(symbol)
# Each symbol gets its own HMM model, risk manager instance, and bar feed
```

**Key consideration:** Topstep X account limits apply across all instruments. The risk manager must be extended to track total account PnL across symbols, not just per-symbol PnL.

---

## 7. Walk-Forward Optimization for Backtesting

**Why this is needed:** The initial backtest trains the HMM on the full historical dataset and tests on the same data — this is in-sample and will overstate performance. Walk-forward optimization trains on a window, tests on the next period, rolls forward, and averages results. This gives a realistic estimate of live performance.

**Implementation approach:**
- 70/30 split: train on first 70% of data, test on last 30%
- Rolling walk-forward: 60-day train window, 10-day test window, step forward 10 days at a time

**Integration point:**
```python
# In backtest.py
# PLUG: walk_forward
# Current: train on all data, test on all data
# Future: wrap backtest in walk_forward_loop(train_window=60, test_window=10)
```

---

## 8. Performance Monitoring and Regime Drift Detection

**Why this is needed:** In live trading, you need to know if the system is performing as expected or if something has broken (market structure changed, API issues, model drift). A monitoring dashboard or alert system prevents silent failures.

**Recommended approach:**
- Daily summary email/Telegram message: PnL, trades, win rate, regime distribution
- Alert if: daily loss > 50% of limit, win rate drops below 30% over rolling 10 days, regime distribution shifts significantly vs training baseline

**Integration point:**
```python
# In main.py, end of session cleanup
# PLUG: monitoring
# Call monitoring.send_daily_summary(trades_today, daily_pnl, regime_counts)
```

---

## 9. Cloud Deployment / Multi-Machine Operation

**Why this is needed:** The initial system runs on a single local machine. For reliability and the ability to run on multiple machines (home + office + VPS), the system needs shared state management.

**Key challenges:**
- Only one machine should be placing orders at a time (prevent duplicate trades)
- Trade log must be accessible from all machines
- Model files must be synchronized after retraining

**Recommended approach:**
- Simple: use a lightweight Redis instance (or even a shared S3/file storage) to hold current state (open positions, daily PnL counter)
- Locking: use Redis distributed lock to ensure only one machine trades at a time
- Model sync: upload `hmm_model.pkl` to S3 after retraining; all machines download at startup

**Integration point:**
```python
# In main.py, initialization
# PLUG: distributed_state
# Replace in-memory risk_manager state with:
# risk_manager = RedisBackedRiskManager(redis_url=config.redis_url)
```

---

## Summary Table

| Feature | Complexity | Estimated Time | Priority |
|---|---|---|---|
| HMM Periodic Retraining | Medium | 4–8 hours | High — implement within first 2 weeks of live trading |
| News Sentiment Integration | Medium | 4–6 hours | Medium |
| Live Economic Calendar | Low | 2–3 hours | Medium |
| Reinforcement Learning Agent | High | 2–3 days | Low — only after strategy is proven profitable |
| 7-State HMM | Low (code) / High (data) | 2 hours + data collection | Low |
| Multi-Instrument | Medium | 4–8 hours | Low |
| Walk-Forward Optimization | Medium | 4 hours | High — implement before scaling position size |
| Performance Monitoring | Low | 2–3 hours | High — implement before going live full-size |
| Cloud / Multi-Machine | High | 1–2 days | Low — only needed if trading from multiple locations |
