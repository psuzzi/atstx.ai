# x01-plan — Initial System Planning

First exploration: full planning phase for the atstx automated trading system targeting the Topstep X platform via the Project X API.

## Scope

Design a Python-based automated trading system that uses a 5-state Hidden Markov Model (HMM) to classify market regimes on 5-min bars, applies deterministic signal rules per regime on 1-min bars, and enforces strict risk management aligned with Topstep X account limits (MNQ, 1 contract).

## Documents

| File | Description |
|---|---|
| `00_claude_code_prompt.md` | Session prompt for the implementation agent: constraints, decision rules, code standards |
| `01_overall_plan.md` | Goals, platform constraints (drawdown, daily loss), timeframe stack, 5-state HMM regime definitions, signal logic per regime, VIX/calendar filters, backtest requirements, target repo structure |
| `02_architecture.md` | Component design and interfaces: API client, feature engineering, HMM detector, signal generator, risk manager, order manager, backtester. Config schema. Data flow diagram. Decision points marked with warnings |
| `03_implementation_tasks.md` | 7 ordered blocks (0-6) from env setup through live integration. Each block has tasks, files touched, and a pass/fail acceptance test. Block 3 (backtest) is a hard gate |
| `04_future_scope.md` | Deferred features with integration points: periodic retraining, news sentiment, live calendar, RL agent, 7-state upgrade, multi-instrument, walk-forward optimization, monitoring, cloud deployment |

## Key Decisions

- Python 3.11+, hmmlearn GaussianHMM, 5 regime states
- 3 timeframes: 30min/1hr (macro bias), 5min (regime), 1min (entry)
- Deterministic rules per regime (RL deferred)
- Conservative risk: $750 daily loss limit, max 3 consecutive losses, 1 contract
- All future extensions use `# PLUG:` comments at defined integration points
