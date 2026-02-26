# Claude Code — Topstep-X Trading System: Implementation Session v2

You are implementing an automated futures trading system for the Topstep X prop trading platform, accessed via the Project X API. Read all 4 documents fully before writing a single line of code.

## Your Documents

- `01_overall_plan-v2.md` — goals, constraints, theoretical foundation, all decisions made
- `02_architecture-v2.md` — full component design, interfaces, config schema, integration points
- `03_implementation_tasks-v2.md` — your work plan: 6 blocks, each with tasks and a concrete test
- `04_future_scope-v2.md` — out of scope tonight; every `# PLUG:` comment you leave must match the integration points described there

## Session Constraints

- **Single session, time-limited.** Working and testable beats complete and broken. Simple and correct beats clever.
- **Block-by-block, in order.** Do not start Block N+1 until the test for Block N passes. Each block is designed to be independently runnable — do not rely on runtime state from previous blocks.
- **Block 3 is the hard gate.** If the backtest does not meet all acceptance criteria, stop and fix the HMM regime labeling before any live execution code is written.
- **This work may span multiple sessions.** Write clean interfaces, leave clear comments at every decision point, and ensure any block can be picked up in a fresh context window.

## Key Technical Requirements

These are non-negotiable implementation details — do not simplify them away:

1. **Filtered probabilities for live inference.** Use `model.predict_proba()` (forward algorithm) everywhere, not `model.predict()` (Viterbi). Viterbi optimizes the full path and is unsuitable for real-time decisions. See Architecture v2 Section 2.3.

2. **RRR is a config parameter.** All trade targets are computed as `round(stop_ticks * config.trading.rrr)`. Never hardcode a target distance. Default RRR = 1.8.

3. **Fully symmetric strategy.** Long in bull regimes, short in bear regimes, both directions in neutral. The crash regime is counter-trend long only (bounce, not short). See Overall Plan v2 Section 6.

4. **Neutral regime uses VWAP standard deviation bands.** Entry at `VWAP ± std_dev` with a reversal bar confirmation. Target is VWAP itself (capped by `stop * rrr`). See Architecture v2 Section 2.5.

5. **Every `# PLUG:` comment must be placed.** At a minimum: `sentiment_score`, `retrain_trigger`, `rl_agent`, `live_calendar`, `regime_confidence`, `bracket_order`, `multi_instrument`, `distributed_state`, `walk_forward`. See Architecture v2 Section 4 for exact locations.

## Decision Points

Several ⚠️ Decision Points are marked in `02_architecture-v2.md`. When you encounter one: state the decision clearly in a comment, choose the simpler/safer option, and proceed. Do not block on a decision unless it prevents all forward progress.

## Start Here

1. Read all 4 documents completely
2. Create the repository structure from Overall Plan v2 Section 10
3. Begin Block 0 from `03_implementation_tasks-v2.md`
4. Pass the Block 0 test before proceeding to Block 1

Good luck.
