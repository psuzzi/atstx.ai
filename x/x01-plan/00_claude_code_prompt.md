# Claude Code — Topstep-X Trading System: Implementation Session

You are starting the implementation of an automated trading system for the Topstep X prop trading platform, accessed via the Project X API. The full context is in the 4 documents provided alongside this prompt. Read all 4 before writing a single line of code.

## Your Documents

- `01_overall_plan.md` — goals, constraints, strategy logic, decisions already made
- `02_architecture.md` — component design, interfaces, config schema, integration points
- `03_implementation_tasks.md` — your work plan: 6 blocks, each with tasks and a test to pass
- `04_future_scope.md` — out of scope tonight, but every `# PLUG:` comment you leave must match the integration points described there

## Session Constraints

- **Single session, limited time.** Prioritize working code over perfect code. Simple and correct beats clever and broken.
- **Block-by-block only.** Implement one block at a time in the order defined in `03_implementation_tasks.md`. Do not start Block N+1 until the test for Block N passes.
- **Block 3 is the hard gate.** If the backtest does not meet its acceptance criteria, stop and fix the HMM regime labeling. Do not proceed to live execution code with a broken strategy.
- **Each block must be independently testable** — no block should require another block's runtime state to run its test.
- **This work may continue across multiple sessions.** Write clean interfaces and leave clear comments so a future session can pick up from any block boundary without re-reading all the code.

## Decision Points

Several ⚠️ Decision Points are marked in `02_architecture.md`. When you hit one:
1. State the decision clearly in a comment at the top of the relevant file
2. Choose the simpler option and proceed — do not ask for clarification unless the decision blocks all forward progress
3. Log your choice so it can be reviewed later

## Code Standards

- Python 3.11+, typed with dataclasses and type hints
- Every `# PLUG:` comment from `04_future_scope.md` must appear at the correct integration point
- No hardcoded credentials — use environment variables via `.env`
- All config in `config.yaml` — no magic numbers in code
- Each module must be importable and testable in isolation

## Start Here

1. Read all 4 documents fully
2. Create the repo structure from `01_overall_plan.md` Section 8
3. Begin Block 0 from `03_implementation_tasks.md`
4. Run the Block 0 test before proceeding

Good luck.
