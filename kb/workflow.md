# Workflow Conventions

## Incremental Development

The atstx project follows an incremental development approach:

1. **One issue at a time** — each issue is scoped to a single deliverable
2. **Issue-first** — always create a GitHub issue before starting work
3. **Small commits** — each commit references its issue number
4. **Working state** — main branch should always be in a working state after each issue is closed
5. **No big-bang changes** — large features are broken into a sequence of small issues

## Repository Layout

- **atstx** (github.com/psuzzi/atstx) — main project repo, public-facing code
- **atstx.ai** (github.com/psuzzi/atstx.ai) — AI companion repo, cloned as `.ai/` inside atstx (gitignored)

## .ai Structure

- `x/` — explorations (x01, x02, ...) for open-ended research
- `s/` — sessions (s01, s02, ...) tied to specific issues
- `kb/` — knowledge base: architecture, setup, prompts, conventions

## Issue Tracking

- atstx issues track project implementation work
- atstx.ai issues track AI-related content (explorations, sessions, kb updates)
