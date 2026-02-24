# Safety Conventions

## Rule: Confirm Before Non-Reversible Actions

Never execute non-reversible or hard-to-reverse actions without explicit user confirmation. Always show what will happen and wait for approval.

### Actions that require confirmation

- **Git**: commit, push, force-push, branch deletion, tag creation
- **Filesystem**: deleting files or folders, overwriting uncommitted changes
- **GitHub**: creating/closing issues, creating/merging PRs, posting comments
- **Infrastructure**: database changes, deployment, config changes on shared systems
- **Processes**: killing processes, restarting services

### Actions that do NOT require confirmation

- Reading files, searching, exploring the codebase
- Creating or editing local files (reviewable before commit)
- Running tests, linters, build commands
- Installing dependencies locally (`uv sync`)

### How to confirm

Show the user what you intend to do, then wait. For example:

- "Ready to commit with message `...` and push to origin/main. Proceed?"
- "This will delete `data/old_model.pkl`. Confirm?"
- "Will close issue #3 on atstx.ai. OK?"
