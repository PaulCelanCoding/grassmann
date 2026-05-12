# Agent guidelines — Grassmann GS

Rules for agent-assisted work on this repository. Human-facing project
docs are in `README.md`.

## Communication

- Always reply in English.
- Be brief. Prefer direct, sober prose; do not pad answers with caveats
  or restate the question.
- Do not glaze the user. Match the work to the request without
  congratulating it.
- In plan mode (or whenever discussing approach), ask multiple
  non-trivial clarifying questions via `AskUserQuestion` covering
  semantic processing flow, architecture, data flow, main entry
  points, data structures, testing, and other affected areas of the
  codebase. End every question round with a final multi-select
  "Weitere Fragen?" gate so the user can request follow-ups or sign
  off with "Keine weiteren Fragen".

## Code

- No `Enhanced` / `Advanced` / `V2` variants. Upgrade the existing
  class or function directly.
- Never fake results. Failing fast is better than a green path that
  hides a regression. Every fallback must be explicit, documented in
  a comment, and emit a dedicated warning at runtime.
- Don't add comments that just restate code; comment the *why* and any
  non-obvious invariants. Cite `v7-doc Sec. X.Y` / `Prop X.Y` when
  implementing a specific operator from the math spec.

## Workflow

- Track multi-step work via the `TaskCreate` / `TaskUpdate` tools; mark
  the active task `in_progress` before starting it.
- After each removal commit, run the per-commit gate:

  ```bash
  grep -rn "<removed-symbol>" grassmann scripts tests docs results   # must be empty
  pytest -x tests/ -q                                                # must stay green
  python scripts/train_mono.py --help | grep -E "<removed-flag>"     # must be empty
  ```

- Do not push to the default branch without explicit user approval. The
  cleanup work should land on a topic branch first.
