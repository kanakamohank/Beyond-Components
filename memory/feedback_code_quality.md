---
name: always validate code before running
description: user requires tests, sanity checks, and self-review to pass before executing any new or modified code
type: feedback
---

Never run broken code. Before executing any new or modified script, always:
1. Complete a self-review pass for correctness, data leaks, and silent failures.
2. Add/run sanity checks (input shapes, expected ranges, fixture outputs).
3. Verify unit/integration tests pass where they exist.
4. Confirm the output semantics match what the script claims to measure.

**Why:** User has been burned by assistants shipping code that "runs" but silently measures the wrong thing (e.g., ActAdd direction captured at the wrong residual position), producing paper-grade numbers from broken pipelines. Discovered during WinoGender eval scaffolding (2026-04-21).

**How to apply:** When writing any new experiment script, halt before the first run. Complete the review. Report findings. Only execute after the user confirms or obvious-correctness is established. For modifications to existing code, re-verify the invariants the script already relied on.