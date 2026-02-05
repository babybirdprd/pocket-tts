---
name: implement-execplan
description: >-
  Execute a pending execution plan from the .agent folder.
  Use when user asks to implement an execplan, execute the plan,
  run the pending plan, or mentions execplan-pending.md.
---

# Implement ExecPlan

Execute the pending execution plan located at `.agent/execplan-pending.md` in the current repository.

## Workflow

1. Read `.agent/execplan-pending.md` at the repo root
2. Implement each step in order, marking progress as you go
3. When uncertain about the best approach, make a reasonable decision and proceed
4. Use web search when external information would help (API docs, library usage, etc.)
5. Continue until the entire plan is complete
6. Once the execplan is implemented completely, rename it (with a name that makes sense), and move it to .agent/done. If the 'done' folder does not exist, create it.

## Decision Making

If a step is ambiguous or has multiple valid approaches proceed with your best judgment rather than stopping to ask
