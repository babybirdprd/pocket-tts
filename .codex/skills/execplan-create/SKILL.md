---
name: execplan-create
description: >-
  Create an ExecPlan (execution plan) from a PRD, RFC, voice note, or
  brainstorming blurb, following the repo's PLANS.md. Use when the user asks
  for an exec plan, execution plan, or ExecPlan, or wants a PRD/RFC turned into
  a step-by-step plan.
---

# ExecPlan Authoring

## Required inputs

- The user must provide a PRD, RFC, or a detailed problem statement. If missing or unclear, ask for clarification before writing the plan.

## Source of truth

- Read `{baseDir}/PLANS.md` in full before drafting.
- If `{baseDir}/PLANS.md` is missing, copy this skill's `PLANS.md` to `{baseDir}/.agent/PLANS.md`, then read that copy as the source of truth.
- Follow PLANS.md exactly. If any instruction conflicts with this skill, PLANS.md wins.

## Output location

- Write the ExecPlan to `.agent/execplan-pending.md` in the target repo.
- If `.agent/` does not exist, create it before writing the file.

## Format rules

- Because `.agent/execplan-pending.md` contains only the ExecPlan, do not wrap it in outer triple backticks.

## Authoring workflow

1. Read the user’s input and identify the concrete outcomes and acceptance criteria.
2. Inspect the repo to understand relevant files and structure; note paths explicitly in the plan.
3. Draft the ExecPlan using the skeleton and rules from PLANS.md.
4. Ensure required sections exist and are self-contained, novice-friendly, and behavior-focused.
5. Save to `.agent/execplan-pending.md`.
