---
name: bead-creator
description: >-
  Analyze a code repository (README.md, ARCHITECTURE.md, and lightweight codebase search)
  and create 3–5 new Beads issues via br (beads_rust). Use when the user asks to
  create/generate beads, seed a backlog, open tasks/issues, or wants 3–5 task ideas focused
  on reliability, new features, performance, security, refactors, docs, or marketing for a
  specific repo.
---

# Bead Creator

Create a small, high-quality set of new Beads (“br” issues) for a repo based on quick repo understanding + a user-specified focus.

## Quick start

1. **Confirm repo target**
   - If the user gave a path, use that.
   - Otherwise assume the current working directory is inside the repo.
   - Determine the repo root by walking up until you find a `.git/` directory.

2. **Confirm focus** (unless the user already specified)
   Ask a single question:
   - “What should these beads optimize for: reliability, new features, performance, security, marketing, docs, refactor/tech-debt, or something else?”

3. **Ensure Beads is initialized**
   - If `{repoRoot}/.beads/` does not exist: run `br init` from `{repoRoot}`.

4. **Fast repo understanding**
   - Read (if present): `README.md`, `ARCHITECTURE.md`.
   - Build a lightweight *mental model* of the repo using quick searches (don’t deep-read everything).
   - Do **not** create new documentation files or “repo map” artifacts; keep any notes in the chat response only.

5. **Create 3–5 beads**
   - Keep each bead small and action-oriented.
   - Use `br create ... --json` and capture the IDs.
   - Add labels (optional but recommended) to encode focus/area.

6. **Report back**
   - Post a short list: bead id → title → why it matters → “next step” / verification hint.

## Repo scan checklist (lightweight)

Run these from `{repoRoot}` (use `--no-color` when piping).

### Core docs
- `README.md` (how to run/test/deploy; stated goals)
- `ARCHITECTURE.md` (components/boundaries; operational constraints)

### Structure / hotspots
- Top-level layout: `ls -la`
- Language/build clues:
  - `package.json`, `pnpm-lock.yaml`, `yarn.lock`, `tsconfig.json`
  - `pyproject.toml`, `requirements.txt`
  - `Cargo.toml`, `go.mod`
  - `Dockerfile`, `docker-compose.yml`
  - `.github/workflows/*`, `Makefile`, `Justfile`

### Quick search targets
- Work debt markers:
  - `rg -n "TODO|FIXME|HACK|XXX" .`
- Stability markers:
  - `rg -n "retry|backoff|timeout|circuit|rate limit" .`
- Observability markers:
  - `rg -n "log(ger)?\\.|metrics|tracing|sentry|otel|prometheus" .`
- Tests:
  - `find . -maxdepth 4 -type d \( -name test -o -name tests -o -name __tests__ \)`

## Bead writing rules

### 1) Prefer small, parallelizable work
Each bead should be roughly **30–120 minutes** and have minimal shared surface area.

### 2) Include acceptance criteria
A bead without acceptance criteria becomes a conversation, not a task.

### 3) Avoid speculative refactors
Unless the focus explicitly asks for refactors, prefer beads that have a clear “why now”.

## Bead template (use this in `--description`)

Use a compact structure:

- **Goal:**
- **Context:** (files/dirs; what you observed)
- **Approach:** (high level)
- **Acceptance:** (bullet list)
- **Verify:** (command(s) or observable behavior)

Example description:

Goal: Improve reliability of <component> under transient failures.
Context: Found <file/path> with <behavior> and no timeout/retry.
Approach: Add timeout + bounded retries with jitter; surface errors clearly.
Acceptance:
- Requests time out after <N>s
- Retries capped at <M>
- Error is logged with request id
Verify:
- Run <command>
- Simulate failure by <method>

## Creating beads (commands)

Create 3–5 beads. Prefer `task` unless it’s clearly a `bug` or `feature`.

```bash
br create "<title>" --type task --priority 2 --description "<template>" --json
```

Optional labels (use consistent tokens):
```bash
br label add <id> focus:reliability area:observability
```

## Focus guidance

If the user specifies a focus, bias bead selection accordingly:

- **Reliability:** timeouts/retries, idempotency, backoff, failure modes, CI stability, error handling
- **Performance:** obvious hotspots, N+1 patterns, caching, build performance, large bundle size
- **Security:** secrets handling, authz checks, dependency hygiene, SSRF/path traversal hazards
- **New features:** “thin vertical slice” improvements that fit existing architecture
- **Docs:** missing runbooks, setup steps, troubleshooting, architecture diagrams, onboarding
- **Marketing:** landing page copy debt, SEO pages, product docs that convert (only if repo is marketing-related)
- **Refactor/tech debt:** simplify modules, remove duplication, improve boundaries, add types/tests

## Guardrails

- **Do not change repo code** unless the user asked for implementation; this skill’s job is to create beads.
- **Do not create new documentation files** (no README/ARCHITECTURE updates, no new “repo-map” docs). Any repo understanding stays in the chat response.
- **Only touch `.beads/`** (via `br`).
- **Do not run git commands** unless the user explicitly asks.
- If repo context is insufficient, ask 1–2 targeted questions rather than guessing.
