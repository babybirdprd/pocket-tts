---
name: refactor-something
description: Suggest one consolidation refactor that reduces surface area (fewer concepts, fewer files) after deep repo analysis. Use when the user asks to simplify architecture, consolidate modules, reduce conceptual surface area, merge similar subsystems, or produce an ExecPlan for a chosen simplification.
---

# Consolidation Refactor Plan

## Mindset: Think Like a Principal Software Architect

Approach this task as a principal software architect would. Your job is NOT to find every possible refactor—it's to identify the ONE refactor that delivers the highest value relative to its cost and risk.

**Be decisive. Do not ask clarifying questions.** You have access to the codebase—use it. Make reasonable assumptions based on evidence you find. State your assumptions explicitly, then move forward with your best recommendation. The user wants a recommendation, not a Q&A session.

**Prioritization criteria (in order of weight):**
1. **Blast radius vs. payoff** — A refactor touching 3 files that eliminates an entire abstraction layer beats a "cleaner" refactor touching 30 files.
2. **Cognitive load reduction** — Fewer concepts a new engineer must learn > fewer lines of code.
3. **Bug surface area** — Consolidating code paths that diverge for no good reason removes entire classes of bugs.
4. **Velocity unlock** — Refactors that unblock future work (e.g., removing a leaky abstraction blocking a feature) get priority.
5. **Risk tolerance** — Favor refactors with clear rollback paths and low data-loss potential.

Do NOT propose refactors that are cosmetic, speculative, or require rearchitecting stable, working code without a clear win.

---

## Workflow

### Step 1: Establish Scope and Constraints

Determine scope from context. If the user specified a directory, use that. If not, assume the current workspace root.

Infer these from the codebase and user context:
- **Target repo/directory** — Default to workspace root if unspecified.
- **Boundaries** — Assume all code is in scope unless obviously third-party or generated.
- **Risk tolerance** — Assume production-level caution unless the repo is clearly experimental (no CI, no tests, README says "prototype").
- **Known pain points** — If the user mentioned specific friction, prioritize those areas. Otherwise, let the evidence guide you.

**Do not ask for clarification.** State your assumptions and proceed.

### Step 2: Deep Repo Analysis (Evidence-Based)

Read the codebase systematically:
1. Start with `README`, `ARCHITECTURE.md`, or similar docs.
2. Identify the top 5-10 most imported/referenced modules (use grep or semantic search).
3. Trace call paths for 2-3 core user flows to understand actual coupling.
4. Look for these specific smells:
   - **Duplicate abstractions** — Two classes/modules doing nearly the same thing with slight variations.
   - **Thin wrappers** — A module that just passes through to another with no added value.
   - **Shotgun surgery** — A single concept scattered across 5+ files that change together.
   - **Dead code** — Modules with no callers or feature flags that will never flip.
   - **Leaky abstractions** — An interface that forces callers to know implementation details.

**Output a mental model** (you can share this with the user):
- Core concepts and their file locations
- Dependency graph highlights (what depends on what)
- Identified smells with file evidence

### Step 3: Rank Candidate Refactors

Generate 2-4 candidate consolidation refactors. For each candidate, score:

| Criteria | Weight | Score (1-5) |
|----------|--------|-------------|
| Payoff (concepts removed, code deleted) | 30% | |
| Blast radius (files touched, risk) | 25% | |
| Cognitive load reduction | 20% | |
| Velocity unlock potential | 15% | |
| Ease of validation/rollback | 10% | |

Calculate weighted scores. Present the top candidate with justification.

### Step 4: Propose the Single Best Refactor

For the chosen refactor, provide:
1. **Current state** — What exists today and why it's problematic (with file paths).
2. **Proposed change** — What will be merged, removed, or restructured.
3. **Files/dirs impacted** — Explicit list.
4. **Expected outcome** — New shape of the code (diagram or description).
5. **Acceptance criteria** — 2-4 measurable success conditions.
6. **Risks and mitigations** — What could go wrong and how you'll handle it.

### Step 5: Make the Call

If multiple candidates score similarly, pick the one with the smallest blast radius. Tie-breaker: choose the refactor that is easiest to validate (has clear before/after tests).

**Do not ask the user to choose.** Present your single best recommendation with clear reasoning. The user can redirect if they disagree.

### Step 6: Produce the ExecPlan

After presenting your recommendation:
1. Use the `execplan-authoring` skill to write the plan.
2. The problem statement = your consolidation summary + acceptance criteria.
3. Follow `PLANS.md` format exactly.
4. Write to `.agent/execplan-pending.md`
5. Include affected paths, test steps, and validation criteria.

---

## Anti-Patterns to Avoid

- **Asking clarifying questions** — You have the codebase. Read it. Make assumptions and state them.
- **Presenting multiple options without a recommendation** — Pick one. Justify it. Move on.
- **Boiling the ocean** — Proposing a refactor that touches every file in the repo.
- **Cosmetic cleanup** — Renaming or reformatting without structural improvement.
- **Speculative abstraction** — "We might need this pattern someday."
- **Ignoring working code** — Stable, battle-tested code doesn't need refactoring just because it's "old."
- **No evidence** — Every claim must reference specific files or code paths.
