---

name: add-codex-cli-support

description: After deep repo analysis, inventory every integration/lock-in point to the Claude Agent SDK (Claude Code foundation) and produce a single best implementation plan to add OpenAI Codex CLI support—either via a provider adapter or by refactoring to a provider-neutral core. Conclude by authoring an ExecPlan.

---



\# Multi-Provider Agent Support Plan (Claude Agent SDK + OpenAI Codex CLI)



\## Mindset: Think Like a Principal Software Architect + Integration Engineer



Your job is NOT to brainstorm every possible integration. It’s to:

1\) \*\*Find all connection points\*\* to the Claude Agent SDK / Claude-Code-like runtime in this repo, and  

2\) \*\*Make one decisive recommendation\*\* for how to add \*\*OpenAI Codex CLI\*\* support with the best payoff-to-risk ratio.



\*\*Be decisive. Do not ask clarifying questions.\*\* You have access to the full codebase—use it. Make reasonable assumptions based on evidence you find. State assumptions explicitly, then proceed.



\### Prioritization criteria (in order)

1\. \*\*Compatibility preservation\*\* — existing Claude-based flows must keep working with minimal behavioral drift.

2\. \*\*Lock-in removal (or isolation)\*\* — provider-specific concepts must be pushed behind a narrow interface.

3\. \*\*Small blast radius\*\* — prefer changes that touch fewer files and have clear rollback paths.

4\. \*\*Feature parity for core UX\*\* — streaming, tool execution, sessions, permissions must remain coherent across providers.

5\. \*\*Testability\*\* — the path should enable automated validation (even if initially via a thin integration test harness).



Do NOT propose cosmetic refactors. Do NOT propose a rewrite. This is an integration + decoupling plan.



---



\## Workflow



\### Step 1: Establish Repo Scope and Constraints (Evidence-Based)



Determine scope from codebase reality:

\- \*\*Target repo/directory\*\*: default to workspace root.

\- \*\*Likely hotspots to inspect first\*\* (verify in repo):

&nbsp; - `packages/shared/src/agent/` (agent orchestration)

&nbsp; - `packages/shared/src/sessions/` (session persistence + formats)

&nbsp; - `packages/shared/src/config/` + credentials/auth (keys, tokens, provider selection)

&nbsp; - `apps/electron/src/main/` (runtime hosting + filesystem/tool execution)

&nbsp; - UI streaming/event plumbing (renderer and shared event types)

\- \*\*Runtime constraints\*\*: verify whether the agent runtime is expected to run in Electron main, a local server, or shared library context. Note any Node/Bun/Electron constraints that affect Codex integration.



State assumptions explicitly. Do not ask the user.



---



\### Step 2: Deep Repo Analysis — Claude SDK Connection-Point Inventory



Systematically map every Claude/Anthropic-specific coupling. You must produce an \*\*Integration Map\*\* with file-path evidence.



\#### 2.1 Locate all direct imports and hard dependencies

Search for (examples; adapt to what you find):

\- `@anthropic-ai/claude-agent-sdk` imports

\- any “claude” runtime/client wrappers, factories, or providers

\- provider-specific env vars / config keys / credential storage

\- “tool” / “command” execution glue that is shaped to Claude’s expectations

\- streaming event schemas that match Claude agent events



\#### 2.2 Trace 2–3 core user flows end-to-end (call paths)

Pick real flows used by the app (based on code):

1\) “Start new session → send user message → stream responses”

2\) “Tool invocation / command execution → permission gating → result streaming”

3\) “Session persistence → resume session → continue conversation”



For each flow:

\- Identify the top-level entrypoint(s)

\- Identify where provider-specific types/events leak into shared logic or UI

\- Identify where state/thread/session identity is handled



\#### 2.3 Produce the “Provider Integration Map” (required output)

Create a structured inventory table like:



| Category | Connection Point | File(s) | Symbol(s) | Coupling Type (hard/soft) | Notes |

|---|---|---|---|---|---|

| Client creation | Anthropic SDK client init | … | … | hard | … |

| Auth/keys | API key lookup / OAuth | … | … | hard | … |

| Message model | prompt/message formatting | … | … | hard | … |

| Streaming | event types / reducers | … | … | hard | … |

| Tools | tool schema + execution glue | … | … | hard | … |

| Sessions | persistence format + resume | … | … | soft/hard | … |

| UI | provider event rendering assumptions | … | … | soft | … |

| Errors | retry/backoff, rate limits | … | … | soft | … |



Also include:

\- A concise dependency graph highlight: “what depends on what”

\- A short list of “highest-friction lock-ins” (with file evidence)



---



\### Step 3: Codex CLI / Codex SDK Capability Mapping



You must determine the cleanest integration surface for OpenAI Codex support:

\- \*\*Option 1: Codex SDK (`@openai/codex-sdk`)\*\* running server-side to control local Codex agents (threads, `run()`, resume by thread ID, etc.).

\- \*\*Option 2: Codex CLI subprocess adapter\*\* (spawn `codex` and communicate via supported modes/IO).

\- \*\*Option 3: Codex App Server\*\* (if present/required in this repo’s constraints; confirm via docs and code needs).



For each, list:

\- Required runtime environment (Node/Electron main process vs elsewhere)

\- State model (thread/session IDs) and how it maps to this repo’s session concept

\- Streaming model and how it maps to the app’s streaming/event pipeline

\- Tool/command execution semantics and permission gating implications



Output a concise “Codex Fit Assessment” that directly references the integration map from Step 2.



---



\### Step 4: Identify the Minimal Abstraction Seam



Based on the Integration Map, propose the \*\*narrowest\*\* provider interface that can support both providers without leaking provider concepts.



You must produce:

1\) A proposed internal interface (pseudocode is fine) such as:

&nbsp;  - `ProviderClient.startSession(...)`

&nbsp;  - `ProviderClient.sendMessage(...)` / `streamEvents(...)`

&nbsp;  - `ProviderClient.listTools()` / `invokeTool(...)` (if tools are provider-driven)

&nbsp;  - `ProviderClient.resumeSession(...)`

2\) A provider-neutral internal event schema (or an adapter that normalizes provider events)

3\) A plan for where provider selection lives (config, workspace settings, runtime flags)



Your interface must be shaped by real call sites you found (not a speculative abstraction).



---



\### Step 5: Rank Integration Strategies (Pick One Winner)



Generate 2–3 candidate strategies, then score them:



| Criteria | Weight | Score (1-5) |

|---|---:|---:|

| Preserves current Claude behavior | 30% | |

| Minimizes blast radius | 25% | |

| Removes/isolates lock-in | 20% | |

| Enables Codex support cleanly | 15% | |

| Ease of validation/rollback | 10% | |



Candidates should look like:

\- \*\*A. Provider Adapter Layer\*\*: Introduce `AgentProvider` interface; keep existing internal event model; implement `ClaudeProvider` using current code; add `CodexProvider` using Codex SDK/CLI.

\- \*\*B. Provider-Neutral Core Refactor\*\*: Extract a provider-agnostic “Agent Runtime” and migrate call sites; providers become plugins.

\- \*\*C. Minimal CLI Bridge\*\*: Implement Codex only via subprocess first, postponing deeper decoupling (only if it wins on blast radius and risk).



\*\*Pick the single best strategy\*\*. If close, choose the one with smallest blast radius and clearest validation path.



---



\### Step 6: Propose the Implementation (Single Best Plan)



For the chosen strategy, provide:



1\. \*\*Current state (evidence-based)\*\*  

&nbsp;  - Where Claude SDK is coupled (top 5–10 items from your map)

2\. \*\*Proposed change\*\*  

&nbsp;  - What new modules/interfaces are introduced

&nbsp;  - What existing modules are modified

&nbsp;  - What gets deleted or simplified (if any)

3\. \*\*Files/dirs impacted\*\*  

&nbsp;  - Explicit list (group by package)

4\. \*\*New architecture shape\*\*  

&nbsp;  - Short diagram or bullet “before → after”

5\. \*\*Acceptance criteria (measurable)\*\*  

&nbsp;  Must include:

&nbsp;  - A config switch to choose provider (Claude vs Codex)

&nbsp;  - Claude provider continues working unchanged for core flows

&nbsp;  - Codex provider can run at least one end-to-end flow (message → plan → tool usage if applicable → streamed output)

&nbsp;  - Session/thread resume works (Claude session + Codex thread mapping)

6\. \*\*Risks and mitigations\*\*  

&nbsp;  Include:

&nbsp;  - Runtime compatibility (Electron/Bun/Node constraints)

&nbsp;  - Streaming/event normalization mismatches

&nbsp;  - Permission/tool execution safety differences

&nbsp;  - Rollback plan (feature flag, provider toggle, adapter isolation)



---



\### Step 7: Validation Plan



Provide:

\- Unit tests for provider interface + event normalization

\- An integration test harness that runs a short “hello agent” flow for each provider

\- Manual QA checklist for the UI flow(s) you traced in Step 2



---



\### Step 8: Produce the ExecPlan



After presenting your recommendation:

1\. Use the `execplan-authoring` skill to write the plan.

2\. The problem statement = your integration summary + acceptance criteria.

3\. Follow `PLANS.md` format exactly.

4\. Write to `.agent/execplan-pending.md`

5\. Include affected paths, test steps, and validation criteria.



---



\## Anti-Patterns to Avoid



\- \*\*Asking clarifying questions\*\* — you have the repo; analyze it.

\- \*\*Implementing Codex by scattering `if (provider === ...)` across the codebase\*\* — provider logic must be centralized.

\- \*\*Designing speculative abstractions\*\* — every interface method must be justified by real call sites.

\- \*\*Breaking session formats without migration\*\* — changes must be backward-compatible or include a migration strategy.

\- \*\*Skipping streaming parity\*\* — the UI must remain responsive and coherent across providers.

\- \*\*No evidence\*\* — every claim must reference specific files or call paths.



