---
name: update-architecture-docs
description: >-
  Update ARCHITECTURE.md after implementing an execution plan.
  Use when user asks to update architecture docs, sync architecture,
  refresh architecture after implementation, or mentions updating docs
  post-execplan.
---

# Update Architecture Docs

Synchronize ARCHITECTURE.md with changes made during execplan implementation.

## Workflow

1. Read the completed execplan `.agent/execplan-pending.md`
2. Locate ARCHITECTURE.md at the repo root
3. Analyze what was implemented in the execplan and identify architectural impacts:
   - New modules or bounded contexts
   - Changed data flows or boundaries
   - New integrations or dependencies
   - Updated directory structure
   - New cross-cutting concerns
4. Update only the affected sections of ARCHITECTURE.md
5. If diagrams exist, update Mermaid diagrams to reflect new components/flows
6. After all of your updates, stage any unstaged changes, then commit. 

## Update Principles

- Preserve existing structure and formatting
- Add to existing sections rather than rewriting
- Update Key Design Decisions with new decisions made during implementation
- Move resolved items from Open Questions to appropriate sections
- Keep the same level of detail as the existing doc

## If No ARCHITECTURE.md Exists

Create one using the architecture-docs-creator skill template, populated with the current state post-implementation.
