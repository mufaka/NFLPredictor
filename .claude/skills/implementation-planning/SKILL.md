---
name: implementation-planning
description: Creates or refines a phased implementation plan from a formal specification and, when helpful, the original idea document. Use this when requirements are defined and the work should be organized into logical phases with checkbox-tracked work items that guide execution without duplicating the specification.
compatibility: Intended for repositories that use an idea -> specification -> implementation plan workflow.
---

# Purpose

Use this skill when a feature has a specification and is ready to be organized for execution.

This skill turns a specification into a practical implementation plan that breaks work into logical phases, preserves execution order, and provides checkbox-based tracking. The implementation plan should be detailed enough to guide development and validation, but it should not restate the entire specification or drift into low-value task fragmentation.

## Use this skill when

- The user wants a phased plan derived from a specification.
- The repository tracks execution through implementation plan documents.
- The work spans multiple vertical slices or capability areas.
- The team benefits from explicit sequencing, dependencies, and completion tracking.

## Do not use this skill when

- The idea is still exploratory and no stable requirements exist.
- A formal specification still needs to be written.
- The task is a very small, self-contained code change.
- The user wants immediate code changes rather than planning.

## Inputs

Prefer these sources in order:

1. The formal specification document.
2. The related idea document for additional context and rationale.
3. Existing implementation plans in the repository for formatting and phase conventions.

Use the idea document to improve understanding, but treat the specification as the primary source of truth when the two differ.

## Core behavior

- Treat the implementation plan as an execution artifact.
- Derive phases from the specification’s capability areas, dependencies, and architectural boundaries.
- Organize work into vertical slices where possible.
- Use checkboxes for trackable work items.
- Keep each phase internally coherent and realistically executable.
- Include tests as part of every relevant phase.
- Avoid duplicating the specification line-by-line.
- Avoid collapsing everything into a single giant checklist.

## Workflow

1. Read the specification first.
2. Read the idea document if it helps clarify intent, sequencing, or hidden dependencies.
3. Review existing implementation plans in the repository to match style and granularity.
4. Identify the major dependency layers, such as:
   - foundation and infrastructure
   - data model and migrations
   - services
   - controllers, endpoints, tools, prompts, or resources
   - UI
   - tests
   - polish and hardening
5. Group work into logical phases that each deliver meaningful progress.
6. Within each phase, create subsection work items that are concrete and checkable.
7. Include cross-cutting validation, integration testing, documentation, and hardening where appropriate.
8. If the repository uses phase numbering, continue the established numbering scheme when appropriate.

## Writing rules

- Use a short introduction that explains what the plan is based on.
- Include a progress summary table when it fits the repository style.
- Include a current state section when it helps orient future work.
- Include guiding principles when the repository’s plan format expects them.
- Use phase headings with short descriptions.
- Under each phase, use checkbox lists for concrete work items.
- Prefer grouping by outcome rather than by file.
- Keep work items implementation-facing and actionable.
- Make tests explicit. No feature phase should feel complete without validation.

## TeamWare-style patterns to emulate

When matching the TeamWare implementation-plan style, prefer this overall shape:

1. Title
2. Intro paragraph describing the plan’s relationship to the specification
3. Progress Summary table
4. Current State
5. Guiding Principles
6. Phase sections, each with:
   - short phase summary
   - numbered subsections
   - checkbox work items
7. Branch Strategy if relevant to the repository
8. References

## Phase design guidance

Good phases are:
- dependency-aware
- end-to-end where possible
- large enough to matter
- small enough to review and complete

Useful patterns include:
- foundation first
- read before write
- shared infrastructure before feature surfaces
- primary feature delivery before polish and hardening
- documentation and cross-cutting tests near the end, unless needed earlier

## Checkbox guidance

Use checkboxes for work that can be marked complete during execution.

Good checkbox items:
- create a model, service, controller, tool, or prompt
- add a migration
- wire DI registration
- add tests for a capability
- verify a cross-cutting behavior

Avoid checkboxes that are too vague:
- “implement feature”
- “do backend”
- “finish UI”

Avoid checkboxes that are too granular unless the repository clearly prefers that level.

## Boundary with specification

The implementation plan should not redefine requirements.

Instead, it should answer:
- in what order should this be built
- how should the work be grouped
- what concrete implementation items need to be completed
- where do tests and hardening belong

If a requirement is unclear enough that planning becomes guesswork, pause and fix the specification rather than inventing plan details.

## Output expectations

A strong implementation plan should:

- clearly sequence the work
- reflect the specification faithfully
- make progress trackable with checkboxes
- support issue creation or execution handoff
- include testing and hardening explicitly

See `references/implementation-plan-template.md` for a reusable starter format.
