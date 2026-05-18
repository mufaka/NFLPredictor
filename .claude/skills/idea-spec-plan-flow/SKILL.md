---
name: idea-spec-plan-flow
description: Orchestrates the full software feature workflow from collaborative idea exploration to formal specification to phased implementation plan. Use this when a user wants to take a feature from an early concept through clarified requirements and into an execution-ready plan using consistent repository conventions.
compatibility: Intended for repositories that use an idea -> specification -> implementation plan workflow.
---

# Purpose

Use this skill when the user wants support for the full end-to-end workflow of shaping a feature before implementation begins.

This is a meta skill. It does not replace the individual skills for idea collaboration, specification creation, or implementation planning. Instead, it coordinates the transitions between those stages so the work stays structured, traceable, and appropriately detailed at each step.

## Use this skill when

- The user has a new feature idea and wants to drive it through the full workflow.
- The repository benefits from separate idea, specification, and implementation plan documents.
- The user wants clear transition rules between exploration, formalization, and execution planning.
- The team wants a consistent artifact trail that can be reused across projects.

## Do not use this skill when

- The task is a small, already-defined code change.
- The user only wants one stage of the workflow, such as just a specification or just an implementation plan.
- The repository does not benefit from separate planning artifacts.
- The user wants immediate implementation without intermediate documentation.

## Core model

The workflow has three distinct stages:

1. **Idea collaboration**
   - Purpose: explore the concept with a human-in-the-loop document
   - Output: an idea document that stays neutral, expands options, and asks open questions
   - Main concern: understanding the problem space without prematurely locking decisions

2. **Specification creation**
   - Purpose: convert clarified ideas and decisions into a formal, implementation-facing specification
   - Output: a structured specification with numbered sections, requirement tables, and clear scope
   - Main concern: defining what must exist and how it should behave

3. **Implementation planning**
   - Purpose: organize the specification into executable phases and checkbox-tracked work items
   - Output: a phased implementation plan with logical sequencing and validation steps
   - Main concern: determining the order and grouping of work without redefining requirements

## Stage boundaries

Respect these boundaries strictly:

- The **idea document** is collaborative and exploratory.
  - It should present alternatives neutrally.
  - It should surface questions and tradeoffs.
  - It should not act like a final decision document unless the human has explicitly decided something.

- The **specification** is formal and requirement-oriented.
  - It should capture decisions, constraints, and scoped behavior.
  - It should use stable structure and testable requirement language.
  - It should not become a task checklist or phase plan.

- The **implementation plan** is execution-oriented.
  - It should derive from the specification first and the idea document second.
  - It should break work into logical phases and checkbox items.
  - It should not redefine requirements or reopen design exploration.

## Workflow orchestration

Follow this progression:

1. Start with the user’s raw concept.
2. Create or refine the idea document.
3. Iterate with the human until the major open questions are answered well enough to formalize the feature.
4. Create the specification from the idea document and confirmed human direction.
5. Verify that the specification is complete enough for execution planning.
6. Create the implementation plan from the specification, using the idea document only for added rationale and sequencing context.
7. Keep links and references between the three artifacts so the lineage is clear.

## Transition rules

### Move from idea to specification when

- the feature goals are clear
- the main tradeoffs have been narrowed or decided
- the human has answered the most important open questions
- the feature boundaries and constraints are stable enough to write testable requirements

### Stay in idea mode when

- multiple major directions are still intentionally open
- key workflow, permissions, data, or UX questions remain unanswered
- the agent would need to invent major product decisions to write a specification

### Move from specification to implementation plan when

- the major behavior is captured as requirements
- the scope is concrete enough to sequence work
- the architecture and capability areas are clear enough to group into phases

### Stay in specification mode when

- requirement gaps would force the implementation plan to guess
- data model or authorization rules are still materially unclear
- the work cannot be sequenced without redefining the feature

## Artifact expectations

A healthy full-flow output usually produces three separate documents:

- `<Feature>Idea.md`
- `<Feature>Specification.md`
- `<Feature>ImplementationPlan.md`

The exact filenames should follow repository conventions.

## Repository-style expectations

When following the TeamWare-style workflow:

- Idea documents are collaborative and human-in-the-loop.
- Specifications use numbered sections, formal prose, and requirement tables.
- Implementation plans use phased sections with checkbox tracking.
- Each stage references the previous stage.
- Reusable templates, checklists, and supporting artifacts should be captured so the workflow can be reused in other projects.

## Recommended operating pattern

1. Gather repository context and existing related artifacts.
2. Determine which stage the current request belongs to.
3. If the user asked for the whole flow, begin at the earliest incomplete stage.
4. Do not skip directly to later stages unless the earlier stage is already sufficiently documented.
5. Keep documents distinct rather than merging all stages into one file.
6. Preserve traceability with references between idea, specification, and plan documents.

## Output expectations

A strong full-flow workflow should:

- keep exploration, formalization, and execution planning separate
- reduce ambiguity before planning begins
- make decisions traceable across documents
- produce artifacts that are easy for humans and agents to reuse
- support consistent execution across projects

See `references/workflow-checklist.md` for a reusable checklist for moving through the full flow.
