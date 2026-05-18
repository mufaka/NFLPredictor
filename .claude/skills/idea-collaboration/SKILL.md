---
name: idea-collaboration
description: Creates or refines a collaborative software idea document from an early feature concept. Use this when the user has a rough idea and wants a human-in-the-loop artifact that expands options, grounds them in the current codebase, and surfaces questions before writing a formal specification or implementation plan.
compatibility: Intended for repositories that use an idea -> specification -> implementation plan workflow.
---

# Purpose

Use this skill when the user wants to explore an idea before committing to a design.

The idea document is a collaborative working artifact between the agent and a human stakeholder. The agent should expand the initial idea using knowledge of the current repository, existing product patterns, and relevant technical context, while keeping the document exploratory rather than prescriptive.

## Use this skill when

- The user has an early feature, product, integration, or workflow idea.
- The user wants to brainstorm options before writing a specification.
- The repository already uses idea documents as a precursor to specifications and implementation plans.
- The user wants help surfacing hidden assumptions, tradeoffs, and unanswered questions.

## Do not use this skill when

- The user already wants a formal requirements specification.
- The user wants an implementation plan or task breakdown.
- The problem is already decided and only needs coding.
- The task is a narrow bug fix or a small, well-defined code change.

## Core behavior

- Treat the idea document as a human-in-the-loop collaboration artifact.
- Expand the user’s initial prompt using:
  - the current codebase and architecture
  - existing product conventions and patterns
  - relevant platform, framework, and domain knowledge
- Stay neutral. Present alternatives, tradeoffs, and open questions.
- Do not turn the document into a specification, implementation plan, or unilateral decision log.
- Do not declare a preferred solution unless the human explicitly asks for a recommendation.
- If the human has already decided something, record it as **Human-Provided Direction**, **Confirmed Constraints**, or a similarly explicit heading.

## Workflow

1. Capture the raw idea in plain language.
2. Inspect the current repository for related features, constraints, naming patterns, and architectural boundaries.
3. Identify the design dimensions that matter for this idea, such as:
   - user workflow
   - UX and entry points
   - data model implications
   - permissions and visibility rules
   - integrations
   - operational concerns
   - migration or compatibility concerns
4. Draft or refine the idea document so it:
   - explains the problem and motivation
   - clarifies goals and non-goals
   - explores multiple viable approaches or sub-concepts
   - highlights tradeoffs and risks
   - asks concrete questions a human can answer
5. Keep unresolved items explicitly open.
6. When the human answers questions, revise the idea document rather than jumping immediately to implementation planning.
7. Recommend moving to a specification only after the idea is sufficiently clear.

## Writing rules

- Prefer concise but substantive sections over shallow bullets.
- Separate known facts from possibilities.
- Use neutral language such as:
  - possible approaches
  - potential advantages
  - potential concerns
  - open questions
- Avoid turning exploratory content into requirements.
- Avoid detailed implementation tasks, phase breakdowns, or commit-level planning.
- Avoid overly rigid language such as “must,” “shall,” or “will” unless quoting an explicit human requirement.
- Keep the document readable to technical and non-technical stakeholders.
- If draft data models, endpoints, or UI flows are included, label them as exploratory.

## Suggested structure

Adapt this structure as needed for the idea:

1. Title
2. Overview
3. Human-Provided Direction or Constraints
4. Problem Statement
5. Goals
6. Non-Goals
7. Current System Context
8. Concepts or Possible Approaches
9. Relevant UX, Data, Security, and Integration Considerations
10. Tradeoffs or Risks
11. Open Questions for the Human
12. References or Related Artifacts

See `references/idea-document-template.md` for a reusable starter format.

## Iteration guidance

- If an idea document already exists, preserve useful context and refine it.
- Fold human answers back into the document visibly.
- Convert answered questions into confirmed direction or narrowed options.
- Remove stale questions only when they are actually resolved.
- Keep the document collaborative throughout the iteration cycle.

## Output expectations

A strong idea document should help a human:

- react to the concept quickly
- spot hidden assumptions
- compare directions without pressure to choose too early
- provide missing constraints
- decide when the idea is ready to become a specification

If the task shifts from exploration to commitment, recommend creating a formal specification next.
