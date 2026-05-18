---
name: spec-creation
description: Creates or refines a formal software specification from an existing idea document or clarified feature concept. Use this when exploratory collaboration is complete enough that the work should be converted into structured requirements, scoped sections, requirement IDs, and implementation-facing detail without yet turning it into an implementation plan.
compatibility: Intended for repositories that use an idea -> specification -> implementation plan workflow.
---

# Purpose

Use this skill when an idea has matured beyond brainstorming and is ready to be captured as a formal specification.

This skill converts collaborative idea material into a precise, structured, implementation-facing specification. The output should preserve human decisions, resolve ambiguity where the idea phase already settled it, and organize the result into numbered sections, requirement tables, and clearly scoped design details.

## Use this skill when

- The user has an idea document and wants a formal specification.
- The exploratory questions have been answered well enough to define requirements.
- The repository uses specification documents as the source for later implementation planning.
- The user wants a durable, reviewable artifact that describes what the feature is, how it fits, and what must be built.

## Do not use this skill when

- The work is still exploratory and multiple directions remain intentionally open.
- The user wants brainstorming, tradeoff exploration, or human-facing open questions.
- The user wants a phased task breakdown or implementation plan.
- The task is a tiny code change that does not need a formal specification.

## Core behavior

- Treat the specification as a formal artifact, not a brainstorming document.
- Base the specification on:
  - the idea document
  - explicit human decisions
  - existing repository conventions
  - current architecture and service boundaries
- Convert explored concepts into concrete requirements.
- Remove unresolved brainstorming language where decisions have been made.
- Keep the document implementation-aware, but not implementation-task-oriented.
- Do not produce a phase-by-phase plan, checklist, or backlog in the specification.

## Workflow

1. Read the relevant idea document and any related specifications.
2. Identify what has already been decided versus what is still unresolved.
3. Infer the specification structure from repository conventions.
4. Organize the document into numbered sections.
5. Convert behavior into requirement tables with stable requirement IDs.
6. Capture supporting design details such as:
   - scope
   - definitions and acronyms
   - technology additions
   - data model changes
   - service layer design
   - endpoint, UI, security, and testing requirements
   - non-functional requirements
7. Preserve traceability back to the idea and related documents.
8. If critical ambiguities remain, either:
   - call them out explicitly in a limited, formal way, or
   - ask the human for clarification before finalizing the specification.

## Writing rules

- Use formal, structured language.
- Prefer numbered sections and subsection headings.
- Use requirement tables with stable IDs in a consistent prefix family for the feature.
- Write requirements as testable statements.
- Use decisive language in requirements, typically “shall,” when the behavior is actually decided.
- Keep explanatory prose around the requirements concise and focused.
- Distinguish between requirements, design notes, and future considerations.
- Keep the specification aligned with existing repository patterns rather than inventing a new format.

## TeamWare-style patterns to emulate

When the repository uses the TeamWare style shown in documents such as `McpServerSpecification.md`, prefer this overall shape:

1. Introduction
   - Purpose
   - Scope
   - Definitions and Acronyms
   - Design Principles
2. Technology Additions
3. Functional Requirements
   - grouped by capability area
   - each group expressed as a requirement table with IDs
4. Data Model
   - new entities
   - modified entities
   - seeded configuration if applicable
5. Service Layer Design
6. Endpoint / Integration / Tooling Design as applicable
7. Changes to Existing Requirements
8. Non-Functional Requirements
9. UI Requirements if applicable
10. Testing Requirements
11. Security Considerations
12. Future Considerations
13. References

Not every section is required for every feature, but the document should feel structurally similar.

## Requirement ID guidance

- Use a short, feature-specific prefix, such as `MCP`, `PAT`, or `LOUNGE`.
- Group IDs by concern when useful.
- Keep numbering stable once published unless there is a strong reason to renumber.
- Prefer concise, testable requirements over broad narrative paragraphs.

Examples:
- `MCP-01`
- `MCP-TEST-03`
- `MCP-SEC-05`
- `LOUNGE-42`

## Converting idea content into spec content

When translating from an idea document:

- Convert goals and decisions into formal requirements.
- Convert optional approaches into a single chosen direction only if the human or prior documents have actually decided it.
- Move remaining unresolved items out of the main flow, either by resolving them with the human or by capturing them as explicit assumptions, exclusions, or future considerations.
- Replace exploratory phrases like “possible,” “might,” or “could” with concrete language only when justified.
- Preserve important rationale briefly where it helps readers understand scope or constraints.

## Output expectations

A strong specification should:

- define the feature clearly enough for planning and implementation
- minimize ambiguity
- fit the repository’s existing specification style
- support testing and review
- make downstream implementation planning easier

See `references/specification-template.md` for a reusable starter format.

## Boundary with implementation plans

The specification describes what must exist and how it should behave.

It should not include:
- phased delivery plans
- branch strategy
- commit sequencing
- checklists of coding tasks
- developer workflow steps

If the user wants execution sequencing, create an implementation plan after the specification is accepted.
