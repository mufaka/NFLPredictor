# Idea -> Specification -> Implementation Plan Workflow Checklist

Use this checklist to decide what stage the work is in and what should happen next.

## 1. Idea Collaboration Stage

Use this stage when the feature is still being explored.

Checklist:
- Capture the initial concept and problem statement.
- Expand the idea using repository context and relevant technical knowledge.
- Identify goals and non-goals.
- Explore viable approaches neutrally.
- Surface tradeoffs, risks, and open questions.
- Record explicit human choices under a heading such as **Human-Provided Direction**.
- Keep the document collaborative rather than prescriptive.

Ready to move on when:
- the problem is clear
- the feature boundaries are mostly understood
- the most important open questions have been answered
- writing a formal specification would not require inventing major product decisions

## 2. Specification Creation Stage

Use this stage when the idea is mature enough to formalize.

Checklist:
- Use the idea document as the main source of explored context.
- Convert confirmed direction into formal scope and requirements.
- Use numbered sections and consistent headings.
- Use requirement tables with stable IDs.
- Capture data model, service, UI, testing, and security implications as needed.
- Separate requirements from future considerations.
- Reference the idea document and related specifications.

Ready to move on when:
- the feature behavior is described in testable terms
- major constraints and permissions are defined
- the work can be sequenced without reinterpreting requirements

## 3. Implementation Planning Stage

Use this stage when the specification is stable enough to execute.

Checklist:
- Use the specification as the primary source of truth.
- Use the idea document only for additional rationale or sequencing context.
- Break work into logical, dependency-aware phases.
- Use checkbox-tracked work items.
- Make testing explicit in each relevant phase.
- Add cross-cutting validation and polish phases where appropriate.
- Reference the specification and idea documents.

A good implementation plan should answer:
- what gets built first
- what depends on what
- how work is grouped
- where testing and hardening happen

## 4. Meta Rules Across the Whole Flow

- Do not let idea documents become specifications.
- Do not let specifications become implementation checklists.
- Do not let implementation plans redefine requirements.
- Preserve traceability between all documents.
- Match the repository’s naming, structure, and formatting conventions.
- Capture reusable patterns so the workflow can be applied in future projects.
