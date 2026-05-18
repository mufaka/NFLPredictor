# <Feature Name> Specification

## 1. Introduction

### 1.1 Purpose

Describe why this specification exists and what feature or capability it defines.

### 1.2 Scope

Describe the feature boundaries, intended outcomes, and major capability areas covered by this specification.

### 1.3 Definitions and Acronyms

| Term | Definition |
|------|-----------|
| Example Term | Example definition |

### 1.4 Design Principles

- Principle one
- Principle two
- Principle three

---

## 2. Technology Additions

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Example Layer | Example Technology | Example purpose |

---

## 3. Functional Requirements

### 3.1 <Capability Area>

| ID | Requirement |
|----|------------|
| FEAT-01 | The system shall ... |
| FEAT-02 | The system shall ... |

### 3.2 <Capability Area>

| ID | Requirement |
|----|------------|
| FEAT-10 | The system shall ... |
| FEAT-11 | The system shall ... |

---

## 4. Data Model

### 4.1 New Entities

#### <Entity Name>

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| Id | int | PK | Primary key |

**Indexes:**
- `IX_...`

**Relationships:**
- `<Entity>.<Property>` -> `<OtherEntity>.<Property>`

### 4.2 Modified Entities

#### <Existing Entity>

| Property | Type | Description |
|----------|------|-------------|
| ExampleProperty | string | Example change |

---

## 5. Service Layer Design

### 5.1 <Service or Interface>

| Method | Parameters | Returns | Description |
|--------|-----------|---------|-------------|
| ExampleMethod | `...` | `...` | Example behavior |

### 5.2 Existing Service Reuse

| Service | Used By |
|---------|---------|
| `IExampleService` | `ExampleFeature` |

---

## 6. Integration / Endpoint / Tooling Design

Use this section only if relevant.

### 6.1 <Relevant Subsection>

Describe the integration, endpoint, tool, prompt, resource, background process, or client interaction contract.

---

## 7. Changes to Existing Requirements

Describe whether this feature modifies any previously specified behavior. If none, say so explicitly.

---

## 8. Non-Functional Requirements

| ID | Requirement |
|----|------------|
| FEAT-NF-01 | The system shall ... |
| FEAT-NF-02 | The system shall ... |

---

## 9. UI Requirements

Use only if the feature has a user-facing surface.

| ID | Requirement |
|----|------------|
| FEAT-UI-01 | The UI shall ... |
| FEAT-UI-02 | The UI shall ... |

---

## 10. Testing Requirements

| ID | Requirement |
|----|------------|
| FEAT-TEST-01 | The feature shall have tests for ... |
| FEAT-TEST-02 | Integration tests shall verify ... |

---

## 11. Security Considerations

| ID | Consideration |
|----|--------------|
| FEAT-SEC-01 | Describe a security constraint or expectation |
| FEAT-SEC-02 | Describe another security constraint or expectation |

---

## 12. Future Considerations

List explicitly out-of-scope items or possible future enhancements.

- Future item one
- Future item two
- Future item three

---

## 13. References

- [RelatedIdea.md](../RelatedIdea.md) — Source idea document
- [Specification.md](Specification.md) — Main project specification
- External references as needed

---

## Notes for the agent

- Convert idea content into concrete, testable requirements.
- Match the repository’s existing specification format.
- Use stable requirement IDs with a feature-specific prefix.
- Keep implementation sequencing out of the specification.
- If unresolved ambiguities remain, surface them explicitly instead of silently inventing requirements.
