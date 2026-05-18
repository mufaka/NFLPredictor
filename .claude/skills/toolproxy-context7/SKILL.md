---
name: toolproxy-context7
description: Look up current documentation for third-party libraries, frameworks, SDKs, CLIs, and cloud services via Context7 — even well-known ones like React, Next.js, Prisma, Express, Tailwind, Django, or Spring Boot. Use this skill any time the user asks about a library's API, configuration, setup, version migration, or library-specific behavior ("how do I configure X in Next.js 15", "what's the new Prisma schema syntax", "show me Tailwind v4 config"). Prefer over web search and over recall for library docs — model training data lags releases. Skip for general programming concepts, refactoring, business-logic debugging, or code review. For Microsoft / Azure / .NET docs use `toolproxy-ms-docs` instead.
---

# Context7: library documentation lookup (via ToolProxy)

Up-to-date documentation and code snippets for third-party libraries, indexed by Context7. Two tools, used in sequence: resolve a library name to a Context7 ID, then query that library's docs.

## Operating principles for library lookups

- **Two-step flow.** `resolve-library-id` first, then `query-docs` with the resolved ID. Skip resolution only when the user has handed you an explicit ID in `/org/project` or `/org/project/version` form.
- **Use proper library names when resolving.** `Next.js`, `Three.js`, `Customer.io` — not `nextjs`, `threejs`, `customerio`. Resolution is name-matched; sloppy names get sloppy matches.
- **Pass a real query, not a keyword.** Both tools take a `query` string. Specific beats vague: `"How to set up JWT auth in Express.js with refresh tokens"` produces a useful answer; `"auth"` does not. The same query goes to `resolve-library-id` (for ranking) and `query-docs` (for the actual answer) — write it once, well.
- **Pin the version when the user named one.** When resolving, look at the returned `Versions` list; if the user mentioned a specific version, request the `/org/project/version` form on the follow-up query. Library APIs drift between majors.
- **`researchMode` is the retry, not the default.** Call `query-docs` once without `researchMode`. Only set `researchMode: true` on a retry if the first answer was thin or off-target. It spins up sandboxed agents and live web search — slower and may require an API key.
- **Cap at 3 calls per tool per question.** Both tools enforce this. If you've burned three resolutions or three queries, stop and use the best result you have.
- **Don't reach for this server for non-library questions.** Refactoring, writing scripts from scratch, debugging the user's own business logic, code review, general programming concepts — none of these are library lookups. The host's normal tools are better.
- **Don't put secrets in the query.** Both `query` strings are sent to the Context7 API. Strip API keys, credentials, or proprietary code before forming the query.
- **Cite the library ID in the reply.** When the answer comes from Context7, mention the resolved ID (e.g., `/vercel/next.js/v15.0.0`) so the user can reproduce the lookup or pin the same version.

## How to invoke

All calls go through the ToolProxy `call_external_tool` dispatcher. The envelope is always:

```json
{
  "server": "context7",
  "tool": "<tool name>",
  "arguments": { /* tool-specific parameters */ }
}
```

Every example below repeats the full envelope so it can be copied without inferring structure from elsewhere.

## Tools

### `resolve-library-id` — find the Context7 ID for a library

Searches Context7's library index for a name and returns matching libraries with IDs, descriptions, snippet counts, source reputation, benchmark scores, and available versions. Almost always the first call — skip only when the user gave you an explicit `/org/project` ID.

**Key parameters:**
- `libraryName` — the official library name with proper punctuation (`Next.js`, not `nextjs`).
- `query` — the user's actual question; used to rank candidate libraries by relevance.

**Example — resolving a library by name:**

```json
{
  "server": "context7",
  "tool": "resolve-library-id",
  "arguments": {
    "libraryName": "Next.js",
    "query": "How do I configure middleware for route groups in Next.js 15"
  }
}
```

**Picking from results:** prefer exact name match, higher snippet count, higher benchmark score, and `High`/`Medium` source reputation. If the user named a version, pick the matching `/org/project/version` from the `Versions` list.

### `query-docs` — ask a question against a resolved library

Returns documentation and code examples for a specific library ID, scoped to the question.

**Key parameters:**
- `libraryId` — the ID from `resolve-library-id`, in `/org/project` or `/org/project/version` form.
- `query` — the actual question. Specific and detail-rich; same one you used during resolution is usually fine.
- `researchMode` — optional, default false. Set `true` only on retry when the first answer was insufficient. Slower and may require an API key.

**Example — first pass (no research mode):**

```json
{
  "server": "context7",
  "tool": "query-docs",
  "arguments": {
    "libraryId": "/vercel/next.js/v15.0.0",
    "query": "How do I configure middleware for route groups in Next.js 15"
  }
}
```

**Example — retry with research mode after a thin first answer:**

```json
{
  "server": "context7",
  "tool": "query-docs",
  "arguments": {
    "libraryId": "/vercel/next.js/v15.0.0",
    "query": "How do I configure middleware for route groups in Next.js 15, including cookie forwarding and matcher patterns",
    "researchMode": true
  }
}
```

## Common workflows

### Standard library question

1. `resolve-library-id` with the proper library name and the user's question.
2. Pick the best-matching ID from the results (favor name match, snippet count, benchmark, reputation).
3. `query-docs` with that ID and the question.
4. If the answer is solid, reply and cite the library ID. If it's thin, retry once with `researchMode: true`.

### Version-pinned question

1. `resolve-library-id` to retrieve the `Versions` list.
2. Pick the `/org/project/version` matching what the user is on.
3. `query-docs` with the version-pinned ID.

### User supplied an explicit Context7 ID

1. Skip `resolve-library-id` entirely.
2. Call `query-docs` directly with the supplied ID.

## See also

- **`toolproxy-ms-docs`** — for Microsoft / Azure / .NET / M365 documentation. Context7 indexes some Microsoft libraries, but Microsoft Learn is the authoritative source for the Microsoft stack.
- The host's web search — for blog posts, GitHub issues, release-note threads, and other non-doc sources.
