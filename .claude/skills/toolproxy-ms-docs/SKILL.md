---
name: toolproxy-ms-docs
description: Look up official Microsoft and Azure documentation — search Microsoft Learn for concepts, fetch full doc pages as markdown, or pull working code samples from Microsoft's own examples. Use this skill when the user asks about Microsoft / Azure / .NET / M365 / Windows / Power Platform APIs, services, configuration, or SDKs ("how does Azure Functions handle X", "what's the .NET 10 syntax for Y", "show me a Graph API code sample for Z", "is this Azure CLI flag still supported"). Strongly prefer over web search or recall for any Microsoft-stack question — these tools return current, citable, official content. Skip for general programming questions that aren't Microsoft-specific.
---

# Microsoft Learn: documentation lookup (via ToolProxy)

Search and fetch official Microsoft / Azure documentation, plus pull working code samples from Microsoft Learn. Everything returned is from Microsoft's own sources, in clean markdown, with URLs you can cite back to the user.

## Operating principles for Microsoft Learn lookups

- **Microsoft-scoped only.** This server covers Microsoft Learn — Azure, .NET, M365, Windows, Power Platform, Graph, etc. Don't reach for it for general programming questions or non-Microsoft tooling. For other vendors' docs, use `context7` or web search.
- **Trust the docs over training recall.** The Microsoft surface area (especially Azure) shifts often: new SDK versions, deprecated flags, renamed services, preview-to-GA transitions. When the user asks about current behavior, query rather than recall — model knowledge cutoffs lag the live docs.
- **Search before fetch.** `microsoft_docs_search` returns up to 10 short excerpts (~500 tokens each) with title + URL. That's usually enough to answer the question or pick which page is worth pulling in full. `microsoft_docs_fetch` is the expensive option — reach for it only when an excerpt points at a page you genuinely need to read end-to-end (a tutorial, a long config reference, a troubleshooting guide).
- **Code samples have their own search.** If the user wants "how do I do X" with working code, `microsoft_code_sample_search` is more useful than docs search — it returns up to 20 actual snippets rather than prose. Filter by `language` when the user has named one.
- **Search URLs feed fetch.** Every search hit includes a URL. When you decide to fetch, copy the URL straight from the search response — don't reconstruct or guess Microsoft Learn URLs.
- **Cite what you found.** When the answer comes from these tools, include the doc URL(s) in the reply so the user can verify and read further. The whole point of grounding through Microsoft Learn is auditability.

## How to invoke

All calls go through the ToolProxy `call_external_tool` dispatcher. The envelope is always:

```json
{
  "server": "ms-docs",
  "tool": "<tool name>",
  "arguments": { /* tool-specific parameters */ }
}
```

Every example below repeats the full envelope so it can be copied without inferring structure from elsewhere.

## Tools

### `microsoft_docs_search` — find relevant doc excerpts

Searches Microsoft Learn and returns up to 10 high-quality content chunks, each capped at ~500 tokens, with title, URL, and excerpt. The right first call for almost any Microsoft / Azure question.

**Key parameter:**
- `query` — natural-language search string. Be specific; "Azure Functions HTTP trigger authorization levels" beats "azure functions auth".

**Example — concept lookup:**

```json
{
  "server": "ms-docs",
  "tool": "microsoft_docs_search",
  "arguments": {
    "query": "Azure Functions HTTP trigger authorization levels"
  }
}
```

**Example — version-specific syntax:**

```json
{
  "server": "ms-docs",
  "tool": "microsoft_docs_search",
  "arguments": {
    "query": ".NET 10 minimal API route group authentication"
  }
}
```

### `microsoft_code_sample_search` — find working code snippets

Searches Microsoft Learn for actual code samples and returns up to 20 results. Use this when the user wants implementation, not explanation.

**Key parameters:**
- `query` — natural-language description of what the code should do.
- `language` — optional filter (e.g., `csharp`, `python`, `typescript`, `bicep`). Set when the user has named a language; omit to see results across languages.

**Example — language-scoped sample:**

```json
{
  "server": "ms-docs",
  "tool": "microsoft_code_sample_search",
  "arguments": {
    "query": "send email via Microsoft Graph SDK",
    "language": "csharp"
  }
}
```

**Example — cross-language sample search:**

```json
{
  "server": "ms-docs",
  "tool": "microsoft_code_sample_search",
  "arguments": {
    "query": "Azure Storage blob upload with managed identity"
  }
}
```

### `microsoft_docs_fetch` — pull a full doc page as markdown

Fetches a complete Microsoft Learn page and returns it as clean markdown. Use after a search has identified a specific page that needs to be read in full — a tutorial, a long reference, a troubleshooting walkthrough.

**Key parameter:**
- `url` — the Microsoft Learn URL, taken verbatim from a `microsoft_docs_search` or `microsoft_code_sample_search` result.

**Example:**

```json
{
  "server": "ms-docs",
  "tool": "microsoft_docs_fetch",
  "arguments": {
    "url": "https://learn.microsoft.com/azure/azure-functions/functions-bindings-http-webhook-trigger"
  }
}
```

## Common workflows

### Answering a "how does X work" question

1. `microsoft_docs_search` with a precise query.
2. Read the excerpts; if one is enough to answer, do so and cite the URL.
3. If the question needs the full page (tutorial steps, full reference table), `microsoft_docs_fetch` on that URL.

### Producing working code for a Microsoft API

1. `microsoft_code_sample_search` with `language` set to whatever the user is writing.
2. If results are thin or generic, fall back to `microsoft_docs_search` for the SDK overview, then fetch the specific reference page.
3. Adapt the sample to the user's context; cite the source URL.

### Verifying current API surface or deprecation state

1. `microsoft_docs_search` for the specific symbol / flag / service.
2. Check the excerpt for "deprecated", "preview", "GA", or version markers.
3. If unclear, fetch the reference page in full to see the version notes section.

## See also

- **`toolproxy-context7`** (if installed) — non-Microsoft library docs.
- The host's web search — for blog posts, GitHub issues, and other non-official sources when Microsoft Learn doesn't cover the question.
