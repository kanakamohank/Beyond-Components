---
name: read-arxiv-paper
description: >
  Use this skill when asked to read one or more arxiv papers given arxiv URLs.
  Handles any arxiv URL format (/abs/, /pdf/, /src/, or bare IDs).
  Supports both single-paper and batch processing workflows.
  Trigger phrases: "read this paper", "summarize this arxiv", "process these papers",
  or any time one or more arxiv.org URLs are provided.
---

You will be given one or more arxiv paper URLs in any of these formats:
- https://arxiv.org/abs/2502.00873
- https://arxiv.org/pdf/2502.00873
- https://arxiv.org/src/2502.00873
- 2502.00873  (bare ID)

If multiple URLs are provided, process each one sequentially using the steps below.

---

## Step 1: Normalize the URL → Extract arxiv ID

From any input format, extract the numeric arxiv ID (e.g. `2502.00873`).

- Strip trailing suffixes like `.pdf`, `v1`, `v2` etc.
- Examples:
  - `https://arxiv.org/pdf/2502.00873`  → ID: `2502.00873`
  - `https://arxiv.org/abs/2301.05217v2` → ID: `2301.05217`

Then construct the **TeX source URL**:
```
https://arxiv.org/src/{arxiv_id}
```

> Always fetch the TeX source, NOT the PDF. The TeX gives structured, machine-readable content.

---

## Step 2: Check Cache

Before downloading, check if the source already exists:
```
~/.cache/arxiv-papers/{arxiv_id}/
```

If the directory exists and contains `.tex` files, **skip to Step 5** (Read the Paper).

---

## Step 3: Download the Paper Source

Download the `.tar.gz` source bundle:
```bash
mkdir -p ~/.cache/arxiv-papers/{arxiv_id}
curl -L https://arxiv.org/src/{arxiv_id} -o ~/.cache/arxiv-papers/{arxiv_id}.tar.gz
```

If the download fails (network error, no TeX source available), fall back to fetching
the abstract page `https://arxiv.org/abs/{arxiv_id}` and summarizing from the abstract only.
Note this fallback clearly in the output.

---

## Step 4: Unpack the Source

```bash
tar -xzf ~/.cache/arxiv-papers/{arxiv_id}.tar.gz \
    -C ~/.cache/arxiv-papers/{arxiv_id}/
```

---

## Step 5: Locate the Entrypoint

Look for the main `.tex` file. Common names in priority order:
1. `main.tex`
2. Any `.tex` file that contains `\documentclass`
3. Any `.tex` file that contains `\begin{document}`

List all `.tex` files if unsure:
```bash
find ~/.cache/arxiv-papers/{arxiv_id}/ -name "*.tex" | head -30
```

---

## Step 6: Read the Paper

Read the entrypoint `.tex` file, then follow `\input{}` and `\include{}` directives
to read all referenced section files. Focus on:

- **Abstract** — what problem does this solve?
- **Introduction** — motivation and contributions
- **Method / Architecture** — the core technical approach
- **Experiments / Results** — key findings, benchmarks, ablations
- **Conclusion / Limitations** — what the authors acknowledge as gaps
- **Key equations, figures, or algorithms** — describe in plain language

Skip bibliography, appendix boilerplate, and LaTeX preamble setup unless relevant.

---

## Step 7: Generate a Summary File

Write a summary to:
```
./knowledge/summary_{tag}.md
```

Where `{tag}` is a short, descriptive snake_case label you invent based on the paper's
topic (e.g. `speculative_decoding`, `long_context_memory`, `tool_use_agents`).

Before writing, check that the tag doesn't already exist:
```bash
ls ./knowledge/summary_*.md 2>/dev/null
```

If a conflict exists, append a disambiguating suffix (e.g. `_2` or a year).

### Summary Template

Each summary file should follow this structure:

```markdown
# {Full Paper Title}

**arxiv ID:** {id}  
**URL:** https://arxiv.org/abs/{id}  
**Authors:** {authors}  
**Published:** {date}  

---

## TL;DR
One or two sentence plain-English summary of the paper's core contribution.

## Problem
What gap or challenge does this paper address?

## Method
Describe the proposed approach, architecture, or algorithm clearly.
Include key equations or pseudocode if they're central to understanding.

## Key Results
- Bullet list of the most important empirical findings
- Include benchmark names and numbers where available

## Limitations & Open Questions
What do the authors acknowledge as weaknesses? What remains unsolved?

## Relevance & Application Ideas
How might this paper's ideas apply to your current project?
What could you try, adapt, or be inspired by?
Explicitly connect to relevant code or components where applicable.

## Tags
`{topic}` `{method-type}` `{domain}`
```

---

## Batch Processing (Multiple Papers)

When given a list of URLs:

1. Deduplicate the list first (same ID appearing multiple times = process once).
2. Process each paper sequentially through Steps 1–7.
3. After all papers are done, create an **index file** at `./knowledge/index.md`:

```markdown
# Paper Reading Index

| Tag | Title | arxiv ID | Date Added |
|-----|-------|----------|------------|
| [summary_tag](./summary_tag.md) | Paper Title | 2502.00873 | YYYY-MM-DD |
...
```

If `./knowledge/index.md` already exists, **append** new rows rather than overwriting.

---

## Error Handling

| Situation | Action |
|-----------|--------|
| TeX source unavailable | Fall back to abstract page; note "abstract-only" in summary |
| `.tar.gz` is actually a single `.tex` file | Treat it directly as the entrypoint |
| Non-English paper | Summarize in English regardless |
| Duplicate URL in batch | Process once, note deduplication in output |
| `./knowledge/` directory missing | Create it with `mkdir -p ./knowledge` |

---

## Papers Pre-Registered with This Skill

The following papers were used to define this skill. They can be processed immediately
using the workflow above:

| arxiv ID | URL |
|----------|-----|
| 2502.00873 | https://arxiv.org/pdf/2502.00873 |
| 2511.20273 | https://arxiv.org/pdf/2511.20273 |
| 2402.02619 | https://arxiv.org/pdf/2402.02619 |
| 2402.16726 | https://arxiv.org/pdf/2402.16726 |
| 2301.05217 | https://arxiv.org/pdf/2301.05217 |
| 2602.13524 | https://arxiv.org/pdf/2602.13524 |
| 2305.15054 | https://arxiv.org/pdf/2305.15054 |
| 2306.17844 | https://arxiv.org/pdf/2306.17844 |
| 2209.10652 | https://arxiv.org/pdf/2209.10652 |
