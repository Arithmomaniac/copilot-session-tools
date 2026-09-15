# Copilot Session Tools

[![CI](https://github.com/Arithmomaniac/copilot-session-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/Arithmomaniac/copilot-session-tools/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/copilot-session-tools)](https://pypi.org/project/copilot-session-tools/) [![Python](https://img.shields.io/pypi/pyversions/copilot-session-tools)](https://pypi.org/project/copilot-session-tools/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Search, browse, and enrich your GitHub Copilot chat history — across **Copilot CLI**, **VS Code**, and **VS Code Insiders**.

Works out of the box with the Copilot CLI's built-in session store (Chronicle). Sessions are browsable immediately; run `scan` to enrich them with full detail (tool calls, file changes, diffs, thinking blocks) and import VS Code desktop sessions.

![Session list](docs/screenshot-list.png)

## Prerequisites

**Copilot CLI v0.0.412+** is recommended. This version introduced cross-session memory ("Chronicle"), which creates `~/.copilot/session-store.db`. When present, copilot-session-tools can discover CLI sessions from Chronicle for enrichment. Update with:

```bash
copilot --update
```

## Quick Start

### 1. Install

```bash
pip install copilot-session-tools[all]
```

> Also works with `pipx install` or `uv tool install` for isolated environments.

### 2. Browse your chats immediately

No scan needed — the web viewer reads directly from the Copilot CLI's session store:

```bash
copilot-session-tools web
# Open http://127.0.0.1:5000/
```

All your CLI sessions appear immediately with basic data (titles, user messages, assistant responses):

![Unenriched session with Scan Now button](docs/screenshot-unenriched.png)

### 3. Enrich for full detail

Click **Scan Now** on any session, or run a bulk scan to enrich everything at once:

```bash
# Enrich all CLI sessions + import VS Code desktop sessions
copilot-session-tools scan

# Or enrich a single session by ID
copilot-session-tools enrich <session-id>
```

Enriched sessions gain tool invocations, file diffs, command output, thinking blocks, and more:

![Enriched session with tool invocations and file changes](docs/screenshot-session.png)

### 4. Search across all sessions

```bash
copilot-session-tools search "authentication" --full
```

![Search with highlighting](docs/screenshot-search.png)

### 5. Use the AI agent skill

If you use an AI coding agent (Copilot CLI, Claude Code, Cursor, etc.), install the **search-copilot-chats** skill for natural-language chat search:

```bash
npx skills-installer install @Arithmomaniac/copilot-session-tools/search-copilot-chats
```

Then ask your agent: *"search my chats for how I fixed the auth bug"*

Under the hood, the skill calls `copilot-session-tools search` and `export-markdown`.

## How It Works

This tool stores enriched session data in its own database (`~/.copilot/copilot-session-tools.db`), separate from the Copilot CLI's Chronicle database (`session-store.db`). Chronicle is optionally read for CLI session discovery and unenriched fallback — but is never modified.

**Two-tier rendering:**

| Tier | Data Source | What You See | When |
|------|-----------|-------------|------|
| **Unenriched** | Chronicle `sessions` + `turns` tables | Session title, user messages, assistant text | Immediately, no scan needed (requires Chronicle) |
| **Enriched** | `cst_*` tables in `copilot-session-tools.db` | Tool invocations, file diffs, command output, thinking blocks, content blocks | After running `scan` or `enrich` |

If new turns arrive after enrichment (e.g., you continued a conversation), the web viewer appends them below the enriched messages with a "new since last scan" divider.

**VS Code sessions** (Stable and Insiders) are imported during `scan` from workspace storage. They're always enriched on import since there's no built-in tier for VS Code.

### Additional Copilot CLI application sources

One archive can include sessions from multiple applications that embed Copilot CLI. Each named source points to that application's Copilot base directory, which contains its read-only `session-store.db` and session event files.

```bash
# See locally detected presets. scout is shown only when ~/.scout/copilot exists.
copilot-session-tools sources presets

# Register a detected Scout installation explicitly, or provide any compatible base directory.
copilot-session-tools sources add --preset scout --name scout
copilot-session-tools sources add --name another-app --base-dir /path/to/copilot

# Manage the per-archive registry shared by the CLI and web viewer.
copilot-session-tools sources list
copilot-session-tools sources rename another-app --name renamed-app
copilot-session-tools sources disable renamed-app
copilot-session-tools sources enable renamed-app
```

Source identifiers are case-insensitive, stored and rendered in lowercase, and cannot use reserved identifiers such as `cli`, `stable`, `insider`, or `custom`. A session UUID is its global identity across applications, so identical copies collapse to one session and keep the same URL. If different session data is found under the same UUID, the operation fails with an identity-conflict error instead of silently choosing one copy. Disabling a source stops live reads and enrichment while retaining content already stored in the archive.

## Searching

```bash
# Basic search (FTS5 full-text search)
copilot-session-tools search "React hooks" --full

# Filter by role, workspace, edition, date
copilot-session-tools search "role:user workspace:my-project error" --full
copilot-session-tools search "edition:cli start_date:2026-01-01 deploy" --full
copilot-session-tools search "deploy" --source scout --full

# Search only tool invocations or file changes
copilot-session-tools search "git" --tools-only
copilot-session-tools search "Dockerfile" --files-only

# Sort by date instead of relevance
copilot-session-tools search "python" --sort date --limit 50
```

**Search tips:** FTS5 uses AND logic — every keyword must appear in the same message. Start with 1–2 keywords, then narrow. Wrap hyphenated terms in quotes: `'"copilot-session-tools"'`.

## Scanning & Enrichment

```bash
# Scan everything (VS Code Stable + Insiders + enrich CLI sessions)
copilot-session-tools scan

# Scan only one VS Code edition
copilot-session-tools scan --edition stable
copilot-session-tools scan --edition insider

# Force full re-import
copilot-session-tools scan --full

# Enrich a single CLI session
copilot-session-tools enrich <session-id>
copilot-session-tools enrich <native-session-id> --source scout

# Scan only one named CLI source (skips VS Code for this invocation)
copilot-session-tools scan --source scout
```

The web viewer's **Scan Now** button on unenriched sessions triggers single-session enrichment without a full scan.

Scanning is **incremental** by default — only new and changed sessions are processed.

## Exporting

```bash
# Export as Markdown
copilot-session-tools export-markdown --output-dir ./archive --include-diffs

# Export a single session
copilot-session-tools export-markdown --session-id <id> --output-dir .

# Export as self-contained HTML (dark mode, collapsible sections, no server needed)
copilot-session-tools export-html --output-dir ./html-archive

# Export all sessions as JSON
copilot-session-tools export --output chats.json

# Import from JSON
copilot-session-tools import-json chats.json
```

## Web Viewer

```bash
copilot-session-tools web                    # defaults to port 5000
copilot-session-tools web --port 8080        # custom port
copilot-session-tools web --db custom.db     # custom database
```

Features:
- **Full-text search** with keyword highlighting
- **Application filters** with separate counts for `cli`, each named source such as `scout`, and VS Code editions
- **Named CLI sources** managed from the Sources page, including a conditional `scout` preset
- **Repository and workspace filtering**
- **Enrichment status** — see which sessions have full detail vs. basic turns
- **Copy Markdown** toolbar — select message range, include/exclude diffs, tool inputs, thinking
- **Download** markdown or copy session URL
- **Transcript cleanup** — clean up voice-dictated messages using LLM (see below)
- **Dark mode** via CSS `prefers-color-scheme`
- **Incremental refresh** without restarting

## Transcript Cleanup

Voice-dictated user messages often come out garbled (repeated words, filler words, broken sentences). Transcript cleanup uses an LLM to rewrite them, using the assistant's response and reasoning as context for what you actually meant.

```bash
# Clean up a single session (auto-detects voice-dictated messages)
copilot-session-tools cleanup <session-id>

# Preview changes without writing
copilot-session-tools cleanup <session-id> --dry-run

# Clean all user messages (skip auto-detection)
copilot-session-tools cleanup <session-id> --all

# Clean a specific message
copilot-session-tools cleanup <session-id> --message 5

# Revert all cleaned messages to originals
copilot-session-tools cleanup-revert <session-id>

# Revert a single message
copilot-session-tools cleanup-revert <session-id> --message 5

# List sessions with potential voice-dictated messages
copilot-session-tools cleanup
```

Requires the `[llm]` extra (uses [LiteLLM](https://docs.litellm.ai/) with the [GitHub Copilot provider](https://docs.litellm.ai/docs/providers/github_copilot)):

```bash
pip install copilot-session-tools[llm]
```

**How it works:**
1. A pure-Python heuristic pre-filter scores each user message (repeated words, filler density, missing punctuation)
2. Messages above the threshold are batched into a single LLM call with context (assistant response + thinking traces + tool inputs)
3. The LLM classifies each as typed or voice-dictated and cleans the voice-dictated ones
4. Originals are preserved — you can always revert

**Web viewer integration:** Each user message shows a 🎙️ mic icon — gray (not cleaned), green (cleaned), or amber (showing original). Click to open a popover with toggle/revert/clean actions. Session-level cleanup and revert-all buttons are in the toolbar.

![Transcript cleanup with Cleanup/Revert toolbar buttons](docs/screenshot-cleanup.png)

## Session Sources

| Source | Format | Location |
|--------|--------|----------|
| **Copilot CLI** | JSONL events | `~/.copilot/session-state/` |
| **VS Code Stable** | JSON / JSONL | `%APPDATA%/Code/User/workspaceStorage/*/` |
| **VS Code Insiders** | JSON / JSONL | `%APPDATA%/Code - Insiders/User/workspaceStorage/*/` |

On macOS/Linux, replace `%APPDATA%` with `~/Library/Application Support` or `~/.config` respectively.

## Agent Skills

This repository includes [Agent Skills](https://claude-plugins.dev) for AI coding agents:

| Skill | Description |
|-------|-------------|
| **search-copilot-chats** | Search, browse, and export Copilot chat sessions. Triggers on "search my chats", "find in chat history", session GUIDs, or web viewer URLs. Calls `copilot-session-tools search` and `export-markdown` under the hood. |
| **scanner-refresh** | Research recent changes in Copilot CLI/VS Code repos and update the scanner for new event types. |

```bash
# Install for your agent (Claude Code, Cursor, VS Code, Codex, etc.)
npx skills-installer install @Arithmomaniac/copilot-session-tools/search-copilot-chats
npx skills-installer install @Arithmomaniac/copilot-session-tools/search-copilot-chats --client cursor
```

Skills are also available automatically when working in this repository.

## Database Schema

The tool writes to `cst_*` enrichment tables alongside the CLI's built-in tables:

| Table | Contents |
|-------|----------|
| `cst_sessions` | Enriched session metadata (workspace, edition, parser version, source format) |
| `cst_messages` | Parsed messages with content blocks, transcript cleanup (original_content, cleanup_model) |
| `cst_messages_fts` | FTS5 full-text search index |
| `cst_tool_invocations` | Tool calls (name, input, result, status) |
| `cst_file_changes` | File edits with diffs |
| `cst_command_runs` | Shell commands and output |

**Recovery:** If `cst_*` tables get corrupted, just delete the database and re-scan. Chronicle data is completely unaffected since it lives in a separate file:

```bash
rm ~/.copilot/copilot-session-tools.db
copilot-session-tools scan --full
```

## Installation Options

```bash
pip install copilot-session-tools[cli]     # CLI only
pip install copilot-session-tools[web]     # Web viewer only
pip install copilot-session-tools[all]     # CLI + web + LLM
pip install copilot-session-tools[llm]     # Optional: transcript cleanup via LiteLLM (GitHub Copilot API)
pip install copilot-session-tools[vector]  # Optional: hybrid FTS5 + vector search (sqlite-vec + sentence-transformers)
```

## Development

```bash
git clone https://github.com/Arithmomaniac/copilot-session-tools.git
cd copilot-session-tools
uv sync --all-extras
uv run pytest tests/ --ignore=tests/test_webapp_e2e.py -v
uv run ruff check . && uv run ruff format . && uv run ty check
```

## Acknowledgments

This project was informed by several excellent open-source projects:

| Project | Inspiration |
|---------|-------------|
| [simonw/claude-code-transcripts](https://github.com/simonw/claude-code-transcripts) | HTML transcript generation, web viewer design |
| [Arbuzov/copilot-chat-history](https://github.com/Arbuzov/copilot-chat-history) | VS Code session data format |
| [tad-hq/universal-session-viewer](https://github.com/tad-hq/universal-session-viewer) | FTS5 search design |

## License

MIT License — see [LICENSE](LICENSE) for details.
