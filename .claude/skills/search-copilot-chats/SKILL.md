---
name: search-copilot-chats
description: Search archived Copilot chat sessions using Chronicle/session_store SQL as the first hint layer, then copilot-session-tools for targeted lookup/export. Use when the user says "search my chats", "find in chat history", "what did we discuss about X", "look up past sessions", "scan chats", "session history", "recent session where", "earlier conversation", "previous session", "that session where", or supplies a session-state path or session GUID to recover conversation context. Also covers exporting narrowed sessions as markdown/HTML and launching the web viewer. Do not activate merely to read, edit, open, copy, or upload an ordinary artifact stored under session-state/files.
---

# Search Copilot Chats

Search, browse, and export archived GitHub Copilot chat sessions from VS Code (Stable & Insiders) and the Copilot CLI. Use **Chronicle/session_store SQL** as the first hint/index layer when available, then use the **copilot-session-tools** CLI for targeted lookup, export, and web viewing. The CLI is backed by a local SQLite database with FTS5 full-text search. Requires Copilot CLI v0.0.412+ (`uv tool install copilot-session-tools[all]` if not on PATH).

**Do not memorize flags.** Run `copilot-session-tools <command> --help` for flags and examples. This skill only documents **domain knowledge that `--help` cannot teach**.

## When to use

- User asks to search past Copilot conversations ("what did we discuss about X?")
- User supplies a `session-state` path or session GUID and wants archived conversation context
- User pastes a web viewer URL like `http://127.0.0.1:5000/session/{uuid}`
- User wants to find prior decisions, code patterns, or troubleshooting steps from past sessions
- User asks to export a session as markdown, HTML, or JSON

An ordinary artifact under `session-state/files` is not a conversation-history request. For example, "copy the HTML report to OneDrive" uses file-transfer tools, not this skill, even if the report's path contains a session GUID.

## Default workflow: Chronicle first, CLI second

For timeline, "find the session where...", prior-discussion, and session-reference tasks, **start with Chronicle/session_store SQL**. Treat it as the cheap hint/index layer that identifies likely sessions before running heavier CLI searches or exports.

1. Query `session_store` first using the built-in SQL tool when available. Search `sessions`, `turns`, `checkpoints`, `session_files`, `session_refs`, and `search_index` for candidate session IDs, dates, branches, files, or PR/issue refs.
2. Use query expansion yourself: search several related keywords in SQL (`bug OR fix OR error`, `auth OR login OR token`, etc.) rather than firing many broad CLI searches.
3. Run `copilot-session-tools search` only after SQL hints identify likely terms, workspaces, date ranges, or session IDs. Keep it targeted.
4. Export full markdown only after narrowing to a specific candidate session ID.

**Do not** launch broad parallel bursts of `copilot-session-tools search --json` when Chronicle/session_store can first identify candidates. Broad CLI searches are a fallback, not the default.

**Ask before running expensive refresh operations**: `scan`, `--full`, or broad rescans require user confirmation unless the user explicitly asked to refresh/rescan/reindex. If you already have a specific CLI session ID and need current enriched content, prefer the targeted `--rescan-session <session-id>` option on the relevant command; this is a single-session refresh, not a broad rescan.

Prefer CLI execution over writing Python scripts once candidates are known. Only drop into Python or direct SQLite when you need a custom join, aggregation, or programmatic processing the CLI doesn't support.

## Search strategy (critical — not in `--help`)

The FTS5 search uses **AND semantics**: every keyword in the query MUST appear in the **same message**. More keywords = stricter matching = fewer results. This is the #1 cause of empty search results.

**Start broad, narrow later.** Always begin with 1–2 core keywords. Only add more if you get too many results.

**Iterative approach.** If no results are returned, **remove** keywords (don't add more). Try synonyms. Drop field filters. The goal is to widen the net, not narrow it further.

**Field filters don't reduce keyword strictness.** A filter like `workspace:ZTS` narrows the *session pool*, but all FTS keywords still must match within a single message.

**Keyword budget.** Aim for **2–3 keywords max**. Use exact phrases for multi-word concepts (e.g., `"resource graph"` counts as 1 match unit).

### Search dos and don'ts

**❌ DON'T — these all returned zero results in real usage:**

| Failed query | Problem |
|---|---|
| `customer tenant query build SPN` | 5 loose keywords — ALL must appear in one message |
| `workspace:ZTS "resource graph" integration test permission` | Workspace filter + exact phrase + 3 keywords = too strict |
| `"build service" "resource graph" reader` | Two exact phrases + a keyword — very unlikely all in one message |
| `spec driven proposal specs design tasks` | 6 keywords, too many AND conditions |

**✅ DO — these succeeded:**

| Successful query | Why it works |
|---|---|
| `ARG build pipeline` | 3 focused keywords |
| `"resource graph" auth` | 1 exact phrase + 1 keyword |
| `workspace:ZTS ARG RBAC` | Workspace filter + only 2 keywords |
| `docker build` | 2 keywords, broad |

**Iterative search workflow:**

```
Step 0: Query Chronicle/session_store SQL for candidate sessions, dates, branches, or refs
  → Found likely session IDs? Export the best candidate directly.
Step 1: copilot-session-tools search "resource graph" --limit 20
  → Too many results? Add ONE keyword:
Step 2: copilot-session-tools search "resource graph" auth --limit 20
  → Still too many? Add a workspace filter:
Step 3: copilot-session-tools search 'workspace:ZTS "resource graph" auth' --limit 10
  → No results? Back up and try different keyword:
Step 4: copilot-session-tools search 'workspace:ZTS "resource graph" permission' --limit 10
```

## Extract session references from user input

Users reference past sessions in several ways. Recognize and extract the session GUID:

| User provides | Extract |
|---|---|
| `C:\Users\...\session-state\{uuid}` or `~\.copilot\session-state\{uuid}` | The `{uuid}` portion |
| `http://127.0.0.1:5000/session/{uuid}` | The `{uuid}` portion |
| A bare GUID like `67894303-8571-...` | Use directly as session ID |
| A short prefix like `67894303` | Query `session_store.sessions` / `session_store.search_index` for matching IDs first; then targeted CLI search if needed |

Then use `export-markdown --session-id` to retrieve the full session content only after the session ID is specific enough.

## Chronicle/session_store hint queries

Use these patterns with the built-in SQL tool (`database: "session_store"`) before CLI search:

```sql
-- Recent timeline / "what was I doing?"
SELECT id, cwd, repository, branch, summary, created_at
FROM sessions
ORDER BY created_at DESC
LIMIT 20;

-- Prior discussion with expanded keywords
SELECT content, session_id, source_type
FROM search_index
WHERE search_index MATCH 'auth OR login OR token OR JWT OR session'
ORDER BY rank
LIMIT 20;

-- Sessions that touched or mentioned a file/path
SELECT s.id, s.summary, sf.file_path, sf.tool_name
FROM session_files sf
JOIN sessions s ON s.id = sf.session_id
WHERE sf.file_path LIKE '%auth%'
ORDER BY sf.first_seen_at DESC
LIMIT 20;

-- Session linked to a PR/issue/commit
SELECT s.id, s.summary, sr.ref_type, sr.ref_value, sr.created_at
FROM session_refs sr
JOIN sessions s ON s.id = sr.session_id
WHERE sr.ref_value = '42'
ORDER BY sr.created_at DESC;
```

After SQL returns candidates, use targeted CLI commands such as:

```powershell
copilot-session-tools search 'workspace:ZTS "resource graph" auth' --limit 10 --json
copilot-session-tools export-markdown --session-id <candidate-session-id> -o .
copilot-session-tools export-markdown --session-id <candidate-session-id> --rescan-session <candidate-session-id> -o .
```

Use `--rescan-session` when Chronicle found the target session but `cst_*` enrichment may be stale or missing. This can bootstrap a CLI session that has never been enriched by `copilot-session-tools` yet, as long as `~/.copilot/session-state/<session-id>/events.jsonl` still exists.

## Direct SQL for advanced queries

When the CLI search doesn't support your query shape, use SQLite directly on `~/.copilot/copilot-session-tools.db`.

**SQLite syntax reminders** (this is SQLite, not PostgreSQL):
- Use `LIKE`, not `ILIKE` (SQLite LIKE is case-insensitive by default for ASCII)
- Use `datetime('now', '-14 days')`, not `INTERVAL '14 days'` or `now() - INTERVAL`
- Use `datetime('now')`, not `now()`

```powershell
# Find sessions by workspace name
sqlite3 "$HOME/.copilot/copilot-session-tools.db" "SELECT session_id, workspace_name, created_at FROM cst_sessions WHERE workspace_name LIKE '%zts%' ORDER BY created_at DESC LIMIT 20"

# Count messages per session (find long conversations)
sqlite3 "$HOME/.copilot/copilot-session-tools.db" "SELECT s.session_id, s.workspace_name, COUNT(m.id) as msg_count FROM cst_sessions s JOIN cst_messages m ON s.session_id = m.session_id GROUP BY s.session_id ORDER BY msg_count DESC LIMIT 20"

# Find sessions with tool invocations of a specific tool
sqlite3 "$HOME/.copilot/copilot-session-tools.db" "SELECT DISTINCT s.session_id, s.workspace_name, s.created_at FROM cst_sessions s JOIN cst_messages m ON s.session_id = m.session_id JOIN cst_tool_invocations ti ON m.id = ti.message_id WHERE ti.name LIKE '%build%' ORDER BY s.created_at DESC LIMIT 20"

# Find file changes by path pattern
sqlite3 "$HOME/.copilot/copilot-session-tools.db" "SELECT DISTINCT s.session_id, s.workspace_name, fc.path FROM cst_sessions s JOIN cst_messages m ON s.session_id = m.session_id JOIN cst_file_changes fc ON m.id = fc.message_id WHERE fc.path LIKE '%Dockerfile%' ORDER BY s.created_at DESC LIMIT 20"
```

## Database schema (quick reference)

The tool extends `~/.copilot/copilot-session-tools.db` with `cst_*` enrichment tables. Built-in tables are never modified.

**Built-in tables** (managed by Copilot CLI, read-only):

| Table | Key columns |
|-------|-------------|
| `sessions` | id, cwd, repository, branch, summary, created_at |
| `turns` | session_id, turn_index, user_message, assistant_response |

**Enrichment tables** (managed by this tool, prefixed `cst_`):

| Table | Key columns |
|-------|-------------|
| `cst_sessions` | session_id, workspace_name, vscode_edition, custom_title, parser_version, source_format |
| `cst_messages` | session_id, message_index, role, content, timestamp |
| `cst_messages_fts` | content (FTS5 virtual table, synced via triggers) |
| `cst_tool_invocations` | message_id, name, input, result, status |
| `cst_file_changes` | message_id, path, diff, content |
| `cst_command_runs` | message_id, command, result, status, output |

## Workflow: "Continue where we left off"

1. Extract the session GUID from the user's input (see table above)
2. If the user provided only a prefix or indirect reference, query Chronicle/session_store to resolve the likely full session ID
3. Export fresh content: `copilot-session-tools export-markdown --session-id <guid> --rescan-session <guid> -o .`
4. Read the exported file and continue the work

## Workflow: "Find how we solved X before"

1. Query Chronicle/session_store SQL with expanded keywords to find candidate sessions
2. Use targeted `copilot-session-tools search "<topic>" --limit 10 --json` only for the most promising candidates/terms
3. Identify the most relevant session from results
4. Export fresh content: `copilot-session-tools export-markdown --session-id <guid> --rescan-session <guid> -o .`

## Known issues

- **Use `--json` for piped output.** The default Rich console output garbles Unicode box-drawing characters on Windows when piped. `--json` bypasses Rich entirely and is the preferred format for programmatic consumption.
- **Missing sessions**: If Chronicle identifies a CLI session but `copilot-session-tools` lacks enriched details, use `--rescan-session <session-id>` on the targeted search/export command. If a session still doesn't appear, check that the source file exists and wasn't cleaned up by VS Code. Ask before using `scan`, `--full`, or any broad rescan unless the user explicitly requested refresh/rescan.
