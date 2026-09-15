> Created/edited by GitHub Copilot with human review/feedback by avilevin.

# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.18.0] - 2026-08-06

### Changed

- **Scanner / Web Viewer**: External (host-app) tools (`external_tool.requested` / `external_tool.completed`) now render as collapsible tool blocks that expose their parameters, instead of a one-line status breadcrumb. Because the host application runs these tools and never returns a result into the session, the block shows an explicit "Response: Not recorded" note, and an `external tool` badge distinguishes them from CLI tools. Affects the web viewer and HTML export.

## [0.17.1] - 2026-06-30

### Fixed

- **Scanner / Database**: CLI enrichment no longer re-writes already-current sessions on every scan. Staleness is now detected by storing each session's Chronicle turn count (`cst_sessions.builtin_turns`) and comparing it like-for-like, instead of comparing Chronicle turns against the enriched user-message count (an incompatible quantity that never matched, so hundreds of sessions were re-enriched on every scan and the scan never converged).
- **Database**: Deleting a session's full-text-search rows now targets `cst_messages_fts` by `rowid` via the indexed message lookup rather than the `UNINDEXED` `session_id` column, which forced a full scan of every FTS document. On a large database this cuts per-session FTS deletion from ~1 s to ~1.5 ms, making each enrichment write roughly an order of magnitude faster.

### Changed

- **Database**: Schema migrated to v12, adding the nullable `cst_sessions.builtin_turns` column. The first scan after upgrading performs a one-time backfill of existing CLI sessions; subsequent incremental scans skip unchanged sessions.

## [0.17.0] - 2026-06-29

### Added

- **CLI File Changes**: CLI `edit`/`create` tool calls now render as per-message file-change diffs, matching the VS Code view across the web viewer, Markdown export, and HTML export.
- **Per-turn consolidation**: Multiple edits to the same file within a turn collapse into one card — a true net diff when the file is created in-turn, or reconstructed from a prior full-file `view` (including subagent views), with a stacked-hunks fallback when no base is available.

## [0.16.0] - 2026-06-06

### Added

- **Search / Forks**: CLI event IDs are now persisted as message provenance so copied pre-fork history can be identified across forked sessions.

### Changed

- **Search / Forks**: Search now suppresses duplicate hits from copied fork history while keeping fork-specific continuations searchable. Opening or exporting a forked session still shows the full copied conversation.

## [0.15.1] - 2026-06-05

### Added

- **CLI**: Added `--rescan-session <session-id>` to `search`, `export-markdown`, and `export-html` so a single CLI session can be parsed/enriched immediately before the command runs.
- **CLI**: Targeted rescans can bootstrap a missing CST enrichment database for never-scanned sessions when explicitly requested.
- **Agent Workflow**: Updated Copilot chat search guidance to use targeted rescans after Chronicle identifies a specific candidate session.

### Fixed

- **CLI**: Session-specific exports now reject mismatched `--session-id` and `--rescan-session` values instead of silently refreshing one session and exporting another.

## [0.15.0] - 2026-06-02

### Added

- **Scanner / Database**: Track context history for sessions that change working directory or repository over time, while preserving the original scalar workspace/repository fields as the primary context.
- **Search**: Workspace and repository filters now match the context active at each search hit, with repeated `workspace:` and `repo:` filters treated as OR within the same filter kind.
- **Web Viewer**: Search-result snippets show the active workspace/repository only when it differs from the session's root context.

### Changed

- **Web Viewer / HTML Export**: Working-directory changes render inline as transcript status pills instead of a session-level "context changed N times" summary.
- **Markdown Export**: Metadata now includes the primary repository without adding a separate context-history summary.

### Fixed

- **Scanner**: Recognize the current Chronicle session-store schema version so scans no longer warn for schema v4 stores.

## [0.14.0] - 2026-06-02

### Added

- **Web Viewer / HTML Export**: Top-level User, Assistant, and included System messages now render as collapsible sections that are open by default.
- **Web Viewer / HTML Export**: Auxiliary sections inside messages render as collapsible sections while preserving their existing default visibility.
- **Content Types / Exporters**: `system-messages` is now an opt-in CONTENT flag that stays disabled by default in exports and search.

### Changed

- **Search**: Message searches exclude system-role messages by default unless an explicit role filter, such as `role=system`, is provided.

## [0.13.0] - 2026-05-31

### Added

- **Web Viewer / HTML Export**: `role == "system"` messages now render as collapsible `<details>` blocks with a chevron, collapsed by default and expandable inline.
- **Content Types / Exporters**: New `system-messages` content type for `--include/--exclude`. In markdown export, system messages are omitted unless included. In HTML export, system messages always render but are collapsed by default; including `system-messages` opens them by default.

## [0.12.0] - 2026-05-26

### Added

- **Scanner**: Parse CLI `session.info` fork messages and render `Forked as` / `Forked from` status blocks with safe session links.
- **Web Viewer / HTML Export**: Fork status badges can render sanitized Markdown links inline.
- **Agent Workflow**: Replaced the old scanner-refresh skill with a project agent backed by configured `copilot-agent-runtime` or decompiled runtime paths.

### Fixed

- **Security**: Escaped fork links to prevent unsafe HTML injection in fork status rendering.

## [0.11.1] - 2026-05-08

### Fixed

- **Scanner**: Accept non-dict/freeform tool arguments, preserving raw string input for tools such as `apply_patch`.
- **Rendering Helpers**: Guard tool-display formatting against string arguments so parser/enrichment no longer crashes.

## [0.11.0] - 2026-04-28

### Added

- **Scanner / Database**: Attribute root custom-agent sessions and persist their agent metadata through storage and retrieval.
- **Web Viewer / HTML Export**: Surface root custom-agent attribution in rendered sessions.

### Changed

- **Scanner**: Improved custom-agent metadata extraction and utility normalization shared across CLI, HTML export, and web rendering.

## [0.10.4] - 2026-04-14

### Changed

- **Transcript Cleanup**: Added structured output auto-dispatch across GPT chat completions, GPT Responses API text format, and Claude/Gemini tool-choice paths.
- **Transcript Cleanup**: Updated the default cleanup model and LiteLLM minimum version for the structured output integration.

### Fixed

- **Transcript Cleanup**: Pass the system prompt through the correct model-specific parameter and raise a clear error on empty structured-output responses.

## [0.10.3] - 2026-04-05

### Added

- **Web Viewer**: Async shell rendering — `powershell(mode="async")` commands render with amber left-border and "⟳ async" badge (Font Awesome `fa-arrows-rotate`); detached shells (`detach: true`) show "🔗‍💥 detached" badge (`fa-link-slash`) instead
- **Web Viewer**: Grouped IO entries — `read_powershell`, `write_powershell`, `stop_powershell` results are collected inside the parent async shell block as expandable IO cards with Input/Output sections and syntax highlighting
- **Web Viewer**: Bidirectional linking — conversation pills (↩ READ/WRITE/STOP) link into the shell block's IO entry; IO entries link back to conversation context (↗ context). Clicking a pill auto-opens the shell dropdown
- **Web Viewer**: Async shell output simplified to `shellId: X` label (boilerplate `<command started in background>` text hidden)
- **Scanner**: New fields on `CommandRun`: `shell_id`, `is_async`, `is_detached`, `io_entries` for tracking async shell lifecycle
- **Scanner**: New fields on `ToolInvocation`: `is_shell_backlink`, `backlink_shell_id`, `shell_pill_id`, `shell_anchor_id` for bidirectional linking
- **Scanner**: `read_powershell` is no longer skipped as an internal tool — it now renders as a visible shell backlink
- **Scanner**: Shell title map — backlink pills use the launch block's title instead of raw shellId
- **Database**: Schema v8 — new columns on `cst_command_runs` and `cst_tool_invocations` for async shell tracking (auto-migrated from v7)

### Fixed

- **Security**: Shell backlink onclick handlers use `data-*` attributes instead of interpolating shellId into JavaScript string literals (XSS prevention)
- **Web Viewer**: Duplicate shellId handling — when the same shellId is reused across shell launches, IO entries are scoped temporally to the correct parent CommandRun
- **Web Viewer**: IO entry results now use syntax highlighting pipeline (`detect_language` + `highlight_code`), matching main tool output rendering

## [0.10.2] - 2026-04-05

### Added

- **Web Viewer / HTML Export**: Syntax highlighting and JSON prettification for tool inputs and outputs — JSON is auto-detected, pretty-printed with 2-space indentation, and colored using Pygments with GitHub-flavored light/dark themes
- **Web Viewer**: "Syntax highlighting" toggle in View Settings → Display section (persisted in localStorage)
- **Web Viewer / HTML Export**: Unified diff output from tools highlighted with red/green coloring
- **Markdown Export**: Tool inputs with JSON content now use language-tagged code fences (` ```json `) with prettified content
- **Dependencies**: Added `pygments>=2.17.0` as a core dependency for server-side syntax highlighting

## [0.10.1] - 2026-04-02

### Fixed

- **Scanner (VS Code)**: Extract `prompt` from VS Code subagent `toolSpecificData` — prompts were available in the raw JSONL but ignored by the scanner, so VS Code agent blocks never showed their prompt section

## [0.10.0] - 2026-04-01

### Added

- **Web Viewer**: Visual distinction between sync and background agents — sync agents use purple borders, background agents use blue borders with "↗ background" badge
- **Web Viewer**: Agent prompt section — expandable "Prompt" block inside each agent dropdown showing what the agent was asked to do
- **Web Viewer**: State-colored backlink pills for background agents — `read_agent` calls render as clickable ↩ pills colored by state: blue (in progress), green (completed), red (failed), linking back to the agent block
- **Web Viewer**: Skills styled with teal color to distinguish from agent blocks (previously shared purple with sync agents)
- **Web Viewer**: System notification messages (`<system_notification>`) collapsed into compact indicators instead of full message bubbles
- **Scanner**: New fields on `ContentBlock`: `prompt`, `is_background`, `agent_id` for structured agent metadata
- **Scanner**: New fields on `ToolInvocation`: `is_agent_backlink`, `backlink_agent_id` for linking `read_agent` calls back to their agent blocks
- **Database**: Schema v7 — new columns for agent rendering fields (auto-migrated from v6 via ALTER TABLE)

### Changed

- **Web Viewer**: Standalone subagent status pills (`completed`/`failed`) hidden — redundant with agent block headers and backlink pills
- **Web Viewer**: Background agent badge order — "↗ background" now appears before "completed" badge for consistent alignment

## [0.9.0] - 2026-03-29

### Changed

- **Database**: CST enrichment tables now live in their own database file (`~/.copilot/copilot-session-tools.db`) instead of sharing the Copilot CLI's `session-store.db`. Chronicle corruption can no longer take down CST data, and vice versa. No migration needed — just re-run `scan` to populate the new database.
- **Database**: Chronicle's `session-store.db` is ATTACHed read-only for CLI session discovery and unenriched fallback display. Auto-detected as sibling file; gracefully degrades when absent.
- **Performance**: Wrapped VS Code session import in `batch_connection()` — FK=OFF + 64MB cache reduces 100-file update from 35s to 0.6s (57× speedup)
- **CLI**: Default `--db` path changed from `session-store.db` to `copilot-session-tools.db`
- **CLI**: New `--chronicle-db` global option to explicitly specify the Chronicle database path (auto-detected by default)
- **Recovery**: Simplified — just delete `copilot-session-tools.db` and re-scan (no more surgical SQL drops needed)

## [0.8.0] - 2026-03-29

### Added

- **CLI**: `--json` / `-j` flag on `search` and `stats` commands for structured output — bypasses Rich formatting, safe for piping to `jq` or programmatic parsing
- **CLI**: `Examples:` block in `--help` for all 12 subcommands with real invocations
- **CLI**: Actionable error messages — validation errors now include example invocations showing how to fix the issue

### Changed

- **Skill**: Expanded `search-copilot-chats` trigger phrases to cover "session history", "recent session where", "earlier conversation", "previous session", "that session where" (3 missed invocations found in skill audit)
- **Skill**: Trimmed `search-copilot-chats` to domain knowledge only — removed sections now covered by the self-documenting CLI (`--help`, `--json`, actionable errors)
- **CLI**: Refactored `_ensure_db_exists` to eliminate duplicated try/except branches

## [0.7.1] - 2026-03-28

### Changed

- **Performance**: Replaced `orjson` with `ssrjson` (SIMD-accelerated JSON parser) — ~20% faster JSON parsing across all session sizes, benchmarked on 656 real sessions (1 GB)
- **Performance**: Switched file parsing from `ThreadPoolExecutor` to `ProcessPoolExecutor` — bypasses GIL for C-extension JSON parsers, enabling true multi-core parallelism (~10× speedup on full scan)
- **CLI**: Added `--workers` / `-w` flag to `scan` command to control parallel worker process count (default: `min(4, cores/2)`)

## [0.7.0] - 2026-03-27

### Added

- **Schema v6**: Promote subagents to first-class database rows via self-join on `cst_messages` (`parent_message_id`, `child_index`) and `cst_content_blocks.child_message_id`, replacing opaque `nested_data` JSON blobs
- **Schema**: SQL views `cst_messages_tree`, `cst_all_tool_invocations`, `cst_subagent_summary` for convenient querying of agent hierarchies
- **Performance**: `batch_connection()` context manager — reuses a single SQLite connection with `PRAGMA foreign_keys = OFF` and 64MB cache during bulk operations, fixing pre-existing bottlenecks exposed by full re-enrichment (1GB rebuild completes in ~25s)
- **Performance**: Parallel parsing with `ThreadPoolExecutor` (4 workers) — lazy iterator to bound memory, parse in parallel, write serially
- **Performance**: Bulk-fetch in `get_cst_session()` — 5 queries total instead of O(N×4), using in-memory tree assembly
- **Scanner**: Infer subagent completion from `tool.execution_complete` when `subagent.completed` event is absent (fixes all subagents showing "incomplete" in CLI sessions)
- **Storage**: `_delete_session_data()` helper for FK-safe explicit child table deletion (required since CASCADE doesn't fire with FK=OFF)
- **Storage**: Reentrancy guard on `batch_connection()` — raises `RuntimeError` on nested calls
- **Storage**: Per-session error handling in enrichment loop — single failure no longer rolls back entire batch

### Changed

- **Schema**: Drop/recreate migration from v5 to v6 (safe — `cst_*` tables are derived from source files)
- **Search**: FTS indexes all messages; search gates on `agent_nesting_level = 0` by default to avoid duplicate hits from child messages
- **Scanner**: Recursive `_insert_content_blocks_recursive()` handles arbitrary nesting depth for agent-inside-agent patterns
- **UI**: Tighter spacing for tool invocations, collapsibles, and message cards in web viewer

## [0.6.4] - 2026-03-24

### Changed

- **Architecture**: Split `database.py` monolith (2433L) into four focused modules aligned with pipeline stages:
  - `db_search.py` (767L) — FTS5 search, query parsing, result merging
  - `db_retrieval.py` (700L) — session reconstruction, listing, analytics, builtin accessors
  - `db_storage.py` (760L) — schema, ingestion, enrichment, discovery, cleanup
  - `database.py` (515L) — thin facade preserving full backward compatibility
- **Architecture**: Added `db_schema.py` — centralized column-name constants and dataclass↔SQL mapping helpers (`session_to_row`, `row_to_tool`, etc.) closing the schema gap between `models.py` and SQL
- **Architecture**: Decomposed `refresh.py` — extracted `_parse_files_parallel()`, `_classify_and_batch_write()`, `_enrich_session_batch()` from duplicated logic in `run_refresh()` and `run_enrichment()`

## [0.6.3] - 2026-03-23

### Changed

- **Skill**: Rewritten `search-copilot-chats` skill as hybrid read-help pattern
- **Docs**: Refreshed all README screenshots to current UI (PII-free, workspace-filtered)
- **Docs**: Added cleanup screenshot showing Cleanup/Revert All toolbar
- **Docs**: Backfilled CHANGELOG entries for v0.1.1 through v0.6.1

## [0.6.2] - 2026-03-15

### Added

- **Scanner**: Handle `session.title_changed` events — renders status block and sets session display title (highest priority over workspace.yaml and intent blocks)
- **Scanner**: Handle `assistant.usage` events — renders per-API-call token stats (model, input/output tokens, cost, duration, reasoning effort)
- **Scanner**: Extract `intentionSummary` and `toolTitle` from `assistant.message` tool requests for richer tool invocation display
- **Scanner**: Extract `hostType`, `headCommit`, `baseCommit` from `session.start` context
- **Scanner**: ADO sessions (`hostType=ado`) no longer receive incorrect GitHub repository URLs

### Changed

- **Scanner**: Expanded skip list with 22 new event types from CLI v1.0.5 schema (hooks, streaming deltas, plan mode lifecycle, agent selection, internal state tracking)

### Fixed

- **Scanner**: Guard against null token fields in `assistant.usage` events (prevents `TypeError` on nullable `Double` values from SDK)
- **Scanner**: `toolTitle` now correctly used for shell command descriptions (was previously ignored in `cmd_run` branch)

## [0.6.1] - 2026-03-12

### Fixed

- **Scanner**: Extract `reasoningText` from `assistant.message` events as thinking blocks (newer CLI versions embed thinking directly instead of separate `assistant.reasoning` events)

## [0.6.0] - 2026-03-09

### Added

- **Transcript Cleanup**: LLM-powered cleanup of voice-dictated user messages via LiteLLM (GitHub Copilot API)
  - Pure-Python heuristic pre-filter auto-detects garbled messages (repeated words, filler, missing punctuation)
  - Batch processing: all messages sent in a single LLM call per chunk (10 msgs/chunk) with assistant context
  - Original content preserved for revert — cleanup is fully reversible
  - Structured output via `response_format` JSON schema (OpenAI models) with prompt-based fallback
- **CLI**: `cleanup` command with `--dry-run`, `--all`, `--message`, `--force`, `--threshold` options
- **CLI**: `cleanup-revert` command to restore original voice-dictated content
- **Web**: Per-message Font Awesome mic icon with tristate color (gray/green/amber)
- **Web**: Popover with Clean/Toggle/Revert actions, no page reload (AJAX + scroll preservation)
- **Web**: Session-level Cleanup and Revert All toolbar buttons
- **Web**: POST API routes `/api/cleanup/<session_id>` and `/api/cleanup-revert/<session_id>`
- **Database**: Schema v5 migration — `original_content` and `cleanup_model` columns on `cst_messages`
- **Dependencies**: Optional `[llm]` extras group with `litellm>=1.50.0`
- **Benchmark**: `scripts/benchmark_cleanup.py` for comparing pre-filter strategies and LLM models

## [0.5.0] - 2026-03-08

### Added

- **Web**: Content-type filter pills on index page ('Search in:' pills for messages, tools, files, etc.)
- **Web**: Sticky unified toolbar with title, gear icon, and export buttons
- **Web**: Per-message sticky headers with role labels
- **Web**: Workspace/repository sorting by count, searchable repository dropdown
- **CLI**: Unified `--include`/`--exclude` flags on search command with 8 content types (replaces `--tools-only`, `--files-only`)

### Changed

- **Web**: Denser UI — 14px base font, compact headers, unified card padding
- **Web**: Scroll position preservation (CSS `overflow-anchor` + JS fallback)
- **Web**: Full rebuild parallelized with `ThreadPoolExecutor`

### Fixed

- **Database**: `enrichment_version` now correctly stamped for VS Code sessions
- **Scanner**: CLI sessions with missing `events.jsonl` no longer stuck in refresh loop
- **Web**: Zero-message sessions hidden from index page and banner
- **Web**: Tool/file search results now link to correct message anchors

## [0.4.4] - 2026-03-07

### Changed

- **Core**: Extract shared utility module (`utils.py`) — eliminates ~500 lines of duplicated code across exporters and webapp
- **Scanner**: Shared `scanner/shared.py` with `normalize_tool_status`, `extract_command_run`, `normalize_invocation_message`
- **Scanner**: Bump `PARSER_VERSION` to 2 (triggers re-parse of existing sessions)

### Fixed

- **Scanner**: Normalize VS Code tool status (`completed` → `success`, detect errors)
- **Scanner**: Extract `CommandRun` for terminal/shell tools in VS Code subagents
- **Scanner**: Populate block description with `toolId` fallback
- **Scanner**: Normalize invocation messages for built-in tools
- **Scanner**: Guard against null `modelMetrics` in `session.shutdown` handler
- **Scanner**: `ask_user` falls back to `message` field when `question` absent

## [0.4.3] - 2026-03-06

### Added

- **Scanner**: Handle `session.task_complete` events (renders task summary as collapsible block with markdown)
- **Scanner**: Handle `session.shutdown` events (renders session stats: shutdown type, code changes, model metrics)
- **Scanner**: Pretty-format handlers for 13 new CLI tools (`show_file`, `ask_user`, `skill`, `lsp`, `propose_work`, `list_agents`, `read_agent`, `exit_plan_mode`, `fetch_copilot_cli_documentation`, etc.)
- **CLI**: `migrate` subcommand for explicit database schema migration

### Changed

- **Scanner**: Added 10 new ephemeral event types to skip list (permission, elicitation, user_input, external_tool from CLI v0.0.422)

## [0.4.2] - 2026-03-06

### Added

- **CLI**: `--include-agent-details`/`--no-agent-details` flag for markdown export

### Fixed

- **Web**: Structured agent rendering with recursive nested content — internal tools as compact text, MCP tools as collapsible details, agent results in styled quote box
- **Scanner**: Skills render as expandable dropdowns with description

## [0.4.1] - 2026-02-28

### Added

- **Database**: `enrichment_version` column on `cst_sessions` (schema v3) for tracking which parser version enriched each session
- **Web**: Version-refresh banner on session page when enriched with older version
- **Web**: Global version-refresh banner on index page with count of stale sessions
- **Web**: PyPI upgrade check — dismissible banner when newer version available
- **CLI**: PyPI upgrade notice after `scan` and `web` commands (24-hour cached)
- **Web**: App version displayed in header

### Changed

- **Web**: Default title renamed from "Copilot Chat Archive" to "Copilot Session Tools"
- **Web**: Browser tab title uses session display name instead of workspace/session ID

## [0.4.0] - 2026-02-27

### Added

- **Scanner**: Sub-agent/background agent content rendered as collapsible `<details>` blocks with status pills (completed/failed)
- **Scanner**: CLI scanner tracks `subagent.started`/`completed` brackets via stack, tags messages with `agent_id` and `agent_display_name`
- **Scanner**: VS Code scanner detects `toolSpecificData.kind=='subagent'` and groups child tools under parent
- **Web**: Collapsible agent blocks with indigo left border and dark mode support
- **Markdown**: Agent messages rendered as blockquotes with group headers
- **Database**: Schema v2 — `agent_id`, `agent_display_name`, `agent_nesting_level` on `cst_messages`; `subagent_invocation_id` on `cst_tool_invocations`

## [0.3.1] - 2026-02-25

### Changed

- **Core**: Unified CLI and web refresh into shared `refresh.py` module — both now use identical code paths for scan, enrichment, and VS Code import

### Fixed

- **Scanner**: `parse_session_file()` now stamps scan-time mtime/size (not parse-time), preventing unnecessary re-imports
- **CLI**: Verbose output restored via `ProgressCallback`

## [0.3.0] - 2026-02-24

### Added

- **Database**: Two-tier rendering — sessions visible immediately from Copilot CLI's built-in `session-store.db`, enriched with `cst_*` tables on scan
- **CLI**: `enrich` command for single-session enrichment
- **Web**: Enrichment badge (green ✓ Enriched / gray Basic) on session list
- **Web**: "Scan Now" button on unenriched sessions for inline enrichment
- **Web**: Flash message rendering for enrich errors

### Changed

- **Database**: Default DB path changed to `~/.copilot/session-store.db` (Copilot CLI's own database)
- **Database**: All tables renamed to `cst_*` prefix — built-in tables are never modified
- **Database**: WAL mode + `busy_timeout=5000` for concurrent access
- **Scanner**: Added `PARSER_VERSION` tracking and `source_format` field

### Removed

- **CLI**: `rebuild`, `raw-json` commands (superseded by session store integration)
- **Database**: Raw compressed JSON blob storage (`raw_sessions` table)

## [0.2.0] - 2026-02-22

### Added

- **Scanner**: Handle 8 new CLI event types — `subagent.started`/`completed`/`failed`, `session.handoff`/`warning`/`mode_changed`/`context_changed`/`plan_changed`
- **Testing**: Snapshot test infrastructure with `pytest-regressions` golden-file baselines
- **Testing**: Real session fixtures (CLI, VS Code Insiders, VS Code Stable) replacing synthetic test data

### Changed

- **Scanner**: Added 5 internal event types to skip list

## [0.1.3] - 2026-02-14

### Fixed

- **CI/CD**: Auto-tag release trigger — `GITHUB_TOKEN` tags don't trigger workflows; now uses `gh workflow run`

## [0.1.2] - 2026-02-14

### Added

- **CI/CD**: Auto-tag on version bump, pre-push lint hook

## [0.1.1] - 2026-02-13

### Added

- **CI/CD**: PyPI publishing infrastructure

## [0.1.0] - 2025-02-13

### Added

- **Scanner**: Scan VS Code workspace storage (Stable and Insiders editions) to find Copilot chat sessions
- **Scanner**: GitHub Copilot CLI chat history support (JSONL format from `~/.copilot/session-state`)
- **Scanner**: Support for VS Code JSONL append-log format (VS Code >=1.109)
- **Database**: SQLite storage with FTS5 full-text search indexing
- **Database**: Two-layer design with raw compressed JSON as source of truth and derived tables
- **Database**: Incremental scan support (only imports new/changed sessions)
- **CLI**: `scan` command to import sessions from VS Code and CLI
- **CLI**: `search` command with advanced query syntax (field filters, exact phrases, boolean logic)
- **CLI**: `stats` command for database statistics
- **CLI**: `export` command for JSON export
- **CLI**: `export-markdown` command for Markdown export
- **CLI**: `export-html` command for self-contained HTML export
- **CLI**: `import-json` command for JSON import
- **CLI**: `rebuild` command to recreate derived tables from raw JSON
- **Web**: Flask-based web interface for browsing chat sessions
- **Web**: Full-text search with highlighting
- **Web**: Dark mode support via CSS `prefers-color-scheme`
- **Web**: Syntax highlighting for code blocks
- **Web**: Incremental refresh without restarting
- **Tracking**: Tool invocations, file changes, and command runs from chat sessions
