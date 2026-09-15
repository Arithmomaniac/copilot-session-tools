"""Schema constants and column mappings for the cst_* database tables.

Centralizes column names so that renames, additions, or reorderings
only need to change one file.  Both the write path (db_storage.py)
and read path (db_retrieval.py) reference these constants.
"""

from __future__ import annotations

import json
import sqlite3

from .scanner import ChatSession, CommandRun, FileChange, RootAgentInterval, SessionContextEntry, ToolInvocation

# ---------------------------------------------------------------------------
# Column tuples used in INSERT statements (order matters for parameterized queries)
# ---------------------------------------------------------------------------

CST_SESSION_COLUMNS = (
    "session_id",
    "workspace_name",
    "workspace_path",
    "created_at",
    "updated_at",
    "source_file",
    "vscode_edition",
    "custom_title",
    "requester_username",
    "responder_username",
    "source_file_mtime",
    "source_file_size",
    "type",
    "repository_url",
    "parser_version",
    "source_format",
    "enrichment_version",
    "builtin_turns",
    "source_id",
    "native_session_id",
)

CST_MESSAGE_COLUMNS = (
    "session_id",
    "message_index",
    "role",
    "content",
    "timestamp",
    "cached_markdown",
    "source_event_id",
    "agent_id",
    "agent_display_name",
    "agent_nesting_level",
    "original_content",
    "cleanup_model",
    "parent_message_id",
    "child_index",
)

CST_TOOL_INVOCATION_COLUMNS = (
    "message_id",
    "name",
    "input",
    "result",
    "status",
    "start_time",
    "end_time",
    "source_type",
    "invocation_message",
    "subagent_invocation_id",
    "is_agent_backlink",
    "backlink_agent_id",
    "is_shell_backlink",
    "backlink_shell_id",
)

CST_FILE_CHANGE_COLUMNS = (
    "message_id",
    "path",
    "diff",
    "content",
    "explanation",
    "language_id",
)

CST_COMMAND_RUN_COLUMNS = (
    "message_id",
    "command",
    "title",
    "result",
    "status",
    "output",
    "timestamp",
    "shell_id",
    "is_async",
    "is_detached",
)

CST_CONTENT_BLOCK_COLUMNS = (
    "message_id",
    "block_index",
    "kind",
    "content",
    "description",
    "child_message_id",
    "prompt",
    "is_background",
    "agent_id",
)

CST_ROOT_AGENT_INTERVAL_COLUMNS = (
    "session_id",
    "agent_name",
    "agent_display_name",
    "start_timestamp",
    "end_timestamp",
    "tools_json",
)

CST_SESSION_CONTEXT_COLUMNS = (
    "session_id",
    "context_index",
    "message_index",
    "timestamp",
    "workspace_name",
    "workspace_path",
    "repository_url",
    "branch",
    "source",
)

# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------


def insert_sql(table: str, columns: tuple[str, ...]) -> str:
    """Generate a parameterized INSERT statement from a table name and column tuple."""
    cols = ", ".join(columns)
    placeholders = ", ".join("?" * len(columns))
    return f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"  # noqa: S608


# ---------------------------------------------------------------------------
# Dataclass → row helpers  (write path)
# ---------------------------------------------------------------------------


def session_to_row(
    session: ChatSession,
    *,
    enrichment_version: str | None = None,
    updated_at_fallback: str | None = None,
    builtin_turns: int | None = None,
) -> tuple:
    """Map a ChatSession to a parameter tuple matching CST_SESSION_COLUMNS.

    *updated_at_fallback* is used when ``session.updated_at`` is falsy
    (e.g. the enrichment path fills it with a timestamp).

    *builtin_turns* records the Chronicle turn count at enrichment time so
    incremental scans can detect "this CLI session grew" with a like-for-like
    comparison (mirrors how VS Code sessions compare ``source_file_mtime``).
    """
    return (
        session.session_id,
        session.workspace_name,
        session.workspace_path,
        session.created_at,
        session.updated_at or updated_at_fallback,
        session.source_file,
        session.vscode_edition,
        session.custom_title,
        session.requester_username,
        session.responder_username,
        session.source_file_mtime,
        session.source_file_size,
        session.type,
        session.repository_url,
        session.parser_version,
        session.source_format,
        enrichment_version,
        builtin_turns,
        session.source_id,
        session.native_session_id,
    )


def tool_to_row(message_id: int, tool: ToolInvocation) -> tuple:
    """Map a ToolInvocation to a parameter tuple matching CST_TOOL_INVOCATION_COLUMNS."""
    return (
        message_id,
        tool.name,
        tool.input,
        tool.result,
        tool.status,
        tool.start_time,
        tool.end_time,
        tool.source_type,
        tool.invocation_message,
        tool.subagent_invocation_id,
        tool.is_agent_backlink,
        tool.backlink_agent_id,
        tool.is_shell_backlink,
        tool.backlink_shell_id,
    )


def file_change_to_row(message_id: int, change: FileChange) -> tuple:
    """Map a FileChange to a parameter tuple matching CST_FILE_CHANGE_COLUMNS."""
    return (
        message_id,
        change.path,
        change.diff,
        change.content,
        change.explanation,
        change.language_id,
    )


def command_to_row(message_id: int, cmd: CommandRun) -> tuple:
    """Map a CommandRun to a parameter tuple matching CST_COMMAND_RUN_COLUMNS."""
    return (
        message_id,
        cmd.command,
        cmd.title,
        cmd.result,
        cmd.status,
        cmd.output,
        cmd.timestamp,
        cmd.shell_id,
        cmd.is_async,
        cmd.is_detached,
    )


def root_agent_interval_to_row(session_id: str, interval: RootAgentInterval) -> tuple:
    """Map a RootAgentInterval to a parameter tuple matching CST_ROOT_AGENT_INTERVAL_COLUMNS."""
    tools_json = json.dumps(interval.tools) if interval.tools is not None else None
    return (
        session_id,
        interval.agent_name,
        interval.agent_display_name,
        interval.start_timestamp,
        interval.end_timestamp,
        tools_json,
    )


def session_context_to_row(session_id: str, context_index: int, entry: SessionContextEntry) -> tuple:
    """Map a SessionContextEntry to a parameter tuple matching CST_SESSION_CONTEXT_COLUMNS."""
    return (
        session_id,
        context_index,
        entry.message_index,
        entry.timestamp,
        entry.workspace_name,
        entry.workspace_path,
        entry.repository_url,
        entry.branch,
        entry.source,
    )


# ---------------------------------------------------------------------------
# Row → dataclass helpers  (read path)
# ---------------------------------------------------------------------------


def row_to_tool(row: sqlite3.Row) -> ToolInvocation:
    """Map a cst_tool_invocations row to a ToolInvocation."""
    keys = row.keys()
    return ToolInvocation(
        name=row["name"],
        input=row["input"],
        result=row["result"],
        status=row["status"],
        start_time=row["start_time"],
        end_time=row["end_time"],
        source_type=row["source_type"] if "source_type" in keys else None,
        invocation_message=row["invocation_message"] if "invocation_message" in keys else None,
        subagent_invocation_id=row["subagent_invocation_id"] if "subagent_invocation_id" in keys else None,
        is_agent_backlink=bool(row["is_agent_backlink"]) if "is_agent_backlink" in keys and row["is_agent_backlink"] else False,
        backlink_agent_id=row["backlink_agent_id"] if "backlink_agent_id" in keys else None,
        is_shell_backlink=bool(row["is_shell_backlink"]) if "is_shell_backlink" in keys and row["is_shell_backlink"] else False,
        backlink_shell_id=row["backlink_shell_id"] if "backlink_shell_id" in keys else None,
    )


def row_to_root_agent_interval(row: sqlite3.Row) -> RootAgentInterval:
    """Map a cst_root_agent_intervals row to a RootAgentInterval."""
    tools_json = row["tools_json"] if "tools_json" in row.keys() else None  # noqa: SIM118
    tools = None
    if tools_json:
        parsed_tools = json.loads(tools_json)
        if isinstance(parsed_tools, list):
            tools = [str(tool) for tool in parsed_tools]
    return RootAgentInterval(
        agent_name=row["agent_name"],
        agent_display_name=row["agent_display_name"],
        start_timestamp=row["start_timestamp"],
        end_timestamp=row["end_timestamp"],
        tools=tools,
    )


def row_to_session_context(row: sqlite3.Row) -> SessionContextEntry:
    """Map a cst_session_contexts row to a SessionContextEntry."""
    return SessionContextEntry(
        workspace_name=row["workspace_name"],
        workspace_path=row["workspace_path"],
        repository_url=row["repository_url"],
        branch=row["branch"],
        timestamp=row["timestamp"],
        message_index=row["message_index"],
        source=row["source"],
    )


def row_to_file_change(row: sqlite3.Row) -> FileChange:
    """Map a cst_file_changes row to a FileChange."""
    return FileChange(
        path=row["path"],
        diff=row["diff"],
        content=row["content"],
        explanation=row["explanation"],
        language_id=row["language_id"],
    )


def row_to_command(row: sqlite3.Row) -> CommandRun:
    """Map a cst_command_runs row to a CommandRun."""
    keys = row.keys()
    return CommandRun(
        command=row["command"],
        title=row["title"],
        result=row["result"],
        status=row["status"],
        output=row["output"],
        timestamp=row["timestamp"],
        shell_id=row["shell_id"] if "shell_id" in keys else None,
        is_async=bool(row["is_async"]) if "is_async" in keys and row["is_async"] else False,
        is_detached=bool(row["is_detached"]) if "is_detached" in keys and row["is_detached"] else False,
    )
