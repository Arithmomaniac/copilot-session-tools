"""Retrieval module for reading and reconstructing Copilot chat sessions.

Extracted from database.py to isolate read-path concerns:
- Session reconstruction from relational tables
- Session listing (cst_* + built-in deduplication)
- Message markdown rendering
- Analytics (stats, workspaces, repositories)
- Built-in table accessors
- FTS index optimization and JSON export
"""

from __future__ import annotations

import contextlib
import sqlite3

from .db_schema import row_to_command, row_to_file_change, row_to_root_agent_interval, row_to_session_context, row_to_tool
from .markdown_exporter import message_to_markdown
from .scanner import (
    ChatMessage,
    ChatSession,
    ContentBlock,
)


def _deserialize_content_block(row: sqlite3.Row) -> tuple[ContentBlock, int | None]:
    """Deserialize a content block row.

    Returns:
        Tuple of (ContentBlock, child_message_id or None).
        child_message is populated later during tree construction.
    """
    keys = row.keys()
    block = ContentBlock(
        kind=row["kind"],
        content=row["content"] or "",
        description=row["description"] if "description" in keys else None,
        prompt=row["prompt"] if "prompt" in keys else None,
        is_background=bool(row["is_background"]) if "is_background" in keys and row["is_background"] else False,
        agent_id=row["agent_id"] if "agent_id" in keys else None,
    )
    child_message_id = row["child_message_id"] if "child_message_id" in keys else None
    return block, child_message_id


def _link_child_messages(
    cursor: sqlite3.Cursor,
    parent_msg: ChatMessage,
    child_id_map: dict[int, int],
) -> None:
    """Recursively fetch and link child messages to parent content blocks."""
    for block_idx, child_msg_id in child_id_map.items():
        child_row = cursor.execute("SELECT * FROM cst_messages WHERE id = ?", (child_msg_id,)).fetchone()
        if child_row and block_idx < len(parent_msg.content_blocks):
            child_msg, _grandchild_map = reconstruct_message(
                cursor,
                child_msg_id,
                child_row,
                link_children=True,
            )
            parent_msg.content_blocks[block_idx].child_message = child_msg
            # Populate deprecated fields for backward compat
            parent_msg.content_blocks[block_idx].content_blocks = child_msg.content_blocks
            parent_msg.content_blocks[block_idx].tool_invocations = child_msg.tool_invocations
            parent_msg.content_blocks[block_idx].file_changes = child_msg.file_changes
            parent_msg.content_blocks[block_idx].command_runs = child_msg.command_runs
            parent_msg.children.append(child_msg)


def _session_contexts_for_ids(conn: sqlite3.Connection, session_ids: list[str]) -> dict[str, list[dict]]:
    """Return context-history dictionaries keyed by session id."""
    if not session_ids:
        return {}
    placeholders = ",".join("?" * len(session_ids))
    try:
        rows = conn.execute(
            f"""
            SELECT session_id, context_index, message_index, timestamp, workspace_name,
                   workspace_path, repository_url, branch, source
            FROM cst_session_contexts
            WHERE session_id IN ({placeholders})
            ORDER BY session_id, context_index
            """,  # noqa: S608
            session_ids,
        ).fetchall()
    except sqlite3.OperationalError:
        return {}

    contexts: dict[str, list[dict]] = {}
    for row in rows:
        contexts.setdefault(row["session_id"], []).append(dict(row))
    return contexts


def _apply_context_lists(session: dict, contexts: list[dict] | None) -> None:
    """Attach context history and aggregate workspace/repository lists to a session dict."""
    context_entries = contexts or []
    if not context_entries and (session.get("workspace_name") or session.get("workspace_path") or session.get("repository_url")):
        context_entries = [
            {
                "context_index": 0,
                "message_index": 0,
                "timestamp": session.get("created_at"),
                "workspace_name": session.get("workspace_name"),
                "workspace_path": session.get("workspace_path"),
                "repository_url": session.get("repository_url"),
                "branch": None,
                "source": "initial",
            }
        ]
    session["context_entries"] = context_entries
    session["workspace_names"] = list(dict.fromkeys(c["workspace_name"] for c in context_entries if c.get("workspace_name")))
    session["workspace_paths"] = list(dict.fromkeys(c["workspace_path"] for c in context_entries if c.get("workspace_path")))
    session["repository_urls"] = list(dict.fromkeys(c["repository_url"] for c in context_entries if c.get("repository_url")))


def reconstruct_message(
    cursor: sqlite3.Cursor,
    message_id: int,
    msg_row: sqlite3.Row,
    *,
    link_children: bool = False,
) -> tuple[ChatMessage, dict[int, int]]:
    """Reconstruct a ChatMessage from database rows by querying related tables.

    Args:
        cursor: SQLite cursor for querying related tables.
        message_id: The cst_messages.id of the message.
        msg_row: The cst_messages row for this message.
        link_children: If True, recursively fetch and link child messages
            to content blocks. Useful for callers that need the full tree
            without doing their own tree assembly.

    Returns:
        Tuple of (ChatMessage, child_id_map) where child_id_map is
        {block_index: child_message_id} for content blocks that reference
        child messages.
    """
    # Query tool_invocations for this message
    cursor.execute("SELECT * FROM cst_tool_invocations WHERE message_id = ?", (message_id,))
    tool_invocations = [row_to_tool(t) for t in cursor.fetchall()]

    # Query file_changes
    cursor.execute("SELECT * FROM cst_file_changes WHERE message_id = ?", (message_id,))
    file_changes = [row_to_file_change(f) for f in cursor.fetchall()]

    # Query command_runs
    cursor.execute("SELECT * FROM cst_command_runs WHERE message_id = ?", (message_id,))
    command_runs = [row_to_command(c) for c in cursor.fetchall()]

    # Query content_blocks — collect child_message_id mappings
    cursor.execute("SELECT * FROM cst_content_blocks WHERE message_id = ? ORDER BY block_index", (message_id,))
    content_blocks: list[ContentBlock] = []
    child_id_map: dict[int, int] = {}
    for idx, b in enumerate(cursor.fetchall()):
        block, child_msg_id = _deserialize_content_block(b)
        content_blocks.append(block)
        if child_msg_id is not None:
            child_id_map[idx] = child_msg_id

    # Get cached_markdown safely
    cached_md = msg_row["cached_markdown"] if "cached_markdown" in msg_row.keys() else None  # noqa: SIM118

    msg = ChatMessage(
        role=msg_row["role"],
        content=msg_row["content"],
        timestamp=msg_row["timestamp"],
        tool_invocations=tool_invocations,
        file_changes=file_changes,
        command_runs=command_runs,
        content_blocks=content_blocks,
        cached_markdown=cached_md,
        source_event_id=msg_row["source_event_id"] if "source_event_id" in msg_row.keys() else None,  # noqa: SIM118
        agent_id=msg_row["agent_id"] if "agent_id" in msg_row.keys() else None,  # noqa: SIM118
        agent_display_name=msg_row["agent_display_name"] if "agent_display_name" in msg_row.keys() else None,  # noqa: SIM118
        agent_nesting_level=msg_row["agent_nesting_level"] if "agent_nesting_level" in msg_row.keys() else 0,  # noqa: SIM118
        original_content=msg_row["original_content"] if "original_content" in msg_row.keys() else None,  # noqa: SIM118
        cleanup_model=msg_row["cleanup_model"] if "cleanup_model" in msg_row.keys() else None,  # noqa: SIM118
    )
    if link_children and child_id_map:
        _link_child_messages(cursor, msg, child_id_map)
    return msg, child_id_map


def get_cst_session(conn: sqlite3.Connection, session_id: str) -> ChatSession | None:
    """Get a session from cst_* tables by its ID.

    Loads ALL messages and artifacts in bulk (5 queries total regardless
    of message count), then builds the parent→child tree in memory.

    Args:
        conn: SQLite connection.
        session_id: The session ID to look up.

    Returns:
        ChatSession if found, None otherwise.
    """
    from collections import defaultdict

    cursor = conn.cursor()

    cursor.execute("SELECT * FROM cst_sessions WHERE session_id = ?", (session_id,))
    row = cursor.fetchone()
    if not row:
        return None

    # --- Bulk-fetch all data in 5 flat queries (O(1), not O(N)) ---

    # 1) Messages
    cursor.execute(
        "SELECT * FROM cst_messages WHERE session_id = ? ORDER BY message_index, child_index",
        (session_id,),
    )
    msg_rows = cursor.fetchall()
    msg_ids = [r["id"] for r in msg_rows]

    if not msg_ids:
        # Edge case: session exists but has no messages
        msg_ids_placeholder = "(NULL)"
    else:
        msg_ids_placeholder = f"({','.join(str(i) for i in msg_ids)})"

    # 2) Tool invocations (bulk)
    tools_by_msg: dict[int, list] = defaultdict(list)
    for t in cursor.execute(f"SELECT * FROM cst_tool_invocations WHERE message_id IN {msg_ids_placeholder}").fetchall():  # noqa: S608
        tools_by_msg[t["message_id"]].append(row_to_tool(t))

    # 3) File changes (bulk)
    files_by_msg: dict[int, list] = defaultdict(list)
    for f in cursor.execute(f"SELECT * FROM cst_file_changes WHERE message_id IN {msg_ids_placeholder}").fetchall():  # noqa: S608
        files_by_msg[f["message_id"]].append(row_to_file_change(f))

    # 4) Command runs (bulk)
    cmds_by_msg: dict[int, list] = defaultdict(list)
    for c in cursor.execute(f"SELECT * FROM cst_command_runs WHERE message_id IN {msg_ids_placeholder}").fetchall():  # noqa: S608
        cmds_by_msg[c["message_id"]].append(row_to_command(c))

    # 5) Content blocks (bulk, ordered)
    blocks_by_msg: dict[int, list[tuple[ContentBlock, int | None]]] = defaultdict(list)
    for b in cursor.execute(
        f"SELECT * FROM cst_content_blocks WHERE message_id IN {msg_ids_placeholder} ORDER BY message_id, block_index"  # noqa: S608
    ).fetchall():
        block, child_msg_id = _deserialize_content_block(b)
        blocks_by_msg[b["message_id"]].append((block, child_msg_id))

    root_agent_intervals = []
    with contextlib.suppress(sqlite3.OperationalError):
        cursor.execute(
            "SELECT * FROM cst_root_agent_intervals WHERE session_id = ? ORDER BY start_timestamp, id",
            (session_id,),
        )
        root_agent_intervals = [row_to_root_agent_interval(r) for r in cursor.fetchall()]

    context_entries = []
    with contextlib.suppress(sqlite3.OperationalError):
        cursor.execute(
            "SELECT * FROM cst_session_contexts WHERE session_id = ? ORDER BY context_index",
            (session_id,),
        )
        context_entries = [row_to_session_context(r) for r in cursor.fetchall()]

    # --- Reconstruct messages from pre-fetched data ---
    messages_by_id: dict[int, ChatMessage] = {}
    content_block_child_ids: dict[int, dict[int, int]] = {}

    for msg_row in msg_rows:
        msg_id = msg_row["id"]
        content_blocks: list[ContentBlock] = []
        child_id_map: dict[int, int] = {}
        for idx, (block, child_msg_id) in enumerate(blocks_by_msg.get(msg_id, [])):
            content_blocks.append(block)
            if child_msg_id is not None:
                child_id_map[idx] = child_msg_id

        cached_md = msg_row["cached_markdown"] if "cached_markdown" in msg_row.keys() else None  # noqa: SIM118

        msg = ChatMessage(
            role=msg_row["role"],
            content=msg_row["content"],
            timestamp=msg_row["timestamp"],
            tool_invocations=tools_by_msg.get(msg_id, []),
            file_changes=files_by_msg.get(msg_id, []),
            command_runs=cmds_by_msg.get(msg_id, []),
            content_blocks=content_blocks,
            cached_markdown=cached_md,
            source_event_id=msg_row["source_event_id"] if "source_event_id" in msg_row.keys() else None,  # noqa: SIM118
            agent_id=msg_row["agent_id"] if "agent_id" in msg_row.keys() else None,  # noqa: SIM118
            agent_display_name=msg_row["agent_display_name"] if "agent_display_name" in msg_row.keys() else None,  # noqa: SIM118
            agent_nesting_level=msg_row["agent_nesting_level"] if "agent_nesting_level" in msg_row.keys() else 0,  # noqa: SIM118
            original_content=msg_row["original_content"] if "original_content" in msg_row.keys() else None,  # noqa: SIM118
            cleanup_model=msg_row["cleanup_model"] if "cleanup_model" in msg_row.keys() else None,  # noqa: SIM118
        )
        messages_by_id[msg_id] = msg
        if child_id_map:
            content_block_child_ids[msg_id] = child_id_map

    # Build tree: link children to parents
    for msg_row in msg_rows:
        msg_id = msg_row["id"]
        parent_id = msg_row["parent_message_id"] if "parent_message_id" in msg_row.keys() else None  # noqa: SIM118
        if parent_id and parent_id in messages_by_id:
            messages_by_id[parent_id].children.append(messages_by_id[msg_id])

    # Link content blocks to child messages
    for parent_msg_id, block_map in content_block_child_ids.items():
        parent_msg = messages_by_id.get(parent_msg_id)
        if not parent_msg:
            continue
        for block_idx, child_msg_id in block_map.items():
            child_msg = messages_by_id.get(child_msg_id)
            if child_msg and block_idx < len(parent_msg.content_blocks):
                parent_msg.content_blocks[block_idx].child_message = child_msg
                # Populate deprecated fields for backward compat
                parent_msg.content_blocks[block_idx].content_blocks = child_msg.content_blocks
                parent_msg.content_blocks[block_idx].tool_invocations = child_msg.tool_invocations
                parent_msg.content_blocks[block_idx].file_changes = child_msg.file_changes
                parent_msg.content_blocks[block_idx].command_runs = child_msg.command_runs

    # Collect top-level messages only (no parent) in insertion order
    top_level_ordered: list[ChatMessage] = []
    seen: set[int] = set()
    for msg_row in msg_rows:
        parent_id = msg_row["parent_message_id"] if "parent_message_id" in msg_row.keys() else None  # noqa: SIM118
        msg_id = msg_row["id"]
        if parent_id is None and msg_id not in seen:
            seen.add(msg_id)
            top_level_ordered.append(messages_by_id[msg_id])

    # Helper to safely get optional fields from sqlite3.Row
    def safe_get(key):
        try:
            return row[key]
        except (IndexError, KeyError):
            return None

    return ChatSession(
        session_id=row["session_id"],
        workspace_name=row["workspace_name"],
        workspace_path=row["workspace_path"],
        messages=top_level_ordered,
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        source_file=row["source_file"],
        vscode_edition=row["vscode_edition"],
        custom_title=safe_get("custom_title"),
        requester_username=safe_get("requester_username"),
        responder_username=safe_get("responder_username"),
        source_file_mtime=safe_get("source_file_mtime"),
        source_file_size=safe_get("source_file_size"),
        type=safe_get("type") or "vscode",
        repository_url=safe_get("repository_url"),
        root_agent_intervals=root_agent_intervals,
        context_entries=context_entries,
        source_id=safe_get("source_id"),
        native_session_id=safe_get("native_session_id"),
    )


def get_builtin_session(conn: sqlite3.Connection, session_id: str) -> dict | None:
    """Read a session from the Chronicle sessions table.

    Expects Chronicle to be ATTACHed as ``chronicle`` schema.

    Args:
        conn: SQLite connection (with chronicle attached).
        session_id: The session ID to look up.

    Returns:
        Dict with session fields, or None.
    """
    try:
        row = conn.execute(
            "SELECT id, cwd, repository, branch, summary, created_at, updated_at FROM chronicle.sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None
    except sqlite3.OperationalError:
        return None


def get_builtin_turns(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """Read turns from the Chronicle turns table for a session.

    Expects Chronicle to be ATTACHed as ``chronicle`` schema.

    Args:
        conn: SQLite connection (with chronicle attached).
        session_id: The session ID.

    Returns:
        List of turn dicts.
    """
    try:
        rows = conn.execute(
            "SELECT turn_index, user_message, assistant_response FROM chronicle.turns WHERE session_id = ? ORDER BY turn_index",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def get_builtin_checkpoints(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """Read checkpoints from the Chronicle checkpoints table.

    Expects Chronicle to be ATTACHed as ``chronicle`` schema.
    """
    try:
        rows = conn.execute(
            "SELECT checkpoint_number, title, overview, history, work_done, technical_details, "
            "important_files, next_steps FROM chronicle.checkpoints WHERE session_id = ? ORDER BY checkpoint_number",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def get_builtin_files(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """Read file references from the Chronicle session_files table.

    Expects Chronicle to be ATTACHed as ``chronicle`` schema.
    """
    try:
        rows = conn.execute(
            "SELECT file_path, tool_name, turn_index, first_seen_at FROM chronicle.session_files WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def get_builtin_refs(conn: sqlite3.Connection, session_id: str) -> list[dict]:
    """Read refs from the Chronicle session_refs table.

    Expects Chronicle to be ATTACHed as ``chronicle`` schema.
    """
    try:
        rows = conn.execute(
            "SELECT ref_type, ref_value, turn_index, created_at FROM chronicle.session_refs WHERE session_id = ?",
            (session_id,),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def list_builtin_sessions(
    conn: sqlite3.Connection,
    limit: int = 100,
    offset: int = 0,
    *,
    nonempty_only: bool = False,
) -> list[dict]:
    """List sessions from the Chronicle sessions table.

    Expects Chronicle to be ATTACHed as ``chronicle`` schema.
    """
    try:
        nonempty_filter = (
            """
            WHERE EXISTS (
                SELECT 1
                FROM chronicle.turns t
                WHERE t.session_id = chronicle.sessions.id
                  AND (
                      (t.user_message IS NOT NULL AND t.user_message != '')
                      OR (t.assistant_response IS NOT NULL AND t.assistant_response != '')
                  )
            )
        """
            if nonempty_only
            else ""
        )
        rows = conn.execute(
            f"""
            SELECT id, cwd, repository, branch, summary, created_at, updated_at
            FROM chronicle.sessions
            {nonempty_filter}
            ORDER BY updated_at DESC
            LIMIT ? OFFSET ?
            """,  # noqa: S608
            (limit, offset),
        ).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError:
        return []


def count_builtin_turns(conn: sqlite3.Connection, session_id: str) -> int:
    """Count turns for a session in the Chronicle turns table.

    Expects Chronicle to be ATTACHed as ``chronicle`` schema.
    """
    try:
        row = conn.execute(
            "SELECT COUNT(*) FROM chronicle.turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        return row[0] if row else 0
    except sqlite3.OperationalError:
        return 0


def get_builtin_session_as_chat_session(conn: sqlite3.Connection, session_id: str) -> ChatSession | None:
    """Convert Chronicle session/turns data to a ChatSession.

    Expects Chronicle to be ATTACHed as ``chronicle`` schema.

    Args:
        conn: SQLite connection.
        session_id: The session ID to look up in built-in tables.

    Returns:
        ChatSession if found, None otherwise.
    """
    session_data = get_builtin_session(conn, session_id)
    if not session_data:
        return None

    turns = get_builtin_turns(conn, session_id)
    messages: list[ChatMessage] = []
    for turn in turns:
        user_msg = turn.get("user_message")
        if user_msg:
            messages.append(ChatMessage(role="user", content=user_msg))
        assistant_msg = turn.get("assistant_response")
        if assistant_msg:
            messages.append(ChatMessage(role="assistant", content=assistant_msg))

    return ChatSession(
        session_id=session_id,
        workspace_name=session_data.get("repository"),
        workspace_path=session_data.get("cwd"),
        messages=messages,
        created_at=session_data.get("created_at"),
        updated_at=session_data.get("updated_at"),
        vscode_edition="cli",
        custom_title=session_data.get("summary"),
        type="cli",
        repository_url=session_data.get("repository"),
        source_format="cli",
    )


def get_all_session_ids(conn: sqlite3.Connection) -> list[str]:
    """Get all session IDs from cst_sessions.

    Args:
        conn: SQLite connection.

    Returns:
        List of session ID strings.
    """
    rows = conn.execute("SELECT session_id FROM cst_sessions").fetchall()
    return [r[0] for r in rows]


def get_messages_markdown(
    conn: sqlite3.Connection,
    session_id: str,
    start: int | None = None,
    end: int | None = None,
    content_set: set[str] | None = None,
) -> str:
    """Get markdown for specific messages or all messages in a session.

    Args:
        conn: SQLite connection.
        session_id: The session ID to get messages from.
        start: Optional 1-based start message index (inclusive).
        end: Optional 1-based end message index (inclusive).
        content_set: Controls which content types to include.
            If None, uses DEFAULT_INCLUDES from content_types module.

    Returns:
        Combined markdown string for the selected messages.
    """
    from .content_types import DEFAULT_INCLUDES

    if content_set is None:
        content_set = DEFAULT_INCLUDES.copy()

    cursor = conn.cursor()

    # Build query based on range (exclude child messages — they render through parents)
    if start is not None or end is not None:
        # Convert to 0-based indices
        start_idx = (start - 1) if start else 0
        end_idx = (end - 1) if end else 999999  # Large number for "no limit"

        cursor.execute(
            """
            SELECT id, role, content, timestamp, cached_markdown, message_index
            FROM cst_messages
            WHERE session_id = ? AND message_index >= ? AND message_index <= ?
                AND parent_message_id IS NULL
            ORDER BY message_index
            """,
            (session_id, start_idx, end_idx),
        )
    else:
        cursor.execute(
            """
            SELECT id, role, content, timestamp, cached_markdown, message_index
            FROM cst_messages
            WHERE session_id = ? AND parent_message_id IS NULL
            ORDER BY message_index
            """,
            (session_id,),
        )

    rows = cursor.fetchall()
    markdown_parts = []

    # Cached markdown was generated with diffs+tool-inputs ON, thinking OFF,
    # agent-details+tools+commands+file-changes ON.  Use it when the requested content_set matches.
    cache_set = {"diffs", "tool-inputs", "agent-details", "tools", "commands", "file-changes"}
    if content_set == cache_set:
        for row in rows:
            md = row["cached_markdown"]
            if md:
                markdown_parts.append(md)
    else:
        # Need to regenerate markdown with specific options
        for row in rows:
            message_id = row["id"]
            message_index = row["message_index"] + 1  # Convert to 1-based

            # Create message object with child messages linked
            message, _ = reconstruct_message(cursor, message_id, row, link_children=True)

            include_diffs = "diffs" in content_set
            include_tool_inputs = "tool-inputs" in content_set
            include_thinking = "thinking" in content_set
            include_agent_details = "agent-details" in content_set
            include_tools = "tools" in content_set
            include_commands = "commands" in content_set
            include_file_changes = "file-changes" in content_set

            # Generate markdown with specified options
            md = message_to_markdown(
                message,
                message_number=message_index,
                include_diffs=include_diffs,
                include_tool_inputs=include_tool_inputs,
                include_thinking=include_thinking,
                include_agent_details=include_agent_details,
                include_tools=include_tools,
                include_commands=include_commands,
                include_file_changes=include_file_changes,
            )
            markdown_parts.append(md)

    return "\n".join(markdown_parts)


def list_sessions(
    conn: sqlite3.Connection,
    *,
    has_cst: bool,
    has_chronicle: bool = False,
    workspace_name: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    session_type: str | None = None,
) -> list[dict]:
    """List sessions from cst_* tables and (optionally) Chronicle.

    Returns dicts with at minimum: session_id, title, session_type, start_time,
    updated_at, is_enriched, source.  Also includes workspace_name, workspace_path,
    vscode_edition, custom_title, repository_url, message_count, last_message_at,
    first_user_prompt for backward compatibility with cst-sourced rows.

    Deduplicates by session_id — cst_sessions takes precedence over Chronicle.

    Args:
        conn: SQLite connection (with chronicle optionally attached).
        has_cst: Whether cst_* tables exist and should be queried.
        has_chronicle: Whether Chronicle is ATTACHed as ``chronicle`` schema.
        workspace_name: Optional workspace name filter (cst rows only).
        limit: Maximum number of sessions to return.
        offset: Number of sessions to skip.
        session_type: Optional filter: 'cli', 'vscode', etc.

    Returns:
        List of session info dictionaries sorted by updated_at descending.
    """
    results: dict[str, dict] = {}

    # 1. Read from cst_sessions if available
    if has_cst:
        cursor = conn.cursor()

        query = """
            SELECT
                s.session_id,
                s.workspace_name,
                s.workspace_path,
                s.created_at,
                s.updated_at,
                s.vscode_edition,
                s.custom_title,
                s.repository_url,
                s.type as session_type,
                s.source_format,
                s.source_id,
                s.native_session_id,
                src.name AS source_name,
                COUNT(CASE WHEN m.parent_message_id IS NULL THEN 1 END) as message_count,
                MAX(m.timestamp) as last_message_at,
                (SELECT content FROM cst_messages m2
                 WHERE m2.session_id = s.session_id AND m2.role = 'user'
                 ORDER BY m2.message_index LIMIT 1) as first_user_prompt
            FROM cst_sessions s
            LEFT JOIN cst_messages m ON s.session_id = m.session_id
            LEFT JOIN cst_sources src ON src.source_id = s.source_id
        """
        conditions = []
        params: list = []

        if workspace_name:
            conditions.append(
                """(
                    s.workspace_name = ?
                    OR EXISTS (
                        SELECT 1 FROM cst_session_contexts sc
                        WHERE sc.session_id = s.session_id AND sc.workspace_name = ?
                    )
                )"""
            )
            params.extend([workspace_name, workspace_name])
        if session_type:
            conditions.append("s.type = ?")
            params.append(session_type)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " GROUP BY s.session_id HAVING COUNT(CASE WHEN m.parent_message_id IS NULL THEN 1 END) > 0 ORDER BY COALESCE(s.updated_at, MAX(m.timestamp), s.created_at) DESC"
        candidate_limit = offset + limit if limit is not None else None
        if candidate_limit is not None:
            query += " LIMIT ?"
            params.append(candidate_limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        contexts_by_session = _session_contexts_for_ids(conn, [row["session_id"] for row in rows])
        for row in rows:
            d = dict(row)
            _apply_context_lists(d, contexts_by_session.get(d["session_id"]))
            d["title"] = d.get("custom_title") or d.get("workspace_name")
            d["start_time"] = d.get("created_at")
            d["is_enriched"] = True
            d["source"] = "cst"
            if not d.get("session_type"):
                d["session_type"] = d.get("source_format") or "vscode"
            results[d["session_id"]] = d

    # 2. Read from Chronicle sessions (cli type only) if Chronicle is available
    if has_chronicle and (session_type is None or session_type == "cli"):
        builtin = list_builtin_sessions(
            conn,
            limit=offset + limit if limit is not None else 10000,
            nonempty_only=True,
        )
        # Batch-query turn counts for unenriched sessions
        unenriched_sids = [row["id"] for row in builtin if row["id"] not in results]
        turn_counts: dict[str, int] = {}
        if unenriched_sids:
            try:
                placeholders = ",".join("?" * len(unenriched_sids))
                rows = conn.execute(
                    f"SELECT session_id, SUM((user_message IS NOT NULL AND user_message != '') + (assistant_response IS NOT NULL AND assistant_response != '')) "  # noqa: S608
                    f"FROM chronicle.turns WHERE session_id IN ({placeholders}) GROUP BY session_id",
                    unenriched_sids,
                ).fetchall()
                turn_counts = dict(rows)
            except sqlite3.OperationalError:
                pass
        for row in builtin:
            sid = row["id"]
            if sid not in results:  # cst_sessions takes precedence
                tc = turn_counts.get(sid, 0)
                if tc == 0:
                    continue  # Skip empty sessions
                results[sid] = {
                    "session_id": sid,
                    "title": row.get("summary"),
                    "session_type": "cli",
                    "start_time": row.get("created_at"),
                    "updated_at": row.get("updated_at"),
                    "is_enriched": False,
                    "source": "builtin",
                    # Backward-compat fields
                    "workspace_name": row.get("repository"),
                    "workspace_path": row.get("cwd"),
                    "created_at": row.get("created_at"),
                    "vscode_edition": "cli",
                    "custom_title": row.get("summary"),
                    "repository_url": row.get("repository"),
                    "message_count": tc,  # Actual non-null message count
                    "last_message_at": row.get("updated_at"),
                    "first_user_prompt": None,
                }
                _apply_context_lists(results[sid], None)

    # Sort by updated_at desc, apply limit/offset
    sorted_results = sorted(results.values(), key=lambda x: x.get("updated_at") or "", reverse=True)
    effective_limit = limit if limit is not None else len(sorted_results)
    return sorted_results[offset : offset + effective_limit]


def get_workspaces(conn: sqlite3.Connection) -> list[dict]:
    """Get all unique workspaces with activity stats.

    Args:
        conn: SQLite connection.

    Returns:
        List of workspace info dictionaries.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            workspace_name,
            workspace_path,
            COUNT(DISTINCT session_id) as session_count,
            MAX(created_at) as last_activity
        FROM (
            SELECT sc.session_id, sc.workspace_name, sc.workspace_path, s.created_at
            FROM cst_session_contexts sc
            JOIN cst_sessions s ON s.session_id = sc.session_id
            WHERE sc.workspace_name IS NOT NULL
            UNION ALL
            SELECT session_id, workspace_name, workspace_path, created_at
            FROM cst_sessions
            WHERE workspace_name IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM cst_session_contexts sc
                  WHERE sc.session_id = cst_sessions.session_id
              )
        )
        WHERE workspace_name IS NOT NULL
        GROUP BY workspace_name, workspace_path
        ORDER BY session_count DESC, workspace_name ASC
        """
    )
    return [dict(row) for row in cursor.fetchall()]


def get_repositories(conn: sqlite3.Connection) -> list[dict]:
    """Get all unique repositories with activity stats.

    Args:
        conn: SQLite connection.

    Returns:
        List of repository info dictionaries.
    """
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            repository_url,
            COUNT(DISTINCT session_id) as session_count,
            MAX(created_at) as last_activity
        FROM (
            SELECT sc.session_id, sc.repository_url, s.created_at
            FROM cst_session_contexts sc
            JOIN cst_sessions s ON s.session_id = sc.session_id
            WHERE sc.repository_url IS NOT NULL
            UNION ALL
            SELECT session_id, repository_url, created_at
            FROM cst_sessions
            WHERE repository_url IS NOT NULL
              AND NOT EXISTS (
                  SELECT 1 FROM cst_session_contexts sc
                  WHERE sc.session_id = cst_sessions.session_id
              )
        )
        WHERE repository_url IS NOT NULL
        GROUP BY repository_url
        ORDER BY session_count DESC, repository_url ASC
        """
    )
    return [dict(row) for row in cursor.fetchall()]


def get_stats(conn: sqlite3.Connection, *, has_cst: bool, has_chronicle: bool = False) -> dict:
    """Get database statistics.

    Args:
        conn: SQLite connection (with chronicle optionally attached).
        has_cst: Whether cst_* tables exist.
        has_chronicle: Whether Chronicle is ATTACHed as ``chronicle`` schema.

    Returns:
        Dictionary with stats (combines enriched cst_* and Chronicle counts).
    """
    cursor = conn.cursor()

    cst_session_count = 0
    cst_message_count = 0
    workspace_count = 0
    editions: dict = {}

    if has_cst:
        cursor.execute("SELECT COUNT(*) FROM cst_sessions")
        cst_session_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM cst_messages WHERE parent_message_id IS NULL")
        cst_message_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(DISTINCT workspace_name) FROM cst_sessions")
        workspace_count = cursor.fetchone()[0]

        cursor.execute("SELECT vscode_edition, COUNT(*) FROM cst_sessions GROUP BY vscode_edition")
        editions = dict(cursor.fetchall())

    # Count Chronicle sessions and turns not in cst_*
    builtin_only_count = 0
    builtin_message_count = 0
    if has_chronicle:
        try:
            if has_cst:
                cursor.execute("SELECT COUNT(*) FROM chronicle.sessions WHERE id NOT IN (SELECT session_id FROM cst_sessions)")
                builtin_only_count = cursor.fetchone()[0]
                cursor.execute(
                    "SELECT SUM((user_message IS NOT NULL AND user_message != '') + (assistant_response IS NOT NULL AND assistant_response != '')) "
                    "FROM chronicle.turns WHERE session_id NOT IN (SELECT session_id FROM cst_sessions)"
                )
                row = cursor.fetchone()
                builtin_message_count = row[0] or 0
            else:
                cursor.execute("SELECT COUNT(*) FROM chronicle.sessions")
                builtin_only_count = cursor.fetchone()[0]
                cursor.execute("SELECT SUM((user_message IS NOT NULL AND user_message != '') + (assistant_response IS NOT NULL AND assistant_response != '')) FROM chronicle.turns")
                row = cursor.fetchone()
                builtin_message_count = row[0] or 0
        except Exception:  # noqa: S110
            pass

    # Each built-in turn may have user, assistant, or both
    total_message_count = cst_message_count + builtin_message_count

    # Include unenriched CLI sessions in the cli edition count
    if builtin_only_count > 0:
        editions["cli"] = editions.get("cli", 0) + builtin_only_count

    return {
        "session_count": cst_session_count + builtin_only_count,
        "message_count": total_message_count,
        "workspace_count": workspace_count,
        "editions": editions,
        "enriched_count": cst_session_count,
        "unenriched_count": builtin_only_count,
    }


def optimize_fts(conn: sqlite3.Connection) -> dict:
    """Optimize the FTS5 full-text search index for better query performance.

    This merges FTS index segments, reducing fragmentation and improving
    search speed. Should be run periodically, especially after bulk imports.

    Args:
        conn: SQLite connection.

    Returns:
        Dictionary with optimization results including segment counts before/after.
    """
    cursor = conn.cursor()

    # Get segment count before optimization
    cursor.execute("SELECT COUNT(*) FROM cst_messages_fts_data")
    segments_before = cursor.fetchone()[0]

    # Run FTS5 optimize command - merges all segments into one
    cursor.execute("INSERT INTO cst_messages_fts(cst_messages_fts) VALUES('optimize')")

    # Get segment count after optimization
    cursor.execute("SELECT COUNT(*) FROM cst_messages_fts_data")
    segments_after = cursor.fetchone()[0]

    # Also run integrity check
    cursor.execute("INSERT INTO cst_messages_fts(cst_messages_fts) VALUES('integrity-check')")

    return {
        "segments_before": segments_before,
        "segments_after": segments_after,
        "optimized": True,
    }
