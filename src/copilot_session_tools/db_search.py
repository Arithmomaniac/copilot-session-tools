"""Search module for full-text search across Copilot chat sessions.

Extracted from database.py to isolate search concerns:
- Query parsing (field filters, FTS5 syntax)
- SQL query construction for each content type
- Result merging (cst_* + built-in search_index)
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass


@dataclass
class ParsedQuery:
    """Represents a parsed search query with extracted field filters."""

    fts_query: str  # The FTS5 query string for content search
    role: str | None = None  # Extracted role filter (user/assistant)
    workspace: str | None = None  # Extracted workspace filter
    workspaces: list[str] | None = None  # All extracted workspace filters
    title: str | None = None  # Extracted title filter
    repository: str | None = None  # Extracted repository filter
    repositories: list[str] | None = None  # All extracted repository filters
    edition: str | None = None  # Extracted edition filter (stable/insider/cli)
    start_date: str | None = None  # Extracted start date filter (yyyy-mm-dd format, inclusive)
    end_date: str | None = None  # Extracted end date filter (yyyy-mm-dd format, inclusive)


def _validate_date_format(date_str: str) -> str | None:
    """Validate date string is in yyyy-mm-dd format.

    Args:
        date_str: Date string to validate.

    Returns:
        The validated date string if valid, None otherwise.
    """
    if not date_str:
        return None
    # Check format: yyyy-mm-dd
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date_str):
        return None
    # Basic validation of month/day ranges
    try:
        _year, month, day = map(int, date_str.split("-"))
        if month < 1 or month > 12 or day < 1 or day > 31:
            return None
        return date_str
    except (ValueError, AttributeError):
        return None


def _escape_fts5_token(token: str) -> str:
    """Escape a single FTS5 token to prevent syntax errors.

    FTS5 has special operators like:
    - Dash (-) which means NOT
    - Colon (:) for column specification
    - Parentheses, brackets for grouping

    If a token is already quoted, leave it as-is.
    Otherwise, wrap it in quotes if it contains special characters.

    Args:
        token: A single search token (word or phrase).

    Returns:
        The escaped token, safe for FTS5 MATCH queries.
    """
    if not token:
        return token

    # If already quoted, leave as-is
    if token.startswith('"') and token.endswith('"'):
        return token

    # FTS5 special characters that need escaping:
    # - (dash/NOT operator), : (column spec), (, ), [, ]
    # Note: We don't need to escape * (prefix match) or ^ (first token)
    # as these are useful operators users might want to use
    special_chars = ["-", ":", "(", ")", "[", "]"]

    # Check if token contains any special characters
    if any(char in token for char in special_chars):
        # Escape internal quotes by doubling them (FTS5 convention)
        escaped = token.replace('"', '""')
        # Wrap in quotes
        return f'"{escaped}"'

    return token


def parse_search_query(query: str) -> ParsedQuery:
    """Parse a search query to extract field prefixes and convert to FTS5 format.

    Supports:
    - Multiple words: "python function" → matches both words (AND logic)
    - Exact phrases: '"python function"' → matches exact phrase
    - Field prefixes: 'role:user workspace:myproject title:something repository:github.com/owner/repo edition:cli'
    - Date filters: 'start_date:2024-01-01 end_date:2024-12-31' (yyyy-mm-dd format, inclusive)

    Args:
        query: The raw search query string.

    Returns:
        ParsedQuery with extracted field filters and FTS5 query string.
    """
    if not query or not query.strip():
        return ParsedQuery(fts_query="")

    query = query.strip()

    # Extract field prefixes (role:, workspace:, title:, repository:, edition:, start_date:, end_date:)
    role = None
    workspaces: list[str] = []
    title = None
    repositories: list[str] = []
    edition = None
    start_date = None
    end_date = None

    # Pattern for field:value (value can be quoted or unquoted)
    field_pattern = r'\b(role|workspace|title|repository|repo|edition|start_date|end_date):(?:"([^"]*)"|(\S+))'

    def extract_field(match):
        nonlocal role, title, edition, start_date, end_date
        field_name = match.group(1).lower()
        # Value is either in group 2 (quoted) or group 3 (unquoted)
        value = match.group(2) if match.group(2) is not None else match.group(3)

        if field_name == "role":
            role = value.lower()
        elif field_name == "workspace":
            workspaces.append(value)
        elif field_name == "title":
            title = value
        elif field_name in ("repository", "repo"):
            repositories.append(value)
        elif field_name == "edition":
            edition = value.lower()
        elif field_name == "start_date":
            validated = _validate_date_format(value)
            if validated:
                start_date = validated
        elif field_name == "end_date":
            validated = _validate_date_format(value)
            if validated:
                end_date = validated

        return ""  # Remove the field prefix from the query

    # Remove field prefixes and extract their values
    remaining_query = re.sub(field_pattern, extract_field, query, flags=re.IGNORECASE)
    remaining_query = remaining_query.strip()

    # Now process the remaining query for FTS5
    # FTS5 by default uses AND for multiple terms, so we just need to handle:
    # 1. Quoted phrases (keep as-is)
    # 2. Unquoted words (escape special chars and join with spaces for implicit AND)

    if not remaining_query:
        fts_query = ""
    else:
        # Tokenize the query preserving quoted strings
        tokens = []
        # Pattern to match quoted strings or individual words
        token_pattern = r'"[^"]*"|[^\s"]+'

        for match in re.finditer(token_pattern, remaining_query):
            token = match.group(0)
            # Clean up any empty quotes
            if token == '""':
                continue
            # Escape FTS5 special characters in the token
            escaped_token = _escape_fts5_token(token)
            tokens.append(escaped_token)

        # Join tokens with space (FTS5 uses implicit AND)
        fts_query = " ".join(tokens)

    return ParsedQuery(
        fts_query=fts_query,
        role=role,
        workspace=workspaces[0] if workspaces else None,
        workspaces=workspaces or None,
        title=title,
        repository=repositories[0] if repositories else None,
        repositories=repositories or None,
        edition=edition,
        start_date=start_date,
        end_date=end_date,
    )


# SQL LIKE pattern to detect ISO timestamp format (e.g., "2025-01-15T10:30:00Z")
# Pattern matches "YYYY-MM-DD" prefix which is common to all ISO timestamps
_ISO_TIMESTAMP_PATTERN = "____-__-__%"


def _build_date_filter_clause(start_date: str | None, end_date: str | None, date_column: str = "s.created_at") -> tuple[str, list]:
    """Build SQL WHERE clause fragments for date filtering.

    The created_at field can be:
    1. ISO timestamp string like "2025-01-15T10:30:00Z"
    2. Millisecond epoch timestamp like "1704067200000"

    This function handles both formats by using SQLite's date/datetime functions.

    Args:
        start_date: Start date in yyyy-mm-dd format (inclusive).
        end_date: End date in yyyy-mm-dd format (inclusive).
        date_column: The SQL column name to filter on.

    Returns:
        Tuple of (SQL clause string, list of parameters).
    """
    clauses = []
    params = []

    if start_date:
        # Handle both ISO timestamps and millisecond epochs
        # For ISO: compare directly using date extraction
        # For epoch ms: convert to date using SQLite's datetime functions
        clauses.append(f"""
            (
                (TYPEOF({date_column}) = 'text' AND (
                    ({date_column} LIKE '{_ISO_TIMESTAMP_PATTERN}' AND DATE(SUBSTR({date_column}, 1, 10)) >= ?) OR
                    ({date_column} NOT LIKE '{_ISO_TIMESTAMP_PATTERN}' AND DATE({date_column} / 1000, 'unixepoch') >= ?)
                ))
            )
        """)
        params.extend([start_date, start_date])

    if end_date:
        clauses.append(f"""
            (
                (TYPEOF({date_column}) = 'text' AND (
                    ({date_column} LIKE '{_ISO_TIMESTAMP_PATTERN}' AND DATE(SUBSTR({date_column}, 1, 10)) <= ?) OR
                    ({date_column} NOT LIKE '{_ISO_TIMESTAMP_PATTERN}' AND DATE({date_column} / 1000, 'unixepoch') <= ?)
                ))
            )
        """)
        params.extend([end_date, end_date])

    return " AND ".join(clauses) if clauses else "", params


# Allowed sort options with their SQL ORDER BY clauses (whitelist for security)
# Note: For relevance, we combine FTS5 rank (text relevance) with recency.
# FTS5 rank is negative (lower/more negative = better match).
# We subtract a recency bonus to make recent items more negative (better rank).
# The formula: rank - (days_since_2020 * 0.001) gives more weight to text relevance
# while providing a small boost to recent items.
_SORT_ORDER_CLAUSES = {
    "relevance": """ORDER BY (
        rank -
        CASE
            WHEN TYPEOF(s.created_at) = 'text' AND s.created_at LIKE '____-__-__%'
            THEN (JULIANDAY(SUBSTR(s.created_at, 1, 10)) - JULIANDAY('2020-01-01')) * 0.001
            WHEN TYPEOF(s.created_at) = 'text'
            THEN (JULIANDAY(DATETIME(CAST(s.created_at AS REAL) / 1000, 'unixepoch')) - JULIANDAY('2020-01-01')) * 0.001
            ELSE 0
        END
    ), s.session_id, m.message_index""",
    "date": "ORDER BY s.created_at DESC, s.session_id, m.message_index",
}


def _active_context_index_subquery() -> str:
    """Return SQL that selects the context active at the current message row."""
    return """
        COALESCE(
            (
                SELECT sc2.context_index
                FROM cst_session_contexts sc2
                WHERE sc2.session_id = s.session_id
                  AND sc2.message_index <= COALESCE(m.message_index, 0)
                ORDER BY sc2.message_index DESC, sc2.context_index DESC
                LIMIT 1
            ),
            (
                SELECT sc3.context_index
                FROM cst_session_contexts sc3
                WHERE sc3.session_id = s.session_id
                ORDER BY sc3.message_index ASC, sc3.context_index ASC
                LIMIT 1
            )
        )
    """


def _append_session_filters(
    query: str,
    params: list,
    *,
    include_agent_content: bool,
    effective_workspaces: list[str],
    effective_title: str | None,
    effective_repositories: list[str],
    effective_edition: str | None,
    effective_start_date: str | None,
    effective_end_date: str | None,
) -> str:
    """Append common session-level filter clauses to a SQL query.

    Reduces duplication across the five content-type search branches.

    Args:
        query: The SQL query string to append to.
        params: The parameter list (modified in-place).
        include_agent_content: Whether to include agent/subagent content.
        effective_workspaces: Workspace filter values.
        effective_title: Title filter value.
        effective_repositories: Repository filter values.
        effective_edition: Edition filter value.
        effective_start_date: Start date filter.
        effective_end_date: End date filter.

    Returns:
        The query string with appended filter clauses.
    """
    if not include_agent_content:
        query += " AND m.agent_nesting_level = 0"

    if effective_workspaces:
        clauses = []
        active_context_index = _active_context_index_subquery()
        for workspace in effective_workspaces:
            clauses.append(
                f"""(
                    (
                        NOT EXISTS (
                            SELECT 1 FROM cst_session_contexts sc_any
                            WHERE sc_any.session_id = s.session_id
                        )
                        AND (s.workspace_name LIKE ? OR s.workspace_path LIKE ?)
                    )
                    OR EXISTS (
                        SELECT 1 FROM cst_session_contexts sc
                        WHERE sc.session_id = s.session_id
                          AND sc.context_index = ({active_context_index})
                          AND (sc.workspace_name LIKE ? OR sc.workspace_path LIKE ?)
                    )
                )"""  # noqa: S608 -- interpolates static active-context SQL only
            )
            like_value = f"%{workspace}%"
            params.extend([like_value, like_value, like_value, like_value])
        query += " AND (" + " OR ".join(clauses) + ")"

    if effective_title:
        query += " AND (s.workspace_name LIKE ? OR s.custom_title LIKE ?)"
        params.extend([f"%{effective_title}%", f"%{effective_title}%"])

    if effective_repositories:
        clauses = []
        active_context_index = _active_context_index_subquery()
        for repository in effective_repositories:
            clauses.append(
                f"""(
                    (
                        NOT EXISTS (
                            SELECT 1 FROM cst_session_contexts sc_any
                            WHERE sc_any.session_id = s.session_id
                        )
                        AND s.repository_url LIKE ?
                    )
                    OR EXISTS (
                        SELECT 1 FROM cst_session_contexts sc
                        WHERE sc.session_id = s.session_id
                          AND sc.context_index = ({active_context_index})
                          AND sc.repository_url LIKE ?
                    )
                )"""  # noqa: S608 -- interpolates static active-context SQL only
            )
            like_value = f"%{repository}%"
            params.extend([like_value, like_value])
        query += " AND (" + " OR ".join(clauses) + ")"

    if effective_edition:
        query += " AND s.vscode_edition = ?"
        params.append(effective_edition)

    date_clause, date_params = _build_date_filter_clause(effective_start_date, effective_end_date, "s.created_at")
    if date_clause:
        query += f" AND {date_clause}"
        params.extend(date_params)

    return query


def _fork_duplicate_filter_clause(duplicate_content_clause: str = "COALESCE(dm.content, '') = COALESCE(m.content, '')") -> str:
    """Return SQL that keeps only the canonical session row for copied fork events."""
    return f"""
        AND (
            m.source_event_id IS NULL
            OR NOT EXISTS (
                SELECT 1
                FROM cst_messages dm
                JOIN cst_sessions ds ON dm.session_id = ds.session_id
                WHERE dm.source_event_id = m.source_event_id
                  AND dm.session_id != m.session_id
                  AND COALESCE(ds.source_id, '') = COALESCE(s.source_id, '')
                  AND ({duplicate_content_clause})
                  AND (
                      COALESCE(ds.created_at, '') < COALESCE(s.created_at, '')
                      OR (
                          COALESCE(ds.created_at, '') = COALESCE(s.created_at, '')
                          AND ds.session_id < s.session_id
                      )
                  )
            )
        )
    """  # noqa: S608 — duplicate_content_clause is built from hardcoded column names


def _search_builtin_index(conn: sqlite3.Connection, fts_query: str, limit: int) -> dict[str, dict]:
    """Search the Chronicle search_index FTS table for unenriched results.

    Expects Chronicle to be ATTACHed as ``chronicle`` schema.

    Args:
        conn: SQLite connection (with chronicle attached).
        fts_query: FTS5 query string.
        limit: Maximum results.

    Returns:
        Dict mapping session_id → result dict.
    """
    builtin_results: dict[str, dict] = {}
    try:
        rows = conn.execute(
            "SELECT session_id, content, rank FROM chronicle.search_index WHERE search_index MATCH ? ORDER BY rank, session_id LIMIT ?",
            (fts_query, limit * 2),
        ).fetchall()
        for row in rows:
            sid = row["session_id"]
            if sid not in builtin_results:
                builtin_results[sid] = {
                    "session_id": sid,
                    "content": (row["content"] or "")[:500],
                    "highlighted": (row["content"] or "")[:500],
                    "match_type": "builtin_search",
                    "is_enriched": False,
                    "rank": row["rank"],
                }
    except Exception:  # noqa: S110
        pass  # search_index might not exist or chronicle not attached
    return builtin_results


def _attach_active_contexts(conn: sqlite3.Connection, results: list[dict]) -> None:
    """Attach the session context active at each search hit's message index."""
    session_ids = list({r["session_id"] for r in results if r.get("session_id")})
    if not session_ids:
        return
    placeholders = ",".join("?" * len(session_ids))
    try:
        rows = conn.execute(
            f"""
            SELECT session_id, context_index, message_index, timestamp, workspace_name,
                   workspace_path, repository_url, branch, source
            FROM cst_session_contexts
            WHERE session_id IN ({placeholders})
            ORDER BY session_id, message_index, context_index
            """,  # noqa: S608
            session_ids,
        ).fetchall()
    except sqlite3.OperationalError:
        return

    contexts_by_session: dict[str, list[dict]] = {}
    for row in rows:
        contexts_by_session.setdefault(row["session_id"], []).append(dict(row))

    for result in results:
        session_id = result.get("session_id")
        if not isinstance(session_id, str):
            continue
        contexts = contexts_by_session.get(session_id, [])
        if not contexts:
            continue
        message_index = result.get("message_index")
        if message_index is None:
            active = contexts[0]
        else:
            active = contexts[0]
            for context in contexts:
                if context.get("message_index", 0) <= message_index:
                    active = context
                else:
                    break
        result["active_context"] = active
        result["workspace_name"] = active.get("workspace_name") or result.get("workspace_name")
        result["workspace_path"] = active.get("workspace_path")
        result["repository_url"] = active.get("repository_url")


def _search_messages(
    cursor: sqlite3.Cursor,
    fts_query: str,
    has_filters: bool,
    limit: int,
    skip: int,
    sort_by: str,
    effective_role: str | None,
    **filter_kwargs,
) -> list[dict]:
    """Search cst_messages via FTS5 or filter-only query.

    Args:
        cursor: SQLite cursor.
        fts_query: FTS5 query (may be empty).
        has_filters: Whether any field filters are active.
        limit: Max results.
        skip: Results to skip.
        sort_by: Sort order key.
        effective_role: Role filter.
        **filter_kwargs: Passed to _append_session_filters.

    Returns:
        List of result dicts.
    """
    results: list[dict] = []

    if fts_query:
        message_query = """
            SELECT
                m.id,
                m.session_id,
                m.message_index,
                m.role,
                m.content,
                m.parent_message_id,
                s.workspace_name,
                s.custom_title,
                s.created_at,
                s.vscode_edition,
                highlight(cst_messages_fts, 0, '<mark>', '</mark>') as highlighted,
                'message' as match_type,
                rank
            FROM cst_messages_fts
            JOIN cst_messages m ON cst_messages_fts.rowid = m.id
            JOIN cst_sessions s ON m.session_id = s.session_id
            WHERE cst_messages_fts MATCH ?
        """
        params: list = [fts_query]

        if effective_role:
            message_query += " AND m.role = ?"
            params.append(effective_role)
        else:
            message_query += " AND m.role != 'system'"

        message_query = _append_session_filters(message_query, params, **filter_kwargs)
        message_query += _fork_duplicate_filter_clause()

        order_clause = _SORT_ORDER_CLAUSES.get(sort_by, _SORT_ORDER_CLAUSES["relevance"])
        message_query += f" {order_clause} LIMIT ?"
        params.append(limit + skip)

        cursor.execute(message_query, params)
        results.extend([dict(row) for row in cursor.fetchall()])

    elif has_filters:
        message_query = """
            SELECT
                m.id,
                m.session_id,
                m.message_index,
                m.role,
                m.content,
                m.parent_message_id,
                s.workspace_name,
                s.custom_title,
                s.created_at,
                s.vscode_edition,
                m.content as highlighted,
                'message' as match_type
            FROM cst_messages m
            JOIN cst_sessions s ON m.session_id = s.session_id
            WHERE 1=1
        """
        params = []

        if effective_role:
            message_query += " AND m.role = ?"
            params.append(effective_role)
        else:
            message_query += " AND m.role != 'system'"

        message_query = _append_session_filters(message_query, params, **filter_kwargs)
        message_query += _fork_duplicate_filter_clause()

        message_query += " ORDER BY s.created_at DESC LIMIT ?"
        params.append(limit + skip)

        cursor.execute(message_query, params)
        results.extend([dict(row) for row in cursor.fetchall()])

    return results


def _search_tool_invocations(
    cursor: sqlite3.Cursor,
    like_pattern: str,
    remaining: int,
    include_tool_inputs_flag: bool,
    **filter_kwargs,
) -> list[dict]:
    """Search cst_tool_invocations via LIKE matching."""
    tool_conditions = ["(t.name LIKE ? OR t.result LIKE ?)"]
    tool_params: list[str | int] = [like_pattern, like_pattern]
    if include_tool_inputs_flag:
        tool_conditions.append("t.input LIKE ?")
        tool_params.append(like_pattern)

    tool_where = "(" + " OR ".join(tool_conditions) + ")"

    tool_query = f"""
        SELECT
            t.id,
            m.session_id,
            m.message_index,
            'assistant' as role,
            t.name || ': ' || COALESCE(t.input, '') || ' -> ' || COALESCE(t.result, '') as content,
            s.workspace_name,
            s.custom_title,
            s.created_at,
            s.vscode_edition,
            t.name || ': ' || COALESCE(t.input, '') as highlighted,
            'tool_invocation' as match_type
        FROM cst_tool_invocations t
        JOIN cst_messages m ON t.message_id = m.id
        JOIN cst_sessions s ON m.session_id = s.session_id
        WHERE {tool_where}
    """  # noqa: S608 — tool_where is built from hardcoded column names
    params: list = list(tool_params)

    tool_query = _append_session_filters(tool_query, params, **filter_kwargs)
    tool_query += _fork_duplicate_filter_clause(
        """
        EXISTS (
            SELECT 1
            FROM cst_tool_invocations dt
            WHERE dt.message_id = dm.id
              AND COALESCE(dt.name, '') = COALESCE(t.name, '')
              AND COALESCE(dt.input, '') = COALESCE(t.input, '')
              AND COALESCE(dt.result, '') = COALESCE(t.result, '')
        )
        """
    )

    tool_query += " LIMIT ?"
    params.append(remaining)

    cursor.execute(tool_query, params)
    return [dict(row) for row in cursor.fetchall()]


def _search_file_changes(
    cursor: sqlite3.Cursor,
    like_pattern: str,
    remaining: int,
    include_diffs_flag: bool,
    **filter_kwargs,
) -> list[dict]:
    """Search cst_file_changes via LIKE matching."""
    file_conditions = ["(f.path LIKE ? OR f.explanation LIKE ?)"]
    file_params: list[str | int] = [like_pattern, like_pattern]
    if include_diffs_flag:
        file_conditions.append("f.diff LIKE ?")
        file_params.append(like_pattern)

    file_where = "(" + " OR ".join(file_conditions) + ")"

    file_query = f"""
        SELECT
            f.id,
            m.session_id,
            m.message_index,
            'assistant' as role,
            f.path || ': ' || COALESCE(f.explanation, '') as content,
            s.workspace_name,
            s.custom_title,
            s.created_at,
            s.vscode_edition,
            f.path as highlighted,
            'file_change' as match_type
        FROM cst_file_changes f
        JOIN cst_messages m ON f.message_id = m.id
        JOIN cst_sessions s ON m.session_id = s.session_id
        WHERE {file_where}
    """  # noqa: S608 — file_where is built from hardcoded column names
    params: list = list(file_params)

    file_query = _append_session_filters(file_query, params, **filter_kwargs)
    file_query += _fork_duplicate_filter_clause(
        """
        EXISTS (
            SELECT 1
            FROM cst_file_changes df
            WHERE df.message_id = dm.id
              AND COALESCE(df.path, '') = COALESCE(f.path, '')
              AND COALESCE(df.explanation, '') = COALESCE(f.explanation, '')
              AND COALESCE(df.diff, '') = COALESCE(f.diff, '')
        )
        """
    )

    file_query += " LIMIT ?"
    params.append(remaining)

    cursor.execute(file_query, params)
    return [dict(row) for row in cursor.fetchall()]


def _search_command_runs(
    cursor: sqlite3.Cursor,
    like_pattern: str,
    remaining: int,
    **filter_kwargs,
) -> list[dict]:
    """Search cst_command_runs via LIKE matching."""
    cmd_query = """
        SELECT
            c.id,
            m.session_id,
            m.message_index,
            'assistant' as role,
            c.command || ': ' || COALESCE(c.output, '') as content,
            s.workspace_name,
            s.custom_title,
            s.created_at,
            s.vscode_edition,
            c.command as highlighted,
            'command_run' as match_type
        FROM cst_command_runs c
        JOIN cst_messages m ON c.message_id = m.id
        JOIN cst_sessions s ON m.session_id = s.session_id
        WHERE (c.command LIKE ? OR c.output LIKE ?)
    """
    params: list = [like_pattern, like_pattern]

    cmd_query = _append_session_filters(cmd_query, params, **filter_kwargs)
    cmd_query += _fork_duplicate_filter_clause(
        """
        EXISTS (
            SELECT 1
            FROM cst_command_runs dc
            WHERE dc.message_id = dm.id
              AND COALESCE(dc.command, '') = COALESCE(c.command, '')
              AND COALESCE(dc.output, '') = COALESCE(c.output, '')
        )
        """
    )

    cmd_query += " LIMIT ?"
    params.append(remaining)

    cursor.execute(cmd_query, params)
    return [dict(row) for row in cursor.fetchall()]


def _search_thinking_blocks(
    cursor: sqlite3.Cursor,
    like_pattern: str,
    remaining: int,
    **filter_kwargs,
) -> list[dict]:
    """Search cst_content_blocks (thinking kind) via LIKE matching."""
    think_query = """
        SELECT
            cb.id,
            m.session_id,
            m.message_index,
            'assistant' as role,
            cb.content as content,
            s.workspace_name,
            s.custom_title,
            s.created_at,
            s.vscode_edition,
            SUBSTR(cb.content, 1, 200) as highlighted,
            'thinking' as match_type
        FROM cst_content_blocks cb
        JOIN cst_messages m ON cb.message_id = m.id
        JOIN cst_sessions s ON m.session_id = s.session_id
        WHERE cb.kind = 'thinking' AND cb.content LIKE ?
    """
    params: list = [like_pattern]

    think_query = _append_session_filters(think_query, params, **filter_kwargs)
    think_query += _fork_duplicate_filter_clause(
        """
        EXISTS (
            SELECT 1
            FROM cst_content_blocks dcb
            WHERE dcb.message_id = dm.id
              AND dcb.kind = 'thinking'
              AND COALESCE(dcb.content, '') = COALESCE(cb.content, '')
        )
        """
    )

    think_query += " LIMIT ?"
    params.append(remaining)

    cursor.execute(think_query, params)
    return [dict(row) for row in cursor.fetchall()]


def execute_search(
    conn: sqlite3.Connection,
    query: str,
    *,
    limit: int = 50,
    skip: int = 0,
    role: str | None = None,
    search_content_set: set[str],
    session_title: str | None = None,
    sort_by: str = "relevance",
    repository: str | list[str] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    has_chronicle: bool = False,
) -> list[dict]:
    """Execute full-text search across cst_* tables and Chronicle search_index.

    This is the main search entry point, called by Database.search().
    It handles query parsing, content-type dispatch, filter application,
    and result merging.

    Args:
        conn: SQLite connection with row_factory = sqlite3.Row.
        query: The search query (supports field prefixes and quoted phrases).
        limit: Maximum number of results to return.
        skip: Number of results to skip (for pagination).
        role: Filter by message role.
        search_content_set: Set of content type tokens to search.
        session_title: Filter by session title/workspace name.
        sort_by: Sort order - 'relevance' or 'date'.
        repository: Filter by repository URL.
        start_date: Filter results on or after this date (yyyy-mm-dd).
        end_date: Filter results on or before this date (yyyy-mm-dd).

    Returns:
        List of matching result dicts with session info.
    """
    # Derive individual flags from the content set
    include_msgs = "messages" in search_content_set
    include_tools = "tools" in search_content_set
    include_tool_inputs_flag = "tool-inputs" in search_content_set
    include_files = "file-changes" in search_content_set
    include_diffs_flag = "diffs" in search_content_set
    include_commands_flag = "commands" in search_content_set
    include_thinking_flag = "thinking" in search_content_set
    include_agent_content = "agent-details" in search_content_set

    results: list[dict] = []

    # Parse the query to extract field filters and convert to FTS5 format
    parsed = parse_search_query(query)

    # Use parsed field filters, with explicit parameters taking precedence
    effective_role = role if role else parsed.role
    effective_title = session_title if session_title else parsed.title
    effective_workspaces = parsed.workspaces or []
    if isinstance(repository, list):
        effective_repositories = [repo for repo in repository if repo]
    elif repository:
        effective_repositories = [repository]
    else:
        effective_repositories = parsed.repositories or []
    effective_edition = parsed.edition
    effective_start_date = start_date if start_date else parsed.start_date
    effective_end_date = end_date if end_date else parsed.end_date

    fts_query = parsed.fts_query

    # Common filter kwargs shared across all content-type search branches
    filter_kwargs = {
        "include_agent_content": include_agent_content,
        "effective_workspaces": effective_workspaces,
        "effective_title": effective_title,
        "effective_repositories": effective_repositories,
        "effective_edition": effective_edition,
        "effective_start_date": effective_start_date,
        "effective_end_date": effective_end_date,
    }

    # Search Chronicle search_index FTS table for unenriched results
    builtin_results: dict[str, dict] = {}
    if has_chronicle and fts_query and include_msgs:
        builtin_results = _search_builtin_index(conn, fts_query, limit)

    has_filters = bool(effective_role or effective_title or effective_workspaces or effective_repositories or effective_edition or effective_start_date or effective_end_date)

    cursor = conn.cursor()

    # Search messages
    if include_msgs:
        results.extend(
            _search_messages(
                cursor,
                fts_query,
                has_filters,
                limit,
                skip,
                sort_by,
                effective_role,
                **filter_kwargs,
            )
        )

    # For tool/file/command/thinking searches, use LIKE matching
    search_terms = fts_query if fts_query else query
    like_pattern = f"%{search_terms}%"

    if include_tools and len(results) < limit and search_terms:
        results.extend(
            _search_tool_invocations(
                cursor,
                like_pattern,
                limit - len(results),
                include_tool_inputs_flag,
                **filter_kwargs,
            )
        )

    if include_files and len(results) < limit and search_terms:
        results.extend(
            _search_file_changes(
                cursor,
                like_pattern,
                limit - len(results),
                include_diffs_flag,
                **filter_kwargs,
            )
        )

    if include_commands_flag and len(results) < limit and search_terms:
        results.extend(
            _search_command_runs(
                cursor,
                like_pattern,
                limit - len(results),
                **filter_kwargs,
            )
        )

    if include_thinking_flag and len(results) < limit and search_terms:
        results.extend(
            _search_thinking_blocks(
                cursor,
                like_pattern,
                limit - len(results),
                **filter_kwargs,
            )
        )

    # Merge built-in search_index results that weren't covered by cst search
    cst_session_ids = {r["session_id"] for r in results}
    for sid, builtin_row in builtin_results.items():
        if sid not in cst_session_ids:
            results.append(builtin_row)

    _attach_active_contexts(conn, results)

    # Apply skip/limit to merged results for correct pagination
    return results[skip : skip + limit]
