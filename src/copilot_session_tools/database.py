"""Database module for storing and querying Copilot chat sessions.

Schema design inspired by:
- tad-hq/universal-session-viewer: FTS5 full-text search
- jazzyalex/agent-sessions: SQLite indexing patterns

The CST enrichment tables (cst_*) live in their own database file
(~/.copilot/copilot-session-tools.db).  The Copilot CLI's built-in
Chronicle database (~/.copilot/session-store.db) is optionally ATTACHed
read-only when needed for enrichment discovery and unenriched fallback.
"""

import contextlib
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import ClassVar

from . import db_storage
from .db_search import (  # noqa: F401 — re-exported for backward compatibility
    _ISO_TIMESTAMP_PATTERN,
    ParsedQuery,
    _build_date_filter_clause,
    parse_search_query,
)
from .db_storage import (  # noqa: F401 — re-exported for backward compatibility
    CST_FTS_SCHEMA,
    CST_SCHEMA,
    CST_SCHEMA_VERSION,
)
from .scanner import (
    ChatMessage,
    ChatSession,
)
from .sources import (
    DEFAULT_SOURCE_ID,
    CopilotSource,
    SessionIdentityConflictError,
    canonicalize_base_dir,
    new_source_id,
    normalize_source_name,
    validate_chronicle,
)


class Database:
    """SQLite database for storing Copilot chat sessions.

    Uses FTS5 for full-text search (inspired by tad-hq/universal-session-viewer).

    CST enrichment tables live in their own database file.  The Copilot CLI's
    Chronicle database is optionally ATTACHed as ``chronicle`` for enrichment
    discovery and unenriched-session fallback.
    """

    # List of cst_* tables that can be dropped and recreated
    DERIVED_TABLES: ClassVar[list[str]] = [
        "cst_messages_fts",  # FTS table must be dropped first
        "cst_content_blocks",
        "cst_command_runs",
        "cst_file_changes",
        "cst_tool_invocations",
        "cst_root_agent_intervals",
        "cst_session_contexts",
        "cst_messages",
        "cst_sessions",
    ]

    # List of triggers that need to be dropped/recreated with derived tables
    DERIVED_TRIGGERS: ClassVar[list[str]] = [
        "cst_messages_ai",
        "cst_messages_ad",
        "cst_messages_au",
    ]

    def __init__(
        self,
        db_path: str | Path,
        *,
        unenriched_only: bool = False,
        chronicle_db_path: str | Path | None = None,
    ):
        """Initialize the database connection.

        Args:
            db_path: Path to the CST enrichment database file.
            unenriched_only: If True, disable cst_* table reads (use Chronicle tables only).
            chronicle_db_path: Path to the Copilot CLI Chronicle session-store.db.
                If ``None`` (default), auto-detects ``session-store.db`` in the
                same directory as *db_path*.
        """
        self.db_path = Path(db_path)
        self.unenriched_only = unenriched_only
        self._batch_conn: sqlite3.Connection | None = None

        # Auto-detect Chronicle DB as sibling of CST DB
        if chronicle_db_path is not None:
            self.chronicle_db_path: Path | None = Path(chronicle_db_path)
        else:
            candidate = self.db_path.parent / "session-store.db"
            # If db_path IS session-store.db (backward compat), use it as Chronicle too
            if candidate.resolve() == self.db_path.resolve():
                self.chronicle_db_path = self.db_path
            else:
                self.chronicle_db_path = candidate

        self._ensure_schema()
        self._configure_default_source()

    def _configure_default_source(self) -> None:
        """Keep the default source path aligned with Chronicle auto-detection."""
        base_dir = self.chronicle_db_path.parent if self.chronicle_db_path is not None else Path.home() / ".copilot"
        base_dir_key = os.path.normcase(str(base_dir.resolve(strict=False)))
        with self._get_connection() as conn:
            conflicting = conn.execute(
                "SELECT name FROM cst_sources WHERE base_dir_key = ? AND source_id != ?",
                (base_dir_key, DEFAULT_SOURCE_ID),
            ).fetchone()
            if conflicting is not None:
                raise ValueError(f"Default CLI source path is already registered as source '{conflicting['name']}'.")
            conn.execute(
                "UPDATE cst_sources SET base_dir = ?, base_dir_key = ? WHERE source_id = ?",
                (str(base_dir), base_dir_key, DEFAULT_SOURCE_ID),
            )

    @staticmethod
    def _source_from_row(row: sqlite3.Row) -> CopilotSource:
        return CopilotSource(
            source_id=row["source_id"],
            name=row["name"],
            base_dir=Path(row["base_dir"]),
            enabled=bool(row["enabled"]),
            is_default=bool(row["is_default"]),
        )

    def list_sources(self, *, enabled_only: bool = False) -> list[CopilotSource]:
        """List persisted Copilot CLI sources."""
        query = "SELECT * FROM cst_sources"
        if enabled_only:
            query += " WHERE enabled = 1"
        query += " ORDER BY is_default DESC, name_key"
        with self._get_connection() as conn:
            return [self._source_from_row(row) for row in conn.execute(query).fetchall()]

    def get_source(self, name_or_id: str) -> CopilotSource | None:
        """Resolve a source by immutable ID or case-insensitive display name."""
        key = name_or_id.strip().casefold()
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM cst_sources WHERE source_id = ? COLLATE NOCASE OR name_key = ?",
                (key, key),
            ).fetchone()
        return self._source_from_row(row) if row else None

    def add_source(self, name: str, base_dir: str | Path) -> CopilotSource:
        """Validate and persist a named Copilot CLI source."""
        display_name, name_key = normalize_source_name(name)
        if self.get_source(display_name):
            raise ValueError(f"A source named '{display_name}' already exists.")
        resolved_dir, base_dir_key = canonicalize_base_dir(base_dir)
        validate_chronicle(resolved_dir)
        source_id = new_source_id()
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO cst_sources
                        (source_id, name, name_key, base_dir, base_dir_key, enabled, is_default)
                    VALUES (?, ?, ?, ?, ?, 1, 0)
                    """,
                    (source_id, display_name, name_key, str(resolved_dir), base_dir_key),
                )
        except sqlite3.IntegrityError as exc:
            existing = self.get_source(display_name)
            if existing:
                raise ValueError(f"A source named '{display_name}' already exists.") from exc
            raise ValueError(f"Copilot base directory is already registered: {resolved_dir}") from exc
        source = self.get_source(source_id)
        if source is None:
            raise RuntimeError("Source registration did not persist.")
        return source

    def rename_source(self, name_or_id: str, new_name: str) -> CopilotSource:
        """Rename a source without changing its stable identifier."""
        source = self.get_source(name_or_id)
        if source is None:
            raise ValueError(f"Source not found: {name_or_id}")
        if source.is_default:
            raise ValueError("The default Copilot CLI source cannot be renamed.")
        display_name, name_key = normalize_source_name(new_name)
        try:
            with self._get_connection() as conn:
                conn.execute(
                    "UPDATE cst_sources SET name = ?, name_key = ? WHERE source_id = ?",
                    (display_name, name_key, source.source_id),
                )
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"A source named '{display_name}' already exists.") from exc
        renamed = self.get_source(source.source_id)
        if renamed is None:
            raise RuntimeError("Source rename did not persist.")
        return renamed

    def set_source_enabled(self, name_or_id: str, *, enabled: bool) -> CopilotSource:
        """Enable or disable a custom source while retaining archived content."""
        source = self.get_source(name_or_id)
        if source is None:
            raise ValueError(f"Source not found: {name_or_id}")
        if source.is_default and not enabled:
            raise ValueError("The default Copilot CLI source cannot be disabled.")
        with self._get_connection() as conn:
            conn.execute(
                "UPDATE cst_sources SET enabled = ? WHERE source_id = ?",
                (int(enabled), source.source_id),
            )
        updated = self.get_source(source.source_id)
        if updated is None:
            raise RuntimeError("Source state change did not persist.")
        return updated

    @contextmanager
    def _get_connection(self):
        """Get a database connection context manager.

        If a :meth:`batch_connection` is active, reuses that connection
        (no commit/close — the batch context handles that).  Otherwise
        opens a fresh connection, commits on success, and closes.
        """
        if self._batch_conn is not None:
            yield self._batch_conn
            return
        conn = sqlite3.connect(str(self.db_path), uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _has_chronicle(self) -> bool:
        """Check whether the Chronicle DB file exists on disk."""
        return self.chronicle_db_path is not None and self.chronicle_db_path.is_file()

    def _attach_chronicle(self, conn: sqlite3.Connection, source: CopilotSource | None = None) -> bool:
        """ATTACH the Chronicle DB as ``chronicle`` schema on *conn*.

        Uses a read-only URI to prevent accidental writes to Chronicle.
        The connection must be opened with ``uri=True`` for this to work.
        Returns True if attached successfully (or already attached),
        False if Chronicle DB is unavailable or attachment failed.
        """
        chronicle_path = source.chronicle_db_path if source is not None else self.chronicle_db_path
        if chronicle_path is None or not chronicle_path.is_file():
            return False
        # Check if chronicle is already attached (e.g., inside batch_connection)
        try:
            attached_dbs = [row[1] for row in conn.execute("PRAGMA database_list").fetchall()]
            if "chronicle" in attached_dbs:
                return True
        except sqlite3.Error:
            pass
        try:
            chronicle_uri = f"{chronicle_path.resolve().as_uri()}?mode=ro"
            conn.execute("ATTACH DATABASE ? AS chronicle", (chronicle_uri,))
            return True
        except sqlite3.Error:
            return False

    @staticmethod
    def _detach_chronicle(conn: sqlite3.Connection) -> None:
        """DETACH the ``chronicle`` schema from *conn*.  Safe to call even if not attached."""
        with contextlib.suppress(sqlite3.Error):
            conn.execute("DETACH DATABASE chronicle")

    @contextmanager
    def _get_chronicle_connection(self):
        """Get a connection with Chronicle ATTACHed (if available).

        Yields ``(conn, has_chronicle)`` so callers know whether
        ``chronicle.*`` tables are accessible.

        If a batch connection is active, ATTACHes on it.  DETACH is
        skipped inside a batch (SQLite rejects it mid-transaction);
        Chronicle stays attached for the batch's lifetime and is
        released when the batch connection closes.
        """
        if self._batch_conn is not None:
            attached = self._attach_chronicle(self._batch_conn)
            # Don't detach inside batch — DETACH fails mid-transaction,
            # and re-attach is idempotent (checked via pragma_database_list)
            yield self._batch_conn, attached
            return
        conn = sqlite3.connect(str(self.db_path), uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        attached = self._attach_chronicle(conn)
        try:
            yield conn, attached
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def _get_source_connection(self, source: CopilotSource):
        """Get an archive connection with one source's Chronicle DB attached."""
        conn = sqlite3.connect(str(self.db_path), uri=True)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        attached = self._attach_chronicle(conn, source)
        try:
            yield conn, attached
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    @contextmanager
    def batch_connection(self):
        """Hold a single connection open for multiple operations.

        While this context is active, all ``Database`` methods that normally
        open/commit/close per call will reuse the held connection instead.
        A single commit happens when the context exits successfully.

        Foreign key enforcement is disabled during batch operations for
        performance — with FK=ON, CASCADE checks on every DELETE/INSERT
        cause ~38x overhead on large databases.  Write paths explicitly
        delete child rows (via ``_delete_session_data``) so cascades are
        not needed.

        Not reentrant — raises RuntimeError if nested.

        Usage::

            with database.batch_connection():
                for session in sessions:
                    database.enrich_session(session)  # reuses one connection
        """
        if self._batch_conn is not None:
            raise RuntimeError("batch_connection() is not reentrant — already inside a batch")
        conn = sqlite3.connect(str(self.db_path), uri=True)
        conn.row_factory = sqlite3.Row
        # FK enforcement OFF for batch perf — CASCADE checks cause ~38x slowdown
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("PRAGMA busy_timeout = 5000")
        # 64 MB page cache — 27% faster bulk inserts vs default 2 MB
        conn.execute("PRAGMA cache_size = -65536")
        self._batch_conn = conn
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._batch_conn = None
            conn.close()

    def _ensure_schema(self):
        """Ensure the cst_* schema exists in the database."""
        with self._get_connection() as conn:
            db_storage.ensure_schema(conn)
        # Check Chronicle schema version (warns if incompatible)
        if self._has_chronicle():
            with self._get_chronicle_connection() as (conn, has_chronicle):
                if has_chronicle:
                    db_storage.check_builtin_schema_version(conn)

    def has_cst_tables(self) -> bool:
        """Check if cst_* extension tables exist in the database."""
        if self.unenriched_only:
            return False
        with self._get_connection() as conn:
            return db_storage.has_cst_tables(conn)

    def discover_sessions_needing_enrichment(self, source: CopilotSource | None = None) -> list[dict]:
        """Find CLI sessions needing enrichment by comparing Chronicle turns vs cst_messages."""
        resolved_source = source or self.get_source(DEFAULT_SOURCE_ID)
        if resolved_source is None:
            return []
        with self._get_source_connection(resolved_source) as (conn, has_chronicle):
            rows = db_storage.discover_sessions_needing_enrichment(
                conn,
                has_chronicle=has_chronicle,
                source_id=resolved_source.source_id,
            )
        for row in rows:
            row["native_session_id"] = row["session_id"]
            row["source_id"] = resolved_source.source_id
            row["source_base_dir"] = str(resolved_source.base_dir)
        return rows

    @staticmethod
    def _session_content(session: ChatSession) -> dict:
        """Return session data excluding storage location and source provenance."""
        content = asdict(session)
        for key in (
            "source_id",
            "source_name",
            "native_session_id",
            "source_file",
            "source_file_mtime",
            "source_file_size",
        ):
            content.pop(key, None)

        def strip_derived(value):
            if isinstance(value, dict):
                return {key: strip_derived(item) for key, item in value.items() if key != "cached_markdown"}
            if isinstance(value, list):
                return [strip_derived(item) for item in value]
            return value

        return strip_derived(content)

    @staticmethod
    def _raise_identity_conflict(session_id: str, sources: list[str]) -> None:
        source_list = ", ".join(sources)
        raise SessionIdentityConflictError(f"Session UUID {session_id} contains conflicting data in sources: {source_list}.")

    def _check_existing_session_identity(
        self,
        conn: sqlite3.Connection,
        session: ChatSession,
    ) -> bool:
        """Return True for an identical cross-source duplicate, or raise on conflict."""
        from .db_retrieval import get_cst_session

        existing = get_cst_session(conn, session.session_id)
        if existing is None or existing.source_id == session.source_id:
            return False
        if self._session_content(existing) != self._session_content(session):
            existing_source = self.get_source(existing.source_id) if existing.source_id else None
            incoming_source = self.get_source(session.source_id) if session.source_id else None
            self._raise_identity_conflict(
                session.session_id,
                [
                    existing_source.name if existing_source else existing.source_id or "unknown",
                    incoming_source.name if incoming_source else session.source_id or "unknown",
                ],
            )
        return True

    def _resolve_builtin_session(
        self,
        session_id: str,
        *,
        preferred_source: CopilotSource | None = None,
    ) -> tuple[CopilotSource, ChatSession] | None:
        """Resolve one UUID across enabled source stores and reject conflicting copies."""
        from .db_retrieval import get_builtin_session_as_chat_session

        candidates: list[tuple[CopilotSource, ChatSession]] = []
        for source in self.list_sources(enabled_only=True):
            with self._get_source_connection(source) as (conn, has_chronicle):
                if not has_chronicle:
                    continue
                session = get_builtin_session_as_chat_session(conn, session_id)
            if session is None:
                continue
            session.native_session_id = session_id
            session.source_id = source.source_id
            session.source_name = source.name
            candidates.append((source, session))

        if not candidates:
            return None
        expected = self._session_content(candidates[0][1])
        if any(self._session_content(session) != expected for _, session in candidates[1:]):
            self._raise_identity_conflict(session_id, [source.name for source, _ in candidates])
        if preferred_source is not None:
            for candidate in candidates:
                if candidate[0].source_id == preferred_source.source_id:
                    return candidate
        return candidates[0]

    def add_session(self, session: ChatSession) -> bool:
        """Add a chat session to the database.

        Returns True if the session was added, False if it already exists.
        """
        with self._get_connection() as conn:
            if self._check_existing_session_identity(conn, session):
                return False
            return db_storage.add_session(conn, session)

    def add_sessions_batch(self, sessions: list[ChatSession]) -> tuple[int, int]:
        """Add multiple sessions in a single transaction.

        Returns:
            Tuple of (added_count, skipped_count).
        """
        with self._get_connection() as conn:
            accepted: list[ChatSession] = []
            skipped = 0
            for session in sessions:
                if self._check_existing_session_identity(conn, session):
                    skipped += 1
                else:
                    accepted.append(session)
            added, existing = db_storage.add_sessions_batch(conn, accepted)
            return added, skipped + existing

    def _add_session_impl(self, cursor, session: ChatSession):
        """Delegate to db_storage.add_session_impl."""
        db_storage.add_session_impl(cursor, session)

    def update_session(self, session: ChatSession):
        """Update an existing session or add it if it doesn't exist."""
        with self._get_connection() as conn:
            if self._check_existing_session_identity(conn, session):
                return
            db_storage.update_session(conn, session)

    def update_sessions_batch(self, sessions: list[ChatSession]) -> int:
        """Update multiple sessions in a single transaction.

        Returns:
            Number of sessions updated.
        """
        with self._get_connection() as conn:
            accepted = [session for session in sessions if not self._check_existing_session_identity(conn, session)]
            return db_storage.update_sessions_batch(conn, accepted)

    def get_sessions_needing_reparse(self, current_parser_version: int) -> list[dict]:
        """Find cst_sessions with parser_version < current_parser_version."""
        with self._get_connection() as conn:
            return db_storage.get_sessions_needing_reparse(conn, current_parser_version)

    def count_sessions_needing_version_refresh(self, current_version: str) -> int:
        """Count enriched sessions whose enrichment_version differs from current_version."""
        with self._get_connection() as conn:
            return db_storage.count_sessions_needing_version_refresh(conn, current_version)

    def get_sessions_needing_version_refresh(self, current_version: str) -> list[dict]:
        """Find enriched sessions whose enrichment_version is older than current_version."""
        with self._get_connection() as conn:
            return db_storage.get_sessions_needing_version_refresh(conn, current_version)

    def get_session_enrichment_version(self, session_id: str) -> str | None:
        """Get the enrichment_version for a specific session, or None if not enriched."""
        with self._get_connection() as conn:
            return db_storage.get_session_enrichment_version(conn, session_id)

    def update_enrichment_version(self, session_id: str, version: str) -> None:
        """Stamp a session's enrichment_version and parser_version, creating a stub row if needed."""
        with self._get_connection() as conn:
            db_storage.update_enrichment_version(conn, session_id, version)

    def delete_cst_session(self, session_id: str) -> bool:
        """Delete all cst_* data for a session. Returns True if session existed."""
        with self._get_connection() as conn:
            return db_storage.delete_cst_session(conn, session_id)

    def needs_update(self, session_id: str, file_mtime: float | None, file_size: int | None) -> bool:
        """Check if a session needs to be updated based on file metadata."""
        with self._get_connection() as conn:
            return db_storage.needs_update(conn, session_id, file_mtime, file_size)

    def needs_update_by_file(self, source_file: str, file_mtime: float, file_size: int) -> bool:
        """Check if a file needs to be parsed based on its metadata."""
        with self._get_connection() as conn:
            return db_storage.needs_update_by_file(conn, source_file, file_mtime, file_size)

    def get_all_file_metadata(self) -> dict[str, tuple[float, int, int]]:
        """Get all stored file metadata in one query.

        Returns a dict mapping source_file -> (mtime, size, session_count).
        """
        with self._get_connection() as conn:
            return db_storage.get_all_file_metadata(conn)

    def _reconstruct_message(self, cursor, message_id: int, msg_row) -> ChatMessage:
        """Reconstruct a ChatMessage from database rows by querying related tables."""
        from .db_retrieval import reconstruct_message

        msg, _ = reconstruct_message(cursor, message_id, msg_row, link_children=True)
        return msg

    def get_session(self, session_id: str) -> ChatSession | None:
        """Get a session by ID. Checks cst_sessions first (enriched), falls back to Chronicle (unenriched).

        Args:
            session_id: The session ID to look up.

        Returns:
            ChatSession if found, None otherwise.
        """
        from .db_retrieval import get_cst_session

        # Try enriched path first
        if self.has_cst_tables():
            with self._get_connection() as conn:
                session = get_cst_session(conn, session_id)
                if session:
                    if session.source_id:
                        source = self.get_source(session.source_id)
                        session.source_name = source.name if source else None
                    if session.type == "cli":
                        self._resolve_builtin_session(session_id)
                    return session

        resolved = self._resolve_builtin_session(session_id)
        if resolved is None:
            return None
        return resolved[1]

    def get_all_session_ids(self) -> list[str]:
        """Get all session IDs from cst_sessions.

        Returns:
            List of session ID strings.
        """
        if not self.has_cst_tables():
            return []
        from .db_retrieval import get_all_session_ids

        with self._get_connection() as conn:
            return get_all_session_ids(conn)

    def _get_builtin_session_as_chat_session(self, session_id: str) -> ChatSession | None:
        """Convert Chronicle session/turns data to a ChatSession."""
        from .db_retrieval import get_builtin_session_as_chat_session

        with self._get_chronicle_connection() as (conn, has_chronicle):
            if has_chronicle:
                return get_builtin_session_as_chat_session(conn, session_id)
            return None

    def _get_cst_session(self, session_id: str) -> ChatSession | None:
        """Get a session from cst_* tables by its ID."""
        from .db_retrieval import get_cst_session

        with self._get_connection() as conn:
            return get_cst_session(conn, session_id)

    def get_messages_markdown(
        self,
        session_id: str,
        start: int | None = None,
        end: int | None = None,
        content_set: set[str] | None = None,
    ) -> str:
        """Get markdown for specific messages or all messages in a session.

        Args:
            session_id: The session ID to get messages from.
            start: Optional 1-based start message index (inclusive).
            end: Optional 1-based end message index (inclusive).
            content_set: Controls which content types to include.

        Returns:
            Combined markdown string for the selected messages.
        """
        from .db_retrieval import get_messages_markdown

        with self._get_connection() as conn:
            return get_messages_markdown(conn, session_id, start=start, end=end, content_set=content_set)

    def list_sessions(
        self,
        workspace_name: str | None = None,
        limit: int | None = None,
        offset: int = 0,
        session_type: str | None = None,
        source_name: str | None = None,
    ) -> list[dict]:
        """List sessions from cst_* tables and (optionally) Chronicle.

        Deduplicates by session_id — cst_sessions takes precedence over Chronicle.

        Args:
            workspace_name: Optional workspace name filter (cst rows only).
            limit: Maximum number of sessions to return.
            offset: Number of sessions to skip.
            session_type: Optional filter: 'cli', 'vscode', etc.
            source_name: Optional named Copilot application source.

        Returns:
            List of session info dictionaries sorted by updated_at descending.
        """
        from .db_retrieval import list_sessions

        selected_source = self.get_source(source_name) if source_name else None
        if source_name and selected_source is None:
            raise ValueError(f"Source not found: {source_name}")
        fetch_limit = None if selected_source is not None or limit is None else offset + limit
        all_cst_ids = set(self.get_all_session_ids()) if fetch_limit is not None else set()
        has_cst = self.has_cst_tables()
        with self._get_connection() as conn:
            combined = list_sessions(
                conn,
                has_cst=has_cst,
                has_chronicle=False,
                workspace_name=workspace_name,
                limit=fetch_limit,
                offset=0,
                session_type=session_type,
            )
        canonical_by_id = {row["session_id"]: row for row in combined}
        if selected_source is not None:
            combined = [row for row in combined if row.get("source_id") == selected_source.source_id]

        by_id = {row["session_id"]: row for row in combined}
        seen_builtin: dict[str, CopilotSource] = {}
        candidate_sources = [selected_source] if selected_source else self.list_sources(enabled_only=True)
        for source in candidate_sources:
            if source is None or not source.enabled:
                continue
            with self._get_source_connection(source) as (conn, has_chronicle):
                if not has_chronicle:
                    continue
                builtin = list_sessions(
                    conn,
                    has_cst=False,
                    has_chronicle=True,
                    workspace_name=None,
                    limit=fetch_limit,
                    offset=0,
                    session_type=session_type,
                )
            for row in builtin:
                native_id = row["session_id"]
                if native_id in seen_builtin:
                    self._resolve_builtin_session(native_id)
                    continue
                seen_builtin[native_id] = source
                canonical = canonical_by_id.get(native_id)
                if canonical is not None:
                    if selected_source is not None and native_id not in by_id:
                        self._resolve_builtin_session(native_id)
                        selected_row = dict(canonical)
                        selected_row["source_id"] = selected_source.source_id
                        selected_row["source_name"] = selected_source.name
                        by_id[native_id] = selected_row
                    continue
                if native_id in all_cst_ids:
                    continue
                row["native_session_id"] = native_id
                row["source_id"] = source.source_id
                row["source_name"] = source.name
                by_id[native_id] = row
        rows = sorted(
            by_id.values(),
            key=lambda row: row.get("updated_at") or row.get("last_message_at") or row.get("created_at") or "",
            reverse=True,
        )
        effective_limit = limit if limit is not None else len(rows)
        return rows[offset : offset + effective_limit]

    def get_source_session_ids(self, source: CopilotSource) -> set[str]:
        """Return non-empty Chronicle session IDs belonging to one source."""
        with self._get_connection() as conn:
            session_ids = {
                row[0]
                for row in conn.execute(
                    "SELECT session_id FROM cst_sessions WHERE source_id = ? AND type = 'cli'",
                    (source.source_id,),
                )
            }
        with self._get_source_connection(source) as (conn, has_chronicle):
            if not has_chronicle:
                return session_ids
            try:
                rows = conn.execute(
                    """
                    SELECT s.id
                    FROM chronicle.sessions s
                    WHERE EXISTS (
                        SELECT 1
                        FROM chronicle.turns t
                        WHERE t.session_id = s.id
                          AND (
                              (t.user_message IS NOT NULL AND t.user_message != '')
                              OR (t.assistant_response IS NOT NULL AND t.assistant_response != '')
                          )
                    )
                    """
                ).fetchall()
            except sqlite3.OperationalError:
                return session_ids
        session_ids.update(row[0] for row in rows)
        return session_ids

    def search(
        self,
        query: str,
        limit: int = 50,
        skip: int = 0,
        role: str | None = None,
        search_content_set: set[str] | None = None,
        include_messages: bool | None = None,
        include_tool_calls: bool | None = None,
        include_file_changes: bool | None = None,
        session_title: str | None = None,
        sort_by: str = "relevance",
        repository: str | list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        source_name: str | None = None,
    ) -> list[dict]:
        """Search messages using full-text search with field filtering.

        Supports advanced query syntax:
        - Multiple words: "python function" → matches both words (AND logic)
        - Exact phrases: '"python function"' → matches exact phrase
        - Field prefixes: 'role:user', 'workspace:myproject', 'title:something', 'repository:github.com/owner/repo'
        - Date filters: 'start_date:2024-01-01 end_date:2024-12-31' (yyyy-mm-dd format, inclusive)

        Args:
            query: The search query (supports field prefixes and quoted phrases).
            limit: Maximum number of results to return (top).
            skip: Number of results to skip (for pagination).
            role: Filter by message role ('user', 'assistant', 'system', or None for non-system messages).
                  Can also be specified in query as 'role:user', 'role:assistant', or 'role:system'.
            search_content_set: Set of content type tokens controlling which
                data sources are searched.  See ``content_types.SEARCH_CONTENT_TYPES``.
            include_messages: Deprecated — use *search_content_set*.
            include_tool_calls: Deprecated — use *search_content_set*.
            include_file_changes: Deprecated — use *search_content_set*.
            session_title: Filter by session title/workspace name.
                           Can also be specified in query as 'title:...' or 'workspace:...'.
            sort_by: Sort order - 'relevance' (default) or 'date'.
            repository: Filter by repository URL.
                        Can also be specified in query as 'repository:...' or 'repo:...'.
            start_date: Filter results on or after this date (yyyy-mm-dd format, inclusive).
                        Can also be specified in query as 'start_date:yyyy-mm-dd'.
            end_date: Filter results on or before this date (yyyy-mm-dd format, inclusive).
                      Can also be specified in query as 'end_date:yyyy-mm-dd'.

        Also queries the built-in search_index FTS table and merges results.

        Returns:
            List of matching messages with session info.
        """
        from copilot_session_tools.content_types import SEARCH_DEFAULT_INCLUDES

        from .db_search import execute_search

        # Backward compat: build set from old booleans if provided
        if search_content_set is None and any(x is not None for x in [include_messages, include_tool_calls, include_file_changes]):
            search_content_set = set()
            if include_messages is not False:
                search_content_set.add("messages")
            if include_tool_calls is not False:
                search_content_set.update(["tools", "tool-inputs"])
            if include_file_changes is not False:
                search_content_set.update(["file-changes", "diffs"])
            search_content_set.update(["thinking", "agent-details", "commands"])

        if search_content_set is None:
            search_content_set = SEARCH_DEFAULT_INCLUDES

        if not search_content_set:
            return []

        fetch_limit = 10000 if source_name else limit + skip
        with self._get_connection() as conn:
            results = execute_search(
                conn,
                query,
                limit=fetch_limit,
                skip=0,
                role=role,
                search_content_set=search_content_set,
                session_title=session_title,
                sort_by=sort_by,
                repository=repository,
                start_date=start_date,
                end_date=end_date,
                has_chronicle=False,
            )
        session_ids = {row["session_id"] for row in results}
        source_metadata: dict[str, tuple[str | None, str | None, str | None]] = {}
        if session_ids:
            placeholders = ",".join("?" for _ in session_ids)
            with self._get_connection() as conn:
                rows = conn.execute(
                    f"""
                    SELECT s.session_id, s.source_id, s.native_session_id, src.name
                    FROM cst_sessions s
                    LEFT JOIN cst_sources src ON src.source_id = s.source_id
                    WHERE s.session_id IN ({placeholders})
                    """,  # noqa: S608
                    tuple(session_ids),
                ).fetchall()
            source_metadata = {row["session_id"]: (row["source_id"], row["native_session_id"], row["name"]) for row in rows}
        for row in results:
            source_id, native_id, name = source_metadata.get(row["session_id"], (None, None, None))
            row["source_id"] = source_id
            row["native_session_id"] = native_id
            row["source_name"] = name

        selected_source = self.get_source(source_name) if source_name else None
        if source_name and selected_source is None:
            raise ValueError(f"Source not found: {source_name}")
        canonical_results = {row["session_id"]: row for row in results}
        if selected_source:
            results = [row for row in results if row.get("source_id") == selected_source.source_id]

        from .db_search import _search_builtin_index

        parsed = parse_search_query(query)
        candidate_sources = [selected_source] if selected_source else self.list_sources(enabled_only=True)
        result_ids = {row["session_id"] for row in results}
        seen_builtin: set[str] = set()
        if parsed.fts_query and "messages" in search_content_set:
            enriched_ids = set(self.get_all_session_ids())
            for source in candidate_sources:
                if source is None or not source.enabled:
                    continue
                with self._get_source_connection(source) as (conn, attached):
                    if not attached:
                        continue
                    builtin = _search_builtin_index(conn, parsed.fts_query, fetch_limit)
                for native_id, row in builtin.items():
                    if native_id in seen_builtin:
                        self._resolve_builtin_session(native_id)
                        continue
                    seen_builtin.add(native_id)
                    if native_id in result_ids:
                        continue
                    canonical = canonical_results.get(native_id)
                    if selected_source is not None and canonical is not None:
                        self._resolve_builtin_session(native_id)
                        selected_row = dict(canonical)
                        selected_row["source_id"] = selected_source.source_id
                        selected_row["source_name"] = selected_source.name
                        results.append(selected_row)
                        result_ids.add(native_id)
                        continue
                    if native_id in enriched_ids:
                        continue
                    row["session_id"] = native_id
                    row["native_session_id"] = native_id
                    row["source_id"] = source.source_id
                    row["source_name"] = source.name
                    results.append(row)
                    result_ids.add(native_id)

        return results[skip : skip + limit]

    def _get_unenriched_builtin_summaries(self) -> list[dict]:
        enriched_ids = set(self.get_all_session_ids())
        seen_ids = set(enriched_ids)
        summaries = []
        for source in self.list_sources(enabled_only=True):
            with self._get_source_connection(source) as (conn, has_chronicle):
                if not has_chronicle:
                    continue
                try:
                    rows = conn.execute(
                        """
                        SELECT
                            s.id,
                            s.repository,
                            s.cwd,
                            s.created_at,
                            s.updated_at,
                            SUM(
                                (t.user_message IS NOT NULL AND t.user_message != '')
                                + (t.assistant_response IS NOT NULL AND t.assistant_response != '')
                            ) AS message_count
                        FROM chronicle.sessions s
                        LEFT JOIN chronicle.turns t ON t.session_id = s.id
                        GROUP BY s.id
                        HAVING message_count > 0
                        """
                    ).fetchall()
                except sqlite3.OperationalError:
                    continue
            for row in rows:
                session_id = row["id"]
                if session_id in seen_ids:
                    if session_id not in enriched_ids:
                        self._resolve_builtin_session(session_id)
                    continue
                seen_ids.add(session_id)
                summaries.append(dict(row))
        return summaries

    @staticmethod
    def _workspaces_from_sessions(sessions: list[dict]) -> list[dict]:
        workspaces: dict[tuple[str, str | None], dict] = {}
        for session in sessions:
            names = session.get("workspace_names") or [session.get("workspace_name")]
            paths = session.get("workspace_paths") or [session.get("workspace_path")]
            for index, name in enumerate(names):
                if not name:
                    continue
                path = paths[index] if index < len(paths) else session.get("workspace_path")
                key = (name, path)
                item = workspaces.setdefault(
                    key,
                    {"workspace_name": name, "workspace_path": path, "session_count": 0, "last_activity": None},
                )
                item["session_count"] += 1
                item["last_activity"] = max(item["last_activity"] or "", session.get("updated_at") or session.get("created_at") or "")
        return sorted(workspaces.values(), key=lambda item: (-item["session_count"], item["workspace_name"]))

    def get_workspaces(self, sessions: list[dict] | None = None) -> list[dict]:
        """Get all unique workspaces."""
        if sessions is not None:
            return self._workspaces_from_sessions(sessions)

        from .db_retrieval import get_workspaces

        with self._get_connection() as conn:
            workspaces = {(row["workspace_name"], row["workspace_path"]): row for row in get_workspaces(conn)}
        for session in self._get_unenriched_builtin_summaries():
            name = session.get("repository")
            if not name:
                continue
            key = (name, session.get("cwd"))
            item = workspaces.setdefault(
                key,
                {
                    "workspace_name": name,
                    "workspace_path": session.get("cwd"),
                    "session_count": 0,
                    "last_activity": None,
                },
            )
            item["session_count"] += 1
            item["last_activity"] = max(item["last_activity"] or "", session.get("updated_at") or session.get("created_at") or "")
        return sorted(workspaces.values(), key=lambda item: (-item["session_count"], item["workspace_name"]))

    @staticmethod
    def _repositories_from_sessions(sessions: list[dict]) -> list[dict]:
        repositories: dict[str, dict] = {}
        for session in sessions:
            for repository in session.get("repository_urls") or [session.get("repository_url")]:
                if not repository:
                    continue
                item = repositories.setdefault(
                    repository,
                    {"repository_url": repository, "session_count": 0, "last_activity": None},
                )
                item["session_count"] += 1
                item["last_activity"] = max(item["last_activity"] or "", session.get("updated_at") or session.get("created_at") or "")
        return sorted(repositories.values(), key=lambda item: (-item["session_count"], item["repository_url"]))

    def get_repositories(self, sessions: list[dict] | None = None) -> list[dict]:
        """Get all unique repositories."""
        if sessions is not None:
            return self._repositories_from_sessions(sessions)

        from .db_retrieval import get_repositories

        with self._get_connection() as conn:
            repositories = {row["repository_url"]: row for row in get_repositories(conn)}
        for session in self._get_unenriched_builtin_summaries():
            repository = session.get("repository")
            if not repository:
                continue
            item = repositories.setdefault(
                repository,
                {"repository_url": repository, "session_count": 0, "last_activity": None},
            )
            item["session_count"] += 1
            item["last_activity"] = max(item["last_activity"] or "", session.get("updated_at") or session.get("created_at") or "")
        return sorted(repositories.values(), key=lambda item: (-item["session_count"], item["repository_url"]))

    @staticmethod
    def _stats_from_sessions(sessions: list[dict]) -> dict:
        editions: dict[str, int] = {}
        for session in sessions:
            edition = session.get("vscode_edition") or "unknown"
            editions[edition] = editions.get(edition, 0) + 1
        return {
            "session_count": len(sessions),
            "message_count": sum(session.get("message_count") or 0 for session in sessions),
            "workspace_count": len({name for session in sessions for name in (session.get("workspace_names") or [session.get("workspace_name")]) if name}),
            "editions": editions,
            "enriched_count": sum(bool(session.get("is_enriched")) for session in sessions),
            "unenriched_count": sum(not bool(session.get("is_enriched")) for session in sessions),
        }

    def get_stats(self, sessions: list[dict] | None = None) -> dict:
        """Get database statistics."""
        if sessions is not None:
            return self._stats_from_sessions(sessions)

        from .db_retrieval import get_stats

        has_cst = self.has_cst_tables()
        with self._get_connection() as conn:
            stats = get_stats(conn, has_cst=has_cst, has_chronicle=False)
            workspace_names = {
                row[0]
                for row in conn.execute(
                    "SELECT DISTINCT workspace_name FROM cst_sessions WHERE workspace_name IS NOT NULL",
                )
            }
        for session in self._get_unenriched_builtin_summaries():
            stats["session_count"] += 1
            stats["message_count"] += session.get("message_count") or 0
            stats["unenriched_count"] += 1
            stats["editions"]["cli"] = stats["editions"].get("cli", 0) + 1
            if session.get("repository"):
                workspace_names.add(session["repository"])
        stats["workspace_count"] = len(workspace_names)
        return stats

    def export_json(self) -> str:
        """Export all data as JSON."""
        sessions = []
        for session_info in self.list_sessions():
            session = self.get_session(session_info["session_id"])
            if session:
                sessions.append(
                    {
                        "session_id": session.session_id,
                        "workspace_name": session.workspace_name,
                        "workspace_path": session.workspace_path,
                        "created_at": session.created_at,
                        "updated_at": session.updated_at,
                        "vscode_edition": session.vscode_edition,
                        "source_id": session.source_id,
                        "source_name": session.source_name,
                        "native_session_id": session.native_session_id,
                        "messages": [
                            {
                                "role": msg.role,
                                "content": msg.content,
                                "timestamp": msg.timestamp,
                            }
                            for msg in session.messages
                        ],
                    }
                )
        return json.dumps(sessions, indent=2)

    def optimize_fts(self) -> dict:
        """Optimize the FTS5 full-text search index."""
        from .db_retrieval import optimize_fts

        with self._get_connection() as conn:
            return optimize_fts(conn)

    def get_builtin_session(self, session_id: str) -> dict | None:
        """Read a session from the Chronicle sessions table."""
        from .db_retrieval import get_builtin_session

        resolved = self._resolve_builtin_session(session_id)
        if resolved is None:
            return None
        source = resolved[0]
        with self._get_source_connection(source) as (conn, has_chronicle):
            if has_chronicle:
                return get_builtin_session(conn, session_id)
            return None

    def get_builtin_turns(self, session_id: str) -> list[dict]:
        """Read turns from the Chronicle turns table for a session."""
        from .db_retrieval import get_builtin_turns

        resolved = self._resolve_builtin_session(session_id)
        if resolved is None:
            return []
        source = resolved[0]
        with self._get_source_connection(source) as (conn, has_chronicle):
            if has_chronicle:
                return get_builtin_turns(conn, session_id)
            return []

    def get_builtin_checkpoints(self, session_id: str) -> list[dict]:
        """Read checkpoints from the Chronicle checkpoints table."""
        from .db_retrieval import get_builtin_checkpoints

        resolved = self._resolve_builtin_session(session_id)
        if resolved is None:
            return []
        source = resolved[0]
        with self._get_source_connection(source) as (conn, has_chronicle):
            if has_chronicle:
                return get_builtin_checkpoints(conn, session_id)
            return []

    def get_builtin_files(self, session_id: str) -> list[dict]:
        """Read file references from the Chronicle session_files table."""
        from .db_retrieval import get_builtin_files

        resolved = self._resolve_builtin_session(session_id)
        if resolved is None:
            return []
        source = resolved[0]
        with self._get_source_connection(source) as (conn, has_chronicle):
            if has_chronicle:
                return get_builtin_files(conn, session_id)
            return []

    def get_builtin_refs(self, session_id: str) -> list[dict]:
        """Read refs from the Chronicle session_refs table."""
        from .db_retrieval import get_builtin_refs

        resolved = self._resolve_builtin_session(session_id)
        if resolved is None:
            return []
        source = resolved[0]
        with self._get_source_connection(source) as (conn, has_chronicle):
            if has_chronicle:
                return get_builtin_refs(conn, session_id)
            return []

    def list_builtin_sessions(self, limit: int = 100, offset: int = 0) -> list[dict]:
        """List sessions from the Chronicle sessions table."""
        from .db_retrieval import list_builtin_sessions

        rows: list[dict] = []
        by_id: dict[str, dict] = {}
        for source in self.list_sources(enabled_only=True):
            with self._get_source_connection(source) as (conn, has_chronicle):
                if not has_chronicle:
                    continue
                source_rows = list_builtin_sessions(conn, limit=10000, offset=0)
            for row in source_rows:
                native_id = row["id"]
                if native_id in by_id:
                    self._resolve_builtin_session(native_id)
                    continue
                row["native_session_id"] = native_id
                row["source_id"] = source.source_id
                row["source_name"] = source.name
                by_id[native_id] = row
        rows.extend(by_id.values())
        rows.sort(key=lambda row: row.get("updated_at") or "", reverse=True)
        return rows[offset : offset + limit]

    def count_builtin_turns(self, session_id: str) -> int:
        """Count turns for a session in the Chronicle turns table."""
        from .db_retrieval import count_builtin_turns

        resolved = self._resolve_builtin_session(session_id)
        if resolved is None:
            return 0
        source = resolved[0]
        with self._get_source_connection(source) as (conn, has_chronicle):
            if has_chronicle:
                return count_builtin_turns(conn, session_id)
            return 0

    def enrich_session(self, session: ChatSession, source: CopilotSource | None = None) -> None:
        """Write/update cst_* tables for a parsed ChatSession.

        Idempotent: deletes existing data for this session_id, then inserts fresh.
        """
        resolved_source = source
        if resolved_source is None and session.source_id:
            resolved_source = self.get_source(session.source_id)
        if resolved_source is None:
            resolved_source = self.get_source(DEFAULT_SOURCE_ID)
        if resolved_source is not None:
            session.type = "cli"
            session.vscode_edition = "cli"
            session.source_id = resolved_source.source_id
            if session.native_session_id is None:
                session.native_session_id = session.session_id
            session.session_id = session.native_session_id
        with self._get_connection() as conn:
            if self._check_existing_session_identity(conn, session):
                return
            if resolved_source is not None:
                self._attach_chronicle(conn, resolved_source)
            db_storage.enrich_session(conn, session)

    def cleanup_orphaned_cst_sessions(self, source: CopilotSource | None = None) -> list[str]:
        """Find and delete cst_sessions whose session_id doesn't exist in the Chronicle sessions table."""
        if source is not None:
            with self._get_source_connection(source) as (conn, has_chronicle):
                return db_storage.cleanup_orphaned_cst_sessions(
                    conn,
                    has_chronicle=has_chronicle,
                    source_id=source.source_id,
                )

        live_session_ids: set[str] = set()
        sources = self.list_sources()
        for candidate in sources:
            with self._get_source_connection(candidate) as (conn, has_chronicle):
                if not has_chronicle:
                    return []
                live_session_ids.update(row[0] for row in conn.execute("SELECT id FROM chronicle.sessions"))
        with self.batch_connection() as conn:
            return db_storage.cleanup_orphaned_cst_sessions(
                conn,
                live_session_ids=live_session_ids,
            )
