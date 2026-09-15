import json
import sqlite3
from pathlib import Path

import pytest

from copilot_session_tools import ChatMessage, ChatSession, Database, session_to_html, session_to_markdown
from copilot_session_tools.refresh import run_enrichment
from copilot_session_tools.sources import (
    DEFAULT_SOURCE_ID,
    SessionIdentityConflictError,
    discover_source_presets,
    normalize_source_name,
)


def _create_chronicle(base_dir: Path) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(base_dir / "session-store.db") as conn:
        conn.executescript(
            """
            CREATE TABLE schema_version (version INTEGER NOT NULL);
            INSERT INTO schema_version VALUES (6);
            CREATE TABLE sessions (
                id TEXT PRIMARY KEY,
                cwd TEXT,
                repository TEXT,
                host_type TEXT,
                branch TEXT,
                summary TEXT,
                created_at TEXT,
                updated_at TEXT
            );
            CREATE TABLE turns (
                session_id TEXT,
                turn_index INTEGER,
                user_message TEXT,
                assistant_response TEXT
            );
            CREATE VIRTUAL TABLE search_index USING fts5(content, session_id UNINDEXED);
            """
        )


@pytest.fixture
def database(tmp_path: Path) -> Database:
    return Database(tmp_path / "archive.db")


def test_default_source_is_created_for_new_archive(database: Database) -> None:
    sources = database.list_sources()

    assert len(sources) == 1
    assert sources[0].source_id == DEFAULT_SOURCE_ID
    assert sources[0].name == "cli"
    assert sources[0].enabled is True


def test_add_source_validates_name_and_root(database: Database, tmp_path: Path) -> None:
    scout_root = tmp_path / "scout" / "copilot"
    _create_chronicle(scout_root)

    source = database.add_source("ScOuT", scout_root)

    assert source.name == "scout"
    assert database.get_source("SCOUT") == source
    assert database.get_source(source.source_id.upper()) == source
    assert source.base_dir == scout_root.resolve()
    assert source.chronicle_db_path == scout_root.resolve() / "session-store.db"
    assert source.enabled is True

    with pytest.raises(ValueError, match="already exists"):
        database.add_source(" sCoUt ", tmp_path / "other")
    with pytest.raises(ValueError, match="already registered"):
        database.add_source("Other", scout_root)
    for reserved in ("CLI", "Stable", "INSIDER", "Custom", "Copilot CLI"):
        with pytest.raises(ValueError, match="reserved"):
            normalize_source_name(reserved)


def test_rename_disable_and_enable_preserve_source_identity(database: Database, tmp_path: Path) -> None:
    scout_root = tmp_path / "scout"
    _create_chronicle(scout_root)
    source = database.add_source("Scout", scout_root)

    renamed = database.rename_source("SCOUT", "Research Agent")
    disabled = database.set_source_enabled("RESEARCH AGENT", enabled=False)
    enabled = database.set_source_enabled("Research Agent", enabled=True)

    assert renamed.source_id == source.source_id
    assert renamed.name == "research agent"
    assert disabled.source_id == source.source_id
    assert disabled.enabled is False
    assert enabled.enabled is True


def test_scout_preset_is_only_returned_when_present(tmp_path: Path) -> None:
    assert discover_source_presets(tmp_path) == []

    scout_root = tmp_path / ".scout" / "copilot"
    _create_chronicle(scout_root)

    presets = discover_source_presets(tmp_path)
    assert [(preset.preset_id, preset.name, preset.base_dir) for preset in presets] == [("scout", "scout", scout_root)]


def test_same_uuid_in_multiple_sources_is_one_session(database: Database, tmp_path: Path) -> None:
    default_root = tmp_path
    _create_chronicle(default_root)
    scout_root = tmp_path / "scout"
    _create_chronicle(scout_root)
    session_id = "11111111-1111-1111-1111-111111111111"
    for root in (default_root, scout_root):
        with sqlite3.connect(root / "session-store.db") as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, "C:\\work", "owner/repo", "cli", "main", "Shared session", "2026-01-01", "2026-01-02"),
            )
            conn.execute("INSERT INTO turns VALUES (?, 0, ?, ?)", (session_id, "shared question", "shared answer"))
            conn.execute(
                "INSERT INTO search_index (content, session_id) VALUES (?, ?)",
                ("shared question shared answer", session_id),
            )
    database.add_source("Scout", scout_root)

    listed = [row for row in database.list_sessions() if row["session_id"] == session_id]
    scout_listed = database.list_sessions(source_name="Scout")
    session = database.get_session(session_id)
    scout_results = database.search("shared question", source_name="Scout")

    assert len(listed) == 1
    assert [row["session_id"] for row in scout_listed] == [session_id]
    assert scout_listed[0]["source_name"] == "scout"
    assert session is not None
    assert session.session_id == session_id
    assert session.messages[0].content == "shared question"
    assert [row["session_id"] for row in scout_results] == [session_id]


def test_conflicting_copies_of_same_uuid_raise_integrity_error(database: Database, tmp_path: Path) -> None:
    default_root = tmp_path
    _create_chronicle(default_root)
    scout_root = tmp_path / "scout"
    _create_chronicle(scout_root)
    session_id = "11111111-1111-1111-1111-111111111111"
    for root, summary in ((default_root, "Default data"), (scout_root, "Conflicting Scout data")):
        with sqlite3.connect(root / "session-store.db") as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, "C:\\work", "owner/repo", "cli", "main", summary, "2026-01-01", "2026-01-02"),
            )
            conn.execute("INSERT INTO turns VALUES (?, 0, ?, ?)", (session_id, "question", "answer"))
    database.add_source("Scout", scout_root)

    with pytest.raises(SessionIdentityConflictError, match=session_id):
        database.list_sessions()
    with pytest.raises(SessionIdentityConflictError, match=session_id):
        database.get_session(session_id)


def test_conflicting_enriched_copy_of_same_uuid_raises_integrity_error(database: Database, tmp_path: Path) -> None:
    scout_root = tmp_path / "scout"
    _create_chronicle(scout_root)
    scout = database.add_source("Scout", scout_root)
    session_id = "11111111-1111-1111-1111-111111111111"
    database.add_session(
        ChatSession(
            session_id=session_id,
            native_session_id=session_id,
            source_id=DEFAULT_SOURCE_ID,
            workspace_name=None,
            workspace_path=None,
            messages=[ChatMessage(role="user", content="default content")],
            type="cli",
            vscode_edition="cli",
        )
    )
    conflicting = ChatSession(
        session_id=session_id,
        native_session_id=session_id,
        source_id=scout.source_id,
        workspace_name=None,
        workspace_path=None,
        messages=[ChatMessage(role="user", content="scout content")],
        type="cli",
        vscode_edition="cli",
    )

    with pytest.raises(SessionIdentityConflictError, match=session_id):
        database.add_session(conflicting)


def test_identical_enriched_copy_of_same_uuid_is_deduplicated(database: Database, tmp_path: Path) -> None:
    scout_root = tmp_path / "scout"
    _create_chronicle(scout_root)
    scout = database.add_source("Scout", scout_root)
    session_id = "11111111-1111-1111-1111-111111111111"
    original = ChatSession(
        session_id=session_id,
        native_session_id=session_id,
        source_id=DEFAULT_SOURCE_ID,
        workspace_name=None,
        workspace_path=None,
        messages=[ChatMessage(role="user", content="same content")],
        type="cli",
        vscode_edition="cli",
    )
    duplicate = ChatSession(
        session_id=session_id,
        native_session_id=session_id,
        source_id=scout.source_id,
        workspace_name=None,
        workspace_path=None,
        messages=[ChatMessage(role="user", content="same content")],
        type="cli",
        vscode_edition="cli",
    )

    assert database.add_session(original) is True
    assert database.add_session(duplicate) is False


def test_source_filter_uses_enriched_canonical_copy_for_shared_uuid(database: Database, tmp_path: Path) -> None:
    default_root = tmp_path
    _create_chronicle(default_root)
    scout_root = tmp_path / "scout"
    _create_chronicle(scout_root)
    session_id = "12121212-1212-1212-1212-121212121212"
    for root in (default_root, scout_root):
        with sqlite3.connect(root / "session-store.db") as conn:
            conn.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, "C:\\work", "owner/repo", "cli", "main", "Shared session", "2026-01-01", "2026-01-02"),
            )
            conn.execute("INSERT INTO turns VALUES (?, 0, ?, ?)", (session_id, "shared question", "shared answer"))
            conn.execute(
                "INSERT INTO search_index (content, session_id) VALUES (?, ?)",
                ("shared question shared answer", session_id),
            )
    database.add_source("Scout", scout_root)
    database.add_session(
        ChatSession(
            session_id=session_id,
            native_session_id=session_id,
            source_id=DEFAULT_SOURCE_ID,
            workspace_name=None,
            workspace_path=None,
            messages=[
                ChatMessage(role="user", content="shared question"),
                ChatMessage(role="assistant", content="shared answer"),
            ],
            type="cli",
            vscode_edition="cli",
        )
    )

    listed = database.list_sessions(source_name="Scout")
    results = database.search("shared question", source_name="Scout")

    assert [row["session_id"] for row in listed] == [session_id]
    assert listed[0]["is_enriched"] is True
    assert listed[0]["source_name"] == "scout"
    assert [row["session_id"] for row in results] == [session_id]
    assert results[0]["source_name"] == "scout"


def test_v13_qualified_session_id_migrates_back_to_native_uuid(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    database = Database(db_path)
    session_id = "11111111-1111-1111-1111-111111111111"
    qualified_id = "src:legacy-source:11111111-1111-1111-1111-111111111111"
    database.add_session(
        ChatSession(
            session_id=session_id,
            native_session_id=session_id,
            source_id=DEFAULT_SOURCE_ID,
            workspace_name=None,
            workspace_path=None,
            messages=[ChatMessage(role="user", content="migrated content")],
            type="cli",
            vscode_edition="cli",
        )
    )
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = OFF")
        conn.execute("DROP INDEX idx_cst_sessions_native")
        conn.execute(
            "UPDATE cst_sessions SET session_id = ? WHERE session_id = ?",
            (qualified_id, session_id),
        )
        conn.execute(
            "UPDATE cst_messages SET session_id = ? WHERE session_id = ?",
            (qualified_id, session_id),
        )
        conn.execute("UPDATE cst_schema_version SET version = 13")

    migrated = Database(db_path).get_session(session_id)

    assert migrated is not None
    assert migrated.session_id == session_id
    assert migrated.messages[0].content == "migrated content"
    with sqlite3.connect(db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM cst_sessions WHERE session_id = ?",
                (qualified_id,),
            ).fetchone()[0]
            == 0
        )


def test_v14_source_names_migrate_to_lowercase_identifiers(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.db"
    database = Database(db_path)
    scout_root = tmp_path / "scout"
    _create_chronicle(scout_root)
    source = database.add_source("scout", scout_root)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE cst_sources SET name = 'Copilot CLI', name_key = 'copilot cli' WHERE source_id = ?",
            (DEFAULT_SOURCE_ID,),
        )
        conn.execute(
            "UPDATE cst_sources SET name = 'Scout' WHERE source_id = ?",
            (source.source_id,),
        )
        conn.execute("UPDATE cst_schema_version SET version = 14")

    migrated = Database(db_path)
    cli = migrated.get_source("CLI")
    scout = migrated.get_source("SCOUT")

    assert cli is not None
    assert scout is not None
    assert cli.name == "cli"
    assert scout.name == "scout"


def test_custom_source_chronicle_fallback_uses_native_uuid(database: Database, tmp_path: Path) -> None:
    scout_root = tmp_path / "scout"
    _create_chronicle(scout_root)
    session_id = "22222222-2222-2222-2222-222222222222"
    with sqlite3.connect(scout_root / "session-store.db") as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, "C:\\work", "owner/repo", "cli", "main", "Scout session", "2026-01-01", "2026-01-02"),
        )
        conn.execute("INSERT INTO turns VALUES (?, 0, ?, ?)", (session_id, "scout question", "scout answer"))
        conn.execute("INSERT INTO search_index (content, session_id) VALUES (?, ?)", ("scout question scout answer", session_id))
    scout = database.add_source("Scout", scout_root)

    listed = {row["session_id"]: row for row in database.list_sessions()}
    session = database.get_session(session_id)

    assert session_id in listed
    assert listed[session_id]["source_name"] == "scout"
    assert listed[session_id]["native_session_id"] == session_id
    assert session is not None
    assert session.session_id == session_id
    assert session.native_session_id == session_id
    assert session.source_id == scout.source_id
    assert session.messages[0].content == "scout question"
    results = database.search("scout question", source_name="Scout")
    assert [row["session_id"] for row in results] == [session_id]


def test_custom_source_jsonl_is_enriched_under_native_uuid(
    database: Database,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scout_root = tmp_path / "scout"
    _create_chronicle(scout_root)
    native_id = "33333333-3333-3333-3333-333333333333"
    with sqlite3.connect(scout_root / "session-store.db") as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (native_id, "C:\\work", "owner/repo", "cli", "main", "Scout session", "2026-01-01", "2026-01-02"),
        )
        conn.execute("INSERT INTO turns VALUES (?, 0, ?, ?)", (native_id, "question", "answer"))
    events_dir = scout_root / "session-state" / native_id
    events_dir.mkdir(parents=True)
    events = [
        {"type": "session.start", "data": {"sessionId": native_id, "startTime": "2026-01-01T00:00:00Z"}},
        {"type": "user.message", "data": {"content": "rich question"}},
        {"type": "assistant.message", "data": {"content": "rich answer"}},
        {
            "type": "session.info",
            "data": {
                "infoType": "fork",
                "message": 'Forked this session into "child" (44444444-4444-4444-4444-444444444444).',
            },
        },
    ]
    (events_dir / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
    scout = database.add_source("Scout", scout_root)

    def fail_if_constructed(*args, **kwargs):
        raise AssertionError("ProcessPoolExecutor should not be constructed with one worker")

    monkeypatch.setattr("copilot_session_tools.refresh.ProcessPoolExecutor", fail_if_constructed)
    original_enrich_session = database.enrich_session

    def assert_batch_active(*args, **kwargs):
        assert database._batch_conn is not None
        return original_enrich_session(*args, **kwargs)

    monkeypatch.setattr(database, "enrich_session", assert_batch_active)

    result = run_enrichment(database, workers=1)
    session = database.get_session(native_id)

    assert result.enriched == 1
    assert result.failed == 0
    assert session is not None
    assert session.messages[0].content == "rich question"
    assert session.source_id == scout.source_id
    assert session.native_session_id == native_id
    block_text = "\n".join(block.content for message in session.messages for block in message.content_blocks)
    assert "44444444-4444-4444-4444-444444444444" in block_text
    assert "src:" not in block_text
    assert "**Source:** scout" in session_to_markdown(session)
    assert "Application: scout" in session_to_html(session)
    exported = json.loads(database.export_json())
    assert exported[0]["source_name"] == "scout"
    assert exported[0]["native_session_id"] == native_id


def test_enrichment_without_cli_work_does_not_start_process_pool(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_constructed(*args, **kwargs):
        raise AssertionError("ProcessPoolExecutor should not be constructed without CLI work")

    monkeypatch.setattr("copilot_session_tools.refresh.ProcessPoolExecutor", fail_if_constructed)

    run_enrichment(database, workers=1)


def test_cleanup_is_scoped_to_one_source(database: Database, tmp_path: Path) -> None:
    default_db = tmp_path / "session-store.db"
    with sqlite3.connect(default_db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (id TEXT PRIMARY KEY);
            CREATE TABLE turns (session_id TEXT, turn_index INTEGER);
            """
        )
    scout_root = tmp_path / "scout"
    _create_chronicle(scout_root)
    scout = database.add_source("Scout", scout_root)
    native_id = "55555555-5555-5555-5555-555555555555"
    database.add_session(
        ChatSession(
            session_id=native_id,
            native_session_id=native_id,
            source_id=scout.source_id,
            workspace_name=None,
            workspace_path=None,
            messages=[ChatMessage(role="user", content="keep me")],
            type="cli",
            vscode_edition="cli",
        )
    )

    default_source = database.get_source(DEFAULT_SOURCE_ID)
    assert default_source is not None
    database.cleanup_orphaned_cst_sessions(default_source)

    with sqlite3.connect(database.db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM cst_sessions WHERE session_id = ?",
                (native_id,),
            ).fetchone()[0]
            == 1
        )


def test_cleanup_keeps_shared_uuid_present_in_another_source(database: Database, tmp_path: Path) -> None:
    _create_chronicle(tmp_path)
    scout_root = tmp_path / "scout"
    _create_chronicle(scout_root)
    native_id = "56565656-5656-5656-5656-565656565656"
    with sqlite3.connect(scout_root / "session-store.db") as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (native_id, "C:\\work", "owner/repo", "cli", "main", "Shared session", "2026-01-01", "2026-01-02"),
        )
        conn.execute("INSERT INTO turns VALUES (?, 0, ?, ?)", (native_id, "question", "answer"))
    database.add_source("Scout", scout_root)
    database.add_session(
        ChatSession(
            session_id=native_id,
            native_session_id=native_id,
            source_id=DEFAULT_SOURCE_ID,
            workspace_name=None,
            workspace_path=None,
            messages=[ChatMessage(role="user", content="question")],
            type="cli",
            vscode_edition="cli",
        )
    )

    database.cleanup_orphaned_cst_sessions()

    with sqlite3.connect(database.db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM cst_sessions WHERE session_id = ?",
                (native_id,),
            ).fetchone()[0]
            == 1
        )


def test_cleanup_uses_session_id_when_native_uuid_is_missing(database: Database, tmp_path: Path) -> None:
    _create_chronicle(tmp_path)
    session_id = "57575757-5757-5757-5757-575757575757"
    with sqlite3.connect(tmp_path / "session-store.db") as conn:
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, "C:\\work", "owner/repo", "cli", "main", "Live session", "2026-01-01", "2026-01-02"),
        )
        conn.execute("INSERT INTO turns VALUES (?, 0, ?, ?)", (session_id, "question", "answer"))
    database.add_session(
        ChatSession(
            session_id=session_id,
            workspace_name=None,
            workspace_path=None,
            messages=[ChatMessage(role="user", content="question")],
            type="cli",
            vscode_edition="cli",
        )
    )

    database.cleanup_orphaned_cst_sessions()

    with sqlite3.connect(database.db_path) as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM cst_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            == 1
        )


def test_default_source_path_collision_has_actionable_error(database: Database, tmp_path: Path) -> None:
    source_root = tmp_path / "other"
    _create_chronicle(source_root)
    database.add_source("Other", source_root)

    with pytest.raises(ValueError, match="already registered as source 'other'"):
        Database(database.db_path, chronicle_db_path=source_root / "session-store.db")


def test_vscode_reparse_is_not_sent_to_cli_parser(
    database: Database,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session_id = "vscode-old-parser"
    database.add_session(
        ChatSession(
            session_id=session_id,
            workspace_name="workspace",
            workspace_path="C:\\work",
            messages=[ChatMessage(role="user", content="question")],
            vscode_edition="stable",
        )
    )
    with sqlite3.connect(database.db_path) as conn:
        conn.execute("UPDATE cst_sessions SET parser_version = 0 WHERE session_id = ?", (session_id,))

    def fail_if_parsed(entry):
        raise AssertionError(f"VS Code session was sent to CLI parser: {entry['session_id']}")

    monkeypatch.setattr("copilot_session_tools.refresh._parse_cli_entry", fail_if_parsed)

    result = run_enrichment(database, workers=1)

    assert result.failed == 0
