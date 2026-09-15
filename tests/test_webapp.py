"""Tests for the webapp module."""

import sqlite3
import tempfile
from pathlib import Path

import pytest

from copilot_session_tools import ChatMessage, ChatSession, ContentBlock, Database, RootAgentInterval, SessionContextEntry
from copilot_session_tools.sources import SessionIdentityConflictError
from copilot_session_tools.utils import (
    extract_filename as _extract_filename,
)
from copilot_session_tools.utils import (
    markdown_to_html as _markdown_to_html,
)
from copilot_session_tools.utils import (
    parse_diff_stats as _parse_diff_stats,
)
from copilot_session_tools.utils import (
    strip_ansi as _strip_ansi,
)
from copilot_session_tools.web.webapp import (
    create_app,
)


@pytest.fixture
def temp_db():
    """Create a temporary database with sample data."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    db = Database(db_path)

    # Add sample session
    session = ChatSession(
        session_id="webapp-test-session",
        workspace_name="test-workspace",
        workspace_path="/home/user/test",
        messages=[
            ChatMessage(role="user", content="What is Python?"),
            ChatMessage(
                role="assistant",
                content="Python is a programming language.\n\n```python\nprint('Hello')\n```",
            ),
        ],
        created_at="2025-01-15T10:00:00Z",
        vscode_edition="stable",
    )
    db.add_session(session)

    yield db_path

    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def app(temp_db):
    """Create a Flask test app with empty storage paths for fast tests."""
    app = create_app(temp_db, title="Test Archive", storage_paths=[])
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    """Create a Flask test client."""
    return app.test_client()


def test_index_materializes_sessions_once(app, client, monkeypatch):
    original = Database.list_sessions
    calls = []

    def tracked_list_sessions(self, *args, **kwargs):
        calls.append(kwargs.get("source_name"))
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Database, "list_sessions", tracked_list_sessions)

    response = client.get("/")

    assert response.status_code == 200
    assert calls == [None]


class TestMarkdownToHtml:
    """Tests for the markdown to HTML converter."""

    def test_plain_text(self):
        """Test converting plain text."""
        result = _markdown_to_html("Hello, world!")
        assert "Hello, world!" in result


class TestSourceManagement:
    def test_add_rename_disable_and_enable_source(self, client, temp_db, tmp_path):
        source_dir = tmp_path / "scout"
        source_dir.mkdir()
        with sqlite3.connect(source_dir / "session-store.db") as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (id TEXT PRIMARY KEY);
                CREATE TABLE turns (session_id TEXT, turn_index INTEGER);
                """
            )

        added = client.post(
            "/sources/add",
            data={"name": "Scout", "base_dir": str(source_dir)},
            follow_redirects=True,
        )
        database = Database(temp_db)
        scout = database.get_source("Scout")
        assert added.status_code == 200
        assert b"scout" in added.data
        assert scout is not None

        renamed = client.post(
            f"/sources/{scout.source_id}/rename",
            data={"name": "Microsoft Scout"},
            follow_redirects=True,
        )
        disabled = client.post(
            f"/sources/{scout.source_id}/toggle",
            data={"enabled": "false"},
            follow_redirects=True,
        )
        enabled = client.post(
            f"/sources/{scout.source_id}/toggle",
            data={"enabled": "true"},
            follow_redirects=True,
        )

        assert renamed.status_code == 200
        assert b"microsoft scout" in renamed.data
        assert disabled.status_code == 200
        assert b"disabled" in disabled.data
        assert enabled.status_code == 200
        assert b"enabled" in enabled.data

    def test_inline_code(self):
        """Test converting inline code."""
        result = _markdown_to_html("Use `print()` function")
        assert "<code>print()</code>" in result

    def test_code_block(self):
        """Test converting code blocks."""
        text = "```python\nprint('hello')\n```"
        result = _markdown_to_html(text)
        assert "<pre>" in result
        assert "<code" in result
        assert "print" in result


class TestParseDiffStats:
    """Tests for the diff statistics parser."""

    def test_empty_diff(self):
        """Test parsing empty diff returns zeros."""
        result = _parse_diff_stats("")
        assert result == {"additions": 0, "deletions": 0}

    def test_none_diff(self):
        """Test parsing None diff returns zeros."""
        result = _parse_diff_stats(None)
        assert result == {"additions": 0, "deletions": 0}

    def test_additions_only(self):
        """Test parsing diff with only additions."""
        diff = """--- a/file.py
+++ b/file.py
@@ -1,3 +1,5 @@
 existing line
+new line 1
+new line 2
 another existing"""
        result = _parse_diff_stats(diff)
        assert result["additions"] == 2
        assert result["deletions"] == 0

    def test_deletions_only(self):
        """Test parsing diff with only deletions."""
        diff = """--- a/file.py
+++ b/file.py
@@ -1,5 +1,3 @@
 existing line
-removed line 1
-removed line 2
 another existing"""
        result = _parse_diff_stats(diff)
        assert result["additions"] == 0
        assert result["deletions"] == 2

    def test_mixed_changes(self):
        """Test parsing diff with both additions and deletions."""
        diff = """--- a/file.py
+++ b/file.py
@@ -1,4 +1,4 @@
 existing line
-old code
+new code
+another new line
-old line removed
 final line"""
        result = _parse_diff_stats(diff)
        assert result["additions"] == 2
        assert result["deletions"] == 2

    def test_skips_hunk_headers(self):
        """Test that hunk headers (@@ lines) are not counted as deletions."""
        diff = """--- a/file.py
+++ b/file.py
@@ -1,5 +1,6 @@
 line 1
+added line
@@ -10,3 +11,4 @@
 line 10
+another added"""
        result = _parse_diff_stats(diff)
        # Should have 2 additions, 0 deletions (hunk headers should be skipped)
        assert result["additions"] == 2
        assert result["deletions"] == 0


class TestExtractFilename:
    """Tests for the filename extractor."""

    def test_empty_path(self):
        """Test extracting from empty path."""
        assert _extract_filename("") == ""
        assert _extract_filename(None) == ""

    def test_unix_path(self):
        """Test extracting from Unix-style path."""
        assert _extract_filename("/home/user/project/file.py") == "file.py"
        assert _extract_filename("src/main.py") == "main.py"

    def test_windows_path(self):
        """Test extracting from Windows-style path."""
        assert _extract_filename("C:\\Users\\user\\file.py") == "file.py"
        assert _extract_filename("src\\main.py") == "main.py"

    def test_filename_only(self):
        """Test extracting when input is just a filename."""
        assert _extract_filename("file.py") == "file.py"


class TestStripAnsi:
    """Tests for the ANSI escape code stripper."""

    def test_empty_string(self):
        """Test stripping empty string returns empty."""
        assert _strip_ansi("") == ""
        assert _strip_ansi(None) == ""

    def test_no_ansi_codes(self):
        """Test text without ANSI codes is unchanged."""
        text = "Hello, world!"
        assert _strip_ansi(text) == text

    def test_sgr_color_codes(self):
        """Test stripping SGR color codes like [32;1m."""
        # Green bold text: ESC[32;1m ... ESC[0m
        text = "\x1b[32;1mGreen bold text\x1b[0m"
        assert _strip_ansi(text) == "Green bold text"

    def test_multiple_color_codes(self):
        """Test stripping multiple color codes from terminal output."""
        # Simulating typical colored terminal output
        text = "\x1b[32;1mBytesSentSrcToDest\x1b[0m : \x1b[0m218530"
        assert _strip_ansi(text) == "BytesSentSrcToDest : 218530"

    def test_cursor_control_codes(self):
        """Test stripping cursor control codes like [2J."""
        text = "\x1b[2J\x1b[HHello"
        assert _strip_ansi(text) == "Hello"

    def test_real_terminal_output(self):
        """Test stripping ANSI from realistic PowerShell/terminal output."""
        # Example from the bug report
        text = """\x1b[32;1m1mBytesSentSrcToDest\x1b[0m  : \x1b[0m218530
\x1b[32;1m1mPacketsSentSrcToDest\x1b[0m : \x1b[0m2460
\x1b[32;1m1mStartTime\x1b[0m           : \x1b[0m10/12/2025 6:00:00"""
        result = _strip_ansi(text)
        assert "\x1b[" not in result
        assert "BytesSentSrcToDest" in result
        assert "218530" in result

    def test_preserves_newlines(self):
        """Test that newlines are preserved after stripping ANSI."""
        text = "\x1b[32mLine 1\x1b[0m\nLine 2\n\x1b[31mLine 3\x1b[0m"
        result = _strip_ansi(text)
        assert result == "Line 1\nLine 2\nLine 3"


class TestWebappRoutes:
    """Tests for the webapp routes."""

    def test_index_route(self, client):
        """Test the index route returns 200."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"Test Archive" in response.data

    def test_identity_conflict_returns_explicit_error(self, client, monkeypatch):
        def raise_conflict(*args, **kwargs):
            raise SessionIdentityConflictError("Session UUID duplicate contains conflicting data.")

        monkeypatch.setattr(Database, "list_sessions", raise_conflict)

        response = client.get("/")

        assert response.status_code == 409
        assert b"Session UUID duplicate contains conflicting data." in response.data

    def test_application_buttons_separate_cli_from_added_source(
        self,
        client,
        temp_db,
        tmp_path,
    ):
        database = Database(temp_db)
        source_dir = tmp_path / "scout"
        source_dir.mkdir()
        with sqlite3.connect(source_dir / "session-store.db") as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (id TEXT PRIMARY KEY);
                CREATE TABLE turns (
                    session_id TEXT,
                    turn_index INTEGER,
                    user_message TEXT,
                    assistant_response TEXT
                );
                """
            )
        scout = database.add_source("Scout", source_dir)
        database.add_session(
            ChatSession(
                session_id="11111111-1111-1111-1111-111111111111",
                native_session_id="11111111-1111-1111-1111-111111111111",
                source_id="default-cli",
                workspace_name=None,
                workspace_path=None,
                messages=[ChatMessage(role="user", content="default cli content")],
                custom_title="Default CLI session",
                type="cli",
                vscode_edition="cli",
            )
        )
        database.add_session(
            ChatSession(
                session_id="22222222-2222-2222-2222-222222222222",
                native_session_id="22222222-2222-2222-2222-222222222222",
                source_id=scout.source_id,
                workspace_name=None,
                workspace_path=None,
                messages=[ChatMessage(role="user", content="scout content")],
                custom_title="Scout session",
                type="cli",
                vscode_edition="cli",
            )
        )

        unfiltered = client.get("/")
        cli_only = client.get("/?edition=cli")
        scout_only = client.get(f"/?edition=source:{scout.source_id}")

        assert f'value="source:{scout.source_id}"'.encode() in unfiltered.data
        assert b">scout</span>" in unfiltered.data
        assert b"Default CLI session" in cli_only.data
        assert b"Scout session" not in cli_only.data
        assert b"Scout session" in scout_only.data
        assert b"Default CLI session" not in scout_only.data
        scout_card = scout_only.data.split(b"Scout session", 1)[1].split(b"</li>", 1)[0]
        assert b'class="edition-badge cli">scout</span>' in scout_card
        assert b'class="edition-badge cli">cli</span>' not in scout_card

    def test_index_shows_sessions(self, client):
        """Test the index shows sessions."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"test-workspace" in response.data

    def test_index_search(self, client):
        """Test the index search functionality."""
        response = client.get("/?q=Python")
        assert response.status_code == 200
        # Should show search results or the matching session
        assert b"test-workspace" in response.data or b"Python" in response.data

    def test_session_route(self, client):
        """Test the session route returns 200."""
        response = client.get("/session/webapp-test-session")
        assert response.status_code == 200
        assert b"test-workspace" in response.data

    def test_session_shows_messages(self, client):
        """Test the session route shows messages."""
        response = client.get("/session/webapp-test-session")
        assert response.status_code == 200
        assert b"What is Python?" in response.data

    def test_session_top_level_messages_are_collapsible(self, client):
        """User and assistant messages render as open-by-default details."""
        response = client.get("/session/webapp-test-session")
        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert html.count('<details class="message-details" open>') == 2
        assert '<summary class="message-header">' in html

    def test_session_exposes_system_messages_toggle(self, client):
        """System messages are available as an explicit content option."""
        response = client.get("/session/webapp-test-session")
        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert "System messages" in html
        assert 'x-model="systemMessages"' in html

    def test_session_content_set_omits_system_messages_by_default(self, client):
        """Web-viewer export content sets omit system messages unless toggled."""
        response = client.get("/session/webapp-test-session")
        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert "if (this.systemMessages) set.push('system-messages');" in html

    def test_session_system_messages_open_by_default(self, temp_db):
        """System-message parent blocks render open by default in web view."""
        db = Database(temp_db)
        session = ChatSession(
            session_id="webapp-system-session",
            workspace_name="test-workspace",
            workspace_path="/home/user/test",
            messages=[
                ChatMessage(role="user", content="hello"),
                ChatMessage(role="system", content="System instructions go here"),
                ChatMessage(role="assistant", content="ok"),
            ],
            created_at="2025-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)

        app = create_app(temp_db, title="Test Archive", storage_paths=[])
        app.config["TESTING"] = True
        response = app.test_client().get("/session/webapp-system-session")

        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert '<details class="message-details system-message-details" open>' in html

    def test_session_shows_root_agent_transition_pills(self, temp_db):
        """Root custom-agent intervals render as timeline transition pills."""
        db = Database(temp_db)
        session = ChatSession(
            session_id="root-agent-web-session",
            workspace_name="test-workspace",
            workspace_path="/home/user/test",
            messages=[
                ChatMessage(role="user", content="Merge main", timestamp="2026-01-01T00:00:02Z"),
                ChatMessage(role="assistant", content="Merged.", timestamp="2026-01-01T00:00:03Z"),
                ChatMessage(role="user", content="Thanks", timestamp="2026-01-01T00:00:05Z"),
            ],
            root_agent_intervals=[
                RootAgentInterval(
                    agent_name="smart-merge",
                    agent_display_name="smart-merge",
                    start_timestamp="2026-01-01T00:00:01Z",
                    end_timestamp="2026-01-01T00:00:04Z",
                )
            ],
        )
        db.add_session(session)

        app = create_app(temp_db, title="Test Archive", storage_paths=[])
        app.config["TESTING"] = True
        response = app.test_client().get("/session/root-agent-web-session")

        assert response.status_code == 200
        assert b"Entered smart-merge" in response.data
        assert b"Back to default" in response.data

    def test_session_uses_inline_context_change_message(self, temp_db):
        """Context changes render as inline status messages, not a session summary."""
        db = Database(temp_db)
        session = ChatSession(
            session_id="context-inline-session",
            workspace_name="project-one",
            workspace_path="/workspace/one",
            repository_url="github.com/example/one",
            messages=[
                ChatMessage(role="user", content="start"),
                ChatMessage(
                    role="assistant",
                    content="Working directory changed: /workspace/two (feature/x)",
                    content_blocks=[
                        ContentBlock(
                            kind="status",
                            content="Working directory changed: /workspace/two (feature/x)",
                            description="context-change",
                        )
                    ],
                ),
            ],
            context_entries=[
                SessionContextEntry(message_index=0, workspace_name="project-one", workspace_path="/workspace/one", repository_url="github.com/example/one", source="initial"),
                SessionContextEntry(
                    message_index=1,
                    workspace_name="project-two",
                    workspace_path="/workspace/two",
                    repository_url="github.com/example/two",
                    branch="feature/x",
                    source="context_changed",
                ),
            ],
        )
        db.add_session(session)

        app = create_app(temp_db, title="Test Archive", storage_paths=[])
        app.config["TESTING"] = True
        response = app.test_client().get("/session/context-inline-session")

        assert response.status_code == 200
        html = response.data.decode("utf-8")
        assert "Context changed 1 time" not in html
        assert "Working directory changed: /workspace/two (feature/x)" in html
        assert "status-badge model-change context-change" in html

    def test_search_snippet_context_only_when_different_from_root(self, temp_db):
        """Search snippets omit root context but show later active context."""
        db = Database(temp_db)
        session = ChatSession(
            session_id="context-snippet-session",
            workspace_name="project-one",
            workspace_path="/workspace/one",
            repository_url="github.com/example/one",
            messages=[
                ChatMessage(role="user", content="alpha in the first project"),
                ChatMessage(role="assistant", content="beta in the second project"),
            ],
            context_entries=[
                SessionContextEntry(message_index=0, workspace_name="project-one", workspace_path="/workspace/one", repository_url="github.com/example/one", source="initial"),
                SessionContextEntry(
                    message_index=1, workspace_name="project-two", workspace_path="/workspace/two", repository_url="github.com/example/two", source="context_changed"
                ),
            ],
        )
        db.add_session(session)

        app = create_app(temp_db, title="Test Archive", storage_paths=[])
        app.config["TESTING"] = True
        client = app.test_client()

        root_response = client.get("/?q=alpha")
        assert root_response.status_code == 200
        assert '<div class="snippet-context">' not in root_response.data.decode("utf-8")

        changed_response = client.get("/?q=beta")
        assert changed_response.status_code == 200
        changed_html = changed_response.data.decode("utf-8")
        assert '<div class="snippet-context">' in changed_html
        assert "project-two" in changed_html
        assert "github.com/example/two" in changed_html

    def test_session_not_found(self, client):
        """Test 404 for non-existent session."""
        response = client.get("/session/nonexistent-session-id")
        assert response.status_code == 404
        assert b"Session not found" in response.data

    def test_empty_search(self, client):
        """Test empty search query shows all sessions."""
        response = client.get("/?q=")
        assert response.status_code == 200
        assert b"test-workspace" in response.data

    def test_search_with_sort_by_relevance(self, client):
        """Test search with relevance sorting."""
        response = client.get("/?q=Python&sort=relevance")
        assert response.status_code == 200
        # Should show the sort dropdown
        assert b'select name="sort"' in response.data

    def test_search_with_sort_by_date(self, client):
        """Test search with date sorting."""
        response = client.get("/?q=Python&sort=date")
        assert response.status_code == 200
        assert b'<option value="date"' in response.data

    def test_search_help_tips_shown(self, client):
        """Test that search help tips are shown."""
        response = client.get("/")
        assert response.status_code == 200
        assert b"exact phrase" in response.data or b"role:user" in response.data


class TestSearchContentTypeFilters:
    """Tests for search_in content-type filtering on the index route."""

    def test_search_default_no_search_in(self, client):
        """Default search (no search_in) searches everything — backward compat."""
        response = client.get("/?q=Python")
        assert response.status_code == 200
        html = response.data.decode()
        # All content-type checkboxes should be checked by default
        assert 'name="search_in"' in html
        # No hidden search_in inputs should be rendered (empty selected list)
        assert '<input type="hidden" name="search_in"' not in html

    def test_search_with_search_in_params(self, client):
        """search_in params are read and passed to the template."""
        response = client.get("/?q=Python&search_in=tools&search_in=file-changes")
        assert response.status_code == 200
        html = response.data.decode()
        # Hidden inputs should be rendered for the selected types
        assert '<input type="hidden" name="search_in" value="tools">' in html
        assert '<input type="hidden" name="search_in" value="file-changes">' in html

    def test_search_invalid_search_in_ignored(self, client):
        """Invalid search_in values don't crash — fall back to defaults."""
        response = client.get("/?q=Python&search_in=bogus")
        assert response.status_code == 200
        # Should render without error

    def test_search_all_types_selected(self, client):
        """All 8 types selected behaves same as no filter."""
        all_types = [
            "messages",
            "thinking",
            "diffs",
            "tool-inputs",
            "agent-details",
            "tools",
            "commands",
            "file-changes",
        ]
        qs = "&".join(f"search_in={t}" for t in all_types)
        response = client.get(f"/?q=Python&{qs}")
        assert response.status_code == 200
        html = response.data.decode()
        # All checkboxes should be checked
        for t in all_types:
            assert f'value="{t}"' in html

    def test_search_pagination_preserves_search_in(self, client):
        """selected_search_types is available for pagination links."""
        response = client.get("/?q=Python&search_in=tools&page=1")
        assert response.status_code == 200
        html = response.data.decode()
        # Hidden input for search_in should be present so pagination forms preserve it
        assert '<input type="hidden" name="search_in" value="tools">' in html

    def test_search_template_renders_with_selected_types(self, client):
        """Page renders without error when selected_search_types is provided."""
        response = client.get("/?q=Python&search_in=commands&search_in=diffs")
        assert response.status_code == 200
        html = response.data.decode()
        # Content-type filter section should be rendered
        assert "content-type-filters" in html

    def test_search_in_preserved_in_form_hidden_inputs(self, client):
        """Hidden inputs are rendered for each selected search type."""
        response = client.get("/?q=Python&search_in=thinking&search_in=messages")
        assert response.status_code == 200
        html = response.data.decode()
        assert '<input type="hidden" name="search_in" value="thinking">' in html
        assert '<input type="hidden" name="search_in" value="messages">' in html
        # Types NOT selected should NOT have hidden inputs
        assert '<input type="hidden" name="search_in" value="diffs">' not in html

    def test_search_mixed_valid_and_invalid_search_in(self, client):
        """Mix of valid and invalid search_in values — page renders without error.

        Invalid values are passed through to the template (preserving round-trip)
        but are ignored by the search query itself.
        """
        response = client.get("/?q=Python&search_in=tools&search_in=bogus")
        assert response.status_code == 200
        html = response.data.decode()
        # Valid type is preserved
        assert '<input type="hidden" name="search_in" value="tools">' in html


class TestCreateApp:
    """Tests for the create_app function."""

    def test_create_app_with_db(self, temp_db):
        """Test creating an app with a database."""
        app = create_app(temp_db)
        assert app is not None
        assert app.config["DB_PATH"] == temp_db

    def test_create_app_with_title(self, temp_db):
        """Test creating an app with a custom title."""
        app = create_app(temp_db, title="Custom Title")
        assert app.config["ARCHIVE_TITLE"] == "Custom Title"

    def test_app_has_filters(self, temp_db):
        """Test that the app has the required Jinja2 filters."""
        app = create_app(temp_db)
        assert "markdown" in app.jinja_env.filters
        assert "urldecode" in app.jinja_env.filters
        assert "format_timestamp" in app.jinja_env.filters
        assert "parse_diff_stats" in app.jinja_env.filters
        assert "extract_filename" in app.jinja_env.filters


class TestEmptyDatabase:
    """Tests with an empty database."""

    def test_index_empty_db(self, tmp_path):
        """Test index with empty database."""
        db_path = tmp_path / "empty.db"
        _ = Database(db_path)  # Create empty database

        app = create_app(str(db_path))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/")
        assert response.status_code == 200
        assert b"No sessions found" in response.data


class TestRefreshRoute:
    """Tests for the refresh database route."""

    def test_refresh_route_exists(self, client):
        """Test that the refresh route exists and accepts POST."""
        response = client.post("/refresh")
        # Should redirect to index after refresh
        assert response.status_code == 302
        assert "/" in response.headers.get("Location", "")

    def test_refresh_incremental_mode(self, client):
        """Test refresh in incremental mode (default)."""
        response = client.post("/refresh", data={"full": "false"}, follow_redirects=True)
        assert response.status_code == 200
        # Check that the notification shows incremental mode
        assert b"Incremental refresh complete" in response.data

    def test_refresh_full_mode(self, client):
        """Test refresh in full rebuild mode."""
        response = client.post("/refresh", data={"full": "true"}, follow_redirects=True)
        assert response.status_code == 200
        # Check that the notification shows full mode
        assert b"Full refresh complete" in response.data

    def test_refresh_result_display(self, client):
        """Test that refresh result is displayed after redirect."""
        # First do a refresh
        response = client.post("/refresh", data={"full": "false"}, follow_redirects=True)
        assert response.status_code == 200
        # Should contain refresh notification with results
        assert b"refresh complete" in response.data.lower()

    def test_refresh_result_shown_only_once(self, client):
        """Test that refresh result is only shown once (session flash behavior)."""
        # First do a refresh and follow redirects
        response = client.post("/refresh", data={"full": "false"}, follow_redirects=True)
        assert response.status_code == 200
        assert b"refresh complete" in response.data.lower()

        # Navigate to index again - notification should NOT appear
        response = client.get("/")
        assert response.status_code == 200
        assert b"refresh complete" not in response.data.lower()

    def test_index_shows_refresh_buttons(self, client):
        """Test that the index page shows refresh buttons."""
        response = client.get("/")
        assert response.status_code == 200
        # Check for refresh buttons in the HTML
        assert b"Refresh" in response.data
        assert b"Rebuild All" in response.data

    def test_refresh_get_method_not_allowed(self, client):
        """Test that GET method is not allowed for refresh route."""
        response = client.get("/refresh")
        assert response.status_code == 405  # Method Not Allowed


class TestRefreshWithTestData:
    """Tests for refresh functionality with actual test data files."""

    def test_refresh_adds_new_session_from_file(self, tmp_path):
        """Test that refresh correctly adds a new session from a test file."""
        import json

        # Create a temporary database
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        # Create a mock chat sessions directory structure
        workspace_dir = tmp_path / "workspaceStorage" / "abc123"
        chat_dir = workspace_dir / "chatSessions"
        chat_dir.mkdir(parents=True)

        # Create workspace.json
        (workspace_dir / "workspace.json").write_text(json.dumps({"folder": f"file://{tmp_path}/myproject"}))

        # Create a chat session file
        session_file = chat_dir / "session1.json"
        session_file.write_text(
            json.dumps(
                {
                    "sessionId": "test-session-1",
                    "createdAt": "1704110400000",
                    "requests": [{"message": {"text": "Hello, assistant!"}, "timestamp": 1704110400000, "response": [{"value": "Hello! How can I help you?"}]}],
                }
            )
        )

        # Create a Flask app with custom storage paths
        app = create_app(str(db_path), title="Test Archive")
        app.config["TESTING"] = True

        # Verify database is initially empty
        stats = db.get_stats()
        assert stats["session_count"] == 0

        # Manually import the session using the scanner
        from copilot_session_tools.scanner import scan_chat_sessions

        storage_paths = [(str(tmp_path / "workspaceStorage"), "stable")]
        sessions = list(scan_chat_sessions(storage_paths, include_cli=False))

        # Verify we found the session
        assert len(sessions) == 1
        assert sessions[0].session_id == "test-session-1"

        # Add it to the database
        db.add_session(sessions[0])

        # Verify it was added
        stats = db.get_stats()
        assert stats["session_count"] == 1

    def test_refresh_updates_modified_session(self, tmp_path):
        """Test that refresh correctly updates a session when the file changes."""
        import json
        import time

        # Create a temporary database
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        # Create a mock chat sessions directory structure
        workspace_dir = tmp_path / "workspaceStorage" / "abc123"
        chat_dir = workspace_dir / "chatSessions"
        chat_dir.mkdir(parents=True)

        # Create workspace.json
        (workspace_dir / "workspace.json").write_text(json.dumps({"folder": f"file://{tmp_path}/myproject"}))

        # Create a chat session file
        session_file = chat_dir / "session1.json"
        session_file.write_text(
            json.dumps(
                {
                    "sessionId": "update-test-session",
                    "createdAt": "1704110400000",
                    "requests": [{"message": {"text": "First message"}, "timestamp": 1704110400000, "response": [{"value": "First response"}]}],
                }
            )
        )

        # Import initial session
        from copilot_session_tools.scanner import scan_chat_sessions

        storage_paths = [(str(tmp_path / "workspaceStorage"), "stable")]
        sessions = list(scan_chat_sessions(storage_paths, include_cli=False))
        assert len(sessions) == 1
        db.add_session(sessions[0])

        # Get initial session
        initial_session = db.get_session("update-test-session")
        assert initial_session is not None
        assert len(initial_session.messages) == 2  # user + assistant

        # Modify the session file with an additional message
        time.sleep(0.1)  # Ensure mtime changes
        session_file.write_text(
            json.dumps(
                {
                    "sessionId": "update-test-session",
                    "createdAt": "1704110400000",
                    "requests": [
                        {"message": {"text": "First message"}, "timestamp": 1704110400000, "response": [{"value": "First response"}]},
                        {"message": {"text": "Second message"}, "timestamp": 1704110500000, "response": [{"value": "Second response"}]},
                    ],
                }
            )
        )

        # Re-scan and check that needs_update detects the change
        sessions = list(scan_chat_sessions(storage_paths, include_cli=False))
        assert len(sessions) == 1
        updated_session = sessions[0]

        # Check that needs_update returns True for the modified file
        needs_update = db.needs_update(updated_session.session_id, updated_session.source_file_mtime, updated_session.source_file_size)
        assert needs_update, "needs_update should return True for modified file"

        # Update the session
        db.update_session(updated_session)

        # Verify the update
        updated = db.get_session("update-test-session")
        assert updated is not None
        assert len(updated.messages) == 4  # 2 user + 2 assistant messages

    def test_refresh_skips_unchanged_session(self, tmp_path):
        """Test that refresh correctly skips unchanged sessions."""
        import json

        # Create a temporary database
        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        # Create a mock chat sessions directory structure
        workspace_dir = tmp_path / "workspaceStorage" / "abc123"
        chat_dir = workspace_dir / "chatSessions"
        chat_dir.mkdir(parents=True)

        # Create workspace.json
        (workspace_dir / "workspace.json").write_text(json.dumps({"folder": f"file://{tmp_path}/myproject"}))

        # Create a chat session file
        session_file = chat_dir / "session1.json"
        session_file.write_text(
            json.dumps(
                {
                    "sessionId": "skip-test-session",
                    "createdAt": "1704110400000",
                    "requests": [{"message": {"text": "Test message"}, "timestamp": 1704110400000, "response": [{"value": "Test response"}]}],
                }
            )
        )

        # Import session
        from copilot_session_tools.scanner import scan_chat_sessions

        storage_paths = [(str(tmp_path / "workspaceStorage"), "stable")]
        sessions = list(scan_chat_sessions(storage_paths, include_cli=False))
        assert len(sessions) == 1
        db.add_session(sessions[0])

        # Re-scan WITHOUT modifying the file
        sessions = list(scan_chat_sessions(storage_paths, include_cli=False))
        assert len(sessions) == 1
        same_session = sessions[0]

        # Check that needs_update returns False for unchanged file
        needs_update = db.needs_update(same_session.session_id, same_session.source_file_mtime, same_session.source_file_size)
        assert not needs_update, "needs_update should return False for unchanged file"


class TestMarkdownFileUriConversion:
    """Tests for file:// URI to filename conversion in markdown."""

    def test_file_uri_to_monospace(self):
        """Test that file:// URIs with empty link text become monospace filenames."""
        result = _markdown_to_html("Reading [](file:///c%3A/Users/test/file.py)")
        assert "<code>file.py</code>" in result
        assert "[](" not in result  # Empty link should be replaced

    def test_file_uri_with_anchor(self):
        """Test that file:// URIs with anchors extract filename correctly."""
        result = _markdown_to_html("Reading [](file:///c%3A/path/to/SKILL.md#1-1), lines 1 to 100")
        assert "<code>SKILL.md</code>" in result
        assert "lines 1 to 100" in result

    def test_file_uri_url_encoded(self):
        """Test that URL-encoded paths are decoded properly."""
        result = _markdown_to_html("Reading [](file:///c%3A/Users/test%20user/my%20file.py)")
        assert "<code>my file.py</code>" in result

    def test_file_uri_unix_path(self):
        """Test Unix-style file:// URIs."""
        result = _markdown_to_html("Created [](file:///home/user/project/main.py)")
        assert "<code>main.py</code>" in result

    def test_multiple_file_uris(self):
        """Test multiple file:// URIs in the same text."""
        result = _markdown_to_html("Read [](file:///a/b.py) and [](file:///c/d.py)")
        assert "<code>b.py</code>" in result
        assert "<code>d.py</code>" in result


class TestHtmlOutputToolInvocations:
    """Tests for HTML output of tool invocation rendering."""

    @pytest.fixture
    def session_with_tools(self, tmp_path):
        """Create a session with various tool invocations for testing."""
        from copilot_session_tools.scanner import ChatMessage, ChatSession, ContentBlock, ToolInvocation

        db_path = tmp_path / "test_tools.db"
        db = Database(str(db_path))

        # Create session with tool invocations
        session = ChatSession(
            session_id="tool-test-session",
            workspace_name="test-workspace",
            workspace_path="/home/user/test",
            messages=[
                ChatMessage(role="user", content="Run some tools"),
                ChatMessage(
                    role="assistant",
                    content="I'll run some tools for you.",
                    content_blocks=[
                        ContentBlock(kind="text", content="Let me help."),
                        ContentBlock(kind="toolInvocation", content="Running `test_mcp_tool`"),
                        ContentBlock(kind="toolInvocation", content="Reading [](file:///path/to/file.py)"),
                        ContentBlock(kind="toolInvocation", content='Using "Run in Terminal"'),
                    ],
                    tool_invocations=[
                        ToolInvocation(
                            name="mcp_test_tool",
                            status="completed",
                            input='{"query": "test input"}',
                            result="Tool output result",
                            source_type="mcp",
                            invocation_message="Running `test_mcp_tool`",
                        ),
                        ToolInvocation(
                            name="copilot_readFile",
                            status="completed",
                            input=None,
                            result=None,
                            source_type="internal",
                            invocation_message="Reading [](file:///path/to/file.py)",
                        ),
                        ToolInvocation(
                            name="run_in_terminal",
                            status="completed",
                            input="ls -la",
                            result="total 42\ndrwxr-xr-x 5 user user 4096 Jan 10 file.txt",
                            source_type="internal",
                            invocation_message='Using "Run in Terminal"',
                        ),
                    ],
                ),
            ],
            created_at="2025-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)

        return db_path

    def test_mcp_tool_renders_collapsible(self, session_with_tools):
        """Test that MCP tools render with collapsible details."""
        app = create_app(str(session_with_tools))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/tool-test-session")
        html = response.data.decode("utf-8")

        # MCP tool should have collapsible wrapper
        assert "tool-invocation-wrapper" in html
        # Should show the tool name inside
        assert "mcp_test_tool" in html
        # Should have input/output sections
        assert "test input" in html
        assert "Tool output result" in html

    def test_internal_file_tool_renders_inline(self, session_with_tools):
        """Test that internal file tools render inline without collapsible."""
        app = create_app(str(session_with_tools))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/tool-test-session")
        html = response.data.decode("utf-8")

        # Should render the filename in monospace
        assert "<code>file.py</code>" in html

    def test_terminal_tool_renders_collapsible(self, session_with_tools):
        """Test that run_in_terminal renders with collapsible output."""
        app = create_app(str(session_with_tools))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/tool-test-session")
        html = response.data.decode("utf-8")

        # Terminal tool should have collapsible wrapper
        assert "run_in_terminal" in html
        # Should show the command
        assert "ls -la" in html
        # Should have the output
        assert "total 42" in html

    def test_tool_status_badge_rendered(self, session_with_tools):
        """Test that tool status badges are rendered."""
        app = create_app(str(session_with_tools))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/tool-test-session")
        html = response.data.decode("utf-8")

        # Status badges should appear
        assert "status-badge" in html
        assert "completed" in html


class TestHtmlOutputCommandDescriptions:
    """Tests for command run description/title display in web viewer."""

    @pytest.fixture
    def session_with_command_descriptions(self, tmp_path):
        """Create a session with command runs that have titles/descriptions."""
        from copilot_session_tools.scanner import ChatMessage, ChatSession, CommandRun, ContentBlock

        db_path = tmp_path / "test_cmd_desc.db"
        db = Database(str(db_path))

        session = ChatSession(
            session_id="cmd-desc-test",
            workspace_name="test-workspace",
            workspace_path="/home/user/test",
            messages=[
                ChatMessage(role="user", content="Search for something"),
                ChatMessage(
                    role="assistant",
                    content="Searching...",
                    content_blocks=[
                        ContentBlock(kind="text", content="Let me search."),
                        ContentBlock(
                            kind="toolInvocation",
                            content="$ cd /repo && uv run copilot-chat-archive search 'workspace:ZTS ARG test account access' --full --limit 10",
                            description="Search ARG test account access",
                        ),
                        ContentBlock(
                            kind="toolInvocation",
                            content="$ git status",
                            description=None,
                        ),
                    ],
                    command_runs=[
                        CommandRun(
                            command="cd /repo && uv run copilot-chat-archive search 'workspace:ZTS ARG test account access' --full --limit 10",
                            title="Search ARG test account access",
                            status="success",
                            output="No results found",
                        ),
                        CommandRun(
                            command="git status",
                            title=None,
                            status="success",
                            output="On branch main",
                        ),
                    ],
                ),
            ],
            created_at="2026-02-12T08:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)
        return db_path

    def test_inline_command_shows_title(self, session_with_command_descriptions):
        """Test that inline commands show their title/description instead of raw command."""
        app = create_app(str(session_with_command_descriptions))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/cmd-desc-test")
        html = response.data.decode("utf-8")

        # Title should appear in the summary header
        assert "Search ARG test account access" in html

    def test_inline_command_without_title_shows_command(self, session_with_command_descriptions):
        """Test that inline commands without title show the raw command."""
        app = create_app(str(session_with_command_descriptions))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/cmd-desc-test")
        html = response.data.decode("utf-8")

        # git status should appear as raw command since it has no title
        assert "git status" in html


class TestHtmlOutputThinkingBlocks:
    """Tests for HTML output of thinking block rendering."""

    @pytest.fixture
    def session_with_thinking(self, tmp_path):
        """Create a session with thinking blocks for testing."""
        from copilot_session_tools.scanner import ChatMessage, ChatSession, ContentBlock

        db_path = tmp_path / "test_thinking.db"
        db = Database(str(db_path))

        session = ChatSession(
            session_id="thinking-test-session",
            workspace_name="test-workspace",
            workspace_path="/home/user/test",
            messages=[
                ChatMessage(role="user", content="Think about something"),
                ChatMessage(
                    role="assistant",
                    content="Here's my thought process...",
                    content_blocks=[
                        ContentBlock(
                            kind="thinking",
                            content="Let me analyze this carefully...",
                            description="Analyzing the request",
                        ),
                        ContentBlock(kind="text", content="Based on my analysis, here's the answer."),
                    ],
                ),
            ],
            created_at="2025-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)

        return db_path

    def test_thinking_block_renders_collapsible(self, session_with_thinking):
        """Test that thinking blocks render as collapsible sections."""
        app = create_app(str(session_with_thinking))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/thinking-test-session")
        html = response.data.decode("utf-8")

        # Should have thinking block structure
        assert "thinking-block" in html
        assert "Thinking" in html

    def test_thinking_block_shows_description(self, session_with_thinking):
        """Test that thinking block description is shown in header."""
        app = create_app(str(session_with_thinking))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/thinking-test-session")
        html = response.data.decode("utf-8")

        # Description should be in the summary
        assert "Analyzing the request" in html

    def test_thinking_block_content_inside_details(self, session_with_thinking):
        """Test that thinking content is inside the collapsible."""
        app = create_app(str(session_with_thinking))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/thinking-test-session")
        html = response.data.decode("utf-8")

        # Content should be in thinking-content div
        assert "thinking-content" in html
        assert "analyze this carefully" in html


class TestHtmlOutputFileChanges:
    """Tests for HTML output of file changes rendering."""

    @pytest.fixture
    def session_with_file_changes(self, tmp_path):
        """Create a session with file changes for testing."""
        from copilot_session_tools.scanner import ChatMessage, ChatSession, FileChange

        db_path = tmp_path / "test_files.db"
        db = Database(str(db_path))

        session = ChatSession(
            session_id="files-test-session",
            workspace_name="test-workspace",
            workspace_path="/home/user/test",
            messages=[
                ChatMessage(role="user", content="Edit some files"),
                ChatMessage(
                    role="assistant",
                    content="I'll edit the files.",
                    file_changes=[
                        FileChange(
                            path="/src/main.py",
                            language_id="python",
                            explanation="Added error handling",
                            diff="+try:\n+    result = process()\n+except Exception as e:\n+    log_error(e)",
                        ),
                        FileChange(
                            path="/src/utils.js",
                            language_id="javascript",
                            explanation="Fixed bug in helper",
                            diff="-const old = 1;\n+const fixed = 2;",
                        ),
                    ],
                ),
            ],
            created_at="2025-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)

        return db_path

    def test_file_changes_renders_collapsible(self, session_with_file_changes):
        """Test that file changes render in a collapsible section."""
        app = create_app(str(session_with_file_changes))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/files-test-session")
        html = response.data.decode("utf-8")

        # Should have file changes section
        assert "File Changes" in html
        assert "file-change" in html

    def test_file_changes_shows_filename(self, session_with_file_changes):
        """Test that file changes show the filename."""
        app = create_app(str(session_with_file_changes))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/files-test-session")
        html = response.data.decode("utf-8")

        # Filenames should appear
        assert "main.py" in html
        assert "utils.js" in html

    def test_file_changes_shows_language_badge(self, session_with_file_changes):
        """Test that file changes show language badges."""
        app = create_app(str(session_with_file_changes))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/files-test-session")
        html = response.data.decode("utf-8")

        # Language badges should appear
        assert "language-badge" in html
        assert "python" in html
        assert "javascript" in html

    def test_file_changes_shows_diff_stats(self, session_with_file_changes):
        """Test that file changes show addition/deletion statistics."""
        app = create_app(str(session_with_file_changes))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/files-test-session")
        html = response.data.decode("utf-8")

        # Diff stats should be shown
        assert "file-stat-add" in html or "file-stat-del" in html

    def test_file_changes_shows_explanation(self, session_with_file_changes):
        """Test that file changes show explanations."""
        app = create_app(str(session_with_file_changes))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/files-test-session")
        html = response.data.decode("utf-8")

        # Explanations should appear
        assert "Added error handling" in html
        assert "Fixed bug in helper" in html

    def test_file_changes_shows_diff(self, session_with_file_changes):
        """Test that file changes show the diff content."""
        app = create_app(str(session_with_file_changes))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/session/files-test-session")
        html = response.data.decode("utf-8")

        # Diff content should appear
        assert "process()" in html
        assert "log_error" in html


class TestHtmlOutputCodeBlocks:
    """Tests for HTML output of code block rendering."""

    def test_code_block_renders_with_pre(self, client):
        """Test that code blocks render with pre tag."""
        response = client.get("/session/webapp-test-session")
        html = response.data.decode("utf-8")

        # Code block should have pre and code tags
        assert "<pre>" in html or "<pre " in html
        assert "<code" in html

    def test_inline_code_renders_with_code(self):
        """Test that inline code renders with code tag."""
        result = _markdown_to_html("Use the `foo()` function")
        assert "<code>foo()</code>" in result


class TestHtmlOutputMessageStructure:
    """Tests for the overall message HTML structure."""

    def test_messages_have_role_class(self, client):
        """Test that messages have role-based CSS classes."""
        response = client.get("/session/webapp-test-session")
        html = response.data.decode("utf-8")

        # Should have role-based classes
        assert 'class="message user"' in html
        assert 'class="message assistant"' in html

    def test_messages_have_anchors(self, client):
        """Test that messages have anchor links."""
        response = client.get("/session/webapp-test-session")
        html = response.data.decode("utf-8")

        # Should have message anchors
        assert 'id="msg-1"' in html
        assert 'id="msg-2"' in html
        assert 'href="#msg-1"' in html

    def test_session_header_shows_metadata(self, client):
        """Test that session header shows metadata."""
        response = client.get("/session/webapp-test-session")
        html = response.data.decode("utf-8")

        # Should show workspace name
        assert "test-workspace" in html
        # Should show edition badge
        assert "stable" in html


class TestNewTurnsAfterEnrichment:
    """Tests for rendering new turns that arrived after enrichment."""

    @pytest.fixture
    def stale_enrichment_db(self, tmp_path):
        """Create a CST DB with enriched session + newer Chronicle turns in sibling DB."""
        db_path = str(tmp_path / "copilot-session-tools.db")
        chronicle_path = str(tmp_path / "session-store.db")

        db = Database(db_path)

        # Add enriched session with 1 user message (= 1 turn worth of enrichment)
        session = ChatSession(
            session_id="stale-enriched-session",
            workspace_name="test-workspace",
            workspace_path="/home/user/test",
            messages=[
                ChatMessage(role="user", content="First question"),
                ChatMessage(role="assistant", content="First answer"),
            ],
            created_at="2025-01-15T10:00:00Z",
            vscode_edition="cli",
            type="cli",
        )
        db.add_session(session)

        # Now add Chronicle sessions/turns tables in sibling DB with MORE turns than enriched
        import sqlite3

        conn = sqlite3.connect(chronicle_path)
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, cwd TEXT, repository TEXT, branch TEXT, summary TEXT, created_at TEXT, updated_at TEXT)")
        conn.execute(
            "INSERT OR REPLACE INTO sessions (id, cwd, summary, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            ("stale-enriched-session", "/home/user/test", "Test session", "2025-01-15T10:00:00Z", "2025-01-15T11:00:00Z"),
        )
        conn.execute("CREATE TABLE IF NOT EXISTS turns (session_id TEXT, turn_index INTEGER, user_message TEXT, assistant_response TEXT)")
        # Turn 0 = the enriched one
        conn.execute(
            "INSERT INTO turns (session_id, turn_index, user_message, assistant_response) VALUES (?, ?, ?, ?)",
            ("stale-enriched-session", 0, "First question", "First answer"),
        )
        # Turn 1 = new, not yet enriched
        conn.execute(
            "INSERT INTO turns (session_id, turn_index, user_message, assistant_response) VALUES (?, ?, ?, ?)",
            ("stale-enriched-session", 1, "Second question after enrichment", "Second answer after enrichment"),
        )
        conn.commit()
        conn.close()

        yield db_path

    @pytest.fixture
    def stale_client(self, stale_enrichment_db):
        chronicle_path = str(Path(stale_enrichment_db).parent / "session-store.db")
        app = create_app(stale_enrichment_db, title="Test Archive", storage_paths=[], chronicle_db_path=chronicle_path)
        app.config["TESTING"] = True
        return app.test_client()

    def test_new_turns_shown_after_enriched(self, stale_client):
        """New turns since enrichment appear below enriched messages."""
        response = stale_client.get("/session/stale-enriched-session")
        html = response.data.decode("utf-8")

        # Enriched messages should be present
        assert "First question" in html
        assert "First answer" in html

        # New unenriched turn should also appear
        assert "Second question after enrichment" in html
        assert "Second answer after enrichment" in html

    def test_new_turns_banner_shown(self, stale_client):
        """A divider banner should indicate new turns since enrichment."""
        response = stale_client.get("/session/stale-enriched-session")
        html = response.data.decode("utf-8")

        assert "new turn" in html
        assert "since last enrichment" in html

    def test_new_turns_have_scan_now_button(self, stale_client):
        """The new turns banner should include a Scan Now button."""
        response = stale_client.get("/session/stale-enriched-session")
        html = response.data.decode("utf-8")

        assert "Scan Now" in html

    def test_new_turns_message_numbering_continues(self, stale_client):
        """New turn message numbers should continue from enriched messages."""
        response = stale_client.get("/session/stale-enriched-session")
        html = response.data.decode("utf-8")

        # Enriched messages: #1 (user), #2 (assistant)
        assert 'id="msg-1"' in html
        assert 'id="msg-2"' in html
        # New turns: #3 (user), #4 (assistant)
        assert 'id="msg-3"' in html
        assert 'id="msg-4"' in html


class TestWebappPagination:
    """Tests for pagination in the web interface."""

    @pytest.fixture
    def db_with_many_sessions(self, tmp_path):
        """Create a database with many sessions for pagination testing."""
        db_path = tmp_path / "pagination_test.db"
        db = Database(db_path)

        # Create 25 sessions (more than one page of 20)
        for i in range(25):
            session = ChatSession(
                session_id=f"pagination-session-{i}",
                workspace_name=f"project-{i}",
                workspace_path=f"/path/to/project-{i}",
                messages=[ChatMessage(role="user", content=f"Message {i}")],
                created_at=f"2024-01-{(i % 28) + 1:02d}T10:00:00Z",
                vscode_edition="stable",
            )
            db.add_session(session)

        return db_path

    def test_pagination_shows_on_index(self, db_with_many_sessions):
        """Test that pagination controls appear when there are many sessions."""
        app = create_app(str(db_with_many_sessions))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/")
        html = response.data.decode("utf-8")

        # Should show pagination controls
        assert "pagination" in html
        assert "Showing" in html

    def test_pagination_page_parameter(self, db_with_many_sessions):
        """Test that page parameter works correctly."""
        app = create_app(str(db_with_many_sessions))
        app.config["TESTING"] = True
        client = app.test_client()

        # Page 1
        response = client.get("/?page=1")
        assert response.status_code == 200

        # Page 2
        response = client.get("/?page=2")
        assert response.status_code == 200

    def test_pagination_preserves_query_params(self, db_with_many_sessions):
        """Test that pagination preserves search and filter parameters."""
        app = create_app(str(db_with_many_sessions))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/?q=Message&sort=date&page=1")

        # Pagination links should preserve query params
        assert response.status_code == 200

    def test_search_help_includes_date_filters(self, db_with_many_sessions):
        """Test that search help tips include date filter documentation."""
        app = create_app(str(db_with_many_sessions))
        app.config["TESTING"] = True
        client = app.test_client()

        response = client.get("/")
        html = response.data.decode("utf-8")

        # Should include date filter documentation
        assert "start_date" in html
        assert "end_date" in html


class TestMarkdownApiIncludeThinking:
    """Tests for include_thinking parameter on the markdown API endpoint."""

    @pytest.fixture
    def thinking_db(self, tmp_path):
        """Create a database with a session that has thinking blocks."""
        db_path = tmp_path / "thinking_test.db"
        db = Database(str(db_path))

        session = ChatSession(
            session_id="thinking-api-session",
            workspace_name="test-workspace",
            workspace_path="/home/user/test",
            messages=[
                ChatMessage(role="user", content="Think about this"),
                ChatMessage(
                    role="assistant",
                    content="Here's the answer.",
                    content_blocks=[
                        ContentBlock(kind="thinking", content="Deep reasoning here..."),
                        ContentBlock(kind="text", content="Here's the answer."),
                    ],
                ),
            ],
            created_at="2025-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)
        return str(db_path)

    @pytest.fixture
    def thinking_client(self, thinking_db):
        """Create a Flask test client with the thinking session database."""
        app = create_app(thinking_db, storage_paths=[])
        app.config["TESTING"] = True
        return app.test_client()

    def test_get_markdown_include_thinking_true(self, thinking_client):
        """Test that include_thinking=true includes thinking content."""
        response = thinking_client.get("/api/markdown/thinking-api-session?include_thinking=true")
        assert response.status_code == 200
        data = response.get_json()
        assert "markdown" in data
        assert "Deep reasoning here..." in data["markdown"]

    def test_get_markdown_include_thinking_false(self, thinking_client):
        """Test that include_thinking=false omits thinking content."""
        response = thinking_client.get("/api/markdown/thinking-api-session?include_thinking=false")
        assert response.status_code == 200
        data = response.get_json()
        assert "markdown" in data
        assert "Deep reasoning here..." not in data["markdown"]

    def test_get_markdown_include_thinking_default(self, thinking_client):
        """Test that thinking content is omitted by default."""
        response = thinking_client.get("/api/markdown/thinking-api-session")
        assert response.status_code == 200
        data = response.get_json()
        assert "markdown" in data
        assert "Deep reasoning here..." not in data["markdown"]


class TestDownloadMarkdownEndpoint:
    """Tests for the download=true query parameter on the markdown API."""

    def test_download_returns_200(self, client):
        """Test that the download mode returns 200."""
        response = client.get("/api/markdown/webapp-test-session?download=true")
        assert response.status_code == 200

    def test_download_content_type(self, client):
        """Test that the download returns text/markdown content type."""
        response = client.get("/api/markdown/webapp-test-session?download=true")
        assert "text/markdown" in response.content_type

    def test_download_content_disposition(self, client):
        """Test that the download has attachment Content-Disposition with .md filename."""
        response = client.get("/api/markdown/webapp-test-session?download=true")
        disposition = response.headers.get("Content-Disposition", "")
        assert "attachment" in disposition
        assert ".md" in disposition

    def test_download_returns_markdown_not_json(self, client):
        """Test that the download returns raw markdown, not JSON."""
        response = client.get("/api/markdown/webapp-test-session?download=true")
        content = response.data.decode("utf-8")
        # Should contain markdown formatting, not JSON
        assert "Message" in content
        assert "USER" in content or "ASSISTANT" in content or "user" in content.lower()

    def test_download_accepts_include_diffs(self, client):
        """Test that download mode works with include_diffs param."""
        response = client.get("/api/markdown/webapp-test-session?download=true&include_diffs=false")
        assert response.status_code == 200

    def test_download_accepts_include_tool_inputs(self, client):
        """Test that download mode works with include_tool_inputs param."""
        response = client.get("/api/markdown/webapp-test-session?download=true&include_tool_inputs=false")
        assert response.status_code == 200

    def test_download_accepts_include_thinking(self, client):
        """Test that download mode works with include_thinking param."""
        response = client.get("/api/markdown/webapp-test-session?download=true&include_thinking=true")
        assert response.status_code == 200

    def test_download_404_for_nonexistent_session(self, client):
        """Test that download returns 404 for a non-existent session."""
        response = client.get("/api/markdown/nonexistent-session?download=true")
        assert response.status_code == 404

    def test_without_download_returns_json(self, client):
        """Test that without download=true, response is JSON as before."""
        response = client.get("/api/markdown/webapp-test-session")
        assert response.status_code == 200
        data = response.get_json()
        assert "markdown" in data


class TestDownloadHtmlEndpoint:
    """Tests for the /api/html/<session_id> endpoint."""

    def test_download_returns_200(self, client):
        """Test that the HTML download returns 200."""
        response = client.get("/api/html/webapp-test-session")
        assert response.status_code == 200

    def test_download_content_type(self, client):
        """Test that the download returns text/html content type."""
        response = client.get("/api/html/webapp-test-session")
        assert "text/html" in response.content_type

    def test_download_content_disposition(self, client):
        """Test that the download has attachment Content-Disposition with .html filename."""
        response = client.get("/api/html/webapp-test-session")
        disposition = response.headers.get("Content-Disposition", "")
        assert "attachment" in disposition
        assert ".html" in disposition

    def test_download_returns_html_not_json(self, client):
        """Test that the download returns HTML content, not JSON."""
        response = client.get("/api/html/webapp-test-session")
        content = response.data.decode("utf-8")
        assert "<html" in content.lower() or "<!doctype" in content.lower()

    def test_download_404_for_nonexistent_session(self, client):
        """Test that download returns 404 for a non-existent session."""
        response = client.get("/api/html/nonexistent-session")
        assert response.status_code == 404

    def test_download_accepts_content_set(self, client):
        """Test that download accepts content_set param."""
        response = client.get("/api/html/webapp-test-session?content_set=diffs,thinking")
        assert response.status_code == 200

    def test_download_accepts_range_params(self, client):
        """Test that download accepts start and end range params."""
        response = client.get("/api/html/webapp-test-session?start=1&end=1")
        assert response.status_code == 200

    def test_download_invalid_start_returns_400(self, client):
        """Test that invalid start param returns 400."""
        response = client.get("/api/html/webapp-test-session?start=abc")
        assert response.status_code == 400

    def test_api_html_returns_html(self, client):
        """GET /api/html/<session_id> returns an HTML file."""
        response = client.get("/api/html/webapp-test-session")
        assert response.status_code == 200
        assert "text/html" in response.content_type
        assert "attachment" in response.headers.get("Content-Disposition", "")

    def test_api_html_with_content_set(self, client):
        """content_set query param controls content in exported HTML."""
        response = client.get("/api/html/webapp-test-session?content_set=agent-details")
        html = response.data.decode()
        assert "<!DOCTYPE html>" in html or "<html" in html

    def test_api_html_with_range(self, client):
        """start/end params limit message range."""
        response = client.get("/api/html/webapp-test-session?start=1&end=2")
        assert response.status_code == 200

    def test_api_html_not_found(self, client):
        """Returns 404 for unknown session."""
        response = client.get("/api/html/nonexistent-id")
        assert response.status_code == 404

    def test_api_html_is_self_contained(self, client):
        """Exported HTML contains all CSS inline (no external resources)."""
        response = client.get("/api/html/webapp-test-session")
        html = response.data.decode()
        assert "<style>" in html
        assert "cdn.jsdelivr.net" not in html  # No CDN links in static
