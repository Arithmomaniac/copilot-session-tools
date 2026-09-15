"""Tests for the CLI module."""

import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from copilot_session_tools import ChatMessage, ChatSession, Database, __version__
from copilot_session_tools.cli import _default_db_path, _ensure_db_exists, app


@pytest.fixture
def runner():
    """Create a CLI test runner."""
    return CliRunner()


@pytest.fixture
def mock_no_vscode_paths():
    """Mock storage paths to return empty for faster tests.

    We patch at the refresh module level since that's where scan_session_files
    and parse_session_file are now called from.
    """
    with (
        patch("copilot_session_tools.cli.get_vscode_storage_paths", return_value=[]),
        patch("copilot_session_tools.refresh.scan_session_files", return_value=iter([])),
    ):
        yield


@pytest.fixture
def temp_db_with_data(tmp_path):
    """Create a temporary database with sample data."""
    db_path = tmp_path / "test.db"
    db = Database(db_path)

    session = ChatSession(
        session_id="cli-test-session",
        workspace_name="cli-workspace",
        workspace_path="/home/user/cli-test",
        messages=[
            ChatMessage(role="user", content="Hello from CLI test"),
            ChatMessage(role="assistant", content="Hi there!"),
        ],
        created_at="2025-01-15T10:00:00Z",
        vscode_edition="stable",
    )
    db.add_session(session)

    return db_path


class TestCLI:
    """Tests for the CLI commands."""

    def test_version(self, runner):
        """Test --version flag."""
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert __version__ in result.output

    def test_scan_no_paths(self, runner, tmp_path, mock_no_vscode_paths):
        """Test scan command with no valid paths."""
        db_path = tmp_path / "test.db"
        result = runner.invoke(app, ["scan", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "Scanning" in result.output

    def test_sources_add_list_rename_disable_enable(self, runner, tmp_path):
        db_path = tmp_path / "archive.db"
        source_dir = tmp_path / "scout"
        source_dir.mkdir()
        with sqlite3.connect(source_dir / "session-store.db") as conn:
            conn.executescript(
                """
                CREATE TABLE sessions (id TEXT PRIMARY KEY);
                CREATE TABLE turns (session_id TEXT, turn_index INTEGER);
                """
            )

        added = runner.invoke(
            app,
            ["sources", "add", "--db", str(db_path), "--name", "Scout", "--base-dir", str(source_dir)],
        )
        listed = runner.invoke(app, ["sources", "list", "--db", str(db_path)])
        renamed = runner.invoke(
            app,
            ["sources", "rename", "--db", str(db_path), "Scout", "--name", "Microsoft Scout"],
        )
        disabled = runner.invoke(app, ["sources", "disable", "--db", str(db_path), "Microsoft Scout"])
        enabled = runner.invoke(app, ["sources", "enable", "--db", str(db_path), "Microsoft Scout"])

        assert added.exit_code == 0
        assert "scout" in listed.output
        assert renamed.exit_code == 0
        assert disabled.exit_code == 0
        assert enabled.exit_code == 0

    def test_stats_command(self, runner, temp_db_with_data):
        """Test stats command."""
        result = runner.invoke(app, ["stats", "--db", str(temp_db_with_data)])
        assert result.exit_code == 0
        assert "Sessions: 1" in result.output
        assert "Messages: 2" in result.output

    def test_stats_missing_db(self, runner, tmp_path):
        """Test stats command with missing database."""
        result = runner.invoke(app, ["stats", "--db", str(tmp_path / "missing.db")])
        # Typer returns non-zero exit code for parameter validation errors (exists=True)
        assert result.exit_code != 0
        assert "does not exist" in result.output or "not found" in result.output or "Invalid value" in result.output

    def test_search_command(self, runner, temp_db_with_data):
        """Test search command."""
        result = runner.invoke(app, ["search", "--db", str(temp_db_with_data), "Hello"])
        assert result.exit_code == 0
        # Should find results
        assert "CLI test" in result.output or "result" in result.output.lower()

    def test_search_unknown_source_has_friendly_error(self, runner, temp_db_with_data):
        result = runner.invoke(
            app,
            ["search", "--db", str(temp_db_with_data), "--source", "missing", "Hello"],
        )

        assert result.exit_code == 1
        assert "Error: Source not found: missing" in result.output
        assert result.exception is not None
        assert not isinstance(result.exception, ValueError)

    def test_search_no_results(self, runner, temp_db_with_data):
        """Test search command with no results."""
        result = runner.invoke(app, ["search", "--db", str(temp_db_with_data), "nonexistent term xyz123"])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_with_unicode_content(self, runner, tmp_path):
        """Test search works with Unicode content (regression test for cp1252 crash).

        When stdout is piped on Windows, Rich falls back to the system encoding
        (cp1252) which can't handle Unicode. This test ensures the CLI handles
        Unicode content without crashing in non-TTY contexts.
        """
        db_path = tmp_path / "unicode_test.db"
        db = Database(db_path)

        session = ChatSession(
            session_id="unicode-test-session",
            workspace_name="unicode-workspace",
            workspace_path="/home/user/test",
            messages=[
                ChatMessage(role="user", content="Show me emojis 🎉 and symbols ━━━"),
                ChatMessage(role="assistant", content="Here: café, naïve, 日本語, 🚀✨"),
            ],
            created_at="2025-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)

        result = runner.invoke(app, ["search", "--db", str(db_path), "emojis", "--full"])
        assert result.exit_code == 0
        assert "Result" in result.output

    def test_export_command(self, runner, temp_db_with_data):
        """Test export command."""
        result = runner.invoke(app, ["export", "--db", str(temp_db_with_data)])
        assert result.exit_code == 0
        # Output should be valid JSON
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 1

    def test_export_to_file(self, runner, temp_db_with_data, tmp_path):
        """Test export command to file."""
        output_file = tmp_path / "export.json"
        result = runner.invoke(
            app,
            ["export", "--db", str(temp_db_with_data), "--output", str(output_file)],
        )
        assert result.exit_code == 0
        assert output_file.exists()
        data = json.loads(output_file.read_text())
        assert len(data) == 1

    def test_import_json_command(self, runner, tmp_path):
        """Test import-json command."""
        db_path = tmp_path / "import.db"
        json_file = tmp_path / "sessions.json"

        # Create sample JSON
        sessions = [
            {
                "session_id": "imported-session",
                "workspace_name": "imported-workspace",
                "messages": [
                    {"role": "user", "content": "Imported message"},
                ],
            }
        ]
        json_file.write_text(json.dumps(sessions))

        result = runner.invoke(app, ["import-json", "--db", str(db_path), str(json_file)])
        assert result.exit_code == 0
        assert "Added: 1" in result.output

        # Verify import
        db = Database(db_path)
        stats = db.get_stats()
        assert stats["session_count"] == 1

    def test_scan_full_flag(self, runner, tmp_path, mock_no_vscode_paths):
        """Test scan command with --full flag."""
        db_path = tmp_path / "full_test.db"

        # First, create a database with an existing session
        db = Database(db_path)
        session = ChatSession(
            session_id="full-test-session",
            workspace_name="full-workspace",
            workspace_path="/home/user/full-test",
            messages=[
                ChatMessage(role="user", content="Original message"),
            ],
            created_at="2025-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)

        # Test that --full flag is recognized
        result = runner.invoke(app, ["scan", "--db", str(db_path), "--full"])
        assert result.exit_code == 0
        assert "Full mode" in result.output or "Updated:" in result.output

    def test_scan_incremental_default(self, runner, tmp_path, mock_no_vscode_paths):
        """Test that scan command uses incremental mode by default."""
        db_path = tmp_path / "incremental_test.db"

        # Create a database (it will be empty)
        result = runner.invoke(app, ["scan", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "Incremental mode" in result.output

    def test_export_markdown_command(self, runner, temp_db_with_data, tmp_path):
        """Test export-markdown command exports sessions to markdown files."""
        output_dir = tmp_path / "markdown_output"

        result = runner.invoke(app, ["export-markdown", "--db", str(temp_db_with_data), "--output-dir", str(output_dir), "-v"])
        assert result.exit_code == 0
        assert "Exported 1 sessions" in result.output

        # Check that a markdown file was created
        md_files = list(output_dir.glob("*.md"))
        assert len(md_files) == 1

        # Check content of the markdown file
        content = md_files[0].read_text()
        assert "# Chat Session" in content
        assert "cli-workspace" in content
        assert "cli-test-session" in content
        assert "Message 1:" in content
        assert "USER" in content
        assert "ASSISTANT" in content

    def test_export_markdown_single_session(self, runner, temp_db_with_data, tmp_path):
        """Test export-markdown command with specific session ID."""
        output_dir = tmp_path / "markdown_output"

        result = runner.invoke(
            app,
            [
                "export-markdown",
                "--db",
                str(temp_db_with_data),
                "--output-dir",
                str(output_dir),
                "--session-id",
                "cli-test-session",
            ],
        )
        assert result.exit_code == 0
        assert "Exported:" in result.output

        # Check that exactly one markdown file was created
        md_files = list(output_dir.glob("*.md"))
        assert len(md_files) == 1

    def test_export_markdown_missing_session(self, runner, temp_db_with_data, tmp_path):
        """Test export-markdown command with non-existent session ID."""
        output_dir = tmp_path / "markdown_output"

        result = runner.invoke(
            app,
            [
                "export-markdown",
                "--db",
                str(temp_db_with_data),
                "--output-dir",
                str(output_dir),
                "--session-id",
                "nonexistent-session",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_export_markdown_rescan_session_bootstraps_missing_db(self, runner, tmp_path):
        """--rescan-session can create the CST DB and export a never-scanned session."""
        db_path = tmp_path / "missing.db"
        output_dir = tmp_path / "markdown_output"
        session_id = "rescanned-session"

        def fake_parse(requested_session_id):
            return ChatSession(
                session_id=requested_session_id,
                workspace_name="rescanned-workspace",
                workspace_path="/tmp/rescanned",
                messages=[ChatMessage(role="user", content="Fresh bootstrap content")],
                vscode_edition="cli",
                type="cli",
            )

        with patch("copilot_session_tools.cli.parse_single_cli_session", side_effect=fake_parse) as mock_parse:
            result = runner.invoke(
                app,
                [
                    "export-markdown",
                    "--db",
                    str(db_path),
                    "--output-dir",
                    str(output_dir),
                    "--session-id",
                    session_id,
                    "--rescan-session",
                    session_id,
                ],
            )

        assert result.exit_code == 0
        mock_parse.assert_called_once()
        assert db_path.exists()
        md_files = list(output_dir.glob("*.md"))
        assert len(md_files) == 1
        assert "Fresh bootstrap content" in md_files[0].read_text(encoding="utf-8")

    def test_export_markdown_rescan_session_must_match_session_id(self, runner, tmp_path):
        """Session-scoped exports reject mismatched --rescan-session values."""
        db_path = tmp_path / "missing.db"
        result = runner.invoke(
            app,
            [
                "export-markdown",
                "--db",
                str(db_path),
                "--output-dir",
                str(tmp_path / "markdown_output"),
                "--session-id",
                "cli-test-session",
                "--rescan-session",
                "other-session",
            ],
        )

        assert result.exit_code == 1
        assert "must match --session-id" in result.output
        assert not db_path.exists()

    def test_export_markdown_rescan_session_surfaces_enrichment_error(self, runner, tmp_path):
        """Targeted rescan failures stop the command before export."""
        db_path = tmp_path / "missing.db"
        with patch("copilot_session_tools.cli.parse_single_cli_session", return_value="events.jsonl not found for session missing-session"):
            result = runner.invoke(
                app,
                [
                    "export-markdown",
                    "--db",
                    str(db_path),
                    "--output-dir",
                    str(tmp_path / "markdown_output"),
                    "--session-id",
                    "missing-session",
                    "--rescan-session",
                    "missing-session",
                ],
            )

        assert result.exit_code == 1
        assert "events.jsonl not found" in result.output
        assert not db_path.exists()

    def test_export_html_command(self, runner, temp_db_with_data, tmp_path):
        """Test export-html command exports sessions to HTML files."""
        output_dir = tmp_path / "html_output"

        result = runner.invoke(app, ["export-html", "--db", str(temp_db_with_data), "--output-dir", str(output_dir), "-v"])
        assert result.exit_code == 0
        assert "Exported 1 sessions" in result.output

        # Check that an HTML file was created
        html_files = list(output_dir.glob("*.html"))
        assert len(html_files) == 1

        # Check content of the HTML file
        content = html_files[0].read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "cli-workspace" in content
        assert "cli-test-session" in content

    def test_export_html_single_session(self, runner, temp_db_with_data, tmp_path):
        """Test export-html command with specific session ID."""
        output_dir = tmp_path / "html_output"

        result = runner.invoke(
            app,
            [
                "export-html",
                "--db",
                str(temp_db_with_data),
                "--output-dir",
                str(output_dir),
                "--session-id",
                "cli-test-session",
            ],
        )
        assert result.exit_code == 0
        assert "Exported:" in result.output

        html_files = list(output_dir.glob("*.html"))
        assert len(html_files) == 1

    def test_export_html_missing_session(self, runner, temp_db_with_data, tmp_path):
        """Test export-html command with non-existent session ID."""
        output_dir = tmp_path / "html_output"

        result = runner.invoke(
            app,
            [
                "export-html",
                "--db",
                str(temp_db_with_data),
                "--output-dir",
                str(output_dir),
                "--session-id",
                "nonexistent-session",
            ],
        )
        assert result.exit_code == 1
        assert "not found" in result.output

    def test_export_html_static_mode(self, runner, temp_db_with_data, tmp_path):
        """Test that export-html output has no interactive elements."""
        output_dir = tmp_path / "html_output"

        runner.invoke(app, ["export-html", "--db", str(temp_db_with_data), "--output-dir", str(output_dir)])

        html_files = list(output_dir.glob("*.html"))
        content = html_files[0].read_text(encoding="utf-8")

        # Should NOT contain interactive elements
        assert '<div class="copy-markdown-toolbar">' not in content
        assert 'class="message-copy-btn"' not in content
        assert "cdnjs.cloudflare.com" not in content
        assert "buildMarkdownParams" not in content
        assert "Back to all sessions" not in content

        # SHOULD contain core content
        assert "<!DOCTYPE html>" in content
        assert "message-content" in content
        assert "--container-max-width: none" in content

    def test_search_rescan_session_bootstraps_missing_db_with_json_output(self, runner, tmp_path):
        """--rescan-session runs before search without polluting --json stdout."""
        db_path = tmp_path / "missing.db"
        session_id = "search-rescanned-session"

        def fake_parse(requested_session_id):
            return ChatSession(
                session_id=requested_session_id,
                workspace_name="search-rescanned-workspace",
                workspace_path="/tmp/search-rescanned",
                messages=[ChatMessage(role="user", content="uniqueSearchBootstrapToken")],
                vscode_edition="cli",
                type="cli",
            )

        with patch("copilot_session_tools.cli.parse_single_cli_session", side_effect=fake_parse) as mock_parse:
            result = runner.invoke(
                app,
                [
                    "search",
                    "--db",
                    str(db_path),
                    "uniqueSearchBootstrapToken",
                    "--rescan-session",
                    session_id,
                    "--json",
                ],
            )

        assert result.exit_code == 0
        mock_parse.assert_called_once()
        data = json.loads(result.output)
        assert len(data) == 1
        assert data[0]["session_id"] == session_id


class TestSearchIncludeExclude:
    """Tests for --include / --exclude flags on the search command."""

    def test_search_default_no_flags(self, runner, temp_db_with_data):
        """Default search without --include/--exclude works (backward compat)."""
        result = runner.invoke(app, ["search", "--db", str(temp_db_with_data), "Hello"])
        assert result.exit_code == 0
        assert "Result" in result.output or "result" in result.output.lower()

    def test_search_include_tools(self, runner, temp_db_with_data):
        """--include tools narrows search to tool invocations only."""
        result = runner.invoke(app, ["search", "--db", str(temp_db_with_data), "Hello", "--include", "tools"])
        assert result.exit_code == 0
        # Expect no results — fixture has only plain messages, no tool invocations
        assert "No results" in result.output

    def test_search_include_file_changes(self, runner, temp_db_with_data):
        """--include file-changes narrows search to file change content."""
        result = runner.invoke(app, ["search", "--db", str(temp_db_with_data), "Hello", "--include", "file-changes"])
        assert result.exit_code == 0
        assert "No results" in result.output

    def test_search_include_comma_separated(self, runner, temp_db_with_data):
        """--include tools,file-changes accepts comma-separated tokens."""
        result = runner.invoke(
            app,
            ["search", "--db", str(temp_db_with_data), "Hello", "--include", "tools,file-changes"],
        )
        assert result.exit_code == 0
        # No tool or file-change content in fixture
        assert "No results" in result.output

    def test_search_include_messages_finds_content(self, runner, temp_db_with_data):
        """--include messages still finds plain message content."""
        result = runner.invoke(app, ["search", "--db", str(temp_db_with_data), "Hello", "--include", "messages"])
        assert result.exit_code == 0
        assert "Result" in result.output or "CLI test" in result.output

    def test_search_include_repeated_flags(self, runner, temp_db_with_data):
        """Repeated --include flags combine with OR semantics."""
        result = runner.invoke(
            app,
            [
                "search",
                "--db",
                str(temp_db_with_data),
                "Hello",
                "--include",
                "messages",
                "--include",
                "tools",
            ],
        )
        assert result.exit_code == 0
        # messages included → finds the fixture message
        assert "Result" in result.output or "CLI test" in result.output

    def test_search_exclude_messages(self, runner, temp_db_with_data):
        """--exclude messages removes message content from search."""
        result = runner.invoke(app, ["search", "--db", str(temp_db_with_data), "Hello", "--exclude", "messages"])
        assert result.exit_code == 0
        # Fixture only has messages; excluding them yields no results
        assert "No results" in result.output

    def test_search_old_flags_rejected(self, runner, temp_db_with_data):
        """Legacy --no-tools / --tools-only flags are rejected as unknown."""
        for old_flag in ("--no-tools", "--tools-only", "--no-file-changes"):
            result = runner.invoke(app, ["search", "--db", str(temp_db_with_data), "Hello", old_flag])
            assert result.exit_code != 0, f"{old_flag} should be rejected"

    def test_search_include_diffs_auto_includes_file_changes(self, runner, temp_db_with_data):
        """--include diffs auto-includes file-changes parent (no error)."""
        result = runner.invoke(app, ["search", "--db", str(temp_db_with_data), "Hello", "--include", "diffs"])
        # Should succeed — file-changes is auto-added
        assert result.exit_code == 0

    def test_search_include_bogus_token(self, runner, temp_db_with_data):
        """--include bogus warns and falls back to defaults."""
        result = runner.invoke(app, ["search", "--db", str(temp_db_with_data), "Hello", "--include", "bogus"])
        assert result.exit_code == 0
        # Unknown token is warned on stderr; since all tokens were bogus,
        # resolve falls back to defaults and still finds the message
        assert "Result" in result.output or "CLI test" in result.output

    def test_search_exclude_everything_errors(self, runner, temp_db_with_data):
        """Excluding all content types raises an error (empty set)."""
        all_types = "messages,thinking,diffs,tool-inputs,agent-details,tools,commands,file-changes"
        result = runner.invoke(
            app,
            ["search", "--db", str(temp_db_with_data), "Hello", "--exclude", all_types],
        )
        assert result.exit_code != 0

    def test_search_short_flags(self, runner, temp_db_with_data):
        """-i and -x short forms work for --include and --exclude."""
        result = runner.invoke(app, ["search", "--db", str(temp_db_with_data), "Hello", "-i", "messages"])
        assert result.exit_code == 0
        assert "Result" in result.output or "CLI test" in result.output

        result = runner.invoke(app, ["search", "--db", str(temp_db_with_data), "Hello", "-x", "messages"])
        assert result.exit_code == 0
        assert "No results" in result.output


class TestOptimizeCommand:
    """Tests for the optimize CLI command."""

    def test_optimize_command_success(self, runner, tmp_path):
        """Test optimize command with valid database."""
        db_path = tmp_path / "optimize_test.db"
        db = Database(db_path)

        # Add a session to have some data in FTS index
        session = ChatSession(
            session_id="optimize-cli-test",
            workspace_name="optimize-workspace",
            workspace_path="/optimize/path",
            messages=[
                ChatMessage(role="user", content="Test for optimization"),
                ChatMessage(role="assistant", content="Response for optimization"),
            ],
        )
        db.add_session(session)

        result = runner.invoke(app, ["optimize", "--db", str(db_path)])
        assert result.exit_code == 0
        assert "Optimizing" in result.output
        assert "Optimization complete" in result.output
        assert "Index segments" in result.output

    def test_optimize_command_missing_db(self, runner, tmp_path):
        """Test optimize command with non-existent database."""
        result = runner.invoke(app, ["optimize", "--db", str(tmp_path / "nonexistent.db")])
        # Typer returns exit code 2 for validation errors (exists=True on file path)
        assert result.exit_code == 2


class TestDefaultDbPath:
    """Tests for _default_db_path() helper."""

    def test_returns_path_in_home_directory(self):
        """Default DB path should be under user's home directory."""
        result = _default_db_path()
        assert result.parent.parent == Path.home()
        assert result.name == "copilot-session-tools.db"
        assert result.parent.name == ".copilot"

    def test_returns_path_object(self):
        """Should return a Path object, not a string."""
        result = _default_db_path()
        assert isinstance(result, Path)


class TestEnsureDbExists:
    """Tests for _ensure_db_exists() helper."""

    def test_raises_exit_when_db_missing(self, tmp_path):
        """Should raise typer.Exit when database doesn't exist."""
        import typer

        nonexistent = tmp_path / "nonexistent.db"
        with pytest.raises(typer.Exit):
            _ensure_db_exists(nonexistent)

    def test_no_error_when_db_exists(self, tmp_path):
        """Should not raise when database exists."""
        existing = tmp_path / "test.db"
        existing.touch()
        _ensure_db_exists(existing)  # Should not raise
