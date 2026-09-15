"""Snapshot/regression tests for scanner + exporter output stability.

These tests parse fixed session files and compare their exported output
(markdown, HTML) against saved baselines using pytest-regressions.

Fixtures and baselines live in tests/snapshots/ (gitignored).
Tests are skipped when fixtures are absent.

Usage:
    # Normal run (compares against baselines)
    uv run pytest tests/test_snapshot.py -v

    # Regenerate all baselines after intentional changes
    uv run pytest tests/test_snapshot.py --regen-all -v

    # Regenerate only failing baselines
    uv run pytest tests/test_snapshot.py --force-regen -v
"""

from pathlib import Path

import pytest

from copilot_session_tools.html_exporter import session_to_html
from copilot_session_tools.markdown_exporter import session_to_markdown
from copilot_session_tools.scanner import ChatSession, SessionFileInfo, parse_session_file
from copilot_session_tools.scanner.cli import _parse_cli_jsonl_file

SNAPSHOTS_DIR = Path(__file__).parent / "snapshots"
FIXTURES_DIR = SNAPSHOTS_DIR / "fixtures"
BASELINES_DIR = SNAPSHOTS_DIR / "baselines"


# --- CLI Session Snapshots ---

_cli_events = FIXTURES_DIR / "cli" / "events.jsonl"
_cli_fixture_exists = _cli_events.exists() if FIXTURES_DIR.exists() else False


@pytest.mark.skipif(not _cli_fixture_exists, reason="CLI fixture not populated")
class TestCLISessionSnapshot:
    """Snapshot tests for CLI session parsing and export."""

    session: ChatSession

    @pytest.fixture(autouse=True)
    def parse_session(self):
        parsed = _parse_cli_jsonl_file(_cli_events)
        assert parsed is not None, f"Failed to parse CLI fixture: {_cli_events}"
        self.session = parsed
        self.session.source_file = "cli"

    def test_markdown_export(self, file_regression):
        md = session_to_markdown(
            self.session,
            content_set={"diffs", "tool-inputs", "thinking", "agent-details", "tools", "commands", "file-changes"},
        )
        file_regression.check(md, fullpath=BASELINES_DIR / "cli.md", encoding="utf-8")

    def test_html_export(self, file_regression):
        html = session_to_html(self.session)
        file_regression.check(html, fullpath=BASELINES_DIR / "cli.html", encoding="utf-8")


# --- CLI Session with Agents Snapshots ---

_cli_agents_events = FIXTURES_DIR / "cli-with-agents" / "events.jsonl"
_cli_agents_fixture_exists = _cli_agents_events.exists() if FIXTURES_DIR.exists() else False


@pytest.mark.skipif(not _cli_agents_fixture_exists, reason="CLI agents fixture not populated")
class TestCLIAgentSessionSnapshot:
    """Snapshot tests for CLI session with structured agent rendering."""

    session: ChatSession

    @pytest.fixture(autouse=True)
    def parse_session(self):
        parsed = _parse_cli_jsonl_file(_cli_agents_events)
        assert parsed is not None, f"Failed to parse CLI agents fixture: {_cli_agents_events}"
        self.session = parsed
        self.session.source_file = "cli-with-agents"

    def test_markdown_export(self, file_regression):
        md = session_to_markdown(
            self.session,
            content_set={"diffs", "tool-inputs", "thinking", "agent-details", "tools", "commands", "file-changes"},
        )
        file_regression.check(md, fullpath=BASELINES_DIR / "cli-with-agents.md", encoding="utf-8")

    def test_html_export(self, file_regression):
        html = session_to_html(self.session)
        file_regression.check(html, fullpath=BASELINES_DIR / "cli-with-agents.html", encoding="utf-8")


# --- CLI Session with External (host-app) Tools Snapshots ---

_cli_external_tools_events = FIXTURES_DIR / "cli-with-external-tools" / "events.jsonl"
_cli_external_tools_fixture_exists = _cli_external_tools_events.exists() if FIXTURES_DIR.exists() else False


@pytest.mark.skipif(not _cli_external_tools_fixture_exists, reason="CLI external tools fixture not populated")
class TestCLIExternalToolsSnapshot:
    """Snapshot tests for CLI session with external (host-app) tool rendering."""

    session: ChatSession

    @pytest.fixture(autouse=True)
    def parse_session(self):
        parsed = _parse_cli_jsonl_file(_cli_external_tools_events)
        assert parsed is not None, f"Failed to parse CLI external tools fixture: {_cli_external_tools_events}"
        self.session = parsed
        self.session.source_file = "cli-with-external-tools"

    def test_markdown_export(self, file_regression):
        md = session_to_markdown(
            self.session,
            content_set={"diffs", "tool-inputs", "thinking", "agent-details", "tools", "commands", "file-changes"},
        )
        file_regression.check(md, fullpath=BASELINES_DIR / "cli-with-external-tools.md", encoding="utf-8")

    def test_html_export(self, file_regression):
        html = session_to_html(
            self.session,
            content_set={"diffs", "tool-inputs", "thinking", "agent-details", "tools", "commands", "file-changes"},
        )
        file_regression.check(html, fullpath=BASELINES_DIR / "cli-with-external-tools.html", encoding="utf-8")

    def test_external_tools_parsed(self):
        """External tools render as ToolInvocations carrying parameters but no result."""
        external = [t for msg in self.session.messages for t in msg.tool_invocations if t.source_type == "external"]
        names = sorted(t.name for t in external)
        assert names == [
            "annotate_diff_line",
            "create_pull_request",
            "list_sessions_and_chats",
            "rename_branch",
            "rename_session",
            "reply_to_comment",
        ]
        # Every external tool records no response; all but the no-arg one expose parameters.
        for tool in external:
            assert tool.result is None, f"external tool {tool.name} should not record a response"
        assert all(t.input for t in external if t.name != "list_sessions_and_chats")
        # The no-argument external tool still parses as an external ToolInvocation.
        noarg = [t for t in external if t.name == "list_sessions_and_chats"]
        assert len(noarg) == 1

    def test_external_tools_not_status_blocks(self):
        """External tools must not leak through as bare status breadcrumbs."""
        status_descs = [cb.description for msg in self.session.messages for cb in msg.content_blocks if cb.kind == "status"]
        assert "external-tool" not in status_descs


# --- CLI Session with Async Shells Snapshots ---

_cli_async_shells_events = FIXTURES_DIR / "cli-with-async-shells" / "events.jsonl"
_cli_async_shells_fixture_exists = _cli_async_shells_events.exists() if FIXTURES_DIR.exists() else False


@pytest.mark.skipif(not _cli_async_shells_fixture_exists, reason="CLI async shells fixture not populated")
class TestCLIAsyncShellsSnapshot:
    """Snapshot tests for CLI session with async shell rendering."""

    session: ChatSession

    @pytest.fixture(autouse=True)
    def parse_session(self):
        parsed = _parse_cli_jsonl_file(_cli_async_shells_events)
        assert parsed is not None, f"Failed to parse CLI async shells fixture: {_cli_async_shells_events}"
        self.session = parsed
        self.session.source_file = "cli-with-async-shells"

    def test_html_export(self, file_regression):
        html = session_to_html(self.session)
        file_regression.check(html, fullpath=BASELINES_DIR / "cli-with-async-shells.html", encoding="utf-8")

    def test_session_has_async_shells(self):
        """Verify the parsed session has async command runs."""
        async_cmds = [c for msg in self.session.messages for c in msg.command_runs if c.is_async]
        assert len(async_cmds) >= 2, f"Expected at least 2 async shells, got {len(async_cmds)}"
        shell_ids = {c.shell_id for c in async_cmds}
        assert "dev-server" in shell_ids
        assert "tsc-watch" in shell_ids

    def test_detached_shell_detected(self):
        """Verify detached shell has is_detached=True."""
        detached = [c for msg in self.session.messages for c in msg.command_runs if c.is_detached]
        assert len(detached) == 1
        assert detached[0].shell_id == "tsc-watch"

    def test_sync_shell_not_async(self):
        """Verify sync shell (git status) has is_async=False."""
        sync_cmds = [c for msg in self.session.messages for c in msg.command_runs if not c.is_async]
        assert len(sync_cmds) >= 1
        assert any("git status" in c.command for c in sync_cmds)

    def test_shell_backlinks_created(self):
        """Verify read/write/stop_powershell create shell backlinks."""
        backlinks = [t for msg in self.session.messages for t in msg.tool_invocations if t.is_shell_backlink]
        assert len(backlinks) == 3  # read, write, stop
        actions = {t.name for t in backlinks}
        assert actions == {"read_powershell", "write_powershell", "stop_powershell"}
        assert all(t.backlink_shell_id == "dev-server" for t in backlinks)

    def test_io_entries_collected(self):
        """Verify IO entries are collected on the parent async shell."""
        async_cmds = [c for msg in self.session.messages for c in msg.command_runs if c.shell_id == "dev-server"]
        assert len(async_cmds) == 1
        cmd = async_cmds[0]
        assert len(cmd.io_entries) == 3  # read, write, stop
        actions = [io.action for io in cmd.io_entries]
        assert actions == ["read", "write", "stop"]

    def test_write_io_has_input(self):
        """Verify write IO entry captures the input text."""
        async_cmds = [c for msg in self.session.messages for c in msg.command_runs if c.shell_id == "dev-server"]
        write_ios = [io for io in async_cmds[0].io_entries if io.action == "write"]
        assert len(write_ios) == 1
        assert write_ios[0].input_text == "rs{enter}"

    def test_html_has_amber_styling(self):
        """Verify HTML output contains async shell amber CSS classes."""
        html = session_to_html(self.session)
        assert "command-async" in html
        assert "async-shell-badge" in html
        assert "fa-arrows-rotate" in html  # async icon
        assert "fa-link-slash" in html  # detached icon

    def test_html_has_io_entries(self):
        """Verify HTML output renders IO entries inside shell blocks."""
        html = session_to_html(self.session)
        assert "io-entry" in html
        assert "io-dev-server-read-1" in html
        assert "io-dev-server-write-1" in html
        assert "io-dev-server-stop-1" in html

    def test_html_has_bidirectional_links(self):
        """Verify HTML output has pill→IO and IO→pill links."""
        html = session_to_html(self.session)
        assert 'href="#io-dev-server-read-1"' in html  # pill → IO
        assert 'href="#pill-dev-server-read-1"' in html  # IO → pill

    def test_html_has_shell_conv_pills(self):
        """Verify HTML output renders conversation pills for shell backlinks."""
        html = session_to_html(self.session)
        assert "shell-conv-pill" in html
        assert "READ" in html
        assert "WRITE" in html
        assert "STOP" in html


# --- CLI Session with File Changes Snapshots ---

_cli_file_changes_events = FIXTURES_DIR / "cli-file-changes" / "events.jsonl"
_cli_file_changes_fixture_exists = _cli_file_changes_events.exists() if FIXTURES_DIR.exists() else False


@pytest.mark.skipif(not _cli_file_changes_fixture_exists, reason="CLI file-changes fixture not populated")
class TestCLIFileChangesSnapshot:
    """Snapshot tests for CLI edit/create reshaped into FileChange diffs (VS Code parity)."""

    session: ChatSession

    @pytest.fixture(autouse=True)
    def parse_session(self):
        parsed = _parse_cli_jsonl_file(_cli_file_changes_events)
        assert parsed is not None, f"Failed to parse CLI file-changes fixture: {_cli_file_changes_events}"
        self.session = parsed
        self.session.source_file = "cli-file-changes"

    def test_markdown_export(self, file_regression):
        md = session_to_markdown(
            self.session,
            content_set={"diffs", "tool-inputs", "thinking", "agent-details", "tools", "commands", "file-changes"},
        )
        file_regression.check(md, fullpath=BASELINES_DIR / "cli-file-changes.md", encoding="utf-8")

    def test_html_export(self, file_regression):
        html = session_to_html(self.session)
        file_regression.check(html, fullpath=BASELINES_DIR / "cli-file-changes.html", encoding="utf-8")

    def test_successful_edits_become_file_changes(self):
        """create + successful edit on the same file consolidate to ONE diff; failed edit gated out."""
        changes = [fc for m in self.session.messages for fc in m.file_changes]
        assert len(changes) == 1
        fc = changes[0]
        assert fc.path.endswith("utils.py")
        assert "missing.py" not in fc.path
        assert fc.language_id == "python"
        assert fc.diff and "+def add(a, b):" in fc.diff and "+def mul(a, b):" in fc.diff

    def test_html_renders_file_changes_section(self):
        """The web/HTML renderer surfaces the File Changes section with a diff."""
        html = session_to_html(self.session)
        assert "File Changes" in html
        assert "file-change" in html
        assert "+def mul(a, b):" in html


_SESSION_STORE_DB = Path.home() / ".copilot" / "session-store.db"
# Same session as the CLI fixture so enriched/unenriched snapshots are directly comparable
_UNENRICHED_SESSION_ID = "01488532-ff4e-41c6-b137-5f75a48742d3"
_session_store_exists = _SESSION_STORE_DB.exists()


@pytest.mark.skipif(not _session_store_exists, reason="Session store DB not found")
class TestCLIUnenrichedSnapshot:
    """Snapshot test for unenriched CLI session rendering (built-in turns only)."""

    @pytest.fixture(autouse=True)
    def load_from_db(self):
        import sqlite3

        from copilot_session_tools.html_exporter import _get_jinja_env
        from copilot_session_tools.scanner.models import ChatSession

        conn = sqlite3.connect(str(_SESSION_STORE_DB))
        conn.row_factory = sqlite3.Row
        session_row = conn.execute("SELECT * FROM sessions WHERE id = ?", (_UNENRICHED_SESSION_ID,)).fetchone()
        if not session_row:
            conn.close()
            pytest.skip(f"Session {_UNENRICHED_SESSION_ID} not in session store")

        turn_rows = conn.execute(
            "SELECT turn_index, user_message, assistant_response FROM turns WHERE session_id = ? ORDER BY turn_index",
            (_UNENRICHED_SESSION_ID,),
        ).fetchall()
        conn.close()

        self.turns = [dict(r) for r in turn_rows]
        self.session = ChatSession(
            session_id=_UNENRICHED_SESSION_ID,
            workspace_name=session_row["repository"],
            workspace_path=session_row["cwd"],
            messages=[],
            created_at=session_row["created_at"],
            updated_at=session_row["updated_at"],
            vscode_edition="cli",
            custom_title=session_row["summary"],
            type="cli",
            source_format="cli",
        )
        self.env = _get_jinja_env()

    def test_unenriched_html(self, file_regression):
        # Count individual messages (matching how _get_builtin_session_as_chat_session works)
        msg_count = sum((1 if t.get("user_message") else 0) + (1 if t.get("assistant_response") else 0) for t in self.turns)
        template = self.env.get_template("session.html")
        html = template.render(
            title=self.session.custom_title or self.session.workspace_name or "Unenriched Session",
            session=self.session,
            message_count=msg_count,
            first_user_prompt=None,
            message_metadata={},
            static=True,
            is_enriched=False,
            turns=self.turns,
        )
        file_regression.check(html, fullpath=BASELINES_DIR / "cli-unenriched.html", encoding="utf-8")


# --- VS Code Session Snapshots ---

# Discover all vscode-* fixture dirs and their session files
# Each entry is (unique_key, path, file_type) — key includes format suffix for uniqueness
_vscode_fixtures: list[tuple[str, Path, str]] = []
if FIXTURES_DIR.exists():
    for d in sorted(FIXTURES_DIR.iterdir()):
        if not d.is_dir() or not d.name.startswith("vscode-"):
            continue
        for f in sorted(d.iterdir()):
            if f.suffix in (".json", ".jsonl", ".vscdb"):
                # Use dir name as key (dirs are named per-format: vscode-insider-json, etc.)
                # Append file stem if dir has multiple files to avoid collisions
                key = d.name
                _vscode_fixtures.append((key, f, f.suffix.lstrip(".")))


@pytest.mark.skipif(not _vscode_fixtures, reason="VS Code fixtures not populated")
class TestVSCodeSessionSnapshot:
    """Snapshot tests for VS Code session parsing and export."""

    @pytest.fixture(
        autouse=True,
        params=list(range(len(_vscode_fixtures))) if _vscode_fixtures else [0],
        ids=[fx[0] for fx in _vscode_fixtures] if _vscode_fixtures else ["skip"],
    )
    def parse_session(self, request):
        if not _vscode_fixtures:
            pytest.skip("No VS Code fixtures")
        idx = request.param
        name, path, file_type = _vscode_fixtures[idx]
        edition = "insider" if "insider" in name else "stable"
        file_info = SessionFileInfo(
            file_path=path.resolve(),
            file_type=file_type,
            session_type="vscode",
            vscode_edition=edition,
            mtime=path.stat().st_mtime,
            size=path.stat().st_size,
            workspace_name=None,
            workspace_path=None,
        )
        sessions = parse_session_file(file_info)
        if not sessions:
            pytest.skip(f"No sessions parsed from {path}")
        self.session = sessions[0]
        # Neutralize source path to avoid OS/cwd-dependent snapshots
        self.session.source_file = name
        self.fixture_name = name

    def test_markdown_export(self, file_regression):
        md = session_to_markdown(
            self.session,
            content_set={"diffs", "tool-inputs", "thinking", "agent-details", "tools", "commands", "file-changes"},
        )
        file_regression.check(md, fullpath=BASELINES_DIR / f"{self.fixture_name}.md", encoding="utf-8")

    def test_html_export(self, file_regression):
        html = session_to_html(self.session)
        file_regression.check(html, fullpath=BASELINES_DIR / f"{self.fixture_name}.html", encoding="utf-8")


class TestBacklinkStateRendering:
    """Verify that read_agent backlinks are rendered with correct state classes."""

    @staticmethod
    def _make_events(*events):
        import ssrjson

        lines = [ssrjson.dumps({"type": "session.start", "data": {"sessionId": "backlink-state-test", "startTime": "2026-01-01T00:00:00Z"}})]
        for evt in events:
            lines.append(ssrjson.dumps(evt))
        return "\n".join(lines)

    @staticmethod
    def _bg_events_with_read_agent(read_result_content: str):
        """Create events for a bg agent with a read_agent returning the given content."""
        return [
            {
                "type": "assistant.message",
                "data": {
                    "content": "Launching agent.",
                    "toolRequests": [
                        {
                            "toolCallId": "tc-bg",
                            "name": "task",
                            "arguments": {
                                "agent_type": "explore",
                                "description": "Check auth",
                                "prompt": "Review auth module.",
                                "mode": "background",
                            },
                        }
                    ],
                },
            },
            {
                "type": "tool.execution_start",
                "data": {
                    "toolCallId": "tc-bg",
                    "toolName": "task",
                    "arguments": {
                        "agent_type": "explore",
                        "description": "Check auth",
                    },
                },
            },
            {
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": "tc-bg",
                    "success": True,
                    "result": {
                        "content": "Agent started in background with agent_id: agent-0. Use read_agent to check.",
                    },
                },
            },
            {"type": "subagent.started", "data": {"toolCallId": "tc-bg", "agentDisplayName": "Explore Agent"}},
            {
                "type": "assistant.message",
                "data": {
                    "content": "Checking agent.",
                    "toolRequests": [{"toolCallId": "tc-read", "name": "read_agent", "arguments": {"agent_id": "agent-0"}}],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc-read", "toolName": "read_agent", "arguments": {"agent_id": "agent-0"}}},
            {
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": "tc-read",
                    "success": True,
                    "result": {
                        "content": read_result_content,
                    },
                },
            },
            {"type": "assistant.message", "data": {"content": "Done."}},
        ]

    def _render_html(self, tmp_path, read_result_content):
        events = self._bg_events_with_read_agent(read_result_content)
        f = tmp_path / "events.jsonl"
        f.write_text(self._make_events(*events), encoding="utf-8")
        session = _parse_cli_jsonl_file(f)
        assert session is not None
        return session_to_html(session)

    def test_running_agent_gets_in_progress_class(self, tmp_path):
        """read_agent showing 'still running' should render backlink-in-progress."""
        html = self._render_html(tmp_path, "Agent is still running after waiting 60s. agent_id: agent-0, status: running")
        assert "backlink-in-progress" in html
        assert "in progress" in html

    def test_completed_agent_gets_completed_class(self, tmp_path):
        """read_agent showing completed result should render backlink-completed (default)."""
        html = self._render_html(tmp_path, "Agent completed. agent_id: agent-0, status: completed\n\nResult:\nAll good.")
        assert "backlink-completed" in html or ("agent-backlink" in html and "backlink-in-progress" not in html and "backlink-failed" not in html)
        assert "completed" in html

    def test_failed_agent_gets_failed_class(self, tmp_path):
        """read_agent showing failed result should render backlink-failed."""
        html = self._render_html(tmp_path, "Agent failed. agent_id: agent-0, status: failed\n\nError: timeout")
        assert "backlink-failed" in html
        assert "failed" in html

    def test_standalone_subagent_pills_hidden(self, tmp_path):
        """subagent.completed/failed status pills should NOT appear in rendered HTML."""
        events = self._bg_events_with_read_agent("Agent completed. status: completed")
        # Add subagent.completed event
        events.append({"type": "subagent.completed", "data": {"toolCallId": "tc-bg", "agentDisplayName": "Explore Agent"}})
        f = tmp_path / "events.jsonl"
        f.write_text(self._make_events(*events), encoding="utf-8")
        session = _parse_cli_jsonl_file(f)
        assert session is not None
        html = session_to_html(session)
        # The CSS class may exist in stylesheet, but no element should use it
        assert '<span class="subagent-started">' not in html


class TestVSCodePromptRendering:
    """Verify that VS Code subagent prompts are extracted and rendered in HTML."""

    @staticmethod
    def _build_vscode_session_with_subagent(prompt: str | None = None) -> ChatSession:
        """Build a minimal ChatSession with a VS Code subagent containing the given prompt."""
        from copilot_session_tools.scanner.content import _merge_content_blocks
        from copilot_session_tools.scanner.models import ChatMessage
        from copilot_session_tools.scanner.vscode import _process_response_items

        tool_specific_data: dict = {
            "kind": "subagent",
            "agentName": "explore",
            "description": "Search auth patterns",
            "result": "Found JWT auth in 3 files.",
        }
        if prompt is not None:
            tool_specific_data["prompt"] = prompt

        response_items = [
            {
                "kind": "toolInvocationSerialized",
                "toolId": "runSubagent",
                "invocationMessage": "",
                "toolSpecificData": tool_specific_data,
                "isComplete": True,
                "toolCallId": "call_vsc_1",
            },
        ]
        _, raw_blocks, tool_invocations, file_changes, command_runs = _process_response_items(response_items)
        content_blocks = _merge_content_blocks(raw_blocks)

        msg = ChatMessage(
            role="assistant",
            content="Here are the results.",
            tool_invocations=tool_invocations,
            file_changes=file_changes,
            command_runs=command_runs,
            content_blocks=content_blocks,
        )
        return ChatSession(
            session_id="vscode-prompt-test",
            messages=[msg],
            source_file="test-vscode-prompt",
            workspace_name=None,
            workspace_path=None,
        )

    def test_vscode_prompt_rendered_in_html(self):
        """VS Code subagent with prompt should render the subagent-prompt section."""
        session = self._build_vscode_session_with_subagent(prompt="Investigate the authentication module for security issues.")
        html = session_to_html(session)
        assert "subagent-prompt" in html
        assert "Investigate the authentication module" in html

    def test_vscode_no_prompt_omits_section(self):
        """VS Code subagent without prompt should NOT render the subagent-prompt section."""
        session = self._build_vscode_session_with_subagent(prompt=None)
        html = session_to_html(session)
        # The CSS class "subagent-prompt-content" exists in the stylesheet, but no prompt <details> element should be rendered
        assert '<details class="subagent-prompt"' not in html


class TestAsyncShellRendering:
    """Verify that async shell commands render with amber styling and shell backlinks."""

    @staticmethod
    def _make_events(*events):
        import ssrjson

        lines = [ssrjson.dumps({"type": "session.start", "data": {"sessionId": "async-shell-render-test", "startTime": "2026-01-01T00:00:00Z"}})]
        for evt in events:
            lines.append(ssrjson.dumps(evt))
        return "\n".join(lines)

    @staticmethod
    def _async_shell_events():
        """Events for an async shell with a read_powershell interaction."""
        return [
            {
                "type": "assistant.message",
                "data": {
                    "content": "Starting server.",
                    "toolRequests": [
                        {
                            "toolCallId": "tc-shell",
                            "name": "powershell",
                            "arguments": {"command": "npm run dev", "description": "Start dev server", "mode": "async", "shellId": "dev-srv"},
                        }
                    ],
                },
            },
            {
                "type": "tool.execution_start",
                "data": {"toolCallId": "tc-shell", "toolName": "powershell", "arguments": {"command": "npm run dev", "mode": "async", "shellId": "dev-srv"}},
            },
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-shell", "success": True, "result": {"content": "Server started"}}},
            {
                "type": "assistant.message",
                "data": {
                    "content": "Checking output.",
                    "toolRequests": [{"toolCallId": "tc-read", "name": "read_powershell", "arguments": {"shellId": "dev-srv", "delay": 5}}],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc-read", "toolName": "read_powershell", "arguments": {"shellId": "dev-srv"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-read", "success": True, "result": {"content": "Listening on :3000"}}},
            {"type": "assistant.message", "data": {"content": "Ready."}},
        ]

    def _render_html(self, tmp_path, events):
        f = tmp_path / "events.jsonl"
        f.write_text(self._make_events(*events), encoding="utf-8")
        session = _parse_cli_jsonl_file(f)
        assert session is not None
        return session_to_html(session)

    def test_async_shell_renders_with_amber_class(self, tmp_path):
        """Async shell should render with command-async CSS class."""
        html = self._render_html(tmp_path, self._async_shell_events())
        assert "command-async" in html

    def test_async_shell_has_badge(self, tmp_path):
        """Async shell should render with an async badge."""
        html = self._render_html(tmp_path, self._async_shell_events())
        assert "async-shell-badge" in html
        assert "fa-arrows-rotate" in html

    def test_async_shell_has_anchor(self, tmp_path):
        """Async shell block should have an anchor id based on shellId."""
        html = self._render_html(tmp_path, self._async_shell_events())
        assert 'id="shell-dev-srv"' in html

    def test_shell_backlink_rendered(self, tmp_path):
        """read_powershell should render as a shell-conv-pill."""
        html = self._render_html(tmp_path, self._async_shell_events())
        assert "shell-conv-pill" in html
        assert "READ" in html

    def test_shell_backlink_has_href(self, tmp_path):
        """Shell backlink should link to IO entry inside the shell block."""
        html = self._render_html(tmp_path, self._async_shell_events())
        assert "#io-dev-srv-read-1" in html

    def test_detached_shell_shows_link_slash_icon(self, tmp_path):
        """Detached shell should show fa-link-slash icon instead of fa-arrows-rotate."""
        events = [
            {
                "type": "assistant.message",
                "data": {
                    "content": "Starting daemon.",
                    "toolRequests": [
                        {"toolCallId": "tc-det", "name": "powershell", "arguments": {"command": "python server.py", "mode": "async", "detach": True, "shellId": "daemon"}},
                    ],
                },
            },
            {
                "type": "tool.execution_start",
                "data": {"toolCallId": "tc-det", "toolName": "powershell", "arguments": {"command": "python server.py", "mode": "async", "detach": True, "shellId": "daemon"}},
            },
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-det", "success": True, "result": {"content": "Daemon started"}}},
            {"type": "assistant.message", "data": {"content": "Running."}},
        ]
        html = self._render_html(tmp_path, events)
        assert "fa-link-slash" in html
        assert "detached" in html

    def test_io_entry_rendered_inside_shell_block(self, tmp_path):
        """IO entries should appear inside the async shell block."""
        html = self._render_html(tmp_path, self._async_shell_events())
        assert "io-entry" in html
        assert "io-dev-srv-read-1" in html

    def test_bidirectional_links(self, tmp_path):
        """IO entry should link to pill and pill should link to IO entry."""
        html = self._render_html(tmp_path, self._async_shell_events())
        # Pill links to IO entry
        assert 'href="#io-dev-srv-read-1"' in html
        # IO entry links back to pill
        assert 'href="#pill-dev-srv-read-1"' in html

    def test_sync_shell_no_amber(self, tmp_path):
        """Sync shell (no mode) should NOT have async amber styling."""
        events = [
            {
                "type": "assistant.message",
                "data": {
                    "content": "Checking status.",
                    "toolRequests": [{"toolCallId": "tc-sync", "name": "powershell", "arguments": {"command": "git status"}}],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc-sync", "toolName": "powershell", "arguments": {"command": "git status"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-sync", "success": True, "result": {"content": "On branch main"}}},
            {"type": "assistant.message", "data": {"content": "Clean."}},
        ]
        html = self._render_html(tmp_path, events)
        # command-async appears in CSS stylesheet but should not be applied to any element
        assert 'class="tool-invocation-wrapper command-async"' not in html
        assert "async-shell-badge" not in html or html.count("async-shell-badge") == html.count(".async-shell-badge")
