"""Tests for the scanner module."""

import json
import time
from pathlib import Path
from typing import cast

import pytest
from conftest import requires_sample_files

from copilot_session_tools import (
    ChatMessage,
    ChatSession,
    CommandRun,
    FileChange,
    RootAgentInterval,
    ToolInvocation,
    find_copilot_chat_dirs,
    scan_chat_sessions,
)
from copilot_session_tools.scanner import (
    _extract_edit_group_text,
    _extract_inline_reference_name,
    _merge_content_blocks,
    _parse_tool_invocation_serialized,
)
from copilot_session_tools.scanner.models import ContentBlock
from copilot_session_tools.scanner.models import ToolInvocation as ScannerToolInvocation
from copilot_session_tools.scanner.vscode import _process_response_items


@pytest.fixture
def mock_workspace_storage(tmp_path):
    """Create a mock VS Code workspace storage structure."""
    # Create workspace directory with hash-like name
    workspace_dir = tmp_path / "abc123def456"
    workspace_dir.mkdir()

    # Create workspace.json
    workspace_json = workspace_dir / "workspace.json"
    workspace_json.write_text(json.dumps({"folder": "file:///home/user/projects/test-project"}))

    # Create chatSessions directory
    chat_sessions_dir = workspace_dir / "chatSessions"
    chat_sessions_dir.mkdir()

    # Create a sample chat session file
    session_file = chat_sessions_dir / "session-001.json"
    session_data = {
        "sessionId": "session-001",
        "createdAt": "2025-01-15T10:00:00Z",
        "messages": [
            {"role": "user", "content": "What is Python?"},
            {"role": "assistant", "content": "Python is a programming language."},
        ],
    }
    session_file.write_text(json.dumps(session_data))

    return tmp_path


class TestScanner:
    """Tests for the scanner module."""

    def test_find_copilot_chat_dirs(self, mock_workspace_storage):
        """Test finding Copilot chat directories."""
        storage_paths = [(str(mock_workspace_storage), "stable")]
        dirs = list(find_copilot_chat_dirs(storage_paths))

        assert len(dirs) >= 1
        # Should find the chatSessions directory
        chat_dir_found = any("chatSessions" in str(d[0]) for d in dirs)
        assert chat_dir_found

    def test_scan_chat_sessions(self, mock_workspace_storage):
        """Test scanning for chat sessions."""
        storage_paths = [(str(mock_workspace_storage), "stable")]
        sessions = list(scan_chat_sessions(storage_paths))

        assert len(sessions) >= 1
        session = sessions[0]
        assert session.session_id == "session-001"
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert "Python" in session.messages[0].content

    def test_scan_empty_storage(self, tmp_path):
        """Test scanning an empty storage directory."""
        storage_paths = [(str(tmp_path), "stable")]
        # Exclude CLI sessions to test VS Code scanning isolation
        sessions = list(scan_chat_sessions(storage_paths, include_cli=False))
        assert len(sessions) == 0

    def test_scan_nonexistent_path(self, tmp_path):
        """Test scanning a nonexistent path."""
        storage_paths = [(str(tmp_path / "nonexistent"), "stable")]
        # Exclude CLI sessions to test VS Code scanning isolation
        sessions = list(scan_chat_sessions(storage_paths, include_cli=False))
        assert len(sessions) == 0


class TestChatMessage:
    """Tests for the ChatMessage dataclass."""

    def test_create_chat_message(self):
        """Test creating a ChatMessage."""
        msg = ChatMessage(role="user", content="Hello", timestamp="2025-01-15T10:00:00Z")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp == "2025-01-15T10:00:00Z"

    def test_chat_message_defaults(self):
        """Test ChatMessage default values."""
        msg = ChatMessage(role="assistant", content="Hi there")
        assert msg.timestamp is None


class TestChatSession:
    """Tests for the ChatSession dataclass."""

    def test_create_chat_session(self):
        """Test creating a ChatSession."""
        session = ChatSession(
            session_id="test-123",
            workspace_name="my-project",
            workspace_path="/home/user/my-project",
            messages=[
                ChatMessage(role="user", content="Question"),
                ChatMessage(role="assistant", content="Answer"),
            ],
        )
        assert session.session_id == "test-123"
        assert len(session.messages) == 2
        assert session.vscode_edition == "stable"

    def test_chat_session_defaults(self):
        """Test ChatSession default values."""
        session = ChatSession(
            session_id="test-456",
            workspace_name=None,
            workspace_path=None,
            messages=[],
        )
        assert session.created_at is None
        assert session.updated_at is None
        assert session.source_file is None
        assert session.vscode_edition == "stable"


class TestToolInvocationsAndFileChanges:
    """Tests for the new tool invocation and file change data structures."""

    def test_chat_message_with_tool_invocations(self):
        """Test ChatMessage with tool invocations."""
        msg = ChatMessage(
            role="assistant",
            content="Running a command...",
            tool_invocations=[
                ToolInvocation(
                    name="run_command",
                    input="ls -la",
                    result="file1.txt\nfile2.txt",
                    status="success",
                )
            ],
        )
        assert len(msg.tool_invocations) == 1
        assert msg.tool_invocations[0].name == "run_command"
        assert msg.tool_invocations[0].status == "success"

    def test_chat_message_with_file_changes(self):
        """Test ChatMessage with file changes."""
        msg = ChatMessage(
            role="assistant",
            content="Made some changes...",
            file_changes=[
                FileChange(
                    path="/home/user/project/main.py",
                    diff="+ added line",
                    language_id="python",
                )
            ],
        )
        assert len(msg.file_changes) == 1
        assert msg.file_changes[0].path == "/home/user/project/main.py"
        assert msg.file_changes[0].language_id == "python"

    def test_chat_message_with_command_runs(self):
        """Test ChatMessage with command runs."""
        msg = ChatMessage(
            role="assistant",
            content="Executing...",
            command_runs=[
                CommandRun(
                    command="npm install",
                    title="Install dependencies",
                    status="success",
                    output="added 100 packages",
                )
            ],
        )
        assert len(msg.command_runs) == 1
        assert msg.command_runs[0].command == "npm install"
        assert msg.command_runs[0].status == "success"

    def test_chat_session_with_extended_fields(self):
        """Test ChatSession with new extended fields."""
        session = ChatSession(
            session_id="test-extended",
            workspace_name="my-project",
            workspace_path="/home/user/my-project",
            messages=[],
            custom_title="My Important Chat",
            requester_username="user",
            responder_username="copilot",
        )
        assert session.custom_title == "My Important Chat"
        assert session.requester_username == "user"
        assert session.responder_username == "copilot"


class TestResponseItemKinds:
    """Tests for parsing different response item kinds from VS Code Copilot Chat."""

    @pytest.mark.parametrize(
        "kind,item,expected_type",
        [
            # inlineReference with name
            ("inlineReference", {"kind": "inlineReference", "name": "test.py"}, str),
            # inlineReference with nested path
            ("inlineReference", {"kind": "inlineReference", "inlineReference": {"path": "/src/test.py"}}, str),
            # textEditGroup with dict URI
            ("textEditGroup", {"kind": "textEditGroup", "uri": {"path": "/src/file.ts", "scheme": "file"}}, str),
            # textEditGroup with string URI
            ("textEditGroup", {"kind": "textEditGroup", "uri": "file:///src/file.ts"}, str),
            # codeblockUri
            ("codeblockUri", {"kind": "codeblockUri", "uri": {"fsPath": "c:\\src\\file.py"}}, str),
            # toolInvocationSerialized
            (
                "toolInvocationSerialized",
                {"kind": "toolInvocationSerialized", "toolId": "run_command", "isComplete": True},
                ToolInvocation,
            ),
        ],
    )
    def test_response_item_extraction(self, kind, item, expected_type):
        """Test that different response item kinds are correctly parsed."""
        if kind == "inlineReference":
            result = _extract_inline_reference_name(item)
            assert result is not None
            assert isinstance(result, expected_type)
            assert "`" in result  # Should be backtick-formatted
        elif kind in ("textEditGroup", "codeblockUri", "notebookEditGroup"):
            result = _extract_edit_group_text(item)
            assert result is not None
            assert isinstance(result, expected_type)
            assert "`" in result  # Should contain backticked filename
        elif kind == "toolInvocationSerialized":
            result = _parse_tool_invocation_serialized(item)
            assert result is not None
            assert isinstance(result, expected_type)
            assert result.name == "run_command"
            assert result.status == "success"  # D1: normalised from "completed"

    def test_nested_uri_object_handling(self):
        """Test that nested URI objects (common in VS Code data) are correctly parsed."""
        # URI as dict with $mid (VS Code internal format)
        item = {"kind": "textEditGroup", "uri": {"$mid": 1, "path": "/c:/Users/test/project/src/main.py", "scheme": "file", "fsPath": "c:\\Users\\test\\project\\src\\main.py"}}
        result = _extract_edit_group_text(item)
        assert result is not None
        assert "main.py" in result

    def test_uri_string_handling(self):
        """Test that URI strings are correctly parsed."""
        item = {"kind": "textEditGroup", "uri": "file:///c:/Users/test/project/src/main.py"}
        result = _extract_edit_group_text(item)
        assert result is not None
        assert "main.py" in result

    def test_merge_content_blocks_keeps_thinking_separate(self):
        """Test that thinking blocks are not merged with text blocks."""
        blocks: list[tuple[str, str, str | None]] = [
            ("text", "Hello", None),
            ("thinking", "Let me think...", None),
            ("text", "World", None),
        ]
        result = _merge_content_blocks(blocks)
        assert len(result) == 3
        assert result[0].kind == "text"
        assert result[1].kind == "thinking"
        assert result[2].kind == "text"

    def test_merge_content_blocks_merges_consecutive_text(self):
        """Test that consecutive text blocks are merged."""
        blocks: list[tuple[str, str, str | None]] = [
            ("text", "Hello", None),
            ("text", "World", None),
            ("text", "!", None),
        ]
        result = _merge_content_blocks(blocks)
        assert len(result) == 1
        assert result[0].kind == "text"
        assert "Hello" in result[0].content
        assert "World" in result[0].content

    def test_tool_invocation_blocks_stay_separate(self):
        """Test that toolInvocation blocks are never merged."""
        blocks: list[tuple[str, str, str | None]] = [
            ("text", "Starting...", None),
            ("toolInvocation", "Running command", None),
            ("toolInvocation", "Reading file", None),
            ("text", "Done", None),
        ]
        result = _merge_content_blocks(blocks)
        assert len(result) == 4
        assert result[1].kind == "toolInvocation"
        assert result[2].kind == "toolInvocation"

    def test_question_carousel_renders_prompt_options_and_answers(self):
        response_items = [
            {
                "kind": "questionCarousel",
                "message": {"value": "Please answer these questions."},
                "questions": [
                    {
                        "id": "database",
                        "type": "singleSelect",
                        "title": "Database",
                        "message": {"value": "Which database should be used?"},
                        "options": [
                            {"id": "pg", "label": "PostgreSQL", "value": "postgres"},
                            {"id": "sqlite", "label": "SQLite", "value": "sqlite"},
                        ],
                    }
                ],
                "data": {"database": {"selectedValue": "postgres"}},
                "isUsed": True,
            }
        ]

        _, raw_blocks, _, _, _ = _process_response_items(response_items)

        assert raw_blocks == [
            (
                "ask_user",
                "Please answer these questions.\n\n1. **Database**: Which database should be used?\n   Options: PostgreSQL, SQLite\n   Answer: postgres",
                "questions",
            )
        ]

    def test_completed_elicitation_renders_question_and_outcome(self):
        response_items = [
            {
                "kind": "elicitationSerialized",
                "title": {"value": "Continue waiting?"},
                "message": {"value": "Poll for output for two more minutes."},
                "state": "rejected",
                "isHidden": True,
            }
        ]

        _, raw_blocks, _, _, _ = _process_response_items(response_items)

        assert raw_blocks == [
            (
                "ask_user",
                "**Continue waiting?**\n\nPoll for output for two more minutes.\n\nResponse: rejected",
                "elicitationSerialized",
            )
        ]

    @pytest.mark.parametrize(
        ("kind", "item", "expected_content", "expected_description"),
        [
            (
                "progressMessage",
                {"kind": "progressMessage", "content": {"value": "Analyzing answers..."}},
                "Analyzing answers...",
                "progressMessage",
            ),
            (
                "warning",
                {"kind": "warning", "content": {"value": "Chat took too long."}},
                "Chat took too long.",
                "warning",
            ),
            (
                "systemNotification",
                {"kind": "systemNotification", "content": {"value": "Workspace changed."}},
                "Workspace changed.",
                "systemNotification",
            ),
            (
                "hook",
                {"kind": "hook", "hookType": "PostToolUse", "systemMessage": "\x1b[31mHook failed\x1b[0m"},
                "PostToolUse: Hook failed",
                "hook",
            ),
            (
                "autoModeResolution",
                {"kind": "autoModeResolution", "resolved": {"id": "gpt-5.6-sol", "name": "GPT-5.6 Sol"}},
                "Auto selected GPT-5.6 Sol",
                "auto-mode",
            ),
        ],
    )
    def test_structured_status_response_kinds_render(self, kind, item, expected_content, expected_description):
        _, raw_blocks, _, _, _ = _process_response_items([item])

        assert raw_blocks == [("status", expected_content, expected_description)], kind

    def test_plan_review_renders_content_and_result(self):
        response_items = [
            {
                "kind": "planReview",
                "title": "Implementation plan",
                "content": "1. Add parser support\n2. Add tests",
                "data": {"actionId": "approve", "rejected": False},
            }
        ]

        _, raw_blocks, _, _, _ = _process_response_items(response_items)

        assert raw_blocks == [
            (
                "ask_user",
                "**Implementation plan**\n\n1. Add parser support\n2. Add tests\n\nResponse: approve",
                "planReview",
            )
        ]


class TestSampleFilesParsing:
    """Tests using real sample files to validate parsing logic."""

    @requires_sample_files
    def test_sample_session_parses_successfully(self, sample_session_data):
        """Test that sample session JSON can be parsed without errors."""
        assert sample_session_data is not None
        assert isinstance(sample_session_data, dict)

    @requires_sample_files
    def test_sample_session_has_expected_structure(self, sample_session_data):
        """Test that sample session has the expected top-level structure."""
        # Should have version field
        assert "version" in sample_session_data
        # Should have requests array (VS Code Copilot Chat format)
        assert "requests" in sample_session_data
        assert isinstance(sample_session_data["requests"], list)
        # Should have at least one request
        assert len(sample_session_data["requests"]) > 0

    @requires_sample_files
    def test_sample_session_requests_have_messages(self, sample_session_data):
        """Test that requests in sample session have message and response."""
        for request in sample_session_data["requests"]:
            # Each request should have a message with text
            assert "message" in request
            assert isinstance(request["message"], dict)
            # Each request should have a response array
            assert "response" in request
            assert isinstance(request["response"], list)

    @requires_sample_files
    def test_sample_session_scan_integration(self, sample_session_path, tmp_path):
        """Test that sample session can be scanned using the scanner module."""
        from copilot_session_tools.scanner import _parse_chat_session_file

        session = _parse_chat_session_file(sample_session_path, workspace_name="test-workspace", workspace_path=str(tmp_path), edition="stable")
        assert session is not None
        assert isinstance(session, ChatSession)
        assert len(session.messages) > 0
        # Should have both user and assistant messages
        roles = {msg.role for msg in session.messages}
        assert "user" in roles or "assistant" in roles


class TestPerformanceBenchmarks:
    """Performance tests for large session parsing."""

    @requires_sample_files
    def test_large_session_parsing_time(self, all_sample_session_paths):
        """Test that large session files parse within acceptable time limits."""
        from copilot_session_tools.scanner import _parse_chat_session_file

        for sample_path in all_sample_session_paths:
            file_size = sample_path.stat().st_size

            # Only benchmark files larger than 100KB
            if file_size < 100 * 1024:
                continue

            start_time = time.perf_counter()

            # Parse the file
            session = _parse_chat_session_file(sample_path, workspace_name="benchmark", workspace_path="/tmp/benchmark", edition="stable")

            elapsed_time = time.perf_counter() - start_time

            # Log performance metrics (useful for baseline establishment)
            file_size_mb = file_size / (1024 * 1024)
            print(f"\nParsed {sample_path.name}: {file_size_mb:.2f}MB in {elapsed_time:.3f}s")

            # Assert parsing succeeded
            assert session is not None

            # Assert reasonable time limit: 5 seconds per MB as baseline
            max_time = max(5.0, file_size_mb * 5)
            assert elapsed_time < max_time, f"Parsing took {elapsed_time:.2f}s, expected < {max_time:.2f}s"

    @requires_sample_files
    def test_json_parse_performance(self, sample_session_path):
        """Test raw ssrjson parsing performance."""
        import ssrjson

        file_size = sample_session_path.stat().st_size

        start_time = time.perf_counter()
        with open(sample_session_path, "rb") as f:
            data = ssrjson.loads(f.read())
        elapsed_time = time.perf_counter() - start_time

        file_size_mb = file_size / (1024 * 1024)
        print(f"\nssrjson parsed {sample_session_path.name}: {file_size_mb:.2f}MB in {elapsed_time:.3f}s")

        assert data is not None
        # ssrjson should be very fast - less than 1 second per MB
        max_time = max(1.0, file_size_mb * 1)
        assert elapsed_time < max_time


class TestCLIParsing:
    """Tests for GitHub Copilot CLI JSONL parsing."""

    def test_parse_cli_jsonl_event_based_format(self):
        """Test parsing real CLI JSONL session file with event-based format.

        Tests parsing the actual copilot-cli JSONL format with event types like
        session.start, user.message, assistant.message, tool.execution_*, etc.
        """
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        # Use the real sample file from copilot-cli
        sample_file = Path(__file__).parent / "snapshots" / "fixtures" / "cli-66b821d4" / "events.jsonl"

        if not sample_file.exists():
            pytest.skip("Real CLI sample file not found")

        session = _parse_cli_jsonl_file(sample_file)

        assert session is not None
        assert session.type == "cli"
        # Session ID should be a non-empty string (UUID from session.start)
        assert session.session_id and len(session.session_id) > 0
        # Should have a creation timestamp
        assert session.created_at is not None

        # Should have user and assistant messages
        assert len(session.messages) > 0
        user_messages = [m for m in session.messages if m.role == "user"]
        assert len(user_messages) >= 1

        assistant_messages = [m for m in session.messages if m.role == "assistant"]
        assert len(assistant_messages) >= 1

        # Check that tool invocations and command runs are parsed
        all_command_runs = []
        all_content_blocks = []
        for msg in assistant_messages:
            all_command_runs.extend(msg.command_runs)
            all_content_blocks.extend(msg.content_blocks)

        # Should have some command runs or content blocks
        assert len(all_command_runs) > 0 or len(all_content_blocks) > 0

    def test_parse_cli_jsonl_file_simple_format(self):
        """Test parsing CLI JSONL session file with simple format (for backwards compatibility)."""
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        # Use the simple sample file
        sample_file = Path(__file__).parent / "snapshots" / "fixtures" / "cli-simple-format.jsonl"

        if not sample_file.exists():
            pytest.skip("Simple CLI sample file not found")

        # The simple format won't parse with event-based parser, but should not crash
        session = _parse_cli_jsonl_file(sample_file)

        # Simple format doesn't have session.start or user.message events,
        # so it returns None (no messages found)
        # This is expected - the simple format was for testing only
        assert session is None

    def test_parse_cli_jsonl_captures_source_event_ids(self, tmp_path):
        """Test CLI event ids are retained for fork-aware search de-duplication."""
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = [
            {
                "type": "session.start",
                "data": {"sessionId": "source-id-test", "startTime": "2026-01-01T00:00:00Z"},
            },
            {
                "id": "evt-user-1",
                "type": "user.message",
                "timestamp": "2026-01-01T00:00:01Z",
                "data": {"content": "Hello provenance"},
            },
            {
                "id": "evt-assistant-1",
                "type": "assistant.message",
                "timestamp": "2026-01-01T00:00:02Z",
                "data": {"content": "Hi with provenance"},
            },
            {
                "id": "evt-user-2",
                "type": "user.message",
                "timestamp": "2026-01-01T00:00:03Z",
                "data": {"content": "Next turn"},
            },
        ]
        events_file = tmp_path / "events.jsonl"
        events_file.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

        session = _parse_cli_jsonl_file(events_file)

        assert session is not None
        assert [msg.source_event_id for msg in session.messages] == ["evt-user-1", "evt-assistant-1", "evt-user-2"]

    def test_get_cli_storage_paths(self):
        """Test getting CLI storage paths."""
        from copilot_session_tools import get_cli_storage_paths

        paths = get_cli_storage_paths()

        # Should return a list of Path objects
        assert isinstance(paths, list)

        # Paths should be Path objects
        for path in paths:
            assert isinstance(path, Path)

    def test_scan_includes_cli_by_default(self, tmp_path):
        """Test that scan_chat_sessions includes CLI sessions by default."""

        from copilot_session_tools import scan_chat_sessions

        # Mock an empty VS Code storage
        storage_paths = [(str(tmp_path / "nonexistent"), "stable")]

        # We can't easily test actual CLI scanning without mocking home directory,
        # but we can verify the function accepts include_cli parameter
        sessions = list(scan_chat_sessions(storage_paths, include_cli=False))

        # Should work without errors
        assert isinstance(sessions, list)


class TestCLIFileChanges:
    """Tests for reshaping CLI edit/create tool calls into FileChange diffs."""

    @staticmethod
    def _write_events(tmp_path: Path, tool_events: list[dict]) -> Path:
        """Write a minimal CLI events.jsonl with a session.start, a user message,
        and the supplied tool start/complete events, then return its path."""
        sid = "00000000-0000-4000-8000-000000000001"
        lines = [
            {"type": "session.start", "data": {"sessionId": sid, "startTime": "2026-01-01T00:00:00Z", "context": {"cwd": "C:\\proj"}}},
            {"type": "user.message", "data": {"content": "do it"}},
        ]
        lines.extend(tool_events)
        events_file = tmp_path / "events.jsonl"
        events_file.write_text("\n".join(json.dumps(line) for line in lines), encoding="utf-8")
        return events_file

    @staticmethod
    def _tool_pair(tool_call_id: str, tool_name: str, arguments: dict, *, success: bool = True) -> list[dict]:
        return [
            {"type": "tool.execution_start", "data": {"toolCallId": tool_call_id, "toolName": tool_name, "arguments": arguments}},
            {"type": "tool.execution_complete", "data": {"toolCallId": tool_call_id, "success": success, "result": {"content": "ok"}}},
        ]

    @staticmethod
    def _all_file_changes(session) -> list:
        return [fc for m in session.messages for fc in m.file_changes]

    @staticmethod
    def _view(tool_call_id: str, path: str, lines: list[str], view_range: list[int] | None = None) -> list[dict]:
        """A view tool pair whose result is numbered like the real CLI ('N. text')."""
        args = {"path": path}
        if view_range is not None:
            args["view_range"] = view_range
        numbered = "\n".join(f"{i + 1}. {ln}" for i, ln in enumerate(lines))
        return [
            {"type": "tool.execution_start", "data": {"toolCallId": tool_call_id, "toolName": "view", "arguments": args}},
            {"type": "tool.execution_complete", "data": {"toolCallId": tool_call_id, "success": True, "result": {"content": numbered}}},
        ]

    def test_subagent_view_then_edit_reconstructs(self, tmp_path):
        """A subagent's own full-file view seeds reconstruction for its edits (review #1)."""
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = [
            {"type": "assistant.message", "data": {"content": "delegating", "toolRequests": [{"toolCallId": "tc1", "name": "task", "arguments": {"agent_type": "explore"}}]}},
            {"type": "tool.execution_start", "data": {"toolCallId": "tc1", "toolName": "task", "arguments": {}}},
            {"type": "tool.execution_start", "data": {"toolCallId": "c1", "toolName": "view", "parentToolCallId": "tc1", "arguments": {"path": "/p/k.py"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "c1", "success": True, "result": {"content": "1. a = 1\n2. b = 2"}}},
            {
                "type": "tool.execution_start",
                "data": {"toolCallId": "c2", "toolName": "edit", "parentToolCallId": "tc1", "arguments": {"path": "/p/k.py", "old_str": "a = 1\n", "new_str": "a = 9\n"}},
            },
            {"type": "tool.execution_complete", "data": {"toolCallId": "c2", "success": True, "result": {"content": "ok"}}},
            {"type": "subagent.started", "data": {"toolCallId": "tc1", "agentDisplayName": "Explore"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc1", "agentDisplayName": "Explore"}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc1", "success": True, "result": {"content": "done"}}},
        ]
        session = _parse_cli_jsonl_file(self._write_events(tmp_path, events))
        assert session is not None
        nested = [fc for m in session.messages for c in m.children for fc in c.file_changes]
        assert len(nested) == 1
        diff = nested[0].diff
        assert diff is not None
        assert "-a = 1" in diff and "+a = 9" in diff
        assert "b = 2" in diff  # context from the subagent's own view base

    def test_create_produces_all_additions_diff(self, tmp_path):
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = self._tool_pair("t1", "create", {"path": "C:\\proj\\new.py", "file_text": "a = 1\nb = 2\n"})
        session = _parse_cli_jsonl_file(self._write_events(tmp_path, events))

        changes = self._all_file_changes(session)
        assert len(changes) == 1
        fc = changes[0]
        assert fc.path == "C:\\proj\\new.py"
        assert fc.language_id == "python"
        assert fc.content == "a = 1\nb = 2\n"
        assert fc.diff is not None
        assert "+a = 1" in fc.diff
        assert "+b = 2" in fc.diff
        # All-additions: no deletion lines (other than the --- header)
        assert not any(line.startswith("-") and not line.startswith("---") for line in fc.diff.splitlines())

    def test_edit_produces_before_after_diff(self, tmp_path):
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = self._tool_pair("t1", "edit", {"path": "C:\\proj\\mod.ts", "old_str": "let x = 1;\n", "new_str": "let x = 2;\n"})
        session = _parse_cli_jsonl_file(self._write_events(tmp_path, events))

        changes = self._all_file_changes(session)
        assert len(changes) == 1
        fc = changes[0]
        assert fc.language_id == "typescript"
        assert fc.content is None
        assert fc.diff is not None
        assert "-let x = 1;" in fc.diff
        assert "+let x = 2;" in fc.diff

    def test_failed_edit_is_gated_out(self, tmp_path):
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = self._tool_pair("t1", "edit", {"path": "C:\\proj\\mod.py", "old_str": "x\n", "new_str": "y\n"}, success=False)
        session = _parse_cli_jsonl_file(self._write_events(tmp_path, events))

        assert self._all_file_changes(session) == []

    def test_unknown_extension_has_no_language_id(self, tmp_path):
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = self._tool_pair("t1", "create", {"path": "C:\\proj\\notes.xyz", "file_text": "hello\n"})
        session = _parse_cli_jsonl_file(self._write_events(tmp_path, events))

        changes = self._all_file_changes(session)
        assert len(changes) == 1
        assert changes[0].language_id is None

    def test_edit_without_strings_emits_no_file_change(self, tmp_path):
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = self._tool_pair("t1", "edit", {"path": "C:\\proj\\mod.py"})
        session = _parse_cli_jsonl_file(self._write_events(tmp_path, events))

        assert self._all_file_changes(session) == []

    def test_create_then_edits_consolidate_to_one_card(self, tmp_path):
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = []
        events += self._tool_pair("t1", "create", {"path": "C:\\proj\\u.py", "file_text": "a = 1\n"})
        events += self._tool_pair("t2", "edit", {"path": "C:\\proj\\u.py", "old_str": "a = 1\n", "new_str": "a = 1\nb = 2\n"})
        events += self._tool_pair("t3", "edit", {"path": "C:\\proj\\u.py", "old_str": "b = 2\n", "new_str": "b = 2\nc = 3\n"})
        session = _parse_cli_jsonl_file(self._write_events(tmp_path, events))

        changes = self._all_file_changes(session)
        assert len(changes) == 1  # 3 ops to same file → single consolidated card
        fc = changes[0]
        assert fc.diff is not None
        # Net diff: created in-turn, all final lines are additions
        assert "+a = 1" in fc.diff and "+b = 2" in fc.diff and "+c = 3" in fc.diff

    def test_distinct_files_stay_separate(self, tmp_path):
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = []
        events += self._tool_pair("t1", "create", {"path": "C:\\proj\\a.py", "file_text": "x\n"})
        events += self._tool_pair("t2", "create", {"path": "C:\\proj\\b.py", "file_text": "y\n"})
        session = _parse_cli_jsonl_file(self._write_events(tmp_path, events))

        changes = self._all_file_changes(session)
        assert len(changes) == 2
        assert {c.path for c in changes} == {"C:\\proj\\a.py", "C:\\proj\\b.py"}

    def test_edit_only_same_file_stacks_into_one_card(self, tmp_path):
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = []
        events += self._tool_pair("t1", "edit", {"path": "C:\\proj\\m.py", "old_str": "x = 1\n", "new_str": "x = 2\n"})
        events += self._tool_pair("t2", "edit", {"path": "C:\\proj\\m.py", "old_str": "y = 1\n", "new_str": "y = 2\n"})
        session = _parse_cli_jsonl_file(self._write_events(tmp_path, events))

        changes = self._all_file_changes(session)
        assert len(changes) == 1  # not reconstructable (no create) → stacked hunks, one card
        assert changes[0].diff.count("@@") >= 2

    def test_edit_only_with_prior_full_view_reconstructs_net_diff(self, tmp_path):
        """A full-file view earlier in the session lets edit-only sequences form ONE net diff."""
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = []
        events += self._view("v1", "C:\\proj\\app.py", ["x = 1", "y = 1", "z = 1"])  # full read (no range)
        events += self._tool_pair("t1", "edit", {"path": "C:\\proj\\app.py", "old_str": "x = 1\n", "new_str": "x = 2\n"})
        events += self._tool_pair("t2", "edit", {"path": "C:\\proj\\app.py", "old_str": "y = 1\n", "new_str": "y = 9\n"})
        session = _parse_cli_jsonl_file(self._write_events(tmp_path, events))

        changes = self._all_file_changes(session)
        assert len(changes) == 1
        fc = changes[0]
        # Single coherent net diff against the viewed base (line-number prefixes stripped)
        assert "-x = 1" in fc.diff and "+x = 2" in fc.diff
        assert "-y = 1" in fc.diff and "+y = 9" in fc.diff
        assert "z = 1" in fc.diff  # trailing unchanged context preserved → real base was used

    def test_edit_only_partial_view_falls_back_to_stacked(self, tmp_path):
        """A range-limited view is not trusted as a base; stays stacked hunks."""
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = []
        events += self._view("v1", "C:\\proj\\m.py", ["x = 1", "y = 1"], view_range=[1, 2])
        events += self._tool_pair("t1", "edit", {"path": "C:\\proj\\m.py", "old_str": "x = 1\n", "new_str": "x = 2\n"})
        events += self._tool_pair("t2", "edit", {"path": "C:\\proj\\m.py", "old_str": "q = 1\n", "new_str": "q = 2\n"})
        session = _parse_cli_jsonl_file(self._write_events(tmp_path, events))

        changes = self._all_file_changes(session)
        assert len(changes) == 1
        assert changes[0].diff.count("@@") >= 2  # not reconstructed → stacked

    def test_edit_only_view_base_mismatch_falls_back(self, tmp_path):
        """If any old_str doesn't match the viewed base (stale file), fall back safely."""
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events = []
        events += self._view("v1", "C:\\proj\\m.py", ["x = 1", "y = 1"])
        events += self._tool_pair("t1", "edit", {"path": "C:\\proj\\m.py", "old_str": "x = 1\n", "new_str": "x = 2\n"})
        events += self._tool_pair("t2", "edit", {"path": "C:\\proj\\m.py", "old_str": "NOT_PRESENT\n", "new_str": "z\n"})
        session = _parse_cli_jsonl_file(self._write_events(tmp_path, events))

        changes = self._all_file_changes(session)
        assert len(changes) == 1
        assert changes[0].diff.count("@@") >= 2  # mismatch → stacked, never a wrong base


class TestConsolidateCliFileOps:
    """Unit tests for the pure consolidation helper, incl. backfill of existing cases."""

    def _ops(self):
        from copilot_session_tools.scanner.diff import CliFileOp

        return CliFileOp

    @staticmethod
    def _diff(fc) -> str:
        assert fc.diff is not None
        return fc.diff

    def test_single_create_is_all_additions(self):
        from copilot_session_tools.scanner.diff import consolidate_cli_file_ops

        op = self._ops()("create", "/a.py", file_text="a\nb\n")
        out = consolidate_cli_file_ops([op])
        assert len(out) == 1 and out[0].content == "a\nb\n" and "+a" in self._diff(out[0])

    def test_create_led_edits_net_diff(self):
        from copilot_session_tools.scanner.diff import consolidate_cli_file_ops

        C = self._ops()
        ops = [C("create", "/a.py", file_text="a\n"), C("edit", "/a.py", old_str="a\n", new_str="a\nb\n")]
        out = consolidate_cli_file_ops(ops)
        assert len(out) == 1 and "+a" in self._diff(out[0]) and "+b" in self._diff(out[0])

    def test_edit_only_no_base_stacks(self):
        from copilot_session_tools.scanner.diff import consolidate_cli_file_ops

        C = self._ops()
        ops = [C("edit", "/a.py", old_str="x\n", new_str="y\n"), C("edit", "/a.py", old_str="p\n", new_str="q\n")]
        out = consolidate_cli_file_ops(ops)
        assert len(out) == 1 and self._diff(out[0]).count("@@") >= 2

    def test_edit_only_with_base_reconstructs(self):
        from copilot_session_tools.scanner.diff import consolidate_cli_file_ops

        C = self._ops()
        ops = [C("edit", "/a.py", old_str="x\n", new_str="y\n")]
        out = consolidate_cli_file_ops(ops, bases={"/a.py": "x\nkeep\n"})
        assert " keep" in self._diff(out[0])

    def test_base_mismatch_falls_back(self):
        from copilot_session_tools.scanner.diff import consolidate_cli_file_ops

        C = self._ops()
        ops = [C("edit", "/a.py", old_str="zzz\n", new_str="y\n")]
        out = consolidate_cli_file_ops(ops, bases={"/a.py": "x\nkeep\n"})
        assert self._diff(out[0]).count("@@") >= 1 and " keep" not in self._diff(out[0])

    def test_fallback_preserves_create_when_edit_mismatches(self):
        """Mixed create+failed-edit must not drop the create's content (review #3)."""
        from copilot_session_tools.scanner.diff import consolidate_cli_file_ops

        C = self._ops()
        ops = [C("create", "/a.py", file_text="line1\nline2\n"), C("edit", "/a.py", old_str="NOPE\n", new_str="z\n")]
        out = consolidate_cli_file_ops(ops)
        assert len(out) == 1
        assert "+line1" in self._diff(out[0]) and "+line2" in self._diff(out[0])  # creation not lost

    def test_view_base_updates_after_consolidation(self):
        """Reconstructed content is written back so later turns aren't stale (review #2)."""
        from copilot_session_tools.scanner.diff import consolidate_cli_file_ops

        C = self._ops()
        bases = {"/a.py": "x = 1\nkeep\n"}
        consolidate_cli_file_ops([C("edit", "/a.py", old_str="x = 1\n", new_str="x = 2\n")], bases)
        assert bases["/a.py"] == "x = 2\nkeep\n"  # base advanced to post-edit content

    def test_strip_view_line_numbers_keeps_empty_lines_and_trailing_newline(self):
        """Prefix stripping must preserve blank lines and trailing newline (review #4)."""
        from copilot_session_tools.scanner.diff import strip_view_line_numbers

        # line 2 is blank; original ends with newline
        assert strip_view_line_numbers("1. a\n2.\n3. b\n") == "a\n\nb\n"


class TestWorkspaceYamlParsing:
    """Tests for workspace.yaml parsing and CLI session title extraction."""

    def test_parse_workspace_yaml_with_summary(self, tmp_path):
        """Test parsing workspace.yaml extracts summary field."""
        from copilot_session_tools.scanner import _parse_workspace_yaml

        workspace_file = tmp_path / "workspace.yaml"
        workspace_file.write_text(
            "id: 00b8e3a3-f89d-4105-b0e4-a8ab94986035\n"
            "cwd: C:\\_SRC\\ZTS\n"
            "git_root: C:\\_SRC\\ZTS\n"
            "branch: main\n"
            "summary: Remediate AzSecpack On VMSS\n"
            "summary_count: 0\n"
            "created_at: 2026-02-09T09:28:30.798Z\n"
            "updated_at: 2026-02-11T10:13:41.793Z\n",
            encoding="utf-8",
        )

        result = _parse_workspace_yaml(tmp_path)
        assert result["summary"] == "Remediate AzSecpack On VMSS"
        assert result["id"] == "00b8e3a3-f89d-4105-b0e4-a8ab94986035"
        assert result["branch"] == "main"

    def test_parse_workspace_yaml_missing_file(self, tmp_path):
        """Test that missing workspace.yaml returns empty dict."""
        from copilot_session_tools.scanner import _parse_workspace_yaml

        result = _parse_workspace_yaml(tmp_path)
        assert result == {}

    def test_parse_workspace_yaml_no_summary(self, tmp_path):
        """Test parsing workspace.yaml without summary field."""
        from copilot_session_tools.scanner import _parse_workspace_yaml

        workspace_file = tmp_path / "workspace.yaml"
        workspace_file.write_text(
            "id: abc123\ncwd: /home/user/project\n",
            encoding="utf-8",
        )

        result = _parse_workspace_yaml(tmp_path)
        assert "summary" not in result
        assert result["id"] == "abc123"

    def test_parse_workspace_yaml_empty_summary(self, tmp_path):
        """Test parsing workspace.yaml with empty summary field."""
        from copilot_session_tools.scanner import _parse_workspace_yaml

        workspace_file = tmp_path / "workspace.yaml"
        workspace_file.write_text(
            "id: abc123\nsummary:\n",
            encoding="utf-8",
        )

        result = _parse_workspace_yaml(tmp_path)
        assert result["summary"] == ""

    def _make_cli_session_events(self, intent=None):
        """Helper to create minimal CLI JSONL events for title tests."""
        ctx = {"cwd": "/home/user/project"}
        start_data = {"sessionId": "test-id", "startTime": "2026-01-01T00:00:00Z", "context": ctx}
        assistant_data: dict = {"content": "Sure."}
        events: list[dict] = [
            {"type": "session.start", "timestamp": "2026-01-01T00:00:00Z", "data": start_data},
            {"type": "user.message", "timestamp": "2026-01-01T00:00:01Z", "data": {"content": "Help"}},
            {"type": "assistant.message", "timestamp": "2026-01-01T00:00:02Z", "data": assistant_data},
        ]
        if intent:
            intent_args = {"intent": intent}
            assistant_data["toolRequests"] = [{"toolCallId": "tc1", "toolName": "report_intent", "arguments": intent_args}]
            events.append(
                {
                    "type": "tool.execution_start",
                    "timestamp": "2026-01-01T00:00:02Z",
                    "data": {"toolCallId": "tc1", "toolName": "report_intent", "arguments": intent_args},
                }
            )
            events.append(
                {
                    "type": "tool.execution_complete",
                    "timestamp": "2026-01-01T00:00:03Z",
                    "data": {"toolCallId": "tc1", "toolName": "report_intent", "result": ""},
                }
            )
        return events

    def test_cli_session_title_from_workspace_yaml(self, tmp_path):
        """Test that CLI session title is extracted from workspace.yaml summary."""
        import ssrjson

        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        session_dir = tmp_path / "test-session"
        session_dir.mkdir()

        (session_dir / "workspace.yaml").write_text(
            "id: test-id\ncwd: /home/user/project\nsummary: Diagnose ADO Build Failures\n",
            encoding="utf-8",
        )

        events_file = session_dir / "events.jsonl"
        events_file.write_text(
            "\n".join(ssrjson.dumps(e) for e in self._make_cli_session_events()),
            encoding="utf-8",
        )

        session = _parse_cli_jsonl_file(events_file)
        assert session is not None
        assert session.custom_title == "Diagnose ADO Build Failures"

    def test_cli_session_title_fallback_to_intent(self, tmp_path):
        """Test that CLI session title falls back to first report_intent when no workspace.yaml."""
        import ssrjson

        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events_file = tmp_path / "test-session.jsonl"
        events_file.write_text(
            "\n".join(ssrjson.dumps(e) for e in self._make_cli_session_events(intent="Fix failing unit tests")),
            encoding="utf-8",
        )

        session = _parse_cli_jsonl_file(events_file)
        assert session is not None
        assert session.custom_title == "Fix failing unit tests"

    def test_cli_session_title_workspace_yaml_over_intent(self, tmp_path):
        """Test that workspace.yaml summary takes priority over report_intent."""
        import ssrjson

        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        session_dir = tmp_path / "test-session"
        session_dir.mkdir()

        (session_dir / "workspace.yaml").write_text(
            "id: test-id\nsummary: YAML Title Wins\n",
            encoding="utf-8",
        )

        events_file = session_dir / "events.jsonl"
        events_file.write_text(
            "\n".join(ssrjson.dumps(e) for e in self._make_cli_session_events(intent="Intent Title Loses")),
            encoding="utf-8",
        )

        session = _parse_cli_jsonl_file(events_file)
        assert session is not None
        assert session.custom_title == "YAML Title Wins"

    def test_cli_session_title_none_when_no_sources(self, tmp_path):
        """Test that custom_title is None when neither workspace.yaml nor intent exists."""
        import ssrjson

        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        events_file = tmp_path / "test-session.jsonl"
        events_file.write_text(
            "\n".join(ssrjson.dumps(e) for e in self._make_cli_session_events()),
            encoding="utf-8",
        )

        session = _parse_cli_jsonl_file(events_file)
        assert session is not None
        assert session.custom_title is None


class TestAskUserAnswerDisplay:
    """Tests for ask_user tool answer display in parsed sessions."""

    def _make_ask_user_session_events(self, tool_call_id, question, choices, complete_event=None):
        """Create minimal CLI JSONL events with an ask_user tool invocation."""
        events = [
            {"type": "session.start", "data": {"sessionId": "ask-user-test", "timestamp": "2026-01-15T10:00:00Z"}},
            {"type": "user.message", "data": {"content": "Help me pick"}},
            {
                "type": "assistant.message.delta",
                "data": {
                    "toolRequests": [
                        {"toolCallId": tool_call_id, "name": "ask_user", "arguments": {"question": question, "choices": choices}},
                    ]
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": tool_call_id, "toolName": "ask_user", "arguments": {"question": question, "choices": choices}}},
        ]
        if complete_event is not None:
            events.append(complete_event)
        events.append({"type": "assistant.message.delta", "data": {"content": "Great choice!"}})
        return events

    def _parse_events(self, events, tmp_path):
        """Write events to a JSONL file and parse them."""
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        session_file = tmp_path / "ask-user-test.jsonl"
        session_file.write_text("\n".join(json.dumps(e) for e in events))
        return _parse_cli_jsonl_file(session_file)

    def _find_ask_user_block(self, session):
        """Find the ask_user content block in a parsed session."""
        for msg in session.messages:
            for block in msg.content_blocks:
                if block.kind == "ask_user":
                    return block
        return None

    def test_ask_user_with_successful_answer(self, tmp_path):
        """Test that a successful ask_user answer is displayed."""
        events = self._make_ask_user_session_events(
            tool_call_id="toolu_ask1",
            question="Which framework?",
            choices=["React", "Vue", "Angular"],
            complete_event={
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": "toolu_ask1",
                    "success": True,
                    "result": {"content": "User responded: React"},
                },
            },
        )
        session = self._parse_events(events, tmp_path)
        assert session is not None
        block = self._find_ask_user_block(session)
        assert block is not None
        assert "❓ Which framework?" in block.content
        assert "Options: React, Vue, Angular" in block.content
        assert "✅ **Answer:** React" in block.content

    def test_ask_user_with_failed_answer(self, tmp_path):
        """Test that a failed/skipped ask_user shows skipped indicator."""
        events = self._make_ask_user_session_events(
            tool_call_id="toolu_ask2",
            question="Pick a color",
            choices=["Red", "Blue"],
            complete_event={
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": "toolu_ask2",
                    "success": False,
                    "result": {"content": ""},
                },
            },
        )
        session = self._parse_events(events, tmp_path)
        assert session is not None
        block = self._find_ask_user_block(session)
        assert block is not None
        assert "❓ Pick a color" in block.content
        assert "⏭️ *Skipped*" in block.content
        assert "Answer" not in block.content

    def test_ask_user_without_completion_event(self, tmp_path):
        """Test ask_user with no completion event shows question only."""
        events = self._make_ask_user_session_events(
            tool_call_id="toolu_ask3",
            question="Choose a language",
            choices=["Python", "Go"],
            complete_event=None,
        )
        session = self._parse_events(events, tmp_path)
        assert session is not None
        block = self._find_ask_user_block(session)
        assert block is not None
        assert "❓ Choose a language" in block.content
        assert "Answer" not in block.content
        assert "Skipped" not in block.content

    def test_ask_user_answer_strips_prefix(self, tmp_path):
        """Test that 'User responded: ' prefix is stripped from the answer."""
        events = self._make_ask_user_session_events(
            tool_call_id="toolu_ask4",
            question="Which option?",
            choices=[],
            complete_event={
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": "toolu_ask4",
                    "success": True,
                    "result": {"content": "User responded: Option B"},
                },
            },
        )
        session = self._parse_events(events, tmp_path)
        assert session is not None
        block = self._find_ask_user_block(session)
        assert block is not None
        assert "✅ **Answer:** Option B" in block.content
        assert "User responded:" not in block.content

    def test_ask_user_answer_without_prefix(self, tmp_path):
        """Test answer that doesn't have 'User responded: ' prefix is shown as-is."""
        events = self._make_ask_user_session_events(
            tool_call_id="toolu_ask5",
            question="Pick one",
            choices=["A", "B"],
            complete_event={
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": "toolu_ask5",
                    "success": True,
                    "result": {"content": "B"},
                },
            },
        )
        session = self._parse_events(events, tmp_path)
        assert session is not None
        block = self._find_ask_user_block(session)
        assert block is not None
        assert "✅ **Answer:** B" in block.content


class TestRepositoryUrlDetection:
    """Tests for git repository URL detection and normalization."""

    def test_normalize_git_url_https(self):
        """Test normalizing HTTPS git URLs."""
        from copilot_session_tools.scanner import _normalize_git_url

        # Standard HTTPS URL
        result = _normalize_git_url("https://github.com/owner/repo.git")
        assert result == "github.com/owner/repo"

        # Without .git suffix
        result = _normalize_git_url("https://github.com/owner/repo")
        assert result == "github.com/owner/repo"

        # GitLab URL
        result = _normalize_git_url("https://gitlab.com/group/project.git")
        assert result == "gitlab.com/group/project"

    def test_normalize_git_url_ssh(self):
        """Test normalizing SSH git URLs."""
        from copilot_session_tools.scanner import _normalize_git_url

        # Standard SSH URL
        result = _normalize_git_url("git@github.com:owner/repo.git")
        assert result == "github.com/owner/repo"

        # Without .git suffix
        result = _normalize_git_url("git@github.com:owner/repo")
        assert result == "github.com/owner/repo"

        # GitLab SSH URL
        result = _normalize_git_url("git@gitlab.com:group/project.git")
        assert result == "gitlab.com/group/project"

    def test_normalize_git_url_ssh_protocol(self):
        """Test normalizing SSH protocol URLs."""
        from copilot_session_tools.scanner import _normalize_git_url

        # SSH protocol URL
        result = _normalize_git_url("ssh://git@github.com/owner/repo.git")
        assert result == "github.com/owner/repo"

        # Without username
        result = _normalize_git_url("ssh://github.com/owner/repo.git")
        assert result == "github.com/owner/repo"

    def test_normalize_git_url_trailing_slash(self):
        """Test that trailing slashes are handled."""
        from copilot_session_tools.scanner import _normalize_git_url

        result = _normalize_git_url("https://github.com/owner/repo/")
        assert result == "github.com/owner/repo"

    def test_normalize_git_url_unknown_format(self):
        """Test that unknown formats are returned as-is."""
        from copilot_session_tools.scanner import _normalize_git_url

        result = _normalize_git_url("some-unknown-format")
        assert result == "some-unknown-format"

    def test_detect_repository_url_none_workspace(self):
        """Test that None workspace path returns None."""
        from copilot_session_tools.scanner import detect_repository_url

        result = detect_repository_url(None)
        assert result is None

    def test_detect_repository_url_empty_workspace(self):
        """Test that empty workspace path returns None."""
        from copilot_session_tools.scanner import detect_repository_url

        result = detect_repository_url("")
        assert result is None

    def test_detect_repository_url_not_git_repo(self, tmp_path):
        """Test that non-git directory returns None."""
        from copilot_session_tools.scanner import detect_repository_url

        # Create a regular directory that's not a git repo
        workspace = tmp_path / "not-a-repo"
        workspace.mkdir()

        result = detect_repository_url(str(workspace))
        assert result is None

    def test_detect_repository_url_with_git_repo(self, tmp_path):
        """Test detection in an actual git repository."""
        import subprocess

        from copilot_session_tools.scanner import detect_repository_url

        # Create a git repo
        workspace = tmp_path / "test-repo"
        workspace.mkdir()

        # Initialize git repo
        subprocess.run(["git", "init"], cwd=workspace, capture_output=True, check=True)  # noqa: S607

        # Without a remote, should return None
        result = detect_repository_url(str(workspace))
        assert result is None

        # Add a remote
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/test-owner/test-repo.git"],  # noqa: S607
            cwd=workspace,
            capture_output=True,
            check=True,
        )

        # Clear cache so we re-check after adding remote
        from copilot_session_tools.scanner import _clear_repository_url_cache

        _clear_repository_url_cache()

        # Now should return the normalized URL
        result = detect_repository_url(str(workspace))
        assert result == "github.com/test-owner/test-repo"

    def test_chat_session_has_repository_url_field(self):
        """Test that ChatSession dataclass has repository_url field."""
        session = ChatSession(
            session_id="test-session",
            workspace_name="test-workspace",
            workspace_path="/path/to/workspace",
            messages=[],
            repository_url="github.com/owner/repo",
        )

        assert session.repository_url == "github.com/owner/repo"

    def test_chat_session_repository_url_defaults_to_none(self):
        """Test that ChatSession.repository_url defaults to None."""
        session = ChatSession(
            session_id="test-session",
            workspace_name="test-workspace",
            workspace_path="/path/to/workspace",
            messages=[],
        )

        assert session.repository_url is None

    def test_detect_repository_url_exported_from_common(self):
        """Test that detect_repository_url is exported from the common package."""
        from copilot_session_tools import detect_repository_url

        # Should be callable
        assert callable(detect_repository_url)


class TestVSCodeJSONLParsing:
    """Tests for VS Code JSONL append-log format parsing."""

    def test_parse_vscode_jsonl_kind0_only(self, tmp_path):
        """Test parsing JSONL with only a kind=0 snapshot (no incremental ops)."""
        from copilot_session_tools.scanner import _parse_vscode_jsonl_file

        session_data = {
            "kind": 0,
            "v": {
                "version": 3,
                "sessionId": "abc-123",
                "creationDate": "2026-02-01T10:00:00.000Z",
                "customTitle": "Test Session",
                "requests": [
                    {
                        "message": {"text": "What is Python?"},
                        "timestamp": 1738400000000,
                        "response": [{"kind": "markdownContent", "value": {"value": "Python is a language."}}],
                    }
                ],
            },
        }
        jsonl_file = tmp_path / "abc-123.jsonl"
        jsonl_file.write_bytes(json.dumps(session_data).encode("utf-8"))

        session = _parse_vscode_jsonl_file(jsonl_file, "test-workspace", "/home/user/project", "insider")

        assert session is not None
        assert session.session_id == "abc-123"
        assert session.vscode_edition == "insider"
        assert session.workspace_name == "test-workspace"
        assert len(session.messages) == 2
        assert session.messages[0].role == "user"
        assert "Python" in session.messages[0].content
        assert session.messages[1].role == "assistant"
        assert "language" in session.messages[1].content

    def test_parse_vscode_jsonl_with_kind2_push(self, tmp_path):
        """Test parsing JSONL with kind=0 snapshot + kind=2 push (new request appended)."""
        from copilot_session_tools.scanner import _parse_vscode_jsonl_file

        # kind=0: initial snapshot with one request
        line0 = json.dumps(
            {
                "kind": 0,
                "v": {
                    "version": 3,
                    "sessionId": "def-456",
                    "creationDate": "2026-02-01T10:00:00.000Z",
                    "requests": [
                        {
                            "message": {"text": "First question"},
                            "timestamp": 1738400000000,
                            "response": [{"kind": "markdownContent", "value": {"value": "First answer"}}],
                        }
                    ],
                },
            }
        )
        # kind=2: push a new request to the requests array
        line1 = json.dumps(
            {
                "kind": 2,
                "k": ["requests"],
                "v": [
                    {
                        "message": {"text": "Second question"},
                        "timestamp": 1738400060000,
                        "response": [{"kind": "markdownContent", "value": {"value": "Second answer"}}],
                    }
                ],
            }
        )

        jsonl_file = tmp_path / "def-456.jsonl"
        jsonl_file.write_text(line0 + "\n" + line1 + "\n")

        session = _parse_vscode_jsonl_file(jsonl_file, "ws", "/path", "insider")

        assert session is not None
        assert session.session_id == "def-456"
        # Should have 4 messages: 2 user + 2 assistant
        assert len(session.messages) == 4
        user_msgs = [m for m in session.messages if m.role == "user"]
        assert len(user_msgs) == 2
        assert "First question" in user_msgs[0].content
        assert "Second question" in user_msgs[1].content

    def test_parse_vscode_jsonl_with_kind1_set(self, tmp_path):
        """Test parsing JSONL with kind=0 snapshot + kind=1 set (update property)."""
        from copilot_session_tools.scanner import _parse_vscode_jsonl_file

        line0 = json.dumps(
            {
                "kind": 0,
                "v": {
                    "version": 3,
                    "sessionId": "ghi-789",
                    "creationDate": "2026-02-01T10:00:00.000Z",
                    "customTitle": "Original Title",
                    "requests": [
                        {
                            "message": {"text": "Hello"},
                            "timestamp": 1738400000000,
                            "response": [{"kind": "markdownContent", "value": {"value": "Hi!"}}],
                        }
                    ],
                },
            }
        )
        # kind=1: update the custom title
        line1 = json.dumps(
            {
                "kind": 1,
                "k": ["customTitle"],
                "v": "Updated Title",
            }
        )

        jsonl_file = tmp_path / "ghi-789.jsonl"
        jsonl_file.write_text(line0 + "\n" + line1 + "\n")

        session = _parse_vscode_jsonl_file(jsonl_file, None, None, "stable")

        assert session is not None
        assert session.custom_title == "Updated Title"

    def test_parse_vscode_jsonl_empty_file(self, tmp_path):
        """Test parsing an empty JSONL file returns None."""
        from copilot_session_tools.scanner import _parse_vscode_jsonl_file

        jsonl_file = tmp_path / "empty.jsonl"
        jsonl_file.write_text("")

        session = _parse_vscode_jsonl_file(jsonl_file, None, None, "insider")
        assert session is None

    def test_parse_vscode_jsonl_no_kind0(self, tmp_path):
        """Test parsing JSONL without kind=0 snapshot returns None."""
        from copilot_session_tools.scanner import _parse_vscode_jsonl_file

        line = json.dumps({"kind": 1, "k": ["customTitle"], "v": "No Snapshot"})
        jsonl_file = tmp_path / "no-snapshot.jsonl"
        jsonl_file.write_text(line + "\n")

        session = _parse_vscode_jsonl_file(jsonl_file, None, None, "insider")
        assert session is None

    def test_parse_vscode_jsonl_malformed_lines(self, tmp_path):
        """Test that malformed JSONL lines are skipped gracefully."""
        from copilot_session_tools.scanner import _parse_vscode_jsonl_file

        line0 = json.dumps(
            {
                "kind": 0,
                "v": {
                    "version": 3,
                    "sessionId": "mal-001",
                    "creationDate": "2026-02-01T10:00:00.000Z",
                    "requests": [
                        {
                            "message": {"text": "Valid request"},
                            "timestamp": 1738400000000,
                            "response": [{"kind": "markdownContent", "value": {"value": "Valid response"}}],
                        }
                    ],
                },
            }
        )
        jsonl_file = tmp_path / "mal-001.jsonl"
        jsonl_file.write_text(line0 + "\n" + "not valid json\n" + "{broken\n")

        session = _parse_vscode_jsonl_file(jsonl_file, None, None, "insider")
        assert session is not None
        assert session.session_id == "mal-001"

    def test_apply_jsonl_operations_set_nested(self):
        """Test _apply_jsonl_operations with nested path for kind=1 set."""
        from copilot_session_tools.scanner import _apply_jsonl_operations

        base = {"requests": [{"message": {"text": "old"}, "response": []}]}
        ops = [{"kind": 1, "k": ["requests", 0, "message", "text"], "v": "new"}]

        result = _apply_jsonl_operations(base, ops)
        assert result["requests"][0]["message"]["text"] == "new"

    def test_apply_jsonl_operations_push(self):
        """Test _apply_jsonl_operations with kind=2 push to array."""
        from copilot_session_tools.scanner import _apply_jsonl_operations

        base = {"requests": [{"message": {"text": "first"}}]}
        ops = [{"kind": 2, "k": ["requests"], "v": [{"message": {"text": "second"}}]}]

        result = _apply_jsonl_operations(base, ops)
        assert len(result["requests"]) == 2
        assert result["requests"][1]["message"]["text"] == "second"

    def test_apply_jsonl_operations_invalid_path(self):
        """Test _apply_jsonl_operations gracefully handles invalid paths."""
        from copilot_session_tools.scanner import _apply_jsonl_operations

        base = {"requests": []}
        ops = [{"kind": 1, "k": ["nonexistent", "path"], "v": "value"}]

        result = _apply_jsonl_operations(base, ops)
        # Should not crash, just skip the operation
        assert result == {"requests": []}

    def test_scan_chat_sessions_includes_jsonl(self, tmp_path):
        """Test that scan_chat_sessions picks up .jsonl files in VS Code chatSessions dirs."""
        # Create workspace directory
        workspace_dir = tmp_path / "workspace123"
        workspace_dir.mkdir()
        workspace_json = workspace_dir / "workspace.json"
        workspace_json.write_text(json.dumps({"folder": "file:///home/user/project"}))

        chat_dir = workspace_dir / "chatSessions"
        chat_dir.mkdir()

        # Create a VS Code JSONL session file
        session_data = json.dumps(
            {
                "kind": 0,
                "v": {
                    "version": 3,
                    "sessionId": "jsonl-session-001",
                    "creationDate": "2026-02-01T10:00:00.000Z",
                    "requests": [
                        {
                            "message": {"text": "JSONL test question"},
                            "timestamp": 1738400000000,
                            "response": [{"kind": "markdownContent", "value": {"value": "JSONL test answer"}}],
                        }
                    ],
                },
            }
        )
        jsonl_file = chat_dir / "jsonl-session-001.jsonl"
        jsonl_file.write_text(session_data + "\n")

        storage_paths = [(str(tmp_path), "insider")]
        sessions = list(scan_chat_sessions(storage_paths, include_cli=False))

        assert len(sessions) >= 1
        jsonl_sessions = [s for s in sessions if s.session_id == "jsonl-session-001"]
        assert len(jsonl_sessions) == 1
        assert jsonl_sessions[0].vscode_edition == "insider"
        assert len(jsonl_sessions[0].messages) == 2

    def test_scan_session_files_includes_jsonl(self, tmp_path):
        """Test that scan_session_files yields SessionFileInfo for .jsonl files."""
        from copilot_session_tools.scanner import scan_session_files

        workspace_dir = tmp_path / "workspace456"
        workspace_dir.mkdir()
        workspace_json = workspace_dir / "workspace.json"
        workspace_json.write_text(json.dumps({"folder": "file:///home/user/project2"}))

        chat_dir = workspace_dir / "chatSessions"
        chat_dir.mkdir()

        jsonl_file = chat_dir / "test-session.jsonl"
        jsonl_file.write_text('{"kind":0,"v":{"sessionId":"test"}}\n')

        storage_paths = [(str(tmp_path), "insider")]
        file_infos = list(scan_session_files(storage_paths, include_cli=False))

        jsonl_infos = [fi for fi in file_infos if fi.file_type == "jsonl"]
        assert len(jsonl_infos) >= 1
        assert jsonl_infos[0].session_type == "vscode"
        assert jsonl_infos[0].vscode_edition == "insider"

    def test_parse_session_file_vscode_jsonl(self, tmp_path):
        """Test that parse_session_file dispatches vscode jsonl to the correct parser."""
        from copilot_session_tools.scanner import SessionFileInfo, parse_session_file

        jsonl_file = tmp_path / "dispatch-test.jsonl"
        session_data = json.dumps(
            {
                "kind": 0,
                "v": {
                    "version": 3,
                    "sessionId": "dispatch-test-001",
                    "creationDate": "2026-02-01T10:00:00.000Z",
                    "requests": [
                        {
                            "message": {"text": "Dispatch test"},
                            "timestamp": 1738400000000,
                            "response": [{"kind": "markdownContent", "value": {"value": "Dispatched!"}}],
                        }
                    ],
                },
            }
        )
        jsonl_file.write_text(session_data + "\n")

        file_info = SessionFileInfo(
            file_path=jsonl_file,
            file_type="jsonl",
            session_type="vscode",
            vscode_edition="insider",
            mtime=jsonl_file.stat().st_mtime,
            size=jsonl_file.stat().st_size,
            workspace_name="test-ws",
            workspace_path="/test/path",
        )

        sessions = parse_session_file(file_info)
        assert len(sessions) == 1
        assert sessions[0].session_id == "dispatch-test-001"
        assert sessions[0].vscode_edition == "insider"


class TestCLINewEventHandlers:
    """Tests for new CLI event handlers: subagent, handoff, warning, mode/context/plan changes."""

    @staticmethod
    def _make_events_jsonl(*events):
        """Create JSONL string with session.start + given events."""
        import ssrjson

        lines = [
            ssrjson.dumps(
                {
                    "type": "session.start",
                    "data": {
                        "sessionId": "test-session",
                        "startTime": "2026-01-01T00:00:00Z",
                    },
                }
            )
        ]
        for evt in events:
            lines.append(ssrjson.dumps(evt))
        return "\n".join(lines)

    def _parse(self, tmp_path, *events):
        """Write events to a temp file and parse them."""
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        f = tmp_path / "events.jsonl"
        f.write_text(self._make_events_jsonl(*events), encoding="utf-8")
        return _parse_cli_jsonl_file(f)

    def _find_status_blocks(self, session, description=None):
        """Return all status content blocks, optionally filtered by description."""
        blocks = []
        for msg in session.messages:
            for cb in msg.content_blocks:
                if cb.kind == "status" and (description is None or cb.description == description):
                    blocks.append(cb)
        return blocks

    # --- subagent.started (no visible output — completed block replaces it) ---

    def test_subagent_started_emits_no_pill(self, tmp_path):
        """subagent.started creates no visible content block."""
        session = self._parse(
            tmp_path,
            {"type": "assistant.message", "data": {"content": "Dispatching..."}},
            {"type": "subagent.started", "data": {"agentDisplayName": "code-review", "agentName": "cr", "toolCallId": "tc1"}},
        )
        blocks = self._find_status_blocks(session, "subagent")
        assert len(blocks) == 0

    def test_subagent_started_no_agent_metadata_on_messages(self, tmp_path):
        """Messages have no agent_id/agent_display_name set."""
        session = self._parse(
            tmp_path,
            {"type": "assistant.message", "data": {"content": "Dispatching..."}},
            {"type": "subagent.started", "data": {"agentName": "fallback-agent", "toolCallId": "tc2"}},
        )
        for msg in session.messages:
            assert msg.agent_id is None
            assert msg.agent_display_name is None
            assert msg.agent_nesting_level == 0

    # --- root custom-agent intervals ---

    def test_root_agent_selection_creates_interval(self, tmp_path):
        """subagent.selected records a root custom-agent interval without changing message metadata."""
        session = self._parse(
            tmp_path,
            {
                "type": "subagent.selected",
                "timestamp": "2026-01-01T00:00:01Z",
                "data": {"agentName": "smart-merge", "agentDisplayName": "smart-merge", "tools": ["sql", "powershell"]},
            },
            {"type": "user.message", "timestamp": "2026-01-01T00:00:02Z", "data": {"content": "Merge main"}},
            {"type": "assistant.message", "timestamp": "2026-01-01T00:00:03Z", "data": {"content": "Starting merge."}},
            {"type": "subagent.deselected", "timestamp": "2026-01-01T00:00:04Z", "data": {}},
            {"type": "assistant.message", "timestamp": "2026-01-01T00:00:05Z", "data": {"content": "Back to default."}},
        )

        assert session is not None
        assert session.root_agent_intervals == [
            RootAgentInterval(
                agent_name="smart-merge",
                agent_display_name="smart-merge",
                start_timestamp="2026-01-01T00:00:01Z",
                end_timestamp="2026-01-01T00:00:04Z",
                tools=["sql", "powershell"],
            )
        ]
        assert [msg.agent_display_name for msg in session.messages] == [None, None]

    def test_root_agent_reselection_closes_previous_interval(self, tmp_path):
        """A new subagent.selected closes the previous root-agent interval."""
        session = self._parse(
            tmp_path,
            {
                "type": "subagent.selected",
                "timestamp": "2026-01-01T00:00:01Z",
                "data": {"agentName": "smart-merge", "agentDisplayName": "Smart Merge"},
            },
            {"type": "user.message", "timestamp": "2026-01-01T00:00:02Z", "data": {"content": "Merge main"}},
            {
                "type": "subagent.selected",
                "timestamp": "2026-01-01T00:00:03Z",
                "data": {"agentName": "skill-audit", "agentDisplayName": "Skill Audit"},
            },
            {"type": "assistant.message", "timestamp": "2026-01-01T00:00:04Z", "data": {"content": "Auditing."}},
        )

        assert session is not None
        assert session.root_agent_intervals == [
            RootAgentInterval(
                agent_name="smart-merge",
                agent_display_name="Smart Merge",
                start_timestamp="2026-01-01T00:00:01Z",
                end_timestamp="2026-01-01T00:00:03Z",
            ),
            RootAgentInterval(
                agent_name="skill-audit",
                agent_display_name="Skill Audit",
                start_timestamp="2026-01-01T00:00:03Z",
            ),
        ]

    def test_root_agent_selection_without_timestamp_is_ignored(self, tmp_path):
        """A root-agent interval needs a timestamp anchor."""
        session = self._parse(
            tmp_path,
            {"type": "subagent.selected", "data": {"agentName": "smart-merge", "agentDisplayName": "Smart Merge"}},
            {"type": "user.message", "timestamp": "2026-01-01T00:00:02Z", "data": {"content": "Merge main"}},
        )

        assert session is not None
        assert session.root_agent_intervals == []

    def test_root_agent_selection_does_not_affect_nested_subagent_handling(self, tmp_path):
        """Root-agent intervals stay separate from spawned task subagent records."""
        session = self._parse(
            tmp_path,
            {
                "type": "subagent.selected",
                "timestamp": "2026-01-01T00:00:01Z",
                "data": {"agentName": "smart-merge", "agentDisplayName": "smart-merge"},
            },
            {"type": "assistant.message", "timestamp": "2026-01-01T00:00:02Z", "data": {"content": "Spawning explorer."}},
            {
                "type": "tool.execution_start",
                "timestamp": "2026-01-01T00:00:03Z",
                "data": {"toolCallId": "tc1", "toolName": "task", "arguments": {"agent_type": "explore"}},
            },
            {"type": "subagent.started", "timestamp": "2026-01-01T00:00:03Z", "data": {"toolCallId": "tc1", "agentDisplayName": "explore"}},
            {
                "type": "tool.execution_start",
                "timestamp": "2026-01-01T00:00:04Z",
                "data": {"toolCallId": "child1", "toolName": "view", "parentToolCallId": "tc1", "arguments": {"path": "README.md"}},
            },
            {
                "type": "tool.execution_complete",
                "timestamp": "2026-01-01T00:00:05Z",
                "data": {"toolCallId": "tc1", "success": True, "result": {"content": "Found the answer"}},
            },
            {"type": "subagent.completed", "timestamp": "2026-01-01T00:00:05Z", "data": {"toolCallId": "tc1", "agentDisplayName": "explore"}},
        )

        assert session is not None
        assert len(session.root_agent_intervals) == 1
        subagent_blocks = [block for msg in session.messages for block in msg.content_blocks if block.kind == "subagent"]
        assert len(subagent_blocks) == 1
        assert subagent_blocks[0].child_message is not None
        assert subagent_blocks[0].child_message.agent_display_name == "explore"

    # --- subagent.completed (emits subagent content block) ---

    def test_subagent_completed_creates_subagent_block(self, tmp_path):
        """subagent.completed creates a ContentBlock(kind='subagent') with result text."""
        session = self._parse(
            tmp_path,
            {"type": "subagent.started", "data": {"agentDisplayName": "explorer", "toolCallId": "tc3"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc3", "agentDisplayName": "explorer"}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc3", "success": True, "result": {"content": "Found 3 files"}}},
        )
        subagent_blocks = []
        for msg in session.messages:
            for cb in msg.content_blocks:
                if cb.kind == "subagent":
                    subagent_blocks.append(cb)
        assert len(subagent_blocks) == 1
        assert "Found 3 files" in subagent_blocks[0].content
        assert "explorer" in subagent_blocks[0].description

    # --- subagent.failed ---

    def test_subagent_failed(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "assistant.message", "data": {"content": "Running..."}},
            {"type": "subagent.failed", "data": {"agentDisplayName": "builder", "toolCallId": "tc1", "error": "timeout"}},
        )
        blocks = self._find_status_blocks(session, "subagent-error")
        assert len(blocks) == 1
        assert "builder" in blocks[0].content
        assert "failed" in blocks[0].content

    def test_subagent_failed_null_error(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "assistant.message", "data": {"content": "Running..."}},
            {"type": "subagent.failed", "data": {"agentDisplayName": "builder", "toolCallId": "tc2", "error": None}},
        )
        blocks = self._find_status_blocks(session, "subagent-error")
        assert len(blocks) == 1
        assert "failed" in blocks[0].content

    def test_subagent_failed_truncates_long_error(self, tmp_path):
        long_error = "x" * 300
        session = self._parse(
            tmp_path,
            {"type": "assistant.message", "data": {"content": "Running..."}},
            {"type": "subagent.failed", "data": {"agentDisplayName": "builder", "toolCallId": "tc3", "error": long_error}},
        )
        blocks = self._find_status_blocks(session, "subagent-error")
        assert len(blocks) == 1
        assert "failed" in blocks[0].content
        # Error detail is NOT in the pill — it's in the collapsible
        assert len(long_error) > 200  # sanity check

    # --- session.handoff ---

    def test_session_handoff(self, tmp_path):
        session = self._parse(
            tmp_path,
            {
                "type": "session.handoff",
                "data": {
                    "sourceType": "vscode",
                    "repository": {"owner": "octocat", "name": "hello-world", "branch": "main"},
                },
            },
        )
        blocks = self._find_status_blocks(session, "handoff")
        assert len(blocks) == 1
        assert blocks[0].content == "🔄 Session handoff from vscode (octocat/hello-world @ main)"

    def test_session_handoff_null_repository(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "session.handoff", "data": {"sourceType": "cli", "repository": None}},
        )
        blocks = self._find_status_blocks(session, "handoff")
        assert len(blocks) == 1
        assert blocks[0].content == "🔄 Session handoff from cli"

    def test_session_handoff_missing_repository(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "session.handoff", "data": {"sourceType": "cli"}},
        )
        blocks = self._find_status_blocks(session, "handoff")
        assert len(blocks) == 1
        assert blocks[0].content == "🔄 Session handoff from cli"

    def test_session_handoff_no_branch(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "session.handoff", "data": {"sourceType": "vscode", "repository": {"owner": "octocat", "name": "hello-world"}}},
        )
        blocks = self._find_status_blocks(session, "handoff")
        assert len(blocks) == 1
        assert blocks[0].content == "🔄 Session handoff from vscode (octocat/hello-world)"

    # --- session.info fork ---

    def test_session_info_fork_renders_status(self, tmp_path):
        current_session_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        session_id = "12345678-1234-1234-1234-123456789abc"
        session = self._parse(
            tmp_path,
            {"type": "session.start", "data": {"sessionId": current_session_id, "startTime": "2026-01-01T00:00:00Z"}},
            {"type": "session.info", "data": {"infoType": "fork", "message": f'Forked this session into "new-session" ({session_id}).'}},
        )
        blocks = self._find_status_blocks(session, "fork")
        assert len(blocks) == 1
        assert blocks[0].content == f"Forked as [new-session](/session/{session_id})."

    def test_session_info_fork_links_parent_but_not_self(self, tmp_path):
        source_id = "11111111-1111-1111-1111-111111111111"
        fork_id = "22222222-2222-2222-2222-222222222222"
        event_id = "33333333-3333-3333-3333-333333333333"
        session = self._parse(
            tmp_path,
            {"type": "session.start", "data": {"sessionId": fork_id, "startTime": "2026-01-01T00:00:00Z"}},
            {"type": "session.info", "data": {"infoType": "fork", "message": f"Forked from {source_id} before event {event_id} as {fork_id}."}},
        )
        blocks = self._find_status_blocks(session, "fork")
        assert len(blocks) == 1
        assert blocks[0].content == f"Forked from [11111111](/session/{source_id}) before event {event_id}."

    def test_session_info_fork_escapes_markdown_link_text(self, tmp_path):
        current_session_id = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
        fork_id = "11111111-1111-1111-1111-111111111111"
        session = self._parse(
            tmp_path,
            {"type": "session.start", "data": {"sessionId": current_session_id, "startTime": "2026-01-01T00:00:00Z"}},
            {
                "type": "session.info",
                "data": {
                    "infoType": "fork",
                    "message": f'Forked this session into "<img src=x onerror=alert(1)> [oops]" ({fork_id}).',
                },
            },
        )
        blocks = self._find_status_blocks(session, "fork")
        assert len(blocks) == 1
        assert blocks[0].content == f"Forked as [&lt;img src=x onerror=alert(1)&gt; \\[oops\\]](/session/{fork_id})."

    def test_session_info_non_fork_still_skipped(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Hello"}},
            {"type": "assistant.message", "data": {"content": "Hi"}},
            {"type": "session.info", "data": {"infoType": "authentication", "message": "Logged in."}},
        )
        assert session is not None
        assert self._find_status_blocks(session, "fork") == []

    def test_auto_mode_resolution_renders_selected_model(self, tmp_path):
        session = self._parse(
            tmp_path,
            {
                "type": "session.auto_mode_resolved",
                "data": {
                    "chosenModel": "claude-sonnet-5",
                    "reasoningBucket": "high",
                    "categoryScores": {"reasoning": 0.9},
                },
            },
        )

        blocks = self._find_status_blocks(session, "auto-mode")
        assert [block.content for block in blocks] == ["Auto selected claude-sonnet-5 (high reasoning)"]

    def test_fusion_events_reduce_to_one_safe_status(self, tmp_path):
        session = self._parse(
            tmp_path,
            {
                "type": "session.fusion_resolved",
                "data": {
                    "fusionId": "fusion-1",
                    "pattern": "critique",
                    "primaryModel": "gpt-5.6-sol",
                    "secondaryModel": "claude-opus-5",
                },
            },
            {
                "type": "assistant.fusion_phase_completed",
                "data": {
                    "fusionId": "fusion-1",
                    "phaseKind": "primary",
                    "model": "gpt-5.6-sol",
                    "content": "Internal draft that must not render",
                },
            },
            {
                "type": "assistant.fusion_phase_completed",
                "data": {
                    "fusionId": "fusion-1",
                    "phaseKind": "judge",
                    "model": "claude-opus-5",
                    "verdict": "reject",
                    "content": "Internal critique that must not render",
                },
            },
            {
                "type": "session.fusion_handoff",
                "data": {
                    "fusionId": "fusion-1",
                    "targetModel": "gpt-5.6-sol",
                    "message": {"content": "Internal handoff prompt that must not render"},
                },
            },
            {
                "type": "session.fusion_completed",
                "data": {
                    "fusionId": "fusion-1",
                    "outcome": "completed",
                    "finalSourceModel": "gpt-5.6-sol",
                },
            },
            {"type": "assistant.message", "data": {"content": "Final answer"}},
        )

        blocks = self._find_status_blocks(session, "fusion")
        assert [block.content for block in blocks] == ["Fusion · Critique · gpt-5.6-sol + claude-opus-5; Drafted answer; Changes requested; Final response prepared"]
        rendered_content = "\n".join(block.content for message in session.messages for block in message.content_blocks)
        assert "Internal draft" not in rendered_content
        assert "Internal critique" not in rendered_content
        assert "Internal handoff" not in rendered_content

    # --- session.warning ---

    def test_session_warning(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "session.warning", "data": {"message": "Rate limit approaching"}},
        )
        blocks = self._find_status_blocks(session, "warning")
        assert len(blocks) == 1
        assert blocks[0].content == "⚠️ Rate limit approaching"

    def test_session_warning_empty_message_skipped(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "session.warning", "data": {"message": ""}},
        )
        # Empty warning produces no content, so parser may return None (no messages)
        if session is None:
            return  # no messages at all — empty warning correctly skipped
        blocks = self._find_status_blocks(session, "warning")
        assert len(blocks) == 0

    # --- session.mode_changed ---

    def test_session_mode_changed(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "session.mode_changed", "data": {"previousMode": "ask", "newMode": "agent"}},
        )
        blocks = self._find_status_blocks(session, "mode-change")
        assert len(blocks) == 1
        assert blocks[0].content == "Mode changed: ask → agent"

    # --- session.context_changed ---

    def test_session_context_changed(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "session.context_changed", "data": {"cwd": "/home/user/project", "branch": "feature/x"}},
        )
        blocks = self._find_status_blocks(session, "context-change")
        assert len(blocks) == 1
        assert blocks[0].content == "Working directory changed: /home/user/project (feature/x)"

    def test_session_context_changed_no_branch(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "session.context_changed", "data": {"cwd": "/home/user/project"}},
        )
        blocks = self._find_status_blocks(session, "context-change")
        assert len(blocks) == 1
        assert blocks[0].content == "Working directory changed: /home/user/project"

    # --- session.plan_changed ---

    def test_session_plan_changed(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "session.plan_changed", "data": {"operation": "created"}},
        )
        blocks = self._find_status_blocks(session, "plan-change")
        assert len(blocks) == 1
        assert blocks[0].content == "Plan created"

    # --- session.task_complete ---

    def test_session_task_complete(self, tmp_path):
        session = self._parse(
            tmp_path,
            {"type": "session.task_complete", "data": {"summary": "Implemented auth module with JWT tokens."}},
        )
        blocks = self._find_status_blocks(session, "task-complete")
        assert len(blocks) == 1
        assert blocks[0].content == "Implemented auth module with JWT tokens."

    def test_session_task_complete_empty_summary(self, tmp_path):
        """Task complete with empty summary should not produce a block."""
        session = self._parse(
            tmp_path,
            {"type": "session.task_complete", "data": {"summary": ""}},
        )
        if session is None:
            return
        blocks = self._find_status_blocks(session, "task-complete")
        assert len(blocks) == 0

    # --- session.shutdown ---

    def test_session_shutdown(self, tmp_path):
        session = self._parse(
            tmp_path,
            {
                "type": "session.shutdown",
                "data": {
                    "shutdownType": "routine",
                    "codeChanges": {"linesAdded": 100, "linesRemoved": 20, "filesModified": ["a.py", "b.py"]},
                    "modelMetrics": {"claude-sonnet-4": {"requests": {"count": 10, "cost": 5}, "usage": {}}},
                },
            },
        )
        blocks = self._find_status_blocks(session, "shutdown")
        assert len(blocks) == 1
        assert "routine" in blocks[0].content
        assert "+100/-20" in blocks[0].content
        assert "2 files" in blocks[0].content
        assert "claude-sonnet-4" in blocks[0].content

    def test_session_shutdown_minimal(self, tmp_path):
        """Shutdown with only shutdownType should still render."""
        session = self._parse(
            tmp_path,
            {"type": "session.shutdown", "data": {"shutdownType": "crash"}},
        )
        blocks = self._find_status_blocks(session, "shutdown")
        assert len(blocks) == 1
        assert "crash" in blocks[0].content

    # --- skip internal events ---

    def test_skip_internal_events(self, tmp_path):
        """Internal events should produce no content blocks."""
        internal_events = [
            {"type": "assistant.turn_start", "data": {}},
            {"type": "assistant.turn_end", "data": {}},
            {"type": "session.resume", "data": {}},
        ]
        session = self._parse(tmp_path, *internal_events)
        # Internal events add no content, so parser returns None (no messages)
        if session is None:
            return  # correctly produced no messages
        all_blocks = []
        for msg in session.messages:
            all_blocks.extend(msg.content_blocks)
        assert len(all_blocks) == 0

    def test_permission_and_user_input_events_render(self, tmp_path):
        """Prompt-style CLI events should preserve user-visible interaction context."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Do the thing"}},
            {"type": "assistant.message", "data": {"content": "I need confirmation."}},
            {
                "type": "permission.requested",
                "data": {"requestId": "r1", "permissionRequest": {"kind": "shell", "command": "npm test"}},
            },
            {"type": "permission.completed", "data": {"requestId": "r1", "result": {"kind": "approved"}}},
            {"type": "user_input.requested", "data": {"requestId": "r2", "question": "Continue?", "choices": ["yes", "no"]}},
            {"type": "user_input.completed", "data": {"requestId": "r2", "answer": "yes"}},
        )

        assert session is not None
        permission_blocks = self._find_status_blocks(session, "permission")
        assert [block.content for block in permission_blocks] == [
            "Permission requested: shell - npm test",
            "Permission approved",
        ]
        ask_blocks = [block for msg in session.messages for block in msg.content_blocks if block.kind == "ask_user"]
        assert len(ask_blocks) == 1
        assert "Continue?" in ask_blocks[0].content
        assert "yes, no" in ask_blocks[0].content
        user_input_status = self._find_status_blocks(session, "user-input")
        assert user_input_status[0].content == "User answered: yes"

    def test_new_status_events_render(self, tmp_path):
        """Recent runtime status events should not be dropped from CLI sessions."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Run checks"}},
            {"type": "assistant.message", "data": {"content": "Working."}},
            {"type": "external_tool.requested", "data": {"requestId": "r4", "toolName": "ext"}},
            {"type": "command.queued", "data": {"requestId": "r5", "command": "npm test"}},
            {"type": "session.truncation", "data": {"reason": "context_window"}},
            {"type": "session.workspace_file_changed", "data": {"operation": "updated", "path": "src/app.py"}},
            {"type": "system.notification", "data": {"content": "Shell command completed"}},
            {"type": "hook.progress", "data": {"message": "\x1b[33mRunning postToolUse hook\x1b[0m", "temporary": False}},
            {"type": "hook.end", "data": {"hookType": "postToolUse", "success": False, "error": "blocked"}},
        )

        assert session is not None
        assert self._find_status_blocks(session, "external-tool") == []
        assert self._find_status_blocks(session, "command")[0].content == "Command queued: npm test"
        assert self._find_status_blocks(session, "truncation")[0].content == "Context truncated: context_window"
        assert self._find_status_blocks(session, "workspace-file")[0].content == "Workspace file changed: updated src/app.py"
        assert self._find_status_blocks(session, "notification")[0].content == "Shell command completed"
        assert self._find_status_blocks(session, "hook-progress")[0].content == "Hook progress: Running postToolUse hook"
        assert self._find_status_blocks(session, "hook-error")[0].content == "Hook failed: postToolUse - blocked"

    def test_status_event_payloads_strip_ansi_codes(self, tmp_path):
        """Runtime status payloads can contain terminal output with ANSI escapes."""
        session = self._parse(
            tmp_path,
            {"type": "assistant.message", "data": {"content": "Working."}},
            {"type": "hook.end", "data": {"hookType": "postToolUse", "success": False, "error": "\x1b[31mblocked\x1b[0m"}},
        )

        assert session is not None
        assert self._find_status_blocks(session, "hook-error")[0].content == "Hook failed: postToolUse - blocked"

    def _find_tool_invocations(self, session, name=None):
        """Return all tool invocations across messages, optionally filtered by name."""
        tools = []
        for msg in session.messages:
            for tool in msg.tool_invocations:
                if name is None or tool.name == name:
                    tools.append(tool)
        return tools

    def test_external_tool_renders_with_parameters(self, tmp_path):
        """external_tool.requested should render as a tool block exposing its parameters."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Rename it"}},
            {"type": "assistant.message", "data": {"content": "Renaming."}},
            {
                "type": "external_tool.requested",
                "data": {
                    "requestId": "r1",
                    "toolCallId": "tc1",
                    "toolName": "rename_session",
                    "arguments": {"title": "Upstream sync plan"},
                },
            },
            {"type": "external_tool.completed", "data": {"requestId": "r1"}},
        )

        assert session is not None
        # No longer a thin status breadcrumb.
        assert self._find_status_blocks(session, "external-tool") == []
        # Rendered as a proper tool invocation carrying the parameters.
        tools = self._find_tool_invocations(session, "rename_session")
        assert len(tools) == 1
        tool = tools[0]
        assert tool.source_type == "external"
        # Parameters are preserved in the serialized input.
        assert tool.input is not None
        assert "Upstream sync plan" in tool.input
        # The host does not record a response, so result stays unset.
        assert tool.result is None
        # Label falls back to the tool name (same default as builtin tools without a
        # specific format) — the parameters live in the input, not the header.
        assert tool.invocation_message == "rename_session"
        # A matching inline content block exists for template pairing.
        tool_blocks = [cb for msg in session.messages for cb in msg.content_blocks if cb.kind == "toolInvocation"]
        assert any(cb.content == "rename_session" for cb in tool_blocks)

    def test_external_tool_without_arguments(self, tmp_path):
        """An external tool with no arguments still renders as a tool block, not a status line."""
        session = self._parse(
            tmp_path,
            {"type": "assistant.message", "data": {"content": "Working."}},
            {
                "type": "external_tool.requested",
                "data": {"requestId": "r2", "toolCallId": "tc2", "toolName": "list_sessions_and_chats"},
            },
            {"type": "external_tool.completed", "data": {"requestId": "r2"}},
        )

        assert session is not None
        assert self._find_status_blocks(session, "external-tool") == []
        tools = self._find_tool_invocations(session, "list_sessions_and_chats")
        assert len(tools) == 1
        assert tools[0].source_type == "external"
        assert tools[0].result is None


class TestCLISubagentBrackets:
    """Test that CLI subagent events emit status pills and subagent content blocks."""

    @staticmethod
    def _make_events_jsonl(*events):
        """Create JSONL string with session.start + given events."""
        import ssrjson

        lines = [
            ssrjson.dumps(
                {
                    "type": "session.start",
                    "data": {
                        "sessionId": "subagent-test",
                        "startTime": "2026-01-01T00:00:00Z",
                    },
                }
            )
        ]
        for evt in events:
            lines.append(ssrjson.dumps(evt))
        return "\n".join(lines)

    def _parse(self, tmp_path, *events):
        """Write events to a temp file and parse them."""
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        f = tmp_path / "events.jsonl"
        f.write_text(self._make_events_jsonl(*events), encoding="utf-8")
        return _parse_cli_jsonl_file(f)

    def _find_blocks(self, session, kind):
        """Return all content blocks of a given kind."""
        blocks = []
        for msg in session.messages:
            for cb in msg.content_blocks:
                if cb.kind == kind:
                    blocks.append(cb)
        return blocks

    def test_started_emits_no_pill(self, tmp_path):
        """subagent.started emits no visible content block."""
        session = self._parse(
            tmp_path,
            {
                "type": "assistant.message",
                "data": {
                    "content": "Dispatching...",
                    "toolRequests": [
                        {"toolCallId": "tc1", "name": "task", "arguments": {"agent_type": "explore", "description": "Find auth"}},
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc1", "toolName": "task", "arguments": {"agent_type": "explore"}}},
            {"type": "subagent.started", "data": {"toolCallId": "tc1", "agentDisplayName": "Explore Agent", "agentName": "explore"}},
        )
        blocks = [b for b in self._find_blocks(session, "status") if b.description == "subagent"]
        assert len(blocks) == 0

    def test_completed_emits_subagent_block(self, tmp_path):
        """subagent.completed emits a subagent ContentBlock with the result."""
        session = self._parse(
            tmp_path,
            {
                "type": "assistant.message",
                "data": {
                    "content": "Running agent...",
                    "toolRequests": [
                        {"toolCallId": "tc1", "name": "task", "arguments": {"agent_type": "explore", "description": "Find auth"}},
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc1", "toolName": "task", "arguments": {"agent_type": "explore"}}},
            {"type": "subagent.started", "data": {"toolCallId": "tc1", "agentDisplayName": "Explore Agent"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc1", "agentDisplayName": "Explore Agent"}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc1", "success": True, "result": {"content": "Found JWT auth in src/auth.py"}}},
        )
        blocks = self._find_blocks(session, "subagent")
        assert len(blocks) == 1
        assert "Found JWT auth" in blocks[0].content
        assert "Explore Agent" in blocks[0].description

    def test_agent_numbering(self, tmp_path):
        """Agent numbering (agent-0, agent-1) appears in labels based on task dispatch order."""
        session = self._parse(
            tmp_path,
            {
                "type": "assistant.message",
                "data": {
                    "content": "Dispatching two agents...",
                    "toolRequests": [
                        {"toolCallId": "tc1", "name": "task", "arguments": {"agent_type": "explore", "description": "Search code"}},
                        {"toolCallId": "tc2", "name": "task", "arguments": {"agent_type": "task", "description": "Run tests"}},
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc1", "toolName": "task", "arguments": {"agent_type": "explore"}}},
            {"type": "tool.execution_start", "data": {"toolCallId": "tc2", "toolName": "task", "arguments": {"agent_type": "task"}}},
            {"type": "subagent.started", "data": {"toolCallId": "tc1", "agentDisplayName": "Explore Agent"}},
            {"type": "subagent.started", "data": {"toolCallId": "tc2", "agentDisplayName": "Task Agent"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc1", "agentDisplayName": "Explore Agent"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc2", "agentDisplayName": "Task Agent"}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc1", "success": True, "result": {"content": "result1"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc2", "success": True, "result": {"content": "result2"}}},
        )
        # Completed pills exist
        status_blocks = [b for b in self._find_blocks(session, "status") if b.description == "subagent"]
        assert len(status_blocks) == 2
        assert any("completed" in b.content for b in status_blocks)
        # Subagent collapsible blocks at start position
        subagent_blocks = self._find_blocks(session, "subagent")
        assert len(subagent_blocks) == 2
        descs = [b.description for b in subagent_blocks]
        assert any("Explore Agent" in d for d in descs)
        assert any("Task Agent" in d for d in descs)

    def test_parallel_agents_each_get_completed_block(self, tmp_path):
        """Parallel agents each get their own completed subagent block."""
        session = self._parse(
            tmp_path,
            {
                "type": "assistant.message",
                "data": {
                    "content": "Running in parallel...",
                    "toolRequests": [
                        {"toolCallId": "tc1", "name": "task", "arguments": {"agent_type": "explore", "description": "Agent A work"}},
                        {"toolCallId": "tc2", "name": "task", "arguments": {"agent_type": "explore", "description": "Agent B work"}},
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc1", "toolName": "task", "arguments": {}}},
            {"type": "tool.execution_start", "data": {"toolCallId": "tc2", "toolName": "task", "arguments": {}}},
            {"type": "subagent.started", "data": {"toolCallId": "tc1", "agentDisplayName": "Agent A"}},
            {"type": "subagent.started", "data": {"toolCallId": "tc2", "agentDisplayName": "Agent B"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc1", "agentDisplayName": "Agent A"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc2", "agentDisplayName": "Agent B"}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc1", "success": True, "result": {"content": "Result A"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc2", "success": True, "result": {"content": "Result B"}}},
        )
        subagent_blocks = self._find_blocks(session, "subagent")
        assert len(subagent_blocks) == 2
        contents = {b.content for b in subagent_blocks}
        assert "Result A" in contents
        assert "Result B" in contents

    def test_messages_between_brackets_are_normal(self, tmp_path):
        """Messages between subagent.started and subagent.completed have no agent metadata."""
        session = self._parse(
            tmp_path,
            {"type": "assistant.message", "data": {"content": "Before agent"}},
            {"type": "subagent.started", "data": {"toolCallId": "tc1", "agentDisplayName": "Agent"}},
            {"type": "assistant.message", "data": {"content": "During agent"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc1"}},
            {"type": "assistant.message", "data": {"content": "After agent"}},
        )
        for msg in session.messages:
            if msg.role == "assistant":
                assert msg.agent_id is None, f"Message should not have agent_id: {msg.content}"
                assert msg.agent_display_name is None, f"Message should not have agent_display_name: {msg.content}"
                assert msg.agent_nesting_level == 0, f"Message should have nesting_level=0: {msg.content}"


class TestVSCodeSubagentParsing:
    """Test VS Code sub-agent detection from toolSpecificData."""

    def test_subagent_tool_invocation_detected(self):
        """toolSpecificData.kind == 'subagent' should be detected."""
        item = {
            "kind": "toolInvocationSerialized",
            "toolId": "RunSubagentTool",
            "invocationMessage": "",
            "toolSpecificData": {
                "kind": "subagent",
                "agentName": "search",
                "description": "Find auth files",
                "result": "Found 3 auth files",
            },
            "isComplete": True,
            "toolCallId": "call_123",
        }
        inv = _parse_tool_invocation_serialized(item)
        assert inv is not None
        assert inv.result == "Found 3 auth files"
        assert "Agent" in (inv.invocation_message or "")

    def test_subagent_invocation_id_extracted(self):
        """subAgentInvocationId should be read from child tool invocations."""
        item = {
            "kind": "toolInvocationSerialized",
            "toolId": "vscode_readFile",
            "invocationMessage": "Reading file.py",
            "toolSpecificData": {},
            "isComplete": True,
            "toolCallId": "call_456",
            "subAgentInvocationId": "call_123",
        }
        inv = _parse_tool_invocation_serialized(item)
        assert inv is not None
        assert inv.subagent_invocation_id == "call_123"

    def test_subagent_with_no_result(self):
        """Subagent with no result should still parse correctly."""
        item = {
            "kind": "toolInvocationSerialized",
            "toolId": "RunSubagentTool",
            "invocationMessage": "",
            "toolSpecificData": {
                "kind": "subagent",
                "agentName": "explore",
                "description": "Search for patterns",
            },
            "isComplete": False,
            "toolCallId": "call_789",
        }
        inv = _parse_tool_invocation_serialized(item)
        assert inv is not None
        assert inv.status == "pending"
        assert inv.result is None
        assert "Agent" in (inv.invocation_message or "")

    def test_non_subagent_has_no_subagent_invocation_id(self):
        """Regular tool invocations should have subagent_invocation_id == None."""
        item = {
            "kind": "toolInvocationSerialized",
            "toolId": "vscode_readFile",
            "invocationMessage": "Reading file.py",
            "toolSpecificData": {},
            "isComplete": True,
            "toolCallId": "call_regular",
        }
        inv = _parse_tool_invocation_serialized(item)
        assert inv is not None
        assert inv.subagent_invocation_id is None

    def test_child_tools_grouped_under_parent_subagent(self):
        """Child tool invocations with subAgentInvocationId are absorbed into the parent's subagent block."""
        response_items = [
            # Parent subagent tool
            {
                "kind": "toolInvocationSerialized",
                "toolId": "RunSubagentTool",
                "invocationMessage": "",
                "toolSpecificData": {
                    "kind": "subagent",
                    "agentName": "search",
                    "description": "Find auth files",
                    "result": "Found 3 auth files in src/auth/",
                },
                "isComplete": True,
                "toolCallId": "call_parent",
            },
            # Child tool 1
            {
                "kind": "toolInvocationSerialized",
                "toolId": "vscode_readFile",
                "invocationMessage": "Reading auth.py",
                "toolSpecificData": {},
                "isComplete": True,
                "toolCallId": "call_child1",
                "subAgentInvocationId": "call_parent",
            },
            # Child tool 2
            {
                "kind": "toolInvocationSerialized",
                "toolId": "vscode_readFile",
                "invocationMessage": "Reading middleware.py",
                "toolSpecificData": {},
                "isComplete": True,
                "toolCallId": "call_child2",
                "subAgentInvocationId": "call_parent",
            },
            # Regular tool (not part of subagent)
            {
                "kind": "toolInvocationSerialized",
                "toolId": "vscode_readFile",
                "invocationMessage": "Reading README.md",
                "toolSpecificData": {},
                "isComplete": True,
                "toolCallId": "call_regular",
            },
        ]
        _, raw_blocks, tool_invocations, _, _ = _process_response_items(response_items)
        # Should have: one subagent block (with children absorbed) + one regular toolInvocation
        subagent_blocks = [b for b in raw_blocks if b[0] == "subagent"]
        tool_blocks = [b for b in raw_blocks if b[0] == "toolInvocation"]
        assert len(subagent_blocks) == 1, f"Expected 1 subagent block, got {len(subagent_blocks)}: {subagent_blocks}"
        assert len(tool_blocks) == 1, f"Expected 1 regular tool block, got {len(tool_blocks)}: {tool_blocks}"
        assert len(tool_invocations) == 2
        assert tool_invocations[0].name == "RunSubagentTool"
        assert tool_invocations[1].name == "vscode_readFile"
        assert tool_invocations[1].invocation_message == "Reading README.md"
        # The subagent block content should include child tool invocation messages and the result
        subagent_content = subagent_blocks[0][1]
        assert "auth.py" in subagent_content, f"Child tool should be in subagent content: {subagent_content[:200]}"
        assert "middleware.py" in subagent_content
        assert "Found 3 auth files" in subagent_content
        # The subagent title should include the agent name and description
        subagent_block = cast(tuple[str, str, str | None], subagent_blocks[0][:3])
        subagent_title = subagent_block[2]
        assert subagent_title is not None
        assert "search" in subagent_title
        assert "Find auth files" in subagent_title
        # The regular tool should still appear
        assert "README.md" in tool_blocks[0][1]

    def test_duplicate_parent_and_child_updates_collapse_to_one_subagent_block(self):
        """Repeated VS Code updates for the same subagent/toolCallId should render once."""
        response_items = [
            {
                "kind": "toolInvocationSerialized",
                "toolId": "runSubagent",
                "invocationMessage": "",
                "toolSpecificData": {
                    "kind": "subagent",
                    "agentName": "Agent",
                    "description": "Search archive",
                },
                "isComplete": True,
                "toolCallId": "parent_1",
            },
            {
                "kind": "toolInvocationSerialized",
                "toolId": "copilot_readFile",
                "invocationMessage": "Reading skill docs",
                "toolSpecificData": {},
                "isComplete": True,
                "toolCallId": "child_1",
                "subAgentInvocationId": "parent_1",
            },
            {
                "kind": "toolInvocationSerialized",
                "toolId": "copilot_readFile",
                "invocationMessage": "Reading skill docs",
                "toolSpecificData": {},
                "isComplete": True,
                "toolCallId": "child_1",
                "subAgentInvocationId": "parent_1",
            },
            {
                "kind": "toolInvocationSerialized",
                "toolId": "runSubagent",
                "invocationMessage": "",
                "toolSpecificData": {
                    "kind": "subagent",
                    "agentName": "Agent",
                    "description": "Search archive",
                    "result": "Agent completed with no output",
                },
                "isComplete": True,
                "toolCallId": "parent_1",
            },
        ]

        _, raw_blocks, tool_invocations, _, _ = _process_response_items(response_items)
        subagent_blocks = [b for b in raw_blocks if b[0] == "subagent"]
        assert len(subagent_blocks) == 1
        assert [tool.name for tool in tool_invocations] == ["runSubagent"]

        subagent_block = cast(
            tuple[
                str,
                str,
                str | None,
                list[ContentBlock],
                list[ScannerToolInvocation],
                list,
                list,
            ],
            subagent_blocks[0],
        )
        nested_blocks = subagent_block[3]
        nested_tool_invocations = subagent_block[4]
        assert len(nested_tool_invocations) == 1
        assert len(nested_blocks) == 2
        assert nested_blocks[0].content == "Reading skill docs"
        assert nested_blocks[1].kind == "text"
        assert "Agent completed with no output" in nested_blocks[1].content

    def test_subagent_terminal_children_render_inline_via_command_blocks(self):
        """VS Code subagent terminal children without invocation text should still get inline blocks."""
        response_items = [
            {
                "kind": "toolInvocationSerialized",
                "toolId": "runSubagent",
                "invocationMessage": "",
                "toolSpecificData": {
                    "kind": "subagent",
                    "agentName": "Agent",
                    "description": "Search archive",
                    "result": "Done",
                },
                "isComplete": True,
                "toolCallId": "parent_1",
            },
            {
                "kind": "toolInvocationSerialized",
                "toolId": "run_in_terminal",
                "invocationMessage": "",
                "toolSpecificData": {
                    "kind": "terminal",
                    "commandLine": "copilot-session-tools scan --verbose",
                },
                "resultDetails": {
                    "output": [
                        {
                            "value": "scan output",
                        }
                    ]
                },
                "isComplete": True,
                "toolCallId": "child_cmd_1",
                "subAgentInvocationId": "parent_1",
            },
        ]

        _, raw_blocks, _, _, _ = _process_response_items(response_items)
        subagent_blocks = [b for b in raw_blocks if b[0] == "subagent"]
        assert len(subagent_blocks) == 1

        subagent_block = cast(
            tuple[
                str,
                str,
                str | None,
                list[ContentBlock],
                list[ScannerToolInvocation],
                list,
                list,
            ],
            subagent_blocks[0],
        )
        nested_blocks = subagent_block[3]
        nested_command_runs = subagent_block[6]
        assert len(nested_command_runs) == 1
        assert nested_command_runs[0].command == "copilot-session-tools scan --verbose"
        assert nested_blocks[0].content == "$ copilot-session-tools scan --verbose"
        assert nested_blocks[1].kind == "text"

    def test_top_level_terminal_tools_without_invocation_message_render_inline(self):
        """Top-level VS Code terminal tools without invocation text should still get inline blocks."""
        response_items = [
            {
                "kind": "toolInvocationSerialized",
                "toolId": "run_in_terminal",
                "invocationMessage": "",
                "toolSpecificData": {
                    "kind": "terminal",
                    "commandLine": "echo hello",
                },
                "resultDetails": {
                    "output": [
                        {
                            "value": "hello",
                        }
                    ]
                },
                "isComplete": True,
                "toolCallId": "cmd_1",
            },
            {
                "kind": "toolInvocationSerialized",
                "toolId": "run_in_terminal",
                "invocationMessage": "",
                "toolSpecificData": {
                    "kind": "terminal",
                    "commandLine": "echo world",
                },
                "resultDetails": {
                    "output": [
                        {
                            "value": "world",
                        }
                    ]
                },
                "isComplete": True,
                "toolCallId": "cmd_2",
            },
        ]

        _, raw_blocks, tool_invocations, _, _ = _process_response_items(response_items)
        tool_blocks = [b for b in raw_blocks if b[0] == "toolInvocation"]
        assert [block[1] for block in tool_blocks] == ["$ echo hello", "$ echo world"]
        assert [tool.name for tool in tool_invocations] == ["run_in_terminal", "run_in_terminal"]

    def test_subagent_prompt_extracted_from_tool_specific_data(self):
        """Prompt field in toolSpecificData should be extracted into the raw_blocks tuple."""
        response_items = [
            {
                "kind": "toolInvocationSerialized",
                "toolId": "runSubagent",
                "invocationMessage": "",
                "toolSpecificData": {
                    "kind": "subagent",
                    "agentName": "explore",
                    "description": "Search for auth patterns",
                    "prompt": "Find all authentication-related files in the codebase and summarize the patterns used.",
                    "result": "Found 5 auth files using JWT pattern.",
                },
                "isComplete": True,
                "toolCallId": "call_prompt_test",
            },
        ]
        _, raw_blocks, _, _, _ = _process_response_items(response_items)
        subagent_blocks = [b for b in raw_blocks if b[0] == "subagent"]
        assert len(subagent_blocks) == 1
        # 8th element (index 7) is the prompt
        assert len(subagent_blocks[0]) == 8
        block = cast(tuple[str, str, str | None, list, list, list, list, str], subagent_blocks[0])
        assert block[7] == "Find all authentication-related files in the codebase and summarize the patterns used."

    def test_subagent_without_prompt_has_empty_string(self):
        """Subagent blocks without a prompt field should have empty string as prompt."""
        response_items = [
            {
                "kind": "toolInvocationSerialized",
                "toolId": "runSubagent",
                "invocationMessage": "",
                "toolSpecificData": {
                    "kind": "subagent",
                    "agentName": "search",
                    "description": "Find files",
                    "result": "Done",
                },
                "isComplete": True,
                "toolCallId": "call_no_prompt",
            },
        ]
        _, raw_blocks, _, _, _ = _process_response_items(response_items)
        subagent_blocks = [b for b in raw_blocks if b[0] == "subagent"]
        assert len(subagent_blocks) == 1
        assert len(subagent_blocks[0]) == 8
        block = cast(tuple[str, str, str | None, list, list, list, list, str], subagent_blocks[0])
        assert block[7] == ""

    def test_subagent_prompt_surfaces_on_content_block(self):
        """Prompt from raw_blocks should be wired through to the ContentBlock via _merge_content_blocks."""
        response_items = [
            {
                "kind": "toolInvocationSerialized",
                "toolId": "runSubagent",
                "invocationMessage": "",
                "toolSpecificData": {
                    "kind": "subagent",
                    "agentName": "explore",
                    "description": "Check auth",
                    "prompt": "Investigate the authentication module.",
                    "result": "Auth uses OAuth2.",
                },
                "isComplete": True,
                "toolCallId": "call_e2e",
            },
        ]
        _, raw_blocks, _, _, _ = _process_response_items(response_items)
        merged = _merge_content_blocks(raw_blocks)
        subagent_cbs = [cb for cb in merged if cb.kind == "subagent"]
        assert len(subagent_cbs) == 1
        assert subagent_cbs[0].prompt == "Investigate the authentication module."


class TestCLIStructuredSubagentContent:
    """Tests for structured content_blocks, tool_invocations, file_changes on subagent blocks."""

    def _make_events_jsonl(self, *events):
        import ssrjson

        lines = [
            ssrjson.dumps(
                {
                    "type": "session.start",
                    "data": {
                        "sessionId": "structured-subagent-test",
                        "startTime": "2026-01-01T00:00:00Z",
                    },
                }
            )
        ]
        for evt in events:
            lines.append(ssrjson.dumps(evt))
        return "\n".join(lines)

    def _parse(self, tmp_path, *events):
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        f = tmp_path / "events.jsonl"
        f.write_text(self._make_events_jsonl(*events), encoding="utf-8")
        return _parse_cli_jsonl_file(f)

    def _find_blocks(self, session, kind):
        blocks = []
        for msg in session.messages:
            for cb in msg.content_blocks:
                if cb.kind == kind:
                    blocks.append(cb)
        return blocks

    def test_subagent_has_structured_content_blocks(self, tmp_path):
        """Subagent block should have structured content_blocks with child tool invocations."""
        session = self._parse(
            tmp_path,
            {
                "type": "assistant.message",
                "data": {
                    "content": "Let me search for the code.",
                    "toolRequests": [
                        {"toolCallId": "tc1", "name": "task", "arguments": {"agent_type": "explore", "description": "Find auth code"}},
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc1", "toolName": "task", "arguments": {"agent_type": "explore"}}},
            # Child tools with parentToolCallId
            {"type": "tool.execution_start", "data": {"toolCallId": "child1", "toolName": "grep", "parentToolCallId": "tc1", "arguments": {"pattern": "jwt", "path": "src/"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "child1", "success": True, "result": {"content": "src/auth.py:10: import jwt"}}},
            {"type": "tool.execution_start", "data": {"toolCallId": "child2", "toolName": "view", "parentToolCallId": "tc1", "arguments": {"path": "src/auth.py"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "child2", "success": True, "result": {"content": "file contents..."}}},
            {"type": "subagent.started", "data": {"toolCallId": "tc1", "agentDisplayName": "Explore Agent"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc1", "agentDisplayName": "Explore Agent"}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc1", "success": True, "result": {"content": "Found JWT auth in src/auth.py"}}},
        )
        blocks = self._find_blocks(session, "subagent")
        assert len(blocks) == 1
        block = blocks[0]
        # Should have structured content_blocks
        assert len(block.content_blocks) > 0, "Subagent should have nested content_blocks"
        # Should have toolInvocation blocks for child tools
        tool_blocks = [cb for cb in block.content_blocks if cb.kind == "toolInvocation"]
        assert len(tool_blocks) == 2, f"Expected 2 child tool blocks, got {len(tool_blocks)}"
        # Should have a text block for the result
        text_blocks = [cb for cb in block.content_blocks if cb.kind == "text"]
        assert len(text_blocks) == 1
        assert "Found JWT auth" in text_blocks[0].content
        # Should have tool_invocations list
        assert len(block.tool_invocations) == 2
        assert block.tool_invocations[0].name == "grep"
        assert block.tool_invocations[1].name == "view"

    def test_subagent_child_command_runs(self, tmp_path):
        """Subagent with shell command child tools should have command_runs."""
        session = self._parse(
            tmp_path,
            {
                "type": "assistant.message",
                "data": {
                    "content": "Running tests...",
                    "toolRequests": [
                        {"toolCallId": "tc1", "name": "task", "arguments": {"agent_type": "task", "description": "Run tests"}},
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc1", "toolName": "task", "arguments": {"agent_type": "task"}}},
            # Shell command as child
            {
                "type": "tool.execution_start",
                "data": {
                    "toolCallId": "child1",
                    "toolName": "powershell",
                    "parentToolCallId": "tc1",
                    "arguments": {"command": "uv run pytest tests/ -q"},
                },
            },
            {"type": "tool.execution_complete", "data": {"toolCallId": "child1", "success": True, "result": {"content": "5 passed"}}},
            {"type": "subagent.started", "data": {"toolCallId": "tc1", "agentDisplayName": "Task Agent"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc1", "agentDisplayName": "Task Agent"}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc1", "success": True, "result": {"content": "All tests pass."}}},
        )
        blocks = self._find_blocks(session, "subagent")
        assert len(blocks) == 1
        block = blocks[0]
        # Shell commands become CommandRuns, not ToolInvocations
        assert len(block.command_runs) == 1
        assert "uv run pytest" in block.command_runs[0].command
        assert block.command_runs[0].output == "5 passed"

    def test_subagent_fts_content(self, tmp_path):
        """Subagent content should be in child_message.content, NOT in parent flat content."""
        session = self._parse(
            tmp_path,
            {
                "type": "assistant.message",
                "data": {
                    "content": "Delegating...",
                    "toolRequests": [
                        {"toolCallId": "tc1", "name": "task", "arguments": {"agent_type": "explore", "description": "Search"}},
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc1", "toolName": "task", "arguments": {}}},
            {"type": "subagent.started", "data": {"toolCallId": "tc1", "agentDisplayName": "Agent"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc1", "agentDisplayName": "Agent"}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc1", "success": True, "result": {"content": "Found unique_searchable_term_xyz"}}},
        )
        # Parent flat content should NOT contain subagent text
        for msg in session.messages:
            if msg.role == "assistant":
                assert "unique_searchable_term_xyz" not in msg.content
        # The subagent result should be in child_message.content
        blocks = self._find_blocks(session, "subagent")
        assert len(blocks) == 1
        assert blocks[0].child_message is not None
        assert "unique_searchable_term_xyz" in blocks[0].child_message.content
        # Also accessible via msg.children
        parent = next(m for m in session.messages if m.children)
        assert any("unique_searchable_term_xyz" in c.content for c in parent.children)

    def test_subagent_db_roundtrip(self, tmp_path):
        """Structured subagent data should survive database save/load cycle."""
        from copilot_session_tools.database import Database

        session = self._parse(
            tmp_path,
            {
                "type": "assistant.message",
                "data": {
                    "content": "Using agent...",
                    "toolRequests": [
                        {"toolCallId": "tc1", "name": "task", "arguments": {"agent_type": "explore", "description": "Find bugs"}},
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc1", "toolName": "task", "arguments": {"agent_type": "explore"}}},
            {"type": "tool.execution_start", "data": {"toolCallId": "child1", "toolName": "grep", "parentToolCallId": "tc1", "arguments": {"pattern": "TODO", "path": "src/"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "child1", "success": True, "result": {"content": "3 matches"}}},
            {"type": "subagent.started", "data": {"toolCallId": "tc1", "agentDisplayName": "Explore Agent"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc1", "agentDisplayName": "Explore Agent"}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc1", "success": True, "result": {"content": "Found 3 TODOs"}}},
        )

        # Save to DB
        db_path = tmp_path / "test.db"
        # Create a minimal session-store.db with required schema
        import sqlite3

        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER)")
        conn.execute("INSERT INTO schema_version VALUES (1)")
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)")
        conn.commit()
        conn.close()

        db = Database(str(db_path))
        db.add_session(session)

        # Load back
        loaded = db.get_session(session.session_id)
        assert loaded is not None

        # Find the subagent block
        subagent_blocks = []
        for msg in loaded.messages:
            for cb in msg.content_blocks:
                if cb.kind == "subagent":
                    subagent_blocks.append(cb)
        assert len(subagent_blocks) == 1
        block = subagent_blocks[0]
        # Basic round-trip: kind, content, and description survive
        assert "Found 3 TODOs" in block.content
        assert "Explore Agent" in block.description
        # Note: child_message and deprecated nested fields may not survive DB
        # round-trip until the DB layer is updated to serialize them.


class TestSharedHelpers:
    """Tests for the shared scanner helpers in scanner/shared.py."""

    # --- normalize_tool_status ---

    def test_completed_maps_to_success(self):
        """VS Code 'completed' should normalise to 'success'."""
        from copilot_session_tools.scanner.shared import normalize_tool_status

        assert normalize_tool_status("completed") == "success"

    def test_complete_maps_to_success(self):
        """VS Code 'complete' (no -d) should also normalise to 'success'."""
        from copilot_session_tools.scanner.shared import normalize_tool_status

        assert normalize_tool_status("complete") == "success"

    def test_is_complete_true(self):
        """is_complete=True with no raw_status should return 'success'."""
        from copilot_session_tools.scanner.shared import normalize_tool_status

        assert normalize_tool_status(None, is_complete=True) == "success"

    def test_is_complete_false(self):
        """is_complete=False with no raw_status should return 'pending'."""
        from copilot_session_tools.scanner.shared import normalize_tool_status

        assert normalize_tool_status(None, is_complete=False) == "pending"

    def test_has_error(self):
        """has_error=True should always return 'error'."""
        from copilot_session_tools.scanner.shared import normalize_tool_status

        assert normalize_tool_status(None, has_error=True) == "error"

    def test_error_overrides_is_complete(self):
        """has_error should override is_complete=True."""
        from copilot_session_tools.scanner.shared import normalize_tool_status

        assert normalize_tool_status(None, is_complete=True, has_error=True) == "error"

    def test_none_returns_none(self):
        """No inputs at all should return None."""
        from copilot_session_tools.scanner.shared import normalize_tool_status

        assert normalize_tool_status(None) is None

    def test_passthrough_unknown(self):
        """Unknown status strings should be passed through unchanged."""
        from copilot_session_tools.scanner.shared import normalize_tool_status

        assert normalize_tool_status("custom_status") == "custom_status"

    # --- extract_command_run ---

    def test_shell_tool_creates_command_run(self):
        """Shell tool names should create a CommandRun."""
        from copilot_session_tools.scanner.shared import extract_command_run

        result = extract_command_run("powershell", "ls")
        assert result is not None
        assert result.command == "ls"

    def test_non_shell_returns_none(self):
        """Non-shell tool names should return None."""
        from copilot_session_tools.scanner.shared import extract_command_run

        assert extract_command_run("read_file", "path") is None

    def test_empty_command_returns_none(self):
        """Empty command string should return None."""
        from copilot_session_tools.scanner.shared import extract_command_run

        assert extract_command_run("bash", "") is None

    def test_all_shell_tools_recognized(self):
        """Every member of SHELL_TOOL_NAMES should produce a CommandRun."""
        from copilot_session_tools.scanner.shared import SHELL_TOOL_NAMES, extract_command_run

        for name in SHELL_TOOL_NAMES:
            result = extract_command_run(name, "echo hi")
            assert result is not None, f"SHELL_TOOL_NAMES member '{name}' did not produce a CommandRun"
            assert result.command == "echo hi"

    # --- normalize_invocation_message ---

    def test_generic_message_replaced(self):
        """Generic 'Using \"Read File\"' message should be replaced with 'Viewing `filename`'."""
        from copilot_session_tools.scanner.shared import normalize_invocation_message

        tool_data = {"file": {"uri": {"fsPath": "/home/user/project/main.py", "path": "/home/user/project/main.py"}}}
        result = normalize_invocation_message("copilot_readFile", tool_data, 'Using "Read File"')
        assert result == "Viewing `main.py`"

    def test_non_generic_message_preserved(self):
        """Non-generic (custom) messages should be preserved as-is."""
        from copilot_session_tools.scanner.shared import normalize_invocation_message

        result = normalize_invocation_message("copilot_readFile", {"file": {"uri": {"fsPath": "/a/b.py"}}}, "Custom description for read")
        assert result == "Custom description for read"

    def test_unknown_tool_preserved(self):
        """Unknown tool IDs should preserve the message unchanged."""
        from copilot_session_tools.scanner.shared import normalize_invocation_message

        result = normalize_invocation_message("unknown_tool", {}, 'Using "Unknown"')
        assert result == 'Using "Unknown"'

    def test_none_message_returns_none(self):
        """None input message should return None."""
        from copilot_session_tools.scanner.shared import normalize_invocation_message

        assert normalize_invocation_message("copilot_readFile", {}, None) is None


class TestVSCodeRenderingAlignment:
    """Tests for VS Code rendering alignment changes D1-D6 through the scanner."""

    def test_d1_status_success(self):
        """D1: isComplete=True should produce status='success'."""
        item = {
            "kind": "toolInvocationSerialized",
            "toolId": "run_command",
            "invocationMessage": "Running command",
            "isComplete": True,
            "toolSpecificData": {"commandLine": "ls -la"},
            "resultDetails": {},
        }
        inv = _parse_tool_invocation_serialized(item)
        assert inv is not None
        assert inv.status == "success"

    def test_d1_status_pending(self):
        """D1: isComplete=False (no error) should produce status='pending'."""
        item = {
            "kind": "toolInvocationSerialized",
            "toolId": "run_command",
            "invocationMessage": "Running command",
            "isComplete": False,
            "toolSpecificData": {"commandLine": "ls -la"},
            "resultDetails": {},
        }
        inv = _parse_tool_invocation_serialized(item)
        assert inv is not None
        assert inv.status == "pending"

    def test_d1_status_error(self):
        """D1: isComplete=False with errorMessage should produce status='error'."""
        item = {
            "kind": "toolInvocationSerialized",
            "toolId": "run_command",
            "invocationMessage": "Running command",
            "isComplete": False,
            "toolSpecificData": {"commandLine": "ls -la"},
            "resultDetails": {"errorMessage": "fail"},
        }
        inv = _parse_tool_invocation_serialized(item)
        assert inv is not None
        assert inv.status == "error"

    def test_d4_readfile_invocation_normalized(self):
        """D4: Generic 'Using \"Read File\"' message should be normalised for copilot_readFile."""
        item = {
            "kind": "toolInvocationSerialized",
            "toolId": "copilot_readFile",
            "invocationMessage": 'Using "Read File"',
            "isComplete": True,
            "toolSpecificData": {"file": {"uri": {"fsPath": "/home/user/project/main.py", "path": "/home/user/project/main.py"}}},
            "resultDetails": {},
        }
        inv = _parse_tool_invocation_serialized(item)
        assert inv is not None
        assert inv.invocation_message == "Viewing `main.py`"

    def test_d6_input_from_tool_specific_data(self):
        """D6: toolSpecificData.input should be extracted when commandLine also exists."""
        item = {
            "kind": "toolInvocationSerialized",
            "toolId": "run_command",
            "invocationMessage": "Running command",
            "isComplete": True,
            "toolSpecificData": {
                "commandLine": "npm test",
                "input": "extra-input-data",
            },
            "resultDetails": {},
        }
        inv = _parse_tool_invocation_serialized(item)
        assert inv is not None
        # commandLine is the preferred input source
        assert inv.input == "npm test"

    def test_d6_input_fallback_from_tool_specific_data(self):
        """D6: toolSpecificData.input used as fallback when commandLine is absent."""
        item = {
            "kind": "toolInvocationSerialized",
            "toolId": "copilot_readFile",
            "invocationMessage": "Reading file",
            "isComplete": True,
            "toolSpecificData": {
                "input": "/path/to/file.py",
            },
            "resultDetails": {},
        }
        inv = _parse_tool_invocation_serialized(item)
        assert inv is not None
        assert inv.input == "/path/to/file.py"


class TestCLIv105EventHandlers:
    """Tests for CLI v1.0.5 scanner refresh: new events, new fields, expanded skip list."""

    @staticmethod
    def _make_events_jsonl(*events, start_data=None):
        """Create JSONL string with session.start + given events.

        Args:
            *events: Event dicts to append after session.start.
            start_data: Optional dict to merge into the session.start data.
        """
        import ssrjson

        base_start = {
            "sessionId": "test-session",
            "startTime": "2026-01-01T00:00:00Z",
        }
        if start_data:
            base_start.update(start_data)
        lines = [ssrjson.dumps({"type": "session.start", "data": base_start})]
        for evt in events:
            lines.append(ssrjson.dumps(evt))
        return "\n".join(lines)

    def _parse(self, tmp_path, *events, start_data=None):
        """Write events to a temp file and parse them."""
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        f = tmp_path / "events.jsonl"
        f.write_text(self._make_events_jsonl(*events, start_data=start_data), encoding="utf-8")
        return _parse_cli_jsonl_file(f)

    def _find_status_blocks(self, session, description=None):
        """Return all status content blocks, optionally filtered by description."""
        blocks = []
        for msg in session.messages:
            for cb in msg.content_blocks:
                if cb.kind == "status" and (description is None or cb.description == description):
                    blocks.append(cb)
        return blocks

    # ---- T1/T3: Skip list — ephemeral and internal events produce no output ----

    @pytest.mark.parametrize(
        "event_type",
        [
            "hook.start",
            "hook.end",
            "assistant.message_delta",
            "assistant.message_start",
            "assistant.reasoning_delta",
            "assistant.streaming_delta",
            "capabilities.changed",
            "commands.changed",
            "exit_plan_mode.requested",
            "exit_plan_mode.completed",
            "pending_messages.modified",
            "sampling.completed",
            "sampling.requested",
            "session.background_tasks_changed",
            "session.canvas.opened",
            "session.canvas.closed",
            "session.canvas.registry_changed",
            "session.custom_notification",
            "session.custom_agents_updated",
            "session.extensions_loaded",
            "session.extensions.attachments_pushed",
            "session.idle",
            "session.import_legacy",
            "session.mcp_server_status_changed",
            "session.mcp_servers_loaded",
            "session.autopilot_objective_changed",
            "session.binary_asset",
            "session.permissions_changed",
            "session.skills_loaded",
            "session.snapshot_rewind",
            "session.todos_changed",
            "session.tools_updated",
            "session.usage_info",
            "subagent.selected",
            "subagent.deselected",
        ],
    )
    def test_skipped_events_produce_no_content(self, tmp_path, event_type):
        """New skip-list events should produce no status blocks or content."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Hello"}},
            {"type": "assistant.message", "data": {"content": "Hi"}},
            {"type": event_type, "data": {"some": "data"}},
        )
        assert session is not None
        # Should have exactly user + assistant messages, no extra from skipped event
        blocks = self._find_status_blocks(session)
        # Skipped events should not generate status blocks
        for b in blocks:
            assert b.description != event_type, f"Skipped event {event_type} should not create a status block"

    def test_hook_events_with_real_data_skipped(self, tmp_path):
        """hook.start/end with realistic data should be silently skipped."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Hello"}},
            {"type": "assistant.message", "data": {"content": "Working..."}},
            {
                "type": "hook.start",
                "data": {
                    "hookInvocationId": "fe46353b-6596-40e5-9230-276a82390ca4",
                    "hookType": "postToolUse",
                    "input": {"toolName": "report_intent", "toolResult": {"resultType": "success"}},
                },
            },
            {
                "type": "hook.end",
                "data": {
                    "hookInvocationId": "fe46353b-6596-40e5-9230-276a82390ca4",
                    "hookType": "postToolUse",
                    "success": True,
                },
            },
        )
        assert session is not None
        all_blocks = []
        for msg in session.messages:
            all_blocks.extend(msg.content_blocks)
        # Only the text block from assistant.message, no hook-related blocks
        status_blocks = [b for b in all_blocks if b.kind == "status"]
        assert len(status_blocks) == 0

    def test_remote_steerable_changed_renders_status(self, tmp_path):
        """session.remote_steerable_changed should render the remote steering state."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Start remote work"}},
            {"type": "assistant.message", "data": {"content": "Starting."}},
            {"type": "session.remote_steerable_changed", "data": {"remoteSteerable": True}},
        )
        blocks = self._find_status_blocks(session, "remote-steering")
        assert len(blocks) == 1
        assert blocks[0].content == "Remote steering enabled"

    def test_freeform_tool_string_arguments_parse(self, tmp_path):
        """Freeform tools like apply_patch can store arguments as a raw string."""
        patch = "*** Begin Patch\n*** Add File: example.txt\n+hello\n*** End Patch\n"
        tool_call_id = "custom_call_string_args"

        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Add the file"}},
            {"type": "assistant.message", "data": {"content": "Applying the patch."}},
            {
                "type": "tool.execution_start",
                "data": {
                    "toolCallId": tool_call_id,
                    "toolName": "apply_patch",
                    "arguments": patch,
                },
            },
            {
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": tool_call_id,
                    "success": True,
                    "result": {"content": "Modified 1 file(s): example.txt"},
                },
            },
        )

        assert session is not None
        assistant = session.messages[1]
        assert assistant.tool_invocations[0].name == "apply_patch"
        assert assistant.tool_invocations[0].input == patch
        assert assistant.tool_invocations[0].status == "success"
        tool_blocks = [block for block in assistant.content_blocks if block.kind == "toolInvocation"]
        assert tool_blocks[0].content == "apply_patch"

    # ---- T7: session.title_changed ----

    def test_title_changed_renders_status_block(self, tmp_path):
        """session.title_changed should create a status block with the title."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Fix the auth bug"}},
            {"type": "assistant.message", "data": {"content": "On it."}},
            {"type": "session.title_changed", "data": {"title": "Fix auth middleware bug"}},
        )
        blocks = self._find_status_blocks(session, "title-changed")
        assert len(blocks) == 1
        assert "Fix auth middleware bug" in blocks[0].content
        assert "📝" in blocks[0].content

    def test_title_changed_sets_custom_title(self, tmp_path):
        """session.title_changed should override custom_title on the ChatSession."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Help me"}},
            {"type": "assistant.message", "data": {"content": "Sure."}},
            {"type": "session.title_changed", "data": {"title": "Important Session Title"}},
        )
        assert session is not None
        assert session.custom_title == "Important Session Title"

    def test_title_changed_last_wins(self, tmp_path):
        """If multiple session.title_changed events, the last one wins for custom_title."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Help"}},
            {"type": "assistant.message", "data": {"content": "Sure."}},
            {"type": "session.title_changed", "data": {"title": "First Title"}},
            {"type": "session.title_changed", "data": {"title": "Second Title"}},
        )
        assert session is not None
        assert session.custom_title == "Second Title"
        # Both should render as status blocks
        blocks = self._find_status_blocks(session, "title-changed")
        assert len(blocks) == 2

    def test_title_changed_overrides_workspace_yaml(self, tmp_path):
        """session.title_changed should take priority over workspace.yaml summary."""
        import ssrjson

        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        session_dir = tmp_path / "test-session"
        session_dir.mkdir()
        (session_dir / "workspace.yaml").write_text(
            "id: test-id\ncwd: /home/user/project\nsummary: Workspace YAML Title\n",
            encoding="utf-8",
        )
        events = [
            {"type": "session.start", "data": {"sessionId": "test-session", "startTime": "2026-01-01T00:00:00Z"}},
            {"type": "user.message", "data": {"content": "Help"}},
            {"type": "assistant.message", "data": {"content": "Sure."}},
            {"type": "session.title_changed", "data": {"title": "Event Title Wins"}},
        ]
        (session_dir / "events.jsonl").write_text("\n".join(ssrjson.dumps(e) for e in events), encoding="utf-8")
        session = _parse_cli_jsonl_file(session_dir / "events.jsonl")
        assert session is not None
        assert session.custom_title == "Event Title Wins"

    def test_title_changed_empty_title_ignored(self, tmp_path):
        """session.title_changed with empty title should not create a status block."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Help"}},
            {"type": "assistant.message", "data": {"content": "Sure."}},
            {"type": "session.title_changed", "data": {"title": ""}},
        )
        blocks = self._find_status_blocks(session, "title-changed")
        assert len(blocks) == 0

    def test_title_changed_missing_title_ignored(self, tmp_path):
        """session.title_changed with missing title field should not crash."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Help"}},
            {"type": "assistant.message", "data": {"content": "Sure."}},
            {"type": "session.title_changed", "data": {}},
        )
        blocks = self._find_status_blocks(session, "title-changed")
        assert len(blocks) == 0

    # ---- T6: assistant.usage ----

    def test_assistant_usage_renders_status_block(self, tmp_path):
        """assistant.usage should render as a status block with token stats."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Help"}},
            {"type": "assistant.message", "data": {"content": "Sure."}},
            {
                "type": "assistant.usage",
                "data": {
                    "model": "claude-sonnet-4.5",
                    "inputTokens": 1234,
                    "outputTokens": 567,
                    "cost": 2,
                    "duration": 3200,
                },
            },
        )
        blocks = self._find_status_blocks(session, "usage")
        assert len(blocks) == 1
        assert "claude-sonnet-4.5" in blocks[0].content
        assert "1,234 in" in blocks[0].content
        assert "567 out" in blocks[0].content
        assert "cost 2" in blocks[0].content
        assert "3.2s" in blocks[0].content
        assert "📊" in blocks[0].content

    def test_assistant_usage_with_reasoning_effort(self, tmp_path):
        """assistant.usage with reasoningEffort should include effort in display."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Help"}},
            {"type": "assistant.message", "data": {"content": "Thinking..."}},
            {
                "type": "assistant.usage",
                "data": {
                    "model": "claude-opus-4.6",
                    "inputTokens": 5000,
                    "outputTokens": 2000,
                    "reasoningEffort": "high",
                },
            },
        )
        blocks = self._find_status_blocks(session, "usage")
        assert len(blocks) == 1
        assert "effort=high" in blocks[0].content

    def test_assistant_usage_minimal_data(self, tmp_path):
        """assistant.usage with only model should still render."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Help"}},
            {"type": "assistant.message", "data": {"content": "Sure."}},
            {"type": "assistant.usage", "data": {"model": "gpt-5"}},
        )
        blocks = self._find_status_blocks(session, "usage")
        assert len(blocks) == 1
        assert "gpt-5" in blocks[0].content

    def test_assistant_usage_empty_data(self, tmp_path):
        """assistant.usage with empty data should not render."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Help"}},
            {"type": "assistant.message", "data": {"content": "Sure."}},
            {"type": "assistant.usage", "data": {}},
        )
        blocks = self._find_status_blocks(session, "usage")
        assert len(blocks) == 0

    def test_assistant_usage_null_tokens(self, tmp_path):
        """assistant.usage with null token fields should not crash."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Help"}},
            {"type": "assistant.message", "data": {"content": "Sure."}},
            {
                "type": "assistant.usage",
                "data": {
                    "model": "gpt-5",
                    "inputTokens": None,
                    "outputTokens": 100,
                },
            },
        )
        blocks = self._find_status_blocks(session, "usage")
        assert len(blocks) == 1
        assert "100 out" in blocks[0].content

    # ---- T4: session.start context fields ----

    def test_session_start_host_type_github(self, tmp_path):
        """session.start with hostType=github should produce a github repo URL."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Hello"}},
            {"type": "assistant.message", "data": {"content": "Hi"}},
            start_data={
                "context": {"cwd": "/home/user/project", "hostType": "github", "repository": "owner/repo"},
            },
        )
        assert session is not None
        assert session.repository_url == "github.com/owner/repo"
        assert session.context_entries[0].repository_url == "github.com/owner/repo"

    def test_session_start_host_type_ado(self, tmp_path):
        """session.start with hostType=ado should NOT generate any repository URL."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Hello"}},
            {"type": "assistant.message", "data": {"content": "Hi"}},
            start_data={
                "context": {"cwd": "/home/user/project", "hostType": "ado", "repository": "org/project/repo"},
            },
        )
        assert session is not None
        # ADO repos should not get any URL — no GitHub URL and no detect_repository_url fallback
        assert session.repository_url is None

    def test_session_start_without_host_type(self, tmp_path):
        """session.start without hostType should default to github URL as before."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Hello"}},
            {"type": "assistant.message", "data": {"content": "Hi"}},
            start_data={
                "context": {"cwd": "/home/user/project", "repository": "owner/repo"},
            },
        )
        assert session is not None
        assert session.repository_url == "github.com/owner/repo"

    def test_session_start_repository_matches_detected_context_format(self, tmp_path):
        """Initial GitHub metadata uses the same normalized form as detected context changes."""
        import subprocess

        workspace = tmp_path / "repo"
        workspace.mkdir()
        subprocess.run(["git", "init"], cwd=workspace, capture_output=True, check=True)  # noqa: S607
        subprocess.run(
            ["git", "remote", "add", "origin", "https://github.com/owner/repo.git"],  # noqa: S607
            cwd=workspace,
            capture_output=True,
            check=True,
        )

        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Hello"}},
            {"type": "session.context_changed", "data": {"cwd": str(workspace), "branch": "main"}},
            {"type": "assistant.message", "data": {"content": "Hi"}},
            start_data={
                "context": {"cwd": "/home/user/project", "hostType": "github", "repository": "owner/repo"},
            },
        )

        assert session is not None
        assert session.repository_url == "github.com/owner/repo"
        assert [entry.repository_url for entry in session.context_entries] == [
            "github.com/owner/repo",
            "github.com/owner/repo",
        ]

    # ---- T5: intentionSummary and toolTitle on toolRequests ----

    def test_intention_summary_used_in_tool_display(self, tmp_path):
        """intentionSummary from toolRequests should appear in the tool invocation display."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Read the file"}},
            {
                "type": "assistant.message",
                "data": {
                    "content": "",
                    "toolRequests": [
                        {
                            "toolCallId": "tc1",
                            "name": "view",
                            "arguments": {"path": "/home/user/project/src/auth.ts"},
                            "intentionSummary": "view the auth middleware implementation",
                        },
                    ],
                },
            },
            {
                "type": "tool.execution_start",
                "data": {"toolCallId": "tc1", "toolName": "view", "arguments": {"path": "/home/user/project/src/auth.ts"}},
            },
            {
                "type": "tool.execution_complete",
                "data": {"toolCallId": "tc1", "success": True, "result": {"content": "file contents"}},
            },
        )
        assert session is not None
        # Find the tool invocation content block
        tool_blocks = []
        for msg in session.messages:
            for cb in msg.content_blocks:
                if cb.kind == "toolInvocation":
                    tool_blocks.append(cb)
        assert len(tool_blocks) >= 1
        # The intentionSummary should be used as the description
        found = any("view the auth middleware" in (b.description or "") for b in tool_blocks)
        assert found, f"intentionSummary should appear in tool block description. Got: {[(b.content, b.description) for b in tool_blocks]}"

    def test_intention_summary_null_ignored(self, tmp_path):
        """intentionSummary=null should not break or override normal display."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Read"}},
            {
                "type": "assistant.message",
                "data": {
                    "content": "",
                    "toolRequests": [
                        {
                            "toolCallId": "tc1",
                            "name": "view",
                            "arguments": {"path": "/src/file.py"},
                            "intentionSummary": None,
                        },
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc1", "toolName": "view", "arguments": {"path": "/src/file.py"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc1", "success": True, "result": {"content": "ok"}}},
        )
        assert session is not None
        # Should still render the tool, just without intentionSummary
        tool_blocks = [cb for msg in session.messages for cb in msg.content_blocks if cb.kind == "toolInvocation"]
        assert len(tool_blocks) >= 1

    def test_tool_title_used_as_description_fallback(self, tmp_path):
        """toolTitle from toolRequests should be used as description when intentionSummary is absent."""
        session = self._parse(
            tmp_path,
            {"type": "user.message", "data": {"content": "Search"}},
            {
                "type": "assistant.message",
                "data": {
                    "content": "",
                    "toolRequests": [
                        {
                            "toolCallId": "tc1",
                            "name": "grep",
                            "arguments": {"pattern": "TODO"},
                            "toolTitle": "Search Code",
                        },
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc1", "toolName": "grep", "arguments": {"pattern": "TODO"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc1", "success": True, "result": {"content": "3 matches"}}},
        )
        assert session is not None
        tool_blocks = [cb for msg in session.messages for cb in msg.content_blocks if cb.kind == "toolInvocation"]
        assert len(tool_blocks) >= 1
        # toolTitle should appear somewhere in the display
        found = any("Search Code" in (b.description or "") or "Search Code" in b.content for b in tool_blocks)
        assert found, f"toolTitle should appear in tool display. Got: {[(b.content, b.description) for b in tool_blocks]}"


class TestCLIBackgroundAgentFields:
    """Tests for background agent rendering fields: prompt, is_background, agent_id, backlinks."""

    @staticmethod
    def _make_events_jsonl(*events):
        import ssrjson

        lines = [ssrjson.dumps({"type": "session.start", "data": {"sessionId": "bg-agent-test", "startTime": "2026-01-01T00:00:00Z"}})]
        for evt in events:
            lines.append(ssrjson.dumps(evt))
        return "\n".join(lines)

    def _parse(self, tmp_path, *events):
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        f = tmp_path / "events.jsonl"
        f.write_text(self._make_events_jsonl(*events), encoding="utf-8")
        return _parse_cli_jsonl_file(f)

    def _bg_agent_events(self):
        """Return events for a session with one background agent."""
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
                                "description": "Check auth module",
                                "prompt": "Review the auth module for security issues.",
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
                        "description": "Check auth module",
                        "prompt": "Review the auth module for security issues.",
                    },
                },
            },
            {
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": "tc-bg",
                    "success": True,
                    "result": {
                        "content": "Agent started in background with agent_id: agent-0. You can use read_agent tool with this agent_id to check status and retrieve results.",
                    },
                },
            },
            {"type": "subagent.started", "data": {"toolCallId": "tc-bg", "agentDisplayName": "Explore Agent", "agentName": "explore"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc-bg", "agentDisplayName": "Explore Agent"}},
            {
                "type": "assistant.message",
                "data": {
                    "content": "Let me check results.",
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
                        "content": "Status: idle\n\nResult:\nFound 2 security issues in the auth module.",
                        "detailedContent": "Found 2 security issues: hardcoded secret, no rate limiting.",
                    },
                },
            },
            {"type": "assistant.message", "data": {"content": "Review complete."}},
        ]

    def test_background_agent_has_prompt(self, tmp_path):
        """Background agent subagent block should have prompt text from task args."""
        session = self._parse(tmp_path, *self._bg_agent_events())
        subagent_blocks = [cb for msg in session.messages for cb in msg.content_blocks if cb.kind == "subagent"]
        assert len(subagent_blocks) == 1
        assert "security issues" in subagent_blocks[0].prompt

    def test_background_agent_is_background_flag(self, tmp_path):
        """Background agent should have is_background=True."""
        session = self._parse(tmp_path, *self._bg_agent_events())
        subagent_blocks = [cb for msg in session.messages for cb in msg.content_blocks if cb.kind == "subagent"]
        assert subagent_blocks[0].is_background is True

    def test_background_agent_has_agent_id(self, tmp_path):
        """Background agent should have agent_id from placeholder text."""
        session = self._parse(tmp_path, *self._bg_agent_events())
        subagent_blocks = [cb for msg in session.messages for cb in msg.content_blocks if cb.kind == "subagent"]
        assert subagent_blocks[0].agent_id == "agent-0"

    def test_background_agent_result_populated(self, tmp_path):
        """Background agent result should come from read_agent, not placeholder."""
        session = self._parse(tmp_path, *self._bg_agent_events())
        subagent_blocks = [cb for msg in session.messages for cb in msg.content_blocks if cb.kind == "subagent"]
        assert "Agent started in background" not in subagent_blocks[0].content
        assert "security issues" in subagent_blocks[0].content

    def test_read_agent_marked_as_backlink(self, tmp_path):
        """read_agent tool invocations should be marked as backlinks."""
        session = self._parse(tmp_path, *self._bg_agent_events())
        read_tools = [t for msg in session.messages for t in msg.tool_invocations if t.name == "read_agent"]
        assert len(read_tools) == 1
        assert read_tools[0].is_agent_backlink is True
        assert read_tools[0].backlink_agent_id == "agent-0"

    def test_sync_agent_not_background(self, tmp_path):
        """Sync agents should have is_background=False."""
        events = [
            {
                "type": "assistant.message",
                "data": {
                    "content": "Using sync agent.",
                    "toolRequests": [
                        {
                            "toolCallId": "tc-sync",
                            "name": "task",
                            "arguments": {
                                "agent_type": "explore",
                                "description": "Quick check",
                                "prompt": "Check something quickly.",
                            },
                        }
                    ],
                },
            },
            {
                "type": "tool.execution_start",
                "data": {
                    "toolCallId": "tc-sync",
                    "toolName": "task",
                    "arguments": {
                        "agent_type": "explore",
                        "description": "Quick check",
                    },
                },
            },
            {
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": "tc-sync",
                    "success": True,
                    "result": {
                        "content": "Everything looks good.",
                    },
                },
            },
            {"type": "subagent.started", "data": {"toolCallId": "tc-sync", "agentDisplayName": "Explore Agent"}},
            {"type": "subagent.completed", "data": {"toolCallId": "tc-sync", "agentDisplayName": "Explore Agent"}},
            {"type": "assistant.message", "data": {"content": "Done."}},
        ]
        session = self._parse(tmp_path, *events)
        subagent_blocks = [cb for msg in session.messages for cb in msg.content_blocks if cb.kind == "subagent"]
        assert len(subagent_blocks) == 1
        assert subagent_blocks[0].is_background is False
        assert subagent_blocks[0].prompt == "Check something quickly."


class TestCLIAsyncShellFields:
    """Tests for async shell rendering fields: shell_id, is_async, is_detached, shell backlinks."""

    @staticmethod
    def _make_events_jsonl(*events):
        import ssrjson

        lines = [ssrjson.dumps({"type": "session.start", "data": {"sessionId": "async-shell-test", "startTime": "2026-01-01T00:00:00Z"}})]
        for evt in events:
            lines.append(ssrjson.dumps(evt))
        return "\n".join(lines)

    def _parse(self, tmp_path, *events):
        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        f = tmp_path / "events.jsonl"
        f.write_text(self._make_events_jsonl(*events), encoding="utf-8")
        return _parse_cli_jsonl_file(f)

    def _async_shell_events(self):
        """Return events for a session with an async shell + read/write/stop interactions."""
        return [
            {
                "type": "assistant.message",
                "data": {
                    "content": "Starting dev server.",
                    "toolRequests": [
                        {
                            "toolCallId": "tc-shell",
                            "name": "powershell",
                            "arguments": {
                                "command": "npm run dev",
                                "description": "Start dev server",
                                "mode": "async",
                                "shellId": "dev-server",
                            },
                        }
                    ],
                },
            },
            {
                "type": "tool.execution_start",
                "data": {
                    "toolCallId": "tc-shell",
                    "toolName": "powershell",
                    "arguments": {
                        "command": "npm run dev",
                        "description": "Start dev server",
                        "mode": "async",
                        "shellId": "dev-server",
                    },
                },
            },
            {
                "type": "tool.execution_complete",
                "data": {
                    "toolCallId": "tc-shell",
                    "success": True,
                    "result": {"content": "Dev server started on port 3000"},
                },
            },
            {
                "type": "assistant.message",
                "data": {
                    "content": "Server running. Let me check the output.",
                    "toolRequests": [
                        {"toolCallId": "tc-read", "name": "read_powershell", "arguments": {"shellId": "dev-server", "delay": 5}},
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc-read", "toolName": "read_powershell", "arguments": {"shellId": "dev-server"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-read", "success": True, "result": {"content": "Listening on http://localhost:3000"}}},
            {
                "type": "assistant.message",
                "data": {
                    "content": "Need to send a command.",
                    "toolRequests": [
                        {"toolCallId": "tc-write", "name": "write_powershell", "arguments": {"shellId": "dev-server", "input": "rs{enter}"}},
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc-write", "toolName": "write_powershell", "arguments": {"shellId": "dev-server", "input": "rs{enter}"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-write", "success": True, "result": {"content": "Restarted"}}},
            {
                "type": "assistant.message",
                "data": {
                    "content": "Stopping the server.",
                    "toolRequests": [
                        {"toolCallId": "tc-stop", "name": "stop_powershell", "arguments": {"shellId": "dev-server"}},
                    ],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc-stop", "toolName": "stop_powershell", "arguments": {"shellId": "dev-server"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-stop", "success": True, "result": {"content": "Shell stopped"}}},
            {"type": "assistant.message", "data": {"content": "Done."}},
        ]

    def test_async_powershell_detected(self, tmp_path):
        """powershell with mode=async should produce CommandRun with is_async=True."""
        session = self._parse(tmp_path, *self._async_shell_events())
        cmds = [c for msg in session.messages for c in msg.command_runs]
        assert len(cmds) == 1
        assert cmds[0].is_async is True
        assert cmds[0].shell_id == "dev-server"
        assert cmds[0].command == "npm run dev"

    def test_sync_powershell_unchanged(self, tmp_path):
        """powershell without mode should produce CommandRun with is_async=False."""
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
            {"type": "assistant.message", "data": {"content": "Clean tree."}},
        ]
        session = self._parse(tmp_path, *events)
        cmds = [c for msg in session.messages for c in msg.command_runs]
        assert len(cmds) == 1
        assert cmds[0].is_async is False
        assert cmds[0].shell_id is None

    def test_detached_powershell_detected(self, tmp_path):
        """powershell with detach=true should produce CommandRun with is_detached=True."""
        events = [
            {
                "type": "assistant.message",
                "data": {
                    "content": "Starting background server.",
                    "toolRequests": [
                        {
                            "toolCallId": "tc-det",
                            "name": "powershell",
                            "arguments": {"command": "python -m http.server", "mode": "async", "detach": True, "shellId": "http-srv"},
                        }
                    ],
                },
            },
            {
                "type": "tool.execution_start",
                "data": {
                    "toolCallId": "tc-det",
                    "toolName": "powershell",
                    "arguments": {"command": "python -m http.server", "mode": "async", "detach": True, "shellId": "http-srv"},
                },
            },
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-det", "success": True, "result": {"content": "Server started"}}},
            {"type": "assistant.message", "data": {"content": "Server running in background."}},
        ]
        session = self._parse(tmp_path, *events)
        cmds = [c for msg in session.messages for c in msg.command_runs]
        assert len(cmds) == 1
        assert cmds[0].is_async is True
        assert cmds[0].is_detached is True
        assert cmds[0].shell_id == "http-srv"

    def test_read_powershell_not_skipped(self, tmp_path):
        """read_powershell should no longer be skipped — it renders as a shell backlink."""
        session = self._parse(tmp_path, *self._async_shell_events())
        # read_powershell should produce a tool invocation now (not be filtered)
        read_tools = [t for msg in session.messages for t in msg.tool_invocations if t.name == "read_powershell"]
        assert len(read_tools) == 1

    def test_read_powershell_marked_as_shell_backlink(self, tmp_path):
        """read_powershell should be marked as a shell backlink."""
        session = self._parse(tmp_path, *self._async_shell_events())
        read_tools = [t for msg in session.messages for t in msg.tool_invocations if t.name == "read_powershell"]
        assert read_tools[0].is_shell_backlink is True
        assert read_tools[0].backlink_shell_id == "dev-server"

    def test_write_powershell_marked_as_shell_backlink(self, tmp_path):
        """write_powershell should be marked as a shell backlink."""
        session = self._parse(tmp_path, *self._async_shell_events())
        write_tools = [t for msg in session.messages for t in msg.tool_invocations if t.name == "write_powershell"]
        assert len(write_tools) == 1
        assert write_tools[0].is_shell_backlink is True
        assert write_tools[0].backlink_shell_id == "dev-server"

    def test_stop_powershell_marked_as_shell_backlink(self, tmp_path):
        """stop_powershell should be marked as a shell backlink."""
        session = self._parse(tmp_path, *self._async_shell_events())
        stop_tools = [t for msg in session.messages for t in msg.tool_invocations if t.name == "stop_powershell"]
        assert len(stop_tools) == 1
        assert stop_tools[0].is_shell_backlink is True
        assert stop_tools[0].backlink_shell_id == "dev-server"

    def test_shell_backlink_invocation_message(self, tmp_path):
        """Shell backlink should have a descriptive invocation_message using the shell title."""
        session = self._parse(tmp_path, *self._async_shell_events())
        read_tools = [t for msg in session.messages for t in msg.tool_invocations if t.name == "read_powershell"]
        assert "Start dev server" in read_tools[0].invocation_message
        assert "read" in read_tools[0].invocation_message

    def test_duplicate_shellid_scoped_temporally(self, tmp_path):
        """When the same shellId is reused, IO entries attach to the correct CommandRun."""
        events = [
            # First shell with shellId "srv"
            {
                "type": "assistant.message",
                "data": {
                    "content": "Starting first server.",
                    "toolRequests": [
                        {
                            "toolCallId": "tc-srv1",
                            "name": "powershell",
                            "arguments": {
                                "command": "npm run dev",
                                "description": "First server",
                                "mode": "async",
                                "shellId": "srv",
                            },
                        },
                    ],
                },
            },
            {
                "type": "tool.execution_start",
                "data": {
                    "toolCallId": "tc-srv1",
                    "toolName": "powershell",
                    "arguments": {"command": "npm run dev", "mode": "async", "shellId": "srv"},
                },
            },
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-srv1", "success": True, "result": {"content": "started"}}},
            # Write to first shell (same assistant message — no user break)
            {
                "type": "assistant.message",
                "data": {
                    "content": "Sending input to first.",
                    "toolRequests": [{"toolCallId": "tc-w1", "name": "write_powershell", "arguments": {"shellId": "srv", "input": "first-input"}}],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc-w1", "toolName": "write_powershell", "arguments": {"shellId": "srv", "input": "first-input"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-w1", "success": True, "result": {"content": "first-response"}}},
            # Stop first shell
            {
                "type": "assistant.message",
                "data": {
                    "content": "Stopping first.",
                    "toolRequests": [{"toolCallId": "tc-s1", "name": "stop_powershell", "arguments": {"shellId": "srv"}}],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc-s1", "toolName": "stop_powershell", "arguments": {"shellId": "srv"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-s1", "success": True, "result": {"content": "stopped"}}},
            # User message separates the two shell runs
            {"type": "user.message", "data": {"content": "Start another one"}},
            # Second shell reusing shellId "srv"
            {
                "type": "assistant.message",
                "data": {
                    "content": "Starting second server.",
                    "toolRequests": [
                        {
                            "toolCallId": "tc-srv2",
                            "name": "powershell",
                            "arguments": {
                                "command": "npm run dev:hot",
                                "description": "Second server",
                                "mode": "async",
                                "shellId": "srv",
                            },
                        },
                    ],
                },
            },
            {
                "type": "tool.execution_start",
                "data": {
                    "toolCallId": "tc-srv2",
                    "toolName": "powershell",
                    "arguments": {"command": "npm run dev:hot", "mode": "async", "shellId": "srv"},
                },
            },
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-srv2", "success": True, "result": {"content": "started"}}},
            # Write to second shell
            {
                "type": "assistant.message",
                "data": {
                    "content": "Sending input to second.",
                    "toolRequests": [{"toolCallId": "tc-w2", "name": "write_powershell", "arguments": {"shellId": "srv", "input": "second-input"}}],
                },
            },
            {"type": "tool.execution_start", "data": {"toolCallId": "tc-w2", "toolName": "write_powershell", "arguments": {"shellId": "srv", "input": "second-input"}}},
            {"type": "tool.execution_complete", "data": {"toolCallId": "tc-w2", "success": True, "result": {"content": "second-response"}}},
            {"type": "assistant.message", "data": {"content": "Done."}},
        ]
        session = self._parse(tmp_path, *events)
        async_cmds = [c for msg in session.messages for c in msg.command_runs if c.is_async and c.shell_id == "srv"]
        assert len(async_cmds) == 2, f"Expected 2 async shells with shellId 'srv', got {len(async_cmds)}"
        first_cmd, second_cmd = async_cmds
        # First shell should have write + stop IO entries
        assert len(first_cmd.io_entries) == 2, f"First shell expected 2 IO entries, got {len(first_cmd.io_entries)}"
        assert first_cmd.io_entries[0].action == "write"
        assert first_cmd.io_entries[0].result == "first-response"
        assert first_cmd.io_entries[1].action == "stop"
        # Second shell should have only its own write IO entry
        assert len(second_cmd.io_entries) == 1, f"Second shell expected 1 IO entry, got {len(second_cmd.io_entries)}"
        assert second_cmd.io_entries[0].action == "write"
        assert second_cmd.io_entries[0].result == "second-response"
