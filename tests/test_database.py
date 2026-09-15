"""Tests for the database module."""

import json
import sqlite3
import tempfile
from pathlib import Path

import pytest

from copilot_session_tools import ChatMessage, ChatSession, CommandRun, ContentBlock, Database, FileChange, ToolInvocation, parse_search_query
from copilot_session_tools.scanner import RootAgentInterval, SessionContextEntry


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = Database(db_path)
    yield db
    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def sample_session():
    """Create a sample chat session for testing."""
    return ChatSession(
        session_id="test-session-123",
        workspace_name="my-project",
        workspace_path="/home/user/projects/my-project",
        messages=[
            ChatMessage(role="user", content="How do I create a Python function?"),
            ChatMessage(
                role="assistant",
                content="Here's how to create a Python function:\n\n```python\ndef my_function():\n    pass\n```",
            ),
            ChatMessage(role="user", content="Thanks! Can you add parameters?"),
            ChatMessage(
                role="assistant",
                content=("Sure! Here's a function with parameters:\n\n```python\ndef my_function(name, age=18):\n    return f'{name} is {age} years old'\n```"),
            ),
        ],
        created_at="2025-01-15T10:30:00Z",
        updated_at="2025-01-15T10:35:00Z",
        source_file="/path/to/session.json",
        vscode_edition="stable",
    )


class TestDatabase:
    """Tests for the Database class."""

    def test_create_database(self, temp_db):
        """Test that database is created with correct schema."""
        assert temp_db.db_path.exists()
        stats = temp_db.get_stats()
        assert stats["session_count"] == 0
        assert stats["message_count"] == 0

    def test_add_session(self, temp_db, sample_session):
        """Test adding a session to the database."""
        result = temp_db.add_session(sample_session)
        assert result is True

        stats = temp_db.get_stats()
        assert stats["session_count"] == 1
        assert stats["message_count"] == 4

    def test_add_duplicate_session(self, temp_db, sample_session):
        """Test that adding a duplicate session returns False."""
        temp_db.add_session(sample_session)
        result = temp_db.add_session(sample_session)
        assert result is False

        stats = temp_db.get_stats()
        assert stats["session_count"] == 1

    def test_get_session(self, temp_db, sample_session):
        """Test retrieving a session from the database."""
        temp_db.add_session(sample_session)
        retrieved = temp_db.get_session(sample_session.session_id)

        assert retrieved is not None
        assert retrieved.session_id == sample_session.session_id
        assert retrieved.workspace_name == sample_session.workspace_name
        assert len(retrieved.messages) == len(sample_session.messages)
        assert retrieved.messages[0].role == "user"
        assert "Python function" in retrieved.messages[0].content

    def test_session_context_entries_persist_and_retrieve(self, temp_db):
        """Test workspace/repository context history persists with sessions."""
        session = ChatSession(
            session_id="context-session",
            workspace_name="project-one",
            workspace_path="/work/project-one",
            repository_url="github.com/owner/one",
            messages=[
                ChatMessage(role="user", content="Initial repo question"),
                ChatMessage(role="assistant", content="Switching context"),
                ChatMessage(role="user", content="Later repo question"),
            ],
            context_entries=[
                SessionContextEntry(
                    workspace_name="project-one",
                    workspace_path="/work/project-one",
                    repository_url="github.com/owner/one",
                    message_index=0,
                ),
                SessionContextEntry(
                    workspace_name="project-two",
                    workspace_path="/work/project-two",
                    repository_url="github.com/owner/two",
                    branch="main",
                    message_index=2,
                    source="context_changed",
                ),
            ],
        )

        temp_db.add_session(session)

        retrieved = temp_db.get_session("context-session")
        assert retrieved is not None
        contexts = retrieved.effective_context_entries()
        assert [c.workspace_name for c in contexts] == ["project-one", "project-two"]
        assert contexts[1].repository_url == "github.com/owner/two"
        assert contexts[1].message_index == 2

        listed = temp_db.list_sessions(workspace_name="project-two")
        assert [s["session_id"] for s in listed] == ["context-session"]
        assert listed[0]["workspace_names"] == ["project-one", "project-two"]
        assert listed[0]["repository_urls"] == ["github.com/owner/one", "github.com/owner/two"]

    def test_root_agent_intervals_persist_and_retrieve(self, temp_db):
        """Test root custom-agent intervals persist and hydrate with sessions."""
        session = ChatSession(
            session_id="root-agent-session",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="Merge main", timestamp="2026-01-01T00:00:02Z"),
                ChatMessage(
                    role="assistant",
                    content="Starting merge.",
                    timestamp="2026-01-01T00:00:03Z",
                    tool_invocations=[
                        ToolInvocation(
                            name="report_intent",
                            input='{"intent": "Merging upstream"}',
                            result="ok",
                            status="success",
                        )
                    ],
                ),
                ChatMessage(role="assistant", content="Back to default.", timestamp="2026-01-01T00:00:05Z"),
            ],
            root_agent_intervals=[
                RootAgentInterval(
                    agent_name="smart-merge",
                    agent_display_name="Smart Merge",
                    start_timestamp="2026-01-01T00:00:01Z",
                    end_timestamp="2026-01-01T00:00:04Z",
                    tools=["report_intent", "powershell"],
                )
            ],
        )

        temp_db.add_session(session)

        retrieved = temp_db.get_session("root-agent-session")
        assert retrieved is not None
        assert retrieved.root_agent_intervals == session.root_agent_intervals

        with temp_db._get_connection() as conn:
            row = conn.execute(
                "SELECT agent_name, tools_json FROM cst_root_agent_intervals WHERE session_id = ?",
                ("root-agent-session",),
            ).fetchone()
            assert row["agent_name"] == "smart-merge"
            assert json.loads(row["tools_json"]) == ["report_intent", "powershell"]

            messages = conn.execute(
                "SELECT content FROM cst_root_agent_messages WHERE session_id = ? ORDER BY message_index",
                ("root-agent-session",),
            ).fetchall()
            assert [row["content"] for row in messages] == ["Merge main", "Starting merge."]

            usage = conn.execute("SELECT * FROM cst_root_agent_usage WHERE agent_name = ?", ("smart-merge",)).fetchone()
            assert usage["run_count"] == 1
            assert usage["session_count"] == 1
            assert usage["message_count"] == 2
            assert usage["tool_count"] == 1

    def test_root_agent_views_include_nested_subagent_activity(self, temp_db):
        """Root-agent usage includes child subagent work inside the interval."""
        session = ChatSession(
            session_id="root-agent-nested-session",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[
                ChatMessage(
                    role="assistant",
                    content="Launching review.",
                    timestamp="2026-01-01T00:00:02Z",
                    tool_invocations=[
                        ToolInvocation(
                            name="task",
                            input='{"agent_type": "code-review"}',
                            result="started",
                            status="success",
                        )
                    ],
                    content_blocks=[
                        ContentBlock(
                            kind="subagent",
                            content="Review result",
                            description="code-review",
                            child_message=ChatMessage(
                                role="assistant",
                                content="Reviewed file.",
                                timestamp="2026-01-01T00:00:03Z",
                                agent_id="tc1",
                                agent_display_name="code-review",
                                agent_nesting_level=1,
                                tool_invocations=[
                                    ToolInvocation(
                                        name="view",
                                        input='{"path": "src/example.py"}',
                                        result="ok",
                                        status="success",
                                    )
                                ],
                            ),
                        )
                    ],
                )
            ],
            root_agent_intervals=[
                RootAgentInterval(
                    agent_name="smart-merge",
                    agent_display_name="Smart Merge",
                    start_timestamp="2026-01-01T00:00:01Z",
                    end_timestamp="2026-01-01T00:00:04Z",
                )
            ],
        )

        temp_db.add_session(session)

        with temp_db._get_connection() as conn:
            messages = conn.execute(
                "SELECT content FROM cst_root_agent_messages WHERE session_id = ? ORDER BY timestamp, message_id",
                ("root-agent-nested-session",),
            ).fetchall()
            tools = conn.execute(
                "SELECT name FROM cst_root_agent_tool_invocations WHERE session_id = ? ORDER BY timestamp, tool_id",
                ("root-agent-nested-session",),
            ).fetchall()
            usage = conn.execute("SELECT * FROM cst_root_agent_usage WHERE agent_name = ?", ("smart-merge",)).fetchone()

        assert [row["content"] for row in messages] == ["Launching review.", "Reviewed file."]
        assert [row["name"] for row in tools] == ["task", "view"]
        assert usage["message_count"] == 2
        assert usage["tool_count"] == 2

    def test_root_agent_intervals_are_deleted_with_session_refresh(self, temp_db):
        """Replacing a session clears stale root-agent intervals."""
        session = ChatSession(
            session_id="refresh-root-agent-session",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[ChatMessage(role="user", content="Initial", timestamp="2026-01-01T00:00:02Z")],
            root_agent_intervals=[
                RootAgentInterval(
                    agent_name="smart-merge",
                    agent_display_name="Smart Merge",
                    start_timestamp="2026-01-01T00:00:01Z",
                )
            ],
        )
        temp_db.add_session(session)

        refreshed = ChatSession(
            session_id="refresh-root-agent-session",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[ChatMessage(role="user", content="Refreshed", timestamp="2026-01-01T00:01:02Z")],
        )
        temp_db.update_session(refreshed)

        retrieved = temp_db.get_session("refresh-root-agent-session")
        assert retrieved is not None
        assert retrieved.root_agent_intervals == []

    def test_root_agent_intervals_persist_through_enrichment_path(self, temp_db):
        """Test scanner enrichment writes root custom-agent intervals."""
        session = ChatSession(
            session_id="enrich-root-agent-session",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[ChatMessage(role="user", content="Audit skills", timestamp="2026-01-01T00:00:02Z")],
            root_agent_intervals=[
                RootAgentInterval(
                    agent_name="skill-audit",
                    agent_display_name="Skill Audit",
                    start_timestamp="2026-01-01T00:00:01Z",
                )
            ],
        )

        temp_db.enrich_session(session)

        with temp_db._get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM cst_root_agent_intervals WHERE session_id = ?",
                ("enrich-root-agent-session",),
            ).fetchone()[0]
            assert count == 1

    def test_get_nonexistent_session(self, temp_db):
        """Test that getting a nonexistent session returns None."""
        result = temp_db.get_session("nonexistent-id")
        assert result is None

    def test_list_sessions(self, temp_db, sample_session):
        """Test listing sessions."""
        temp_db.add_session(sample_session)

        # Add another session
        session2 = ChatSession(
            session_id="test-session-456",
            workspace_name="other-project",
            workspace_path="/home/user/projects/other",
            messages=[ChatMessage(role="user", content="Hello")],
            vscode_edition="insider",
        )
        temp_db.add_session(session2)

        sessions = temp_db.list_sessions()
        assert len(sessions) == 2

    def test_list_sessions_limit_uses_final_updated_at_order(self, temp_db):
        temp_db.add_session(
            ChatSession(
                session_id="newest-update",
                workspace_name="newest",
                workspace_path="/newest",
                messages=[ChatMessage(role="user", content="newest", timestamp="2026-01-01T00:00:00Z")],
                created_at="2026-01-01T00:00:00Z",
                updated_at="2026-09-15T12:00:00Z",
                vscode_edition="stable",
            )
        )
        temp_db.add_session(
            ChatSession(
                session_id="newest-message",
                workspace_name="older",
                workspace_path="/older",
                messages=[ChatMessage(role="user", content="older", timestamp="2026-09-15T11:00:00Z")],
                created_at="2026-01-02T00:00:00Z",
                updated_at="2026-09-14T12:00:00Z",
                vscode_edition="stable",
            )
        )

        assert temp_db.list_sessions(limit=1) == temp_db.list_sessions()[:1]
        assert temp_db.list_sessions(limit=1)[0]["session_id"] == "newest-update"

    def test_list_sessions_limit_does_not_count_empty_chronicle_sessions(self, tmp_path):
        chronicle_path = str(tmp_path / "session-store.db")
        _create_builtin_only_db(chronicle_path)
        _insert_builtin_session(
            chronicle_path,
            "empty-newest",
            summary="Empty",
            updated_at="2026-09-15T12:00:00Z",
        )
        _insert_builtin_session(
            chronicle_path,
            "nonempty-older",
            summary="Nonempty",
            updated_at="2026-09-14T12:00:00Z",
        )
        _insert_builtin_turn(chronicle_path, "nonempty-older", 0, "question", "answer")
        database = Database(tmp_path / "archive.db")

        sessions = database.list_sessions(limit=1)

        assert [session["session_id"] for session in sessions] == ["nonempty-older"]

    def test_list_sessions_filter_by_workspace(self, temp_db, sample_session):
        """Test filtering sessions by workspace."""
        temp_db.add_session(sample_session)

        session2 = ChatSession(
            session_id="test-session-456",
            workspace_name="other-project",
            workspace_path="/home/user/projects/other",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        temp_db.add_session(session2)

        sessions = temp_db.list_sessions(workspace_name="my-project")
        assert len(sessions) == 1
        assert sessions[0]["workspace_name"] == "my-project"

    def test_search_messages(self, temp_db, sample_session):
        """Test full-text search."""
        temp_db.add_session(sample_session)

        results = temp_db.search("Python function")
        assert len(results) > 0
        assert any("Python" in r["content"] for r in results)

    def test_search_deduplicates_copied_fork_history_by_source_event_id(self, temp_db):
        """Test search suppresses copied fork history while keeping new fork content."""
        original = ChatSession(
            session_id="original-session",
            workspace_name="project",
            workspace_path="/tmp/project",
            created_at="2026-01-01T00:00:00Z",
            messages=[
                ChatMessage(
                    role="user",
                    content="shared fork setup unique text",
                    source_event_id="evt-shared-user",
                ),
                ChatMessage(
                    role="assistant",
                    content="original continuation unique text",
                    source_event_id="evt-original-continuation",
                ),
            ],
        )
        fork = ChatSession(
            session_id="fork-session",
            workspace_name="project-fork",
            workspace_path="/tmp/project-fork",
            created_at="2026-01-01T00:10:00Z",
            messages=[
                ChatMessage(
                    role="user",
                    content="shared fork setup unique text",
                    source_event_id="evt-shared-user",
                ),
                ChatMessage(
                    role="assistant",
                    content="fork continuation unique text",
                    source_event_id="evt-fork-continuation",
                ),
            ],
        )
        temp_db.add_session(original)
        temp_db.add_session(fork)

        shared_results = temp_db.search('"shared fork setup unique text"', search_content_set={"messages"})
        assert [result["session_id"] for result in shared_results] == ["original-session"]

        fork_results = temp_db.search('"fork continuation unique text"', search_content_set={"messages"})
        assert [result["session_id"] for result in fork_results] == ["fork-session"]

        original_results = temp_db.search('"original continuation unique text"', search_content_set={"messages"})
        assert [result["session_id"] for result in original_results] == ["original-session"]

    def test_search_keeps_independent_duplicate_content_with_distinct_event_ids(self, temp_db):
        """Test identical text remains searchable when it is not copied fork history."""
        first = ChatSession(
            session_id="first-independent-session",
            workspace_name="project-a",
            workspace_path="/tmp/project-a",
            created_at="2026-01-01T00:00:00Z",
            messages=[
                ChatMessage(
                    role="user",
                    content="independent duplicate text",
                    source_event_id="evt-first-independent",
                )
            ],
        )
        second = ChatSession(
            session_id="second-independent-session",
            workspace_name="project-b",
            workspace_path="/tmp/project-b",
            created_at="2026-01-01T00:10:00Z",
            messages=[
                ChatMessage(
                    role="user",
                    content="independent duplicate text",
                    source_event_id="evt-second-independent",
                )
            ],
        )
        temp_db.add_session(first)
        temp_db.add_session(second)

        results = temp_db.search('"independent duplicate text"', search_content_set={"messages"}, sort_by="date")

        assert {result["session_id"] for result in results} == {
            "first-independent-session",
            "second-independent-session",
        }

    def test_search_keeps_divergent_content_with_shared_source_event_id(self, temp_db):
        """Test a fork continuation is searchable when only part of an assistant row diverged."""
        original = ChatSession(
            session_id="original-divergent-session",
            workspace_name="project",
            workspace_path="/tmp/project",
            created_at="2026-01-01T00:00:00Z",
            messages=[
                ChatMessage(
                    role="assistant",
                    content="shared assistant content",
                    source_event_id="evt-shared-assistant",
                )
            ],
        )
        fork = ChatSession(
            session_id="fork-divergent-session",
            workspace_name="project-fork",
            workspace_path="/tmp/project-fork",
            created_at="2026-01-01T00:10:00Z",
            messages=[
                ChatMessage(
                    role="assistant",
                    content="shared assistant content plus fork-only divergent text",
                    source_event_id="evt-shared-assistant",
                )
            ],
        )
        temp_db.add_session(original)
        temp_db.add_session(fork)

        results = temp_db.search('"fork-only divergent text"', search_content_set={"messages"})

        assert [result["session_id"] for result in results] == ["fork-divergent-session"]

    def test_search_deduplicates_copied_fork_artifacts_by_source_event_id(self, temp_db):
        """Test copied pre-fork tools, files, commands, and thinking blocks are de-duplicated."""
        shared_message = "shared assistant artifact holder"
        original = ChatSession(
            session_id="original-artifact-session",
            workspace_name="project",
            workspace_path="/tmp/project",
            created_at="2026-01-01T00:00:00Z",
            messages=[
                ChatMessage(
                    role="assistant",
                    content=shared_message,
                    source_event_id="evt-shared-artifacts",
                    tool_invocations=[ToolInvocation(name="search_tool", input="tool duplicate target", result="found copied result")],
                    file_changes=[FileChange(path="src/copied_target.py", explanation="file duplicate target", diff="+ copied diff target")],
                    command_runs=[CommandRun(command="run copied command target", output="command duplicate target")],
                    content_blocks=[
                        ContentBlock(kind="thinking", content="thinking duplicate target"),
                        ContentBlock(kind="text", content=shared_message),
                    ],
                )
            ],
        )
        fork = ChatSession(
            session_id="fork-artifact-session",
            workspace_name="project-fork",
            workspace_path="/tmp/project-fork",
            created_at="2026-01-01T00:10:00Z",
            messages=[
                ChatMessage(
                    role="assistant",
                    content=shared_message,
                    source_event_id="evt-shared-artifacts",
                    tool_invocations=[ToolInvocation(name="search_tool", input="tool duplicate target", result="found copied result")],
                    file_changes=[FileChange(path="src/copied_target.py", explanation="file duplicate target", diff="+ copied diff target")],
                    command_runs=[CommandRun(command="run copied command target", output="command duplicate target")],
                    content_blocks=[
                        ContentBlock(kind="thinking", content="thinking duplicate target"),
                        ContentBlock(kind="text", content=shared_message),
                    ],
                )
            ],
        )
        temp_db.add_session(original)
        temp_db.add_session(fork)

        cases = [
            ("tool duplicate target", {"tools", "tool-inputs"}),
            ("file duplicate target", {"file-changes"}),
            ("copied diff target", {"file-changes", "diffs"}),
            ("command duplicate target", {"commands"}),
            ("thinking duplicate target", {"thinking"}),
        ]
        for query, search_content_set in cases:
            results = temp_db.search(query, search_content_set=search_content_set)
            assert [result["session_id"] for result in results] == ["original-artifact-session"]

    def test_search_no_results(self, temp_db, sample_session):
        """Test search with no matching results."""
        temp_db.add_session(sample_session)

        results = temp_db.search("JavaScript React")
        assert len(results) == 0

    def test_search_tool_invocation_includes_message_index(self, temp_db):
        """Test that tool invocation search results include message_index."""
        session = ChatSession(
            session_id="tool-search-session",
            workspace_name="test-ws",
            workspace_path="/tmp/test",
            messages=[
                ChatMessage(role="user", content="first user message"),
                ChatMessage(
                    role="assistant",
                    content="first reply",
                ),
                ChatMessage(role="user", content="second user message"),
                ChatMessage(
                    role="assistant",
                    content="running a special tool now",
                    tool_invocations=[
                        ToolInvocation(name="grep_search", input="uniqueToolQuery999", result="found it"),
                    ],
                ),
            ],
        )
        temp_db.add_session(session)

        results = temp_db.search("uniqueToolQuery999", include_tool_calls=True)
        tool_results = [r for r in results if r["match_type"] == "tool_invocation"]
        assert len(tool_results) > 0
        for r in tool_results:
            assert "message_index" in r
            assert r["message_index"] == 3  # 0-based index of the 4th message

    def test_search_file_change_includes_message_index(self, temp_db):
        """Test that file change search results include message_index."""
        session = ChatSession(
            session_id="file-search-session",
            workspace_name="test-ws",
            workspace_path="/tmp/test",
            messages=[
                ChatMessage(role="user", content="first user message"),
                ChatMessage(
                    role="assistant",
                    content="first reply",
                ),
                ChatMessage(role="user", content="edit the config file"),
                ChatMessage(
                    role="assistant",
                    content="editing the file now",
                    file_changes=[
                        FileChange(path="src/uniqueFileTarget777.py", explanation="updated config"),
                    ],
                ),
            ],
        )
        temp_db.add_session(session)

        results = temp_db.search("uniqueFileTarget777", include_file_changes=True)
        file_results = [r for r in results if r["match_type"] == "file_change"]
        assert len(file_results) > 0
        for r in file_results:
            assert "message_index" in r
            assert r["message_index"] == 3  # 0-based index of the 4th message

    def test_get_workspaces(self, temp_db, sample_session):
        """Test getting unique workspaces."""
        temp_db.add_session(sample_session)

        session2 = ChatSession(
            session_id="test-session-456",
            workspace_name="my-project",  # Same workspace
            workspace_path="/home/user/projects/my-project",
            messages=[ChatMessage(role="user", content="Another question")],
        )
        temp_db.add_session(session2)

        workspaces = temp_db.get_workspaces()
        assert len(workspaces) == 1
        assert workspaces[0]["workspace_name"] == "my-project"
        assert workspaces[0]["session_count"] == 2

    def test_export_json(self, temp_db, sample_session):
        """Test exporting database as JSON."""
        temp_db.add_session(sample_session)

        json_str = temp_db.export_json()
        data = json.loads(json_str)

        assert isinstance(data, list)
        assert len(data) == 1
        assert data[0]["session_id"] == sample_session.session_id
        assert len(data[0]["messages"]) == 4

    def test_update_session(self, temp_db, sample_session):
        """Test updating an existing session."""
        temp_db.add_session(sample_session)

        # Modify the session
        updated_session = ChatSession(
            session_id=sample_session.session_id,
            workspace_name=sample_session.workspace_name,
            workspace_path=sample_session.workspace_path,
            messages=[
                ChatMessage(role="user", content="Updated message"),
            ],
            vscode_edition="stable",
        )

        temp_db.update_session(updated_session)

        retrieved = temp_db.get_session(sample_session.session_id)
        assert len(retrieved.messages) == 1
        assert retrieved.messages[0].content == "Updated message"


class TestNeedsUpdate:
    """Tests for the needs_update method."""

    def test_needs_update_new_session(self, temp_db):
        """Test that needs_update returns True for a new session."""
        result = temp_db.needs_update("nonexistent-session", 1234567890.0, 1024)
        assert result is True

    def test_needs_update_unchanged_session(self, temp_db):
        """Test that needs_update returns False for an unchanged session."""
        session = ChatSession(
            session_id="unchanged-session",
            workspace_name="test-workspace",
            workspace_path="/test/path",
            messages=[ChatMessage(role="user", content="Hello")],
            source_file="/test/session.json",
            source_file_mtime=1234567890.0,
            source_file_size=1024,
        )
        temp_db.add_session(session)

        result = temp_db.needs_update("unchanged-session", 1234567890.0, 1024)
        assert result is False

    def test_needs_update_modified_mtime(self, temp_db):
        """Test that needs_update returns True when mtime differs."""
        session = ChatSession(
            session_id="mtime-changed-session",
            workspace_name="test-workspace",
            workspace_path="/test/path",
            messages=[ChatMessage(role="user", content="Hello")],
            source_file="/test/session.json",
            source_file_mtime=1234567890.0,
            source_file_size=1024,
        )
        temp_db.add_session(session)

        # mtime changed
        result = temp_db.needs_update("mtime-changed-session", 1234567999.0, 1024)
        assert result is True

    def test_needs_update_modified_size(self, temp_db):
        """Test that needs_update returns True when size differs."""
        session = ChatSession(
            session_id="size-changed-session",
            workspace_name="test-workspace",
            workspace_path="/test/path",
            messages=[ChatMessage(role="user", content="Hello")],
            source_file="/test/session.json",
            source_file_mtime=1234567890.0,
            source_file_size=1024,
        )
        temp_db.add_session(session)

        # size changed
        result = temp_db.needs_update("size-changed-session", 1234567890.0, 2048)
        assert result is True

    def test_needs_update_null_stored_values(self, temp_db):
        """Test that needs_update returns True when stored values are NULL (migration case)."""
        session = ChatSession(
            session_id="null-values-session",
            workspace_name="test-workspace",
            workspace_path="/test/path",
            messages=[ChatMessage(role="user", content="Hello")],
            source_file="/test/session.json",
            # No mtime/size set (simulating migration case)
            source_file_mtime=None,
            source_file_size=None,
        )
        temp_db.add_session(session)

        result = temp_db.needs_update("null-values-session", 1234567890.0, 1024)
        assert result is True

    def test_session_stores_file_metadata(self, temp_db):
        """Test that file metadata is stored and retrieved correctly."""
        session = ChatSession(
            session_id="metadata-session",
            workspace_name="test-workspace",
            workspace_path="/test/path",
            messages=[ChatMessage(role="user", content="Hello")],
            source_file="/test/session.json",
            source_file_mtime=1234567890.123,
            source_file_size=2048,
        )
        temp_db.add_session(session)

        retrieved = temp_db.get_session("metadata-session")
        assert retrieved.source_file_mtime == 1234567890.123
        assert retrieved.source_file_size == 2048


class TestCLISupport:
    """Tests for CLI session support in database."""

    def test_add_cli_session(self, tmp_path):
        """Test adding a CLI session to database."""
        from copilot_session_tools import ChatMessage, ChatSession, Database

        db = Database(tmp_path / "test.db")

        # Create a CLI session
        session = ChatSession(
            session_id="cli-test-123",
            workspace_name=None,
            workspace_path=None,
            messages=[
                ChatMessage(role="user", content="Hello from CLI"),
                ChatMessage(role="assistant", content="Hi there!"),
            ],
            type="cli",
        )

        # Add session
        result = db.add_session(session)
        assert result is True

        # Retrieve session
        retrieved = db.get_session("cli-test-123")
        assert retrieved is not None
        assert retrieved.type == "cli"
        assert retrieved.session_id == "cli-test-123"
        assert len(retrieved.messages) == 2

    def test_vscode_session_type_default(self, tmp_path):
        """Test that VS Code sessions default to 'vscode' type."""
        from copilot_session_tools import ChatMessage, ChatSession, Database

        db = Database(tmp_path / "test.db")

        # Create a session without explicit type (should default to vscode)
        session = ChatSession(
            session_id="vscode-test-456",
            workspace_name="test-workspace",
            workspace_path="/path/to/workspace",
            messages=[
                ChatMessage(role="user", content="Hello from VS Code"),
            ],
        )

        # Add session
        db.add_session(session)

        # Retrieve session
        retrieved = db.get_session("vscode-test-456")
        assert retrieved is not None
        assert retrieved.type == "vscode"

    def test_cli_session_full_workflow(self, tmp_path):
        """Test the full workflow: parse CLI file, add to DB, retrieve."""
        from pathlib import Path

        from copilot_session_tools.scanner import _parse_cli_jsonl_file

        # Use the real sample CLI file with event-based format
        sample_file = Path(__file__).parent / "snapshots" / "fixtures" / "cli-66b821d4" / "events.jsonl"

        if not sample_file.exists():
            pytest.skip("Real CLI sample file not found")

        # Parse CLI file
        session = _parse_cli_jsonl_file(sample_file)
        assert session is not None
        assert session.session_id and len(session.session_id) > 0

        # Add to database
        db = Database(tmp_path / "test.db")
        added = db.add_session(session)
        assert added is True

        # Retrieve and verify
        retrieved = db.get_session(session.session_id)
        assert retrieved is not None
        assert retrieved.type == "cli"
        assert len(retrieved.messages) > 0

        # Verify search works - search for a word from the first user message
        user_msgs = [m for m in session.messages if m.role == "user"]
        if user_msgs:
            # Use first significant word from the user message
            words = user_msgs[0].content.split()
            search_word = next((w for w in words if len(w) > 3), words[0] if words else "test")
            results = db.search(search_word)
            assert len(results) > 0


class TestSortingBehavior:
    """Tests for session sorting behavior."""

    def test_list_sessions_sorted_by_recent_message(self, tmp_path):
        """Test that sessions are sorted by most recent message timestamp."""
        from datetime import datetime, timedelta

        db = Database(tmp_path / "test.db")

        # Create base time
        base_time = datetime(2024, 1, 1, 12, 0, 0)

        # Session 1: Created first, but has most recent message
        session1 = ChatSession(
            session_id="session-1",
            workspace_name="test",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="Old message", timestamp=(base_time + timedelta(hours=1)).isoformat()),
                ChatMessage(
                    role="assistant",
                    content="Recent message",
                    timestamp=(base_time + timedelta(hours=10)).isoformat(),  # Most recent
                ),
            ],
            created_at=(base_time + timedelta(hours=0)).isoformat(),
        )

        # Session 2: Created second, older messages
        session2 = ChatSession(
            session_id="session-2",
            workspace_name="test",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="Older message", timestamp=(base_time + timedelta(hours=2)).isoformat()),
            ],
            created_at=(base_time + timedelta(hours=5)).isoformat(),
        )

        # Session 3: Created last, has middle-aged messages
        session3 = ChatSession(
            session_id="session-3",
            workspace_name="test",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="Middle message", timestamp=(base_time + timedelta(hours=5)).isoformat()),
            ],
            created_at=(base_time + timedelta(hours=8)).isoformat(),
        )

        # Add sessions
        db.add_session(session1)
        db.add_session(session2)
        db.add_session(session3)

        # List sessions - should be sorted by most recent message
        sessions = db.list_sessions()

        # Verify order: session-1 (hour 10), session-3 (hour 5), session-2 (hour 2)
        assert len(sessions) == 3
        assert sessions[0]["session_id"] == "session-1"
        assert sessions[1]["session_id"] == "session-3"
        assert sessions[2]["session_id"] == "session-2"

        # Verify last_message_at is included
        assert "last_message_at" in sessions[0]


class TestParseSearchQuery:
    """Tests for the parse_search_query function using parametrized test cases."""

    @pytest.mark.parametrize(
        "query,expected_fts,expected_role,expected_workspace,expected_title,expected_edition",
        [
            # Empty and simple queries
            ("", "", None, None, None, None),
            ("python", "python", None, None, None, None),
            ("python function", "python function", None, None, None, None),
            # Quoted phrases (for exact match in FTS5)
            ('"python function"', '"python function"', None, None, None, None),
            ('create "python function" parameters', 'create "python function" parameters', None, None, None, None),
            # Field filters
            ("python role:user", "python", "user", None, None, None),
            ("role:assistant function", "function", "assistant", None, None, None),
            ("python workspace:my-project", "python", None, "my-project", None, None),
            ("function title:MySession", "function", None, None, "MySession", None),
            # Edition filter
            ("python edition:cli", "python", None, None, None, "cli"),
            ("edition:stable", "", None, None, None, "stable"),
            ("edition:insider function", "function", None, None, None, "insider"),
            # Quoted field values
            ('workspace:"my project name" python', "python", None, "my project name", None, None),
            # Multiple filters together
            ("python role:user workspace:myproj", "python", "user", "myproj", None, None),
            ("role:user workspace:test", "", "user", "test", None, None),
            ("role:user edition:cli test", "test", "user", None, None, "cli"),
            # Case insensitive field names
            ("Role:user WORKSPACE:test", "", "user", "test", None, None),
            ("EDITION:CLI", "", None, None, None, "cli"),
            # Duplicate scalar field values - last one wins
            ("role:user role:assistant python", "python", "assistant", None, None, None),
            # Repeated workspace filters are preserved as alternatives; workspace keeps the first value for compatibility
            ("workspace:first workspace:second", "", None, "first", None, None),
            # FTS5 special character escaping - dashes
            ("test-driven", '"test-driven"', None, None, None, None),
            ("e-commerce m-commerce", '"e-commerce" "m-commerce"', None, None, None, None),
            # FTS5 special character escaping - colons (outside field prefixes)
            ("C++:17", '"C++:17"', None, None, None, None),
            # FTS5 special character escaping - parentheses and brackets
            ("function(arg)", '"function(arg)"', None, None, None, None),
            ("array[0]", '"array[0]"', None, None, None, None),
            # Mixed special characters
            ("obj.method(arg-name)", '"obj.method(arg-name)"', None, None, None, None),
            # Already quoted phrases should remain quoted
            ('"test-driven development"', '"test-driven development"', None, None, None, None),
            # Combination of escaped and normal tokens
            ("python test-driven", 'python "test-driven"', None, None, None, None),
        ],
    )
    def test_parse_search_query(self, query, expected_fts, expected_role, expected_workspace, expected_title, expected_edition):
        """Test parsing search queries with various formats."""
        result = parse_search_query(query)
        assert result.fts_query == expected_fts
        assert result.role == expected_role
        assert result.workspace == expected_workspace
        assert result.title == expected_title
        assert result.edition == expected_edition

    def test_parse_search_query_preserves_repeated_context_filters(self):
        result = parse_search_query("workspace:first workspace:second repo:one repository:two")

        assert result.workspace == "first"
        assert result.workspaces == ["first", "second"]
        assert result.repository == "one"
        assert result.repositories == ["one", "two"]


@pytest.fixture
def search_test_db(tmp_path):
    """Create a database with multiple sessions for search testing."""
    db = Database(tmp_path / "search_test.db")

    # Session 1: Python project with user and assistant messages
    session1 = ChatSession(
        session_id="session-python",
        workspace_name="python-project",
        workspace_path="/home/user/python-project",
        messages=[
            ChatMessage(role="user", content="How do I create a Python function?"),
            ChatMessage(role="assistant", content="Here's how to create a Python function with def keyword."),
            ChatMessage(role="user", content="Thanks! Can you add parameters?"),
            ChatMessage(role="assistant", content="Sure! Here's a function with parameters."),
        ],
        created_at="1704067200000",  # 2024-01-01
        vscode_edition="stable",
    )
    db.add_session(session1)

    # Session 2: React project with different content
    session2 = ChatSession(
        session_id="session-react",
        workspace_name="react-app",
        workspace_path="/home/user/react-app",
        messages=[
            ChatMessage(role="user", content="How do I use React hooks?"),
            ChatMessage(
                role="assistant",
                content="React hooks like useState and useEffect are used in function components.",
            ),
        ],
        created_at="1704153600000",  # 2024-01-02
        vscode_edition="insider",
    )
    db.add_session(session2)

    # Session 3: Another Python session for testing multi-session results
    session3 = ChatSession(
        session_id="session-python-2",
        workspace_name="python-project",
        workspace_path="/home/user/python-project",
        messages=[
            ChatMessage(role="user", content="What is a Python decorator?"),
            ChatMessage(role="assistant", content="A decorator is a function that modifies another function."),
        ],
        created_at="1704240000000",  # 2024-01-03
        vscode_edition="stable",
    )
    db.add_session(session3)

    return db


class TestAdvancedSearchIntegration:
    """Integration tests for search functionality against actual database."""

    def test_multiple_words_match_all(self, search_test_db):
        """Test that multiple words match messages containing ALL words (AND logic)."""
        # "Python function" should match messages with both words
        results = search_test_db.search("Python function")
        assert len(results) > 0
        for r in results:
            content_lower = r["content"].lower()
            assert "python" in content_lower and "function" in content_lower

    def test_quoted_phrase_exact_match(self, search_test_db):
        """Test that quoted phrases match exactly (verbatim)."""
        # '"Python function"' should match the exact phrase
        results = search_test_db.search('"Python function"')
        assert len(results) > 0
        for r in results:
            assert "Python function" in r["content"]

    def test_mixed_words_and_quoted_phrase(self, search_test_db):
        """Test search with both unquoted words and quoted phrase."""
        # Search for 'create "Python function"' should match messages with
        # the exact phrase "Python function" and the word "create"
        results = search_test_db.search('create "Python function"')
        # Should match messages containing both "create" and the exact phrase
        for r in results:
            content = r["content"]
            assert "Python function" in content
            assert "create" in content.lower()

    @pytest.mark.parametrize(
        "query,expected_role",
        [
            ("role:user Python", "user"),
            ("role:assistant function", "assistant"),
        ],
    )
    def test_role_filter_integration(self, search_test_db, query, expected_role):
        """Test that role filter correctly filters search results."""
        results = search_test_db.search(query)
        assert len(results) > 0
        for r in results:
            if r.get("match_type") == "message":
                assert r["role"] == expected_role

    @pytest.mark.parametrize(
        "query,expected_workspace",
        [
            ("workspace:python-project function", "python-project"),
            ("workspace:react-app hooks", "react-app"),
        ],
    )
    def test_workspace_filter_integration(self, search_test_db, query, expected_workspace):
        """Test that workspace filter correctly filters search results."""
        results = search_test_db.search(query)
        assert len(results) > 0
        for r in results:
            assert r["workspace_name"] == expected_workspace

    def test_duplicate_role_filter_last_wins(self, search_test_db):
        """Test that duplicate field filters use the last value.

        Test data has assistant messages containing 'Python' (e.g.,
        'Here's how to create a Python function with def keyword.').
        """
        # Both role:user and role:assistant in query - last one wins
        results = search_test_db.search("role:user role:assistant Python")
        assert len(results) > 0, "Expected assistant messages with 'Python' in test data"
        for r in results:
            if r.get("match_type") == "message":
                assert r["role"] == "assistant"

    def test_filter_only_no_fts_query(self, search_test_db):
        """Test search with only field filters (no FTS query)."""
        # Search with only workspace filter
        results = search_test_db.search("workspace:python-project")
        assert len(results) > 0
        for r in results:
            assert r["workspace_name"] == "python-project"

    def test_combined_filters(self, search_test_db):
        """Test search with multiple filters combined."""
        results = search_test_db.search("workspace:python-project role:assistant")
        assert len(results) > 0
        for r in results:
            assert r["workspace_name"] == "python-project"
            assert r["role"] == "assistant"

    @pytest.mark.parametrize("sort_by", ["relevance", "date"])
    def test_sort_options(self, search_test_db, sort_by):
        """Test that sort options work correctly."""
        results = search_test_db.search("Python", sort_by=sort_by)
        assert len(results) > 0

    def test_sort_by_date_order(self, search_test_db):
        """Test that date sorting returns results in date order."""
        results = search_test_db.search("Python", sort_by="date")
        assert len(results) > 0
        # Results should be ordered by created_at DESC (newest first)
        dates = [r.get("created_at") for r in results if r.get("created_at")]
        # All dates should be in descending order
        for i in range(len(dates) - 1):
            assert dates[i] >= dates[i + 1]

    def test_no_results_for_non_matching_query(self, search_test_db):
        """Test that non-matching query returns empty results."""
        results = search_test_db.search("nonexistentword12345")
        assert len(results) == 0


class TestRepositoryUrlSupport:
    """Tests for repository_url field in database operations."""

    def test_add_session_with_repository_url(self, temp_db):
        """Test that sessions with repository_url are stored correctly."""
        session = ChatSession(
            session_id="test-repo-session",
            workspace_name="my-project",
            workspace_path="/home/user/projects/my-project",
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi there!"),
            ],
            repository_url="github.com/owner/repo",
        )

        result = temp_db.add_session(session)
        assert result is True

    def test_get_session_returns_repository_url(self, temp_db):
        """Test that get_session returns repository_url."""
        session = ChatSession(
            session_id="test-repo-session-2",
            workspace_name="my-project",
            workspace_path="/home/user/projects/my-project",
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi!"),
            ],
            repository_url="github.com/owner/repo",
        )

        temp_db.add_session(session)
        retrieved = temp_db.get_session("test-repo-session-2")

        assert retrieved is not None
        assert retrieved.repository_url == "github.com/owner/repo"

    def test_session_without_repository_url(self, temp_db):
        """Test that sessions without repository_url work correctly."""
        session = ChatSession(
            session_id="test-no-repo-session",
            workspace_name="my-project",
            workspace_path="/home/user/projects/my-project",
            messages=[
                ChatMessage(role="user", content="Hello"),
            ],
        )

        temp_db.add_session(session)
        retrieved = temp_db.get_session("test-no-repo-session")

        assert retrieved is not None
        assert retrieved.repository_url is None

    def test_update_session_preserves_repository_url(self, temp_db):
        """Test that update_session preserves repository_url."""
        session = ChatSession(
            session_id="test-update-repo-session",
            workspace_name="my-project",
            workspace_path="/home/user/projects/my-project",
            messages=[
                ChatMessage(role="user", content="Hello"),
            ],
            repository_url="gitlab.com/group/project",
        )

        temp_db.add_session(session)

        # Update session with more messages
        updated_session = ChatSession(
            session_id="test-update-repo-session",
            workspace_name="my-project",
            workspace_path="/home/user/projects/my-project",
            messages=[
                ChatMessage(role="user", content="Hello"),
                ChatMessage(role="assistant", content="Hi!"),
            ],
            repository_url="gitlab.com/group/project",
        )

        temp_db.update_session(updated_session)
        retrieved = temp_db.get_session("test-update-repo-session")

        assert retrieved is not None
        assert retrieved.repository_url == "gitlab.com/group/project"
        assert len(retrieved.messages) == 2

    def test_get_repositories(self, temp_db):
        """Test that get_repositories returns distinct repositories with counts."""
        # Add sessions with different repositories
        session1 = ChatSession(
            session_id="test-repo-1",
            workspace_name="project1",
            workspace_path="/path/to/project1",
            messages=[ChatMessage(role="user", content="Hello")],
            repository_url="github.com/owner/repo1",
        )
        session2 = ChatSession(
            session_id="test-repo-2",
            workspace_name="project2",
            workspace_path="/path/to/project2",
            messages=[ChatMessage(role="user", content="Hello")],
            repository_url="github.com/owner/repo1",  # Same repo as session1
        )
        session3 = ChatSession(
            session_id="test-repo-3",
            workspace_name="project3",
            workspace_path="/path/to/project3",
            messages=[ChatMessage(role="user", content="Hello")],
            repository_url="github.com/owner/repo2",  # Different repo
        )
        session4 = ChatSession(
            session_id="test-repo-4",
            workspace_name="project4",
            workspace_path="/path/to/project4",
            messages=[ChatMessage(role="user", content="Hello")],
            repository_url=None,  # No repo URL
        )

        temp_db.add_session(session1)
        temp_db.add_session(session2)
        temp_db.add_session(session3)
        temp_db.add_session(session4)

        repositories = temp_db.get_repositories()

        # Should have 2 unique repositories (excluding None)
        assert len(repositories) == 2

        # Check that session counts are correct
        repo_map = {r["repository_url"]: r["session_count"] for r in repositories}
        assert repo_map.get("github.com/owner/repo1") == 2
        assert repo_map.get("github.com/owner/repo2") == 1

    def test_search_with_repository_filter(self, temp_db):
        """Test that search filters by repository URL."""
        # Add sessions with different repositories
        session1 = ChatSession(
            session_id="search-repo-1",
            workspace_name="project1",
            workspace_path="/path/to/project1",
            messages=[ChatMessage(role="user", content="Hello from repo1")],
            repository_url="github.com/owner/repo1",
        )
        session2 = ChatSession(
            session_id="search-repo-2",
            workspace_name="project2",
            workspace_path="/path/to/project2",
            messages=[ChatMessage(role="user", content="Hello from repo2")],
            repository_url="github.com/owner/repo2",
        )
        temp_db.add_session(session1)
        temp_db.add_session(session2)

        # Search with repository filter
        results = temp_db.search("Hello", repository="github.com/owner/repo1")
        assert len(results) == 1
        assert "repo1" in results[0]["content"]

    def test_search_with_repository_in_query(self, temp_db):
        """Test that search parses repository: or repo: prefix in query."""
        # Add sessions with different repositories
        session1 = ChatSession(
            session_id="query-repo-1",
            workspace_name="project1",
            workspace_path="/path/to/project1",
            messages=[ChatMessage(role="user", content="Hello query test")],
            repository_url="github.com/owner/myrepo",
        )
        session2 = ChatSession(
            session_id="query-repo-2",
            workspace_name="project2",
            workspace_path="/path/to/project2",
            messages=[ChatMessage(role="user", content="Hello query test")],
            repository_url="github.com/other/otherrepo",
        )
        temp_db.add_session(session1)
        temp_db.add_session(session2)

        # Search with repo: prefix
        results = temp_db.search("repo:myrepo Hello")
        assert len(results) == 1

        # Search with repository: prefix
        results = temp_db.search("repository:other Hello")
        assert len(results) == 1

    def test_search_matches_later_context_and_reports_active_context(self, temp_db):
        """Search filters match any session context and report context active at the hit."""
        session = ChatSession(
            session_id="multi-context-search",
            workspace_name="project-one",
            workspace_path="/work/project-one",
            repository_url="github.com/owner/one",
            messages=[
                ChatMessage(role="user", content="Alpha only", timestamp="2025-01-01T00:00:00Z"),
                ChatMessage(role="assistant", content="Changed context", timestamp="2025-01-01T00:01:00Z"),
                ChatMessage(role="user", content="Need beta feature", timestamp="2025-01-01T00:02:00Z"),
            ],
            context_entries=[
                SessionContextEntry(
                    workspace_name="project-one",
                    workspace_path="/work/project-one",
                    repository_url="github.com/owner/one",
                    message_index=0,
                ),
                SessionContextEntry(
                    workspace_name="project-two",
                    workspace_path="/work/project-two",
                    repository_url="github.com/owner/two",
                    message_index=2,
                    source="context_changed",
                ),
            ],
        )
        temp_db.add_session(session)

        results = temp_db.search("repo:two beta")

        assert len(results) == 1
        assert results[0]["session_id"] == "multi-context-search"
        assert results[0]["active_context"]["workspace_name"] == "project-two"
        assert results[0]["repository_url"] == "github.com/owner/two"

        assert temp_db.search("repo:two Alpha") == []
        assert temp_db.search("workspace:project-two Alpha") == []

    def test_repeated_repository_prefixes_are_or_filters(self, temp_db):
        session1 = ChatSession(
            session_id="or-repo-1",
            workspace_name="project1",
            workspace_path="/path/to/project1",
            messages=[ChatMessage(role="user", content="Repeated filter")],
            repository_url="github.com/owner/repo1",
        )
        session2 = ChatSession(
            session_id="or-repo-2",
            workspace_name="project2",
            workspace_path="/path/to/project2",
            messages=[ChatMessage(role="user", content="Repeated filter")],
            repository_url="github.com/owner/repo2",
        )
        session3 = ChatSession(
            session_id="or-repo-3",
            workspace_name="project3",
            workspace_path="/path/to/project3",
            messages=[ChatMessage(role="user", content="Repeated filter")],
            repository_url="github.com/owner/repo3",
        )
        temp_db.add_session(session1)
        temp_db.add_session(session2)
        temp_db.add_session(session3)

        results = temp_db.search("repo:repo1 repo:repo2 Repeated")

        assert {r["session_id"] for r in results} == {"or-repo-1", "or-repo-2"}


class TestDateFiltering:
    """Tests for date filtering in search queries."""

    @pytest.fixture
    def db_with_dated_sessions(self, tmp_path):
        """Create a database with sessions having different dates."""
        db = Database(tmp_path / "date_test.db")

        # Session 1: 2024-01-15 (ISO timestamp format)
        session1 = ChatSession(
            session_id="session-jan-15",
            workspace_name="project1",
            workspace_path="/path/to/project1",
            messages=[ChatMessage(role="user", content="January 15 message")],
            created_at="2024-01-15T10:30:00Z",
        )
        db.add_session(session1)

        # Session 2: 2024-02-20 (ISO timestamp format)
        session2 = ChatSession(
            session_id="session-feb-20",
            workspace_name="project2",
            workspace_path="/path/to/project2",
            messages=[ChatMessage(role="user", content="February 20 message")],
            created_at="2024-02-20T14:00:00Z",
        )
        db.add_session(session2)

        # Session 3: 2024-03-10 (millisecond timestamp format: 1710057600000)
        session3 = ChatSession(
            session_id="session-mar-10",
            workspace_name="project3",
            workspace_path="/path/to/project3",
            messages=[ChatMessage(role="user", content="March 10 message")],
            created_at="1710057600000",  # 2024-03-10 00:00:00 UTC
        )
        db.add_session(session3)

        return db

    def test_parse_search_query_start_date(self):
        """Test parsing start_date from search query."""
        result = parse_search_query("python start_date:2024-01-01")
        assert result.fts_query == "python"
        assert result.start_date == "2024-01-01"

    def test_parse_search_query_end_date(self):
        """Test parsing end_date from search query."""
        result = parse_search_query("python end_date:2024-12-31")
        assert result.fts_query == "python"
        assert result.end_date == "2024-12-31"

    def test_parse_search_query_both_dates(self):
        """Test parsing both start and end dates from search query."""
        result = parse_search_query("python start_date:2024-01-01 end_date:2024-06-30")
        assert result.fts_query == "python"
        assert result.start_date == "2024-01-01"
        assert result.end_date == "2024-06-30"

    def test_parse_search_query_invalid_date_format(self):
        """Test that invalid date formats are ignored."""
        result = parse_search_query("python start_date:01-01-2024")  # Wrong format
        assert result.fts_query == "python"
        assert result.start_date is None

        result = parse_search_query("python start_date:2024/01/01")  # Wrong separator
        assert result.fts_query == "python"
        assert result.start_date is None

    def test_parse_search_query_date_only(self):
        """Test parsing date-only filters without search terms."""
        result = parse_search_query("start_date:2024-01-01 end_date:2024-12-31")
        assert result.fts_query == ""
        assert result.start_date == "2024-01-01"
        assert result.end_date == "2024-12-31"

    def test_search_with_start_date_filter(self, db_with_dated_sessions):
        """Test search with start_date filter."""
        # Search for messages from Feb 1 onwards (should find Feb 20 and Mar 10)
        results = db_with_dated_sessions.search("message start_date:2024-02-01")
        session_ids = {r["session_id"] for r in results}
        assert "session-jan-15" not in session_ids
        assert "session-feb-20" in session_ids
        assert "session-mar-10" in session_ids

    def test_search_with_end_date_filter(self, db_with_dated_sessions):
        """Test search with end_date filter."""
        # Search for messages up to Feb 28 (should find Jan 15 and Feb 20)
        results = db_with_dated_sessions.search("message end_date:2024-02-28")
        session_ids = {r["session_id"] for r in results}
        assert "session-jan-15" in session_ids
        assert "session-feb-20" in session_ids
        assert "session-mar-10" not in session_ids

    def test_search_with_date_range_filter(self, db_with_dated_sessions):
        """Test search with both start and end date filters."""
        # Search for messages between Feb 1 and Feb 28 (should only find Feb 20)
        results = db_with_dated_sessions.search("message start_date:2024-02-01 end_date:2024-02-28")
        assert len(results) == 1
        assert results[0]["session_id"] == "session-feb-20"


class TestSkipAndPagination:
    """Tests for skip/offset pagination in search."""

    @pytest.fixture
    def db_with_many_sessions(self, tmp_path):
        """Create a database with multiple sessions for pagination testing."""
        db = Database(tmp_path / "pagination_test.db")

        # Create 5 sessions with searchable content
        for i in range(5):
            session = ChatSession(
                session_id=f"session-{i}",
                workspace_name=f"project-{i}",
                workspace_path=f"/path/to/project-{i}",
                messages=[ChatMessage(role="user", content=f"Test pagination message number {i}")],
                created_at=f"2024-01-{10 + i:02d}T10:00:00Z",
            )
            db.add_session(session)

        return db

    def test_search_with_skip_parameter(self, db_with_many_sessions):
        """Test search with skip parameter."""
        # Get all results first
        all_results = db_with_many_sessions.search("pagination", limit=10, skip=0)
        assert len(all_results) == 5

        # Skip first 2 results
        skipped_results = db_with_many_sessions.search("pagination", limit=10, skip=2)
        assert len(skipped_results) == 3

        # Skip first 4 results
        skipped_results = db_with_many_sessions.search("pagination", limit=10, skip=4)
        assert len(skipped_results) == 1

    def test_search_with_limit_and_skip(self, db_with_many_sessions):
        """Test search with both limit and skip for pagination."""
        # First page: limit=2, skip=0
        page1 = db_with_many_sessions.search("pagination", limit=2, skip=0)
        assert len(page1) == 2

        # Second page: limit=2, skip=2
        page2 = db_with_many_sessions.search("pagination", limit=2, skip=2)
        assert len(page2) == 2

        # Pages should have different sessions
        page1_ids = {r["session_id"] for r in page1}
        page2_ids = {r["session_id"] for r in page2}
        assert page1_ids.isdisjoint(page2_ids)

    def test_search_skip_exceeds_results(self, db_with_many_sessions):
        """Test search when skip exceeds total results."""
        results = db_with_many_sessions.search("pagination", limit=10, skip=100)
        assert len(results) == 0


class TestFTSOptimization:
    """Tests for FTS5 index optimization."""

    def test_optimize_fts_empty_database(self, temp_db):
        """Test optimize_fts on empty database."""
        result = temp_db.optimize_fts()
        assert result["optimized"] is True
        assert "segments_before" in result
        assert "segments_after" in result

    def test_optimize_fts_with_data(self, temp_db, sample_session):
        """Test optimize_fts after adding data."""
        temp_db.add_session(sample_session)

        result = temp_db.optimize_fts()
        assert result["optimized"] is True
        assert result["segments_before"] >= 0
        assert result["segments_after"] >= 0

    def test_optimize_fts_multiple_sessions(self, tmp_path):
        """Test optimize_fts with multiple sessions (more fragmentation)."""
        db = Database(tmp_path / "multi_session.db")

        # Add multiple sessions to create FTS fragmentation
        for i in range(5):
            session = ChatSession(
                session_id=f"optimize-test-{i}",
                workspace_name=f"project-{i}",
                workspace_path=f"/path/to/project-{i}",
                messages=[
                    ChatMessage(role="user", content=f"Test message number {i} for optimization"),
                    ChatMessage(role="assistant", content=f"Response to message {i}"),
                ],
            )
            db.add_session(session)

        result = db.optimize_fts()
        assert result["optimized"] is True


class TestGetMessagesMarkdownIncludeThinking:
    """Tests for get_messages_markdown with content_set parameter."""

    def test_include_thinking_passes_through(self, temp_db):
        """Test that content_set with 'thinking' is accepted and changes output."""
        session = ChatSession(
            session_id="thinking-md-test",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="Think about this"),
                ChatMessage(
                    role="assistant",
                    content="The answer.",
                    content_blocks=[
                        ContentBlock(kind="thinking", content="Internal reasoning..."),
                        ContentBlock(kind="text", content="The answer."),
                    ],
                ),
            ],
        )
        temp_db.add_session(session)

        # Without 'thinking' in content_set, thinking content should be omitted
        md_without = temp_db.get_messages_markdown("thinking-md-test", content_set={"agent-details"})
        assert "Internal reasoning..." not in md_without

        # With 'thinking' in content_set, thinking content should be included
        md_with = temp_db.get_messages_markdown("thinking-md-test", content_set={"thinking", "agent-details"})
        assert "Internal reasoning..." in md_with


class TestFTS5SpecialCharacterEscaping:
    """Tests for FTS5 special character escaping in search queries."""

    def test_dash_in_query(self, temp_db):
        """Test that dashes in queries don't cause FTS5 syntax errors."""
        session = ChatSession(
            session_id="dash-test",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="How do I use test-driven development?"),
                ChatMessage(role="assistant", content="Test-driven development (TDD) is a practice..."),
            ],
        )
        temp_db.add_session(session)

        # This should not raise an exception
        results = temp_db.search("test-driven")
        assert len(results) > 0
        assert any("test-driven" in r["content"].lower() for r in results)

    def test_multiple_dashes_in_query(self, temp_db):
        """Test query with multiple words containing dashes."""
        session = ChatSession(
            session_id="multi-dash-test",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="What is e-commerce and m-commerce?"),
                ChatMessage(role="assistant", content="E-commerce is electronic commerce, m-commerce is mobile commerce."),
            ],
        )
        temp_db.add_session(session)

        # Multiple dashed words should work
        results = temp_db.search("e-commerce m-commerce")
        assert len(results) > 0

    def test_colon_in_query(self, temp_db):
        """Test that colons in queries (outside field prefixes) are escaped."""
        session = ChatSession(
            session_id="colon-test",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="How do I use C++:17 features?"),
                ChatMessage(role="assistant", content="C++:17 adds many features..."),
            ],
        )
        temp_db.add_session(session)

        # Colon should be escaped to prevent column specification error
        results = temp_db.search("C++:17")
        # Should not raise an exception (main goal)
        assert isinstance(results, list)

    def test_parentheses_in_query(self, temp_db):
        """Test that parentheses in queries are escaped."""
        session = ChatSession(
            session_id="paren-test",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="How do I use function(parameter)?"),
                ChatMessage(role="assistant", content="Call function(parameter) like this..."),
            ],
        )
        temp_db.add_session(session)

        # Parentheses should be escaped
        results = temp_db.search("function(parameter)")
        assert isinstance(results, list)

    def test_brackets_in_query(self, temp_db):
        """Test that brackets in queries are escaped."""
        session = ChatSession(
            session_id="bracket-test",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="How do I use array[0]?"),
                ChatMessage(role="assistant", content="Access array[0] like this..."),
            ],
        )
        temp_db.add_session(session)

        # Brackets should be escaped
        results = temp_db.search("array[0]")
        assert isinstance(results, list)

    def test_mixed_special_characters(self, temp_db):
        """Test query with multiple types of special characters."""
        session = ChatSession(
            session_id="mixed-test",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="How do I use obj.method(arg-name)?"),
                ChatMessage(role="assistant", content="Call obj.method(arg-name) to do..."),
            ],
        )
        temp_db.add_session(session)

        # Complex query with dashes, parentheses, dots
        results = temp_db.search("obj.method(arg-name)")
        assert isinstance(results, list)

    def test_already_quoted_phrase_not_double_escaped(self, temp_db):
        """Test that already quoted phrases are not double-escaped."""
        session = ChatSession(
            session_id="quoted-test",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="What is test-driven development?"),
                ChatMessage(role="assistant", content="Test-driven development is..."),
            ],
        )
        temp_db.add_session(session)

        # Quoted phrase should remain quoted but work
        results = temp_db.search('"test-driven"')
        assert len(results) > 0

    def test_escape_preserves_search_functionality(self, temp_db):
        """Test that escaping doesn't break normal search functionality."""
        session = ChatSession(
            session_id="normal-test",
            workspace_name="test-project",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="How do I create a Python function?"),
                ChatMessage(role="assistant", content="Use def to create a Python function."),
            ],
        )
        temp_db.add_session(session)

        # Normal search without special chars should still work
        results = temp_db.search("Python function")
        assert len(results) > 0
        assert any("Python" in r["content"] for r in results)


class TestGranularSearchContentSet:
    """Tests for search_content_set column-selective search and agent nesting gate."""

    @pytest.fixture
    def rich_db(self, tmp_path):
        """Database with a session containing tool invocations, file changes, commands, thinking, and agent-nested content."""
        db = Database(tmp_path / "rich_search.db")
        session = ChatSession(
            session_id="rich-session",
            workspace_name="rich-project",
            workspace_path="/tmp/rich",
            messages=[
                # msg 0: plain user message
                ChatMessage(role="user", content="Please help me refactor"),
                # msg 1: assistant with tools, files, commands, thinking
                ChatMessage(
                    role="assistant",
                    content="Here is the refactored code",
                    tool_invocations=[
                        ToolInvocation(
                            name="grep_search",
                            input="xyzzyToolInput42 in some file",
                            result="found match in main.py",
                        ),
                    ],
                    file_changes=[
                        FileChange(
                            path="src/main.py",
                            explanation="updated the handler",
                            diff="--- a/src/main.py\n+++ b/src/main.py\n@@ -1 +1 @@\n-old xyzzyDiffOnly99\n+new code",
                        ),
                    ],
                    command_runs=[
                        CommandRun(command="npm test", output="xyzzyCommandOut55 all tests passed"),
                    ],
                    content_blocks=[
                        ContentBlock(kind="thinking", content="xyzzyThinkSecret77 reasoning about approach"),
                        ContentBlock(kind="text", content="Here is the refactored code"),
                    ],
                ),
                # msg 2: user follow-up
                ChatMessage(role="user", content="Now run inside the agent"),
                # msg 3: agent-nested assistant message (nesting_level=1)
                ChatMessage(
                    role="assistant",
                    content="xyzzyAgentNested88 subagent did work",
                    agent_nesting_level=1,
                    agent_id="task-agent-1",
                    agent_display_name="General Purpose Agent",
                    tool_invocations=[
                        ToolInvocation(
                            name="edit_file",
                            input="xyzzyNestedToolInput33",
                            result="xyzzyNestedToolResult33",
                        ),
                    ],
                ),
            ],
            created_at="2025-01-15T10:30:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)
        return db

    # 1. search_content_set parameter — passing a specific set works
    def test_search_content_set_filters_results(self, rich_db):
        """Passing a specific search_content_set restricts which sources are searched."""
        # Search for tool name — should match with tools, not with messages-only
        results_tools = rich_db.search("grep_search", search_content_set={"tools", "tool-inputs", "agent-details"})
        tool_hits = [r for r in results_tools if r["match_type"] == "tool_invocation"]
        assert len(tool_hits) > 0

        # Same query with only messages — tool invocations should not appear
        results_msgs = rich_db.search("grep_search", search_content_set={"messages", "agent-details"})
        tool_hits_msgs = [r for r in results_msgs if r["match_type"] == "tool_invocation"]
        assert len(tool_hits_msgs) == 0

    def test_message_search_excludes_system_messages_by_default(self, tmp_path):
        """Default message search does not return system-role messages."""
        db = Database(tmp_path / "system_search.db")
        db.add_session(
            ChatSession(
                session_id="system-search-session",
                workspace_name="system-search-project",
                workspace_path="/tmp/system-search",
                messages=[
                    ChatMessage(role="user", content="hello"),
                    ChatMessage(role="system", content="xyzzySystemSecret11 internal instructions"),
                ],
            )
        )

        results = db.search("xyzzySystemSecret11", search_content_set={"messages"})
        assert results == []

    def test_role_system_search_finds_system_messages(self, tmp_path):
        """Explicit role=system search still returns system-role messages."""
        db = Database(tmp_path / "system_search.db")
        db.add_session(
            ChatSession(
                session_id="system-search-session",
                workspace_name="system-search-project",
                workspace_path="/tmp/system-search",
                messages=[
                    ChatMessage(role="user", content="hello"),
                    ChatMessage(role="system", content="xyzzySystemSecret11 internal instructions"),
                ],
            )
        )

        results = db.search("xyzzySystemSecret11", role="system", search_content_set={"messages"})
        assert len(results) == 1
        assert results[0]["role"] == "system"

    # 2. Tools-only search
    def test_tools_only_search(self, rich_db):
        """search_content_set={'tools'} only returns tool invocation matches."""
        results = rich_db.search("grep_search", search_content_set={"tools", "agent-details"})
        assert all(r["match_type"] == "tool_invocation" for r in results)
        assert len(results) > 0

    # 3. File-changes-only search
    def test_file_changes_only_search(self, rich_db):
        """search_content_set={'file-changes'} only returns file change matches."""
        results = rich_db.search("main.py", search_content_set={"file-changes", "agent-details"})
        assert all(r["match_type"] == "file_change" for r in results)
        assert len(results) > 0

    # 4. Column-selective: tool-inputs
    def test_tool_input_excluded_without_flag(self, rich_db):
        """With only 'tools' (no 'tool-inputs'), a term only in tool input should NOT match."""
        # xyzzyToolInput42 only exists in tool input, not name or result
        results = rich_db.search("xyzzyToolInput42", search_content_set={"tools", "agent-details"})
        tool_hits = [r for r in results if r["match_type"] == "tool_invocation"]
        assert len(tool_hits) == 0

    def test_tool_input_included_with_flag(self, rich_db):
        """With 'tools' + 'tool-inputs', a term only in tool input SHOULD match."""
        results = rich_db.search("xyzzyToolInput42", search_content_set={"tools", "tool-inputs", "agent-details"})
        tool_hits = [r for r in results if r["match_type"] == "tool_invocation"]
        assert len(tool_hits) > 0

    # 5. Column-selective: diffs
    def test_diff_excluded_without_flag(self, rich_db):
        """With only 'file-changes' (no 'diffs'), a term only in diff should NOT match."""
        # xyzzyDiffOnly99 only exists in the diff, not path or explanation
        results = rich_db.search("xyzzyDiffOnly99", search_content_set={"file-changes", "agent-details"})
        file_hits = [r for r in results if r["match_type"] == "file_change"]
        assert len(file_hits) == 0

    def test_diff_included_with_flag(self, rich_db):
        """With 'file-changes' + 'diffs', a term only in diff SHOULD match."""
        results = rich_db.search("xyzzyDiffOnly99", search_content_set={"file-changes", "diffs", "agent-details"})
        file_hits = [r for r in results if r["match_type"] == "file_change"]
        assert len(file_hits) > 0

    # 6. Commands search
    def test_commands_search(self, rich_db):
        """search_content_set={'commands'} searches cst_command_runs."""
        results = rich_db.search("xyzzyCommandOut55", search_content_set={"commands", "agent-details"})
        cmd_hits = [r for r in results if r["match_type"] == "command_run"]
        assert len(cmd_hits) > 0

    def test_commands_excluded_when_not_in_set(self, rich_db):
        """Command results should not appear when 'commands' is not in the set."""
        results = rich_db.search("xyzzyCommandOut55", search_content_set={"messages", "agent-details"})
        cmd_hits = [r for r in results if r["match_type"] == "command_run"]
        assert len(cmd_hits) == 0

    # 7. Thinking search
    def test_thinking_search(self, rich_db):
        """search_content_set={'thinking'} searches cst_content_blocks WHERE kind='thinking'."""
        results = rich_db.search("xyzzyThinkSecret77", search_content_set={"thinking", "agent-details"})
        think_hits = [r for r in results if r["match_type"] == "thinking"]
        assert len(think_hits) > 0

    def test_thinking_excluded_when_not_in_set(self, rich_db):
        """Thinking results should not appear when 'thinking' is not in the set."""
        results = rich_db.search("xyzzyThinkSecret77", search_content_set={"messages", "agent-details"})
        think_hits = [r for r in results if r["match_type"] == "thinking"]
        assert len(think_hits) == 0

    # 8. Agent nesting gate
    def test_agent_nesting_gate_excludes_nested(self, rich_db):
        """Without 'agent-details', results from agent_nesting_level > 0 are excluded."""
        # xyzzyAgentNested88 is in a message with nesting_level=1
        results = rich_db.search("xyzzyAgentNested88", search_content_set={"messages"})
        assert len(results) == 0

    def test_agent_nesting_gate_includes_nested(self, rich_db):
        """With 'agent-details', results from agent_nesting_level > 0 are included."""
        results = rich_db.search("xyzzyAgentNested88", search_content_set={"messages", "agent-details"})
        msg_hits = [r for r in results if r["match_type"] == "message"]
        assert len(msg_hits) > 0

    def test_agent_nesting_gate_tools(self, rich_db):
        """Without 'agent-details', tool invocations from nested agents are excluded."""
        # xyzzyNestedToolResult33 is in a tool invocation on a nested message
        results = rich_db.search("xyzzyNestedToolResult33", search_content_set={"tools", "tool-inputs"})
        tool_hits = [r for r in results if r["match_type"] == "tool_invocation"]
        assert len(tool_hits) == 0

    def test_agent_nesting_gate_tools_included(self, rich_db):
        """With 'agent-details', tool invocations from nested agents are included."""
        results = rich_db.search(
            "xyzzyNestedToolResult33",
            search_content_set={"tools", "tool-inputs", "agent-details"},
        )
        tool_hits = [r for r in results if r["match_type"] == "tool_invocation"]
        assert len(tool_hits) > 0

    # 9. Backward compatibility
    def test_backward_compat_include_messages_only(self, rich_db):
        """Old-style include_messages=True, include_tool_calls=False still works."""
        results = rich_db.search("refactor", include_messages=True, include_tool_calls=False, include_file_changes=False)
        # Should find the message but not tool/file results
        msg_hits = [r for r in results if r["match_type"] == "message"]
        assert len(msg_hits) > 0
        tool_hits = [r for r in results if r["match_type"] == "tool_invocation"]
        assert len(tool_hits) == 0

    def test_backward_compat_include_tool_calls(self, rich_db):
        """Old-style include_tool_calls=True enables tools + tool-inputs."""
        results = rich_db.search("xyzzyToolInput42", include_messages=False, include_tool_calls=True)
        tool_hits = [r for r in results if r["match_type"] == "tool_invocation"]
        assert len(tool_hits) > 0

    # 10. Empty set
    def test_empty_content_set_returns_empty(self, rich_db):
        """search_content_set=set() returns empty list."""
        results = rich_db.search("refactor", search_content_set=set())
        assert results == []

    # 11. message_index in results
    def test_tool_results_include_message_index(self, rich_db):
        """Tool invocation results include message_index field."""
        results = rich_db.search("grep_search", search_content_set={"tools", "agent-details"})
        tool_hits = [r for r in results if r["match_type"] == "tool_invocation"]
        assert len(tool_hits) > 0
        for r in tool_hits:
            assert "message_index" in r
            assert r["message_index"] == 1  # 0-based, second message

    def test_file_results_include_message_index(self, rich_db):
        """File change results include message_index field."""
        results = rich_db.search("main.py", search_content_set={"file-changes", "agent-details"})
        file_hits = [r for r in results if r["match_type"] == "file_change"]
        assert len(file_hits) > 0
        for r in file_hits:
            assert "message_index" in r
            assert r["message_index"] == 1

    def test_command_results_include_message_index(self, rich_db):
        """Command run results include message_index field."""
        results = rich_db.search("xyzzyCommandOut55", search_content_set={"commands", "agent-details"})
        cmd_hits = [r for r in results if r["match_type"] == "command_run"]
        assert len(cmd_hits) > 0
        for r in cmd_hits:
            assert "message_index" in r
            assert r["message_index"] == 1

    def test_thinking_results_include_message_index(self, rich_db):
        """Thinking block results include message_index field."""
        results = rich_db.search("xyzzyThinkSecret77", search_content_set={"thinking", "agent-details"})
        think_hits = [r for r in results if r["match_type"] == "thinking"]
        assert len(think_hits) > 0
        for r in think_hits:
            assert "message_index" in r
            assert r["message_index"] == 1


class TestRelevanceWithRecency:
    """Tests for relevance algorithm that weighs towards recent items."""

    def test_recent_items_rank_higher_for_similar_relevance(self, tmp_path):
        """Test that more recent items rank higher when text relevance is similar."""
        db = Database(tmp_path / "recency_test.db")

        # Create two sessions with the same content but different dates
        # Older session (2020)
        old_session = ChatSession(
            session_id="old-session",
            workspace_name="test-old",
            workspace_path="/test/old",
            messages=[
                ChatMessage(role="user", content="How do I use Python decorators?"),
                ChatMessage(role="assistant", content="Python decorators are a way to modify functions."),
            ],
            created_at="2020-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(old_session)

        # Recent session (2025)
        new_session = ChatSession(
            session_id="new-session",
            workspace_name="test-new",
            workspace_path="/test/new",
            messages=[
                ChatMessage(role="user", content="How do I use Python decorators?"),
                ChatMessage(role="assistant", content="Python decorators are a way to modify functions."),
            ],
            created_at="2025-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(new_session)

        # Search with relevance sorting
        results = db.search("Python decorators", sort_by="relevance")

        # Both sessions should be returned
        assert len(results) >= 2

        # Extract session IDs from results in order
        session_ids = [r["session_id"] for r in results[:2]]

        # The newer session should rank higher (appear first) due to recency bonus
        assert session_ids[0] == "new-session", "Recent session should rank higher"

    def test_recency_bonus_doesnt_override_strong_relevance(self, tmp_path):
        """Test that recency doesn't override strong text relevance differences."""
        db = Database(tmp_path / "relevance_priority_test.db")

        # Old session with strong match
        old_session = ChatSession(
            session_id="old-strong",
            workspace_name="test-old",
            workspace_path="/test/old",
            messages=[
                ChatMessage(role="user", content="How do I use Python decorators for authentication and authorization?"),
                ChatMessage(role="assistant", content="Python decorators for authentication are perfect. Decorators decorators decorators."),
            ],
            created_at="2020-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(old_session)

        # Recent session with weak match
        new_session = ChatSession(
            session_id="new-weak",
            workspace_name="test-new",
            workspace_path="/test/new",
            messages=[
                ChatMessage(role="user", content="What is Python?"),
                ChatMessage(role="assistant", content="Python is a programming language."),
            ],
            created_at="2025-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(new_session)

        # Search with relevance sorting
        results = db.search("Python decorators", sort_by="relevance")

        # The old session with strong match should rank higher despite being older
        session_ids = [r["session_id"] for r in results]

        # Find positions
        old_pos = session_ids.index("old-strong") if "old-strong" in session_ids else -1
        new_pos = session_ids.index("new-weak") if "new-weak" in session_ids else -1

        # Strong match should rank higher even if older
        if old_pos >= 0 and new_pos >= 0:
            assert old_pos < new_pos, "Strong relevance should override recency"

    def test_date_sorting_still_works(self, tmp_path):
        """Test that explicit date sorting still works correctly."""
        db = Database(tmp_path / "date_sort_test.db")

        # Create sessions with different dates
        old_session = ChatSession(
            session_id="oldest",
            workspace_name="test-old",
            workspace_path="/test/old",
            messages=[
                ChatMessage(role="user", content="Test message old"),
            ],
            created_at="2020-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(old_session)

        new_session = ChatSession(
            session_id="newest",
            workspace_name="test-new",
            workspace_path="/test/new",
            messages=[
                ChatMessage(role="user", content="Test message new"),
            ],
            created_at="2025-01-15T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(new_session)

        # Search with date sorting
        results = db.search("message", sort_by="date")

        # Newest should be first when sorting by date
        session_ids = [r["session_id"] for r in results[:2]]
        assert session_ids[0] == "newest", "Newest session should be first with date sorting"


class TestSubagentSearchBehavior:
    """Tests that FTS and search correctly handle parent/child (subagent) messages.

    Parent message content should NOT contain subagent text.
    Child messages are separate rows with agent_nesting_level=1 and parent_message_id set.
    The agent-details search flag gates visibility of child content.
    """

    @pytest.fixture
    def subagent_db(self, tmp_path):
        """Database with a session containing a parent message and a subagent child."""
        db = Database(tmp_path / "subagent_search.db")
        child_msg = ChatMessage(
            role="assistant",
            content="unique_child_token_xyz from the subagent",
            agent_nesting_level=1,
            agent_id="task-agent-42",
            agent_display_name="General Purpose Agent",
            tool_invocations=[
                ToolInvocation(
                    name="grep",
                    input="unique_tool_pattern_abc",
                    result="found 3 matches",
                ),
            ],
        )
        session = ChatSession(
            session_id="subagent-search-session",
            workspace_name="subagent-project",
            workspace_path="/home/user/subagent-project",
            messages=[
                ChatMessage(role="user", content="Please run the agent"),
                ChatMessage(
                    role="assistant",
                    content="Parent says hello",
                    content_blocks=[
                        ContentBlock(kind="text", content="Parent says hello"),
                        ContentBlock(
                            kind="subagent",
                            content="",
                            child_message=child_msg,
                        ),
                    ],
                ),
            ],
            created_at="2025-01-15T10:30:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)
        return db

    def test_parent_fts_content_excludes_subagent_text(self, subagent_db):
        """FTS search for child-only text without agent-details returns nothing."""
        results = subagent_db.search("unique_child_token_xyz", search_content_set={"messages"})
        assert len(results) == 0

    def test_child_fts_searchable_with_agent_details(self, subagent_db):
        """FTS search for child text with agent-details finds the child message."""
        results = subagent_db.search(
            "unique_child_token_xyz",
            search_content_set={"messages", "agent-details"},
        )
        msg_hits = [r for r in results if r["match_type"] == "message"]
        assert len(msg_hits) > 0
        assert "unique_child_token_xyz" in msg_hits[0]["content"]

    def test_parent_text_still_searchable(self, subagent_db):
        """Parent message text is searchable without agent-details."""
        results = subagent_db.search("Parent says hello", search_content_set={"messages"})
        msg_hits = [r for r in results if r["match_type"] == "message"]
        assert len(msg_hits) > 0
        assert "Parent says hello" in msg_hits[0]["content"]

    def test_child_tool_invocations_searchable(self, subagent_db):
        """Child tool invocations are gated by agent-details."""
        # Without agent-details: child tools hidden
        results_no_agent = subagent_db.search(
            "unique_tool_pattern_abc",
            search_content_set={"tools", "tool-inputs"},
        )
        tool_hits = [r for r in results_no_agent if r["match_type"] == "tool_invocation"]
        assert len(tool_hits) == 0

        # With agent-details: child tools visible
        results_with_agent = subagent_db.search(
            "unique_tool_pattern_abc",
            search_content_set={"tools", "tool-inputs", "agent-details"},
        )
        tool_hits = [r for r in results_with_agent if r["match_type"] == "tool_invocation"]
        assert len(tool_hits) > 0
        assert "unique_tool_pattern_abc" in tool_hits[0]["content"]

    def test_search_result_includes_parent_message_id(self, subagent_db):
        """Child message search results include parent_message_id."""
        results = subagent_db.search(
            "unique_child_token_xyz",
            search_content_set={"messages", "agent-details"},
        )
        msg_hits = [r for r in results if r["match_type"] == "message"]
        assert len(msg_hits) > 0
        hit = msg_hits[0]
        assert "parent_message_id" in hit
        assert hit["parent_message_id"] is not None


# ---------------------------------------------------------------------------
# Built-in schema SQL (mirrors Copilot CLI's own tables)
# ---------------------------------------------------------------------------
BUILTIN_SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version (version) VALUES (1);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    cwd TEXT,
    repository TEXT,
    branch TEXT,
    summary TEXT,
    created_at TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS turns (
    session_id TEXT NOT NULL,
    turn_index INTEGER NOT NULL,
    user_message TEXT,
    assistant_response TEXT,
    timestamp TEXT,
    PRIMARY KEY (session_id, turn_index)
);
"""


def _create_builtin_only_db(path: str) -> None:
    """Create a DB with only built-in tables (no cst_* tables)."""
    conn = sqlite3.connect(path)
    conn.executescript(BUILTIN_SCHEMA)
    conn.close()


def _insert_builtin_session(
    path: str,
    session_id: str,
    *,
    summary: str = "Test session",
    repository: str = "owner/repo",
    cwd: str = "/home/user",
    branch: str = "main",
    created_at: str = "2025-01-15T10:00:00Z",
    updated_at: str = "2025-01-15T10:30:00Z",
) -> None:
    """Insert a session row into the built-in sessions table."""
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO sessions (id, cwd, repository, branch, summary, created_at, updated_at) VALUES (?,?,?,?,?,?,?)",
        (session_id, cwd, repository, branch, summary, created_at, updated_at),
    )
    conn.commit()
    conn.close()


def _insert_builtin_turn(
    path: str,
    session_id: str,
    turn_index: int,
    user_message: str,
    assistant_response: str,
) -> None:
    """Insert a turn row into the built-in turns table."""
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO turns (session_id, turn_index, user_message, assistant_response) VALUES (?,?,?,?)",
        (session_id, turn_index, user_message, assistant_response),
    )
    conn.commit()
    conn.close()


class TestTwoTierRendering:
    """Tests for the two-tier (CST + Chronicle) rendering architecture.

    CST tables live in copilot-session-tools.db, Chronicle tables in
    session-store.db (same directory).  The Database class auto-detects
    the Chronicle sibling.
    """

    # -- fixtures ----------------------------------------------------------

    @pytest.fixture
    def chronicle_db(self, tmp_path):
        """Create a Chronicle DB with built-in tables in tmp_path."""
        chronicle_path = str(tmp_path / "session-store.db")
        _create_builtin_only_db(chronicle_path)
        return chronicle_path

    @pytest.fixture
    def builtin_only_db(self, tmp_path, chronicle_db):
        """DB path for CST, with Chronicle sibling (no cst_* enrichment)."""
        db_path = str(tmp_path / "copilot-session-tools.db")
        return db_path

    @pytest.fixture
    def full_db(self, tmp_path, chronicle_db):
        """DB with CST tables AND a Chronicle sibling with built-in tables."""
        db_path = str(tmp_path / "copilot-session-tools.db")
        # Database() constructor creates cst_* tables automatically
        # Auto-detects session-store.db as sibling
        db = Database(db_path)
        return db

    # -- 1. has_cst_tables = True ------------------------------------------

    def test_has_cst_tables_true(self, full_db):
        """cst_* tables are created by the Database constructor."""
        assert full_db.has_cst_tables() is True

    # -- 2. has_cst_tables = False -----------------------------------------

    def test_has_cst_tables_false(self, builtin_only_db):
        """A raw built-in-only DB reports no cst_* tables via unenriched_only mode."""
        db = Database(builtin_only_db, unenriched_only=True)
        assert db.has_cst_tables() is False

    # -- 3. list_sessions unenriched ---------------------------------------

    def test_list_sessions_unenriched(self, full_db):
        """A session in Chronicle only (no cst_sessions row) is listed with is_enriched=False."""
        chronicle_path = str(full_db.chronicle_db_path)
        _insert_builtin_session(chronicle_path, "sess-builtin-1")
        _insert_builtin_turn(chronicle_path, "sess-builtin-1", 0, "hello", "hi there")

        sessions = full_db.list_sessions()
        matched = [s for s in sessions if s["session_id"] == "sess-builtin-1"]
        assert len(matched) == 1
        assert matched[0]["is_enriched"] is False
        assert matched[0]["source"] == "builtin"

    # -- 4. list_sessions enriched -----------------------------------------

    def test_list_sessions_enriched(self, full_db):
        """A session present in cst_sessions is listed with is_enriched=True."""
        full_db.add_session(
            ChatSession(
                session_id="sess-enriched-1",
                workspace_name="proj",
                workspace_path="/tmp/proj",
                messages=[ChatMessage(role="user", content="hi")],
                created_at="2025-01-15T10:00:00Z",
            )
        )

        sessions = full_db.list_sessions()
        matched = [s for s in sessions if s["session_id"] == "sess-enriched-1"]
        assert len(matched) == 1
        assert matched[0]["is_enriched"] is True
        assert matched[0]["source"] == "cst"

    # -- 5. get_session unenriched -----------------------------------------

    def test_get_session_unenriched(self, full_db):
        """get_session falls back to Chronicle tables when no cst_sessions row exists."""
        chronicle_path = str(full_db.chronicle_db_path)
        _insert_builtin_session(
            chronicle_path,
            "sess-unenriched",
            summary="Unenriched session",
            repository="owner/repo",
        )
        _insert_builtin_turn(chronicle_path, "sess-unenriched", 0, "Q1", "A1")
        _insert_builtin_turn(chronicle_path, "sess-unenriched", 1, "Q2", "A2")

        session = full_db.get_session("sess-unenriched")
        assert session is not None
        assert session.session_id == "sess-unenriched"
        assert session.type == "cli"
        # Chronicle turns become messages (user+assistant pairs)
        assert len(session.messages) == 4
        assert session.messages[0].role == "user"
        assert session.messages[0].content == "Q1"

    # -- 6. get_session enriched -------------------------------------------

    def test_get_session_enriched(self, full_db):
        """get_session returns enriched cst_* data when available."""
        full_db.add_session(
            ChatSession(
                session_id="sess-rich",
                workspace_name="proj",
                workspace_path="/tmp/proj",
                messages=[
                    ChatMessage(role="user", content="enriched question"),
                    ChatMessage(role="assistant", content="enriched answer"),
                ],
                created_at="2025-01-15T10:00:00Z",
                vscode_edition="stable",
            )
        )

        session = full_db.get_session("sess-rich")
        assert session is not None
        assert session.session_id == "sess-rich"
        assert len(session.messages) == 2
        assert session.messages[0].content == "enriched question"

    # -- 7. discover_sessions_needing_enrichment ---------------------------

    def test_discover_sessions_needing_enrichment(self, full_db):
        """Sessions in Chronicle but not in cst_sessions are discovered as needing enrichment."""
        chronicle_path = str(full_db.chronicle_db_path)
        # Session A: in Chronicle only → needs enrichment
        _insert_builtin_session(chronicle_path, "sess-a", summary="A")
        _insert_builtin_turn(chronicle_path, "sess-a", 0, "q", "a")

        # Session B: enriched with the current Chronicle turn count → up to date.
        # Enrichment (not add_session) records builtin_turns for like-for-like checks.
        _insert_builtin_session(chronicle_path, "sess-b", summary="B")
        _insert_builtin_turn(chronicle_path, "sess-b", 0, "q", "a")
        full_db.enrich_session(
            ChatSession(
                session_id="sess-b",
                workspace_name="proj",
                workspace_path="/tmp/proj",
                messages=[ChatMessage(role="user", content="q")],
                created_at="2025-01-15T10:00:00Z",
            )
        )

        needing = full_db.discover_sessions_needing_enrichment()
        ids = [r["session_id"] for r in needing]
        assert "sess-a" in ids
        assert "sess-b" not in ids

    def test_discover_detects_grown_session_not_unchanged(self, full_db):
        """A session is re-flagged only when its Chronicle turn count changes.

        Guards against the prior bug where already-current sessions were
        re-flagged on every scan because Chronicle turns were compared against
        the enriched user-message count (an incompatible quantity).
        """
        chronicle_path = str(full_db.chronicle_db_path)
        _insert_builtin_session(chronicle_path, "sess-grow", summary="G")
        _insert_builtin_turn(chronicle_path, "sess-grow", 0, "q1", "a1")
        _insert_builtin_turn(chronicle_path, "sess-grow", 1, "q2", "a2")

        # Enrich at 2 turns — the enriched session has more user messages than
        # turns, which previously caused a perpetual false positive.
        full_db.enrich_session(
            ChatSession(
                session_id="sess-grow",
                workspace_name="proj",
                workspace_path="/tmp/proj",
                messages=[
                    ChatMessage(role="user", content="q1"),
                    ChatMessage(role="assistant", content="a1"),
                    ChatMessage(role="user", content="q2"),
                ],
                created_at="2025-01-15T10:00:00Z",
            )
        )

        # Unchanged → not flagged (the core regression guard).
        assert "sess-grow" not in [r["session_id"] for r in full_db.discover_sessions_needing_enrichment()]

        # A new Chronicle turn arrives → flagged again.
        _insert_builtin_turn(chronicle_path, "sess-grow", 2, "q3", "a3")
        assert "sess-grow" in [r["session_id"] for r in full_db.discover_sessions_needing_enrichment()]

    def test_discover_converges_after_enrichment(self, full_db):
        """Enriching a discovered session removes it from the next discovery pass."""
        chronicle_path = str(full_db.chronicle_db_path)
        _insert_builtin_session(chronicle_path, "sess-conv", summary="C")
        _insert_builtin_turn(chronicle_path, "sess-conv", 0, "q", "a")

        first = [r["session_id"] for r in full_db.discover_sessions_needing_enrichment()]
        assert "sess-conv" in first

        full_db.enrich_session(
            ChatSession(
                session_id="sess-conv",
                workspace_name="proj",
                workspace_path="/tmp/proj",
                messages=[ChatMessage(role="user", content="q")],
                created_at="2025-01-15T10:00:00Z",
            )
        )

        second = [r["session_id"] for r in full_db.discover_sessions_needing_enrichment()]
        assert "sess-conv" not in second

    # -- 8. delete_cst_session ---------------------------------------------

    def test_delete_cst_session(self, full_db):
        """Deleting a cst_session removes it; get_session still falls back to Chronicle."""
        chronicle_path = str(full_db.chronicle_db_path)
        _insert_builtin_session(chronicle_path, "sess-del")
        _insert_builtin_turn(chronicle_path, "sess-del", 0, "hi", "hello")
        full_db.add_session(
            ChatSession(
                session_id="sess-del",
                workspace_name="proj",
                workspace_path="/tmp/proj",
                messages=[ChatMessage(role="user", content="hi")],
                created_at="2025-01-15T10:00:00Z",
            )
        )

        # Enriched row exists
        assert full_db.delete_cst_session("sess-del") is True
        # Second delete returns False (already gone)
        assert full_db.delete_cst_session("sess-del") is False
        # list_sessions should still find it via Chronicle (unenriched)
        sessions = full_db.list_sessions()
        matched = [s for s in sessions if s["session_id"] == "sess-del"]
        assert len(matched) == 1
        assert matched[0]["is_enriched"] is False

    # -- 9. builtin read methods -------------------------------------------

    def test_builtin_read_methods(self, full_db):
        """list_builtin_sessions, get_builtin_session, get_builtin_turns, count_builtin_turns."""
        chronicle_path = str(full_db.chronicle_db_path)
        _insert_builtin_session(
            chronicle_path,
            "sess-read",
            summary="Read test",
            repository="org/repo",
            cwd="/tmp/proj",
        )
        _insert_builtin_turn(chronicle_path, "sess-read", 0, "Q0", "A0")
        _insert_builtin_turn(chronicle_path, "sess-read", 1, "Q1", "A1")

        # list_builtin_sessions
        listed = full_db.list_builtin_sessions()
        ids = [r["id"] for r in listed]
        assert "sess-read" in ids

        # get_builtin_session
        s = full_db.get_builtin_session("sess-read")
        assert s is not None
        assert s["summary"] == "Read test"
        assert s["repository"] == "org/repo"
        assert s["cwd"] == "/tmp/proj"

        # get_builtin_turns
        turns = full_db.get_builtin_turns("sess-read")
        assert len(turns) == 2
        assert turns[0]["user_message"] == "Q0"
        assert turns[1]["user_message"] == "Q1"

        # count_builtin_turns
        assert full_db.count_builtin_turns("sess-read") == 2

    # -- 10. unenriched_only mode ------------------------------------------

    def test_unenriched_only_mode(self, tmp_path):
        """unenriched_only=True makes has_cst_tables return False even when tables exist."""
        # Create Chronicle sibling
        chronicle_path = str(tmp_path / "session-store.db")
        _create_builtin_only_db(chronicle_path)

        db_path = str(tmp_path / "copilot-session-tools.db")

        # Normal mode — cst_* tables ARE created by constructor
        db_normal = Database(db_path)
        assert db_normal.has_cst_tables() is True

        # unenriched_only mode — same DB, but reports no cst_* tables
        db_unonly = Database(db_path, unenriched_only=True)
        assert db_unonly.has_cst_tables() is False

        # list_sessions still returns Chronicle sessions
        _insert_builtin_session(chronicle_path, "sess-un", summary="Unenriched only")
        _insert_builtin_turn(chronicle_path, "sess-un", 0, "q", "a")
        sessions = db_unonly.list_sessions()
        matched = [s for s in sessions if s["session_id"] == "sess-un"]
        assert len(matched) == 1
        assert matched[0]["is_enriched"] is False


class TestSchemaV6:
    """Tests for the v6 schema and migration from v5."""

    def test_v6_schema_has_new_columns(self):
        """Fresh DB created via ensure_schema has all v6 columns and views."""
        from copilot_session_tools.db_storage import ensure_schema

        conn = sqlite3.connect(":memory:")
        ensure_schema(conn)

        # Verify parent_message_id and child_index on cst_messages
        msg_cols = {row[1] for row in conn.execute("PRAGMA table_info(cst_messages)")}
        assert "parent_message_id" in msg_cols
        assert "child_index" in msg_cols

        # Verify child_message_id on cst_content_blocks
        cb_cols = {row[1] for row in conn.execute("PRAGMA table_info(cst_content_blocks)")}
        assert "child_message_id" in cb_cols
        # nested_data should NOT exist (removed in v6)
        assert "nested_data" not in cb_cols

        # Verify views exist by querying them (empty result is fine)
        for view_name in ("cst_messages_tree", "cst_all_tool_invocations", "cst_subagent_summary"):
            conn.execute(f"SELECT * FROM {view_name} LIMIT 0")  # noqa: S608

        conn.close()

    def test_v5_to_v6_migration(self):
        """Simulating a v5 DB triggers drop/recreate migration to v6."""
        from copilot_session_tools.db_storage import CST_SCHEMA_VERSION, ensure_schema

        conn = sqlite3.connect(":memory:")
        # Bootstrap a v6 schema first, then downgrade version to simulate v5
        ensure_schema(conn)
        conn.execute("UPDATE cst_schema_version SET version = 5")

        # Insert dummy data that should be wiped by migration
        conn.execute(
            "INSERT INTO cst_sessions (session_id, parser_version) VALUES (?, ?)",
            ("old-session", 1),
        )
        assert conn.execute("SELECT COUNT(*) FROM cst_sessions").fetchone()[0] == 1

        # Re-run ensure_schema — should detect v5 and drop/recreate
        ensure_schema(conn)

        # Schema version should now be current
        version = conn.execute("SELECT version FROM cst_schema_version").fetchone()[0]
        assert version == CST_SCHEMA_VERSION

        # Old data should be gone (drop/recreate wipes everything)
        assert conn.execute("SELECT COUNT(*) FROM cst_sessions").fetchone()[0] == 0
        assert (
            conn.execute(
                "SELECT name FROM cst_sources WHERE source_id = 'default-cli'",
            ).fetchone()[0]
            == "cli"
        )

        # New columns should exist
        msg_cols = {row[1] for row in conn.execute("PRAGMA table_info(cst_messages)")}
        assert "parent_message_id" in msg_cols
        assert "child_index" in msg_cols

        conn.close()

    def test_v10_to_v11_migration_adds_source_event_id_before_index(self):
        """Simulating a v10 DB adds source_event_id before replaying current indexes."""
        from copilot_session_tools.db_storage import CST_SCHEMA_VERSION, ensure_schema

        conn = sqlite3.connect(":memory:")
        ensure_schema(conn)
        conn.execute("DROP INDEX IF EXISTS idx_cst_messages_source_event")
        conn.execute("ALTER TABLE cst_messages DROP COLUMN source_event_id")
        conn.execute("UPDATE cst_schema_version SET version = 10")

        ensure_schema(conn)

        msg_cols = {row[1] for row in conn.execute("PRAGMA table_info(cst_messages)")}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(cst_messages)")}
        version = conn.execute("SELECT version FROM cst_schema_version").fetchone()[0]
        assert "source_event_id" in msg_cols
        assert "idx_cst_messages_source_event" in indexes
        assert version == CST_SCHEMA_VERSION

        conn.close()


class TestChronicleSchemaVersion:
    """Tests for Chronicle schema compatibility warnings."""

    def test_known_chronicle_schema_version_does_not_warn(self):
        """Chronicle schema v4 is currently supported by CST's read queries."""
        import warnings

        from copilot_session_tools.db_storage import CHRONICLE_SCHEMA_VERSION, check_builtin_schema_version

        conn = sqlite3.connect(":memory:")
        conn.execute("ATTACH DATABASE ':memory:' AS chronicle")
        conn.execute("CREATE TABLE chronicle.schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO chronicle.schema_version VALUES (?)", (CHRONICLE_SCHEMA_VERSION,))

        with warnings.catch_warnings(record=True) as recorded:
            warnings.simplefilter("always")
            check_builtin_schema_version(conn)

        assert recorded == []
        conn.close()

    def test_future_chronicle_schema_version_warns(self):
        """A future Chronicle schema still warns so parser compatibility is revisited."""
        from copilot_session_tools.db_storage import CHRONICLE_SCHEMA_VERSION, check_builtin_schema_version

        conn = sqlite3.connect(":memory:")
        conn.execute("ATTACH DATABASE ':memory:' AS chronicle")
        conn.execute("CREATE TABLE chronicle.schema_version (version INTEGER NOT NULL)")
        conn.execute("INSERT INTO chronicle.schema_version VALUES (?)", (CHRONICLE_SCHEMA_VERSION + 1,))

        with pytest.warns(UserWarning, match=f"newer than expected \\({CHRONICLE_SCHEMA_VERSION}\\)"):
            check_builtin_schema_version(conn)

        conn.close()


class TestEnrichmentFailureDefault:
    """Verify that enrichment failures do not stamp the version by default."""

    def test_stamp_version_on_failure_defaults_to_false(self):
        """_enrich_session_batch must default stamp_version_on_failure to False."""
        import inspect

        from copilot_session_tools.refresh import _enrich_session_batch

        sig = inspect.signature(_enrich_session_batch)
        assert sig.parameters["stamp_version_on_failure"].default is False


class TestSubagentChildMessages:
    """Tests for subagent child message DB insertion → retrieval roundtrip."""

    @pytest.fixture
    def child_session_db(self, tmp_path):
        """Database with a session containing parent + child subagent messages."""
        db = Database(tmp_path / "child_msg.db")
        child_msg = ChatMessage(
            role="assistant",
            content="Child found 3 files",
            agent_display_name="Explore Agent: Find files",
            agent_nesting_level=1,
            tool_invocations=[ToolInvocation(name="grep", input="pattern", result="3 matches")],
            command_runs=[CommandRun(command="uv run pytest", result="passed")],
            content_blocks=[ContentBlock(kind="text", content="Child found 3 files")],
        )
        parent_msg = ChatMessage(
            role="assistant",
            content="Let me search for files.",
            content_blocks=[
                ContentBlock(kind="text", content="Let me search for files."),
                ContentBlock(
                    kind="subagent",
                    content="Child found 3 files",
                    description="Explore Agent: Find files",
                    child_message=child_msg,
                ),
            ],
            children=[child_msg],
        )
        session = ChatSession(
            session_id="test-child-roundtrip",
            workspace_name="test",
            workspace_path="/tmp/test",
            messages=[
                ChatMessage(role="user", content="Find files"),
                parent_msg,
            ],
            created_at="2025-06-01T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)
        return db

    def test_child_message_roundtrip(self, child_session_db):
        """Insert session with child message, retrieve via get_session, verify tree."""
        retrieved = child_session_db.get_session("test-child-roundtrip")
        assert retrieved is not None
        assert len(retrieved.messages) == 2  # user + parent assistant

        parent_msg = retrieved.messages[1]
        assert len(parent_msg.children) == 1

        # ContentBlock has child_message populated
        subagent_block = parent_msg.content_blocks[1]
        assert subagent_block.kind == "subagent"
        assert subagent_block.child_message is not None

        child = subagent_block.child_message
        assert child.agent_nesting_level == 1
        assert child.agent_display_name == "Explore Agent: Find files"

        # Tool invocations preserved
        assert len(child.tool_invocations) == 1
        assert child.tool_invocations[0].name == "grep"
        assert child.tool_invocations[0].input == "pattern"
        assert child.tool_invocations[0].result == "3 matches"

        # Command runs preserved
        assert len(child.command_runs) == 1
        assert child.command_runs[0].command == "uv run pytest"
        assert child.command_runs[0].result == "passed"

    def test_child_message_parent_link(self, child_session_db):
        """Verify parent_message_id linkage at the DB level."""
        with sqlite3.connect(str(child_session_db.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id, parent_message_id, child_index, agent_nesting_level FROM cst_messages WHERE session_id = ? ORDER BY id",
                ("test-child-roundtrip",),
            )
            rows = [dict(r) for r in cursor.fetchall()]

        # Should have 3 rows: user, parent assistant, child assistant
        assert len(rows) == 3

        # User and parent have no parent_message_id
        parents = [r for r in rows if r["parent_message_id"] is None]
        children = [r for r in rows if r["parent_message_id"] is not None]
        assert len(parents) == 2  # user + parent assistant
        assert len(children) == 1

        child_row = children[0]
        next(r for r in parents if (r["agent_nesting_level"] == 0 and r["id"] != parents[0]["id"]) or r == parents[1])
        # The child's parent_message_id should point to the assistant parent
        assert child_row["parent_message_id"] == parents[1]["id"]
        assert child_row["agent_nesting_level"] == 1
        assert child_row["child_index"] is not None

    def test_content_block_child_message_id(self, child_session_db):
        """Verify cst_content_blocks has child_message_id set on subagent blocks."""
        with sqlite3.connect(str(child_session_db.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # Get the parent assistant message id
            cursor.execute(
                "SELECT id FROM cst_messages WHERE session_id = ? AND parent_message_id IS NULL AND role = 'assistant'",
                ("test-child-roundtrip",),
            )
            parent_id = cursor.fetchone()["id"]

            # Get the child message id
            cursor.execute(
                "SELECT id FROM cst_messages WHERE session_id = ? AND parent_message_id IS NOT NULL",
                ("test-child-roundtrip",),
            )
            child_id = cursor.fetchone()["id"]

            # Check content block for subagent kind
            cursor.execute(
                "SELECT child_message_id FROM cst_content_blocks WHERE kind = 'subagent' AND message_id = ?",
                (parent_id,),
            )
            block_row = cursor.fetchone()
            assert block_row is not None
            assert block_row["child_message_id"] == child_id

            # Non-subagent blocks should have NULL child_message_id
            cursor.execute(
                "SELECT child_message_id FROM cst_content_blocks WHERE kind = 'text' AND message_id = ?",
                (parent_id,),
            )
            text_block = cursor.fetchone()
            assert text_block is not None
            assert text_block["child_message_id"] is None

    def test_message_count_excludes_children(self, tmp_path):
        """list_sessions and get_stats should count only parent messages, not children."""
        db = Database(tmp_path / "count_test.db")
        child_msg = ChatMessage(
            role="assistant",
            content="child work",
            agent_nesting_level=1,
            agent_display_name="Task Agent",
            content_blocks=[ContentBlock(kind="text", content="child work")],
        )
        session = ChatSession(
            session_id="count-test",
            workspace_name="test",
            workspace_path="/tmp/test",
            messages=[
                ChatMessage(role="user", content="msg1"),
                ChatMessage(
                    role="assistant",
                    content="parent1",
                    content_blocks=[
                        ContentBlock(kind="text", content="parent1"),
                        ContentBlock(kind="subagent", content="", child_message=child_msg),
                    ],
                ),
                ChatMessage(role="user", content="msg2"),
                ChatMessage(role="assistant", content="parent2"),
            ],
            created_at="2025-06-01T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)

        # list_sessions should count 4 top-level messages (not 5)
        sessions = db.list_sessions()
        target = [s for s in sessions if s["session_id"] == "count-test"]
        assert len(target) == 1
        assert target[0]["message_count"] == 4

        # get_stats should also exclude children
        stats = db.get_stats()
        assert stats["message_count"] >= 4
        # Verify by direct query
        with sqlite3.connect(str(db.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cst_messages WHERE session_id = 'count-test'")
            total_rows = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM cst_messages WHERE session_id = 'count-test' AND parent_message_id IS NULL")
            parent_rows = cursor.fetchone()[0]
        assert total_rows == 5  # 4 top-level + 1 child
        assert parent_rows == 4

    def test_get_messages_markdown_excludes_children(self, child_session_db):
        """get_messages_markdown should not render child messages as separate top-level sections."""
        md = child_session_db.get_messages_markdown("test-child-roundtrip")
        assert md is not None
        assert len(md) > 0

        # Count top-level message sections ("## Message N:")
        # Should have 2 top-level messages, not 3
        lines = md.split("\n")
        header_lines = [line for line in lines if line.startswith("## Message")]
        assert len(header_lines) == 2

    def test_sql_views(self, child_session_db):
        """Verify SQL views return correct data for parent/child messages."""
        with sqlite3.connect(str(child_session_db.db_path)) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()

            # -- cst_messages_tree --
            cursor.execute(
                "SELECT * FROM cst_messages_tree WHERE session_id = ? ORDER BY id",
                ("test-child-roundtrip",),
            )
            tree_rows = [dict(r) for r in cursor.fetchall()]
            assert len(tree_rows) == 3  # user, parent, child

            child_tree = [r for r in tree_rows if r["parent_message_id"] is not None]
            assert len(child_tree) == 1
            assert child_tree[0]["agent_display_name"] == "Explore Agent: Find files"
            assert child_tree[0]["parent_agent_name"] is None  # parent is a plain assistant (no agent_display_name)
            assert child_tree[0]["agent_nesting_level"] == 1

            parent_tree = [r for r in tree_rows if r["parent_message_id"] is None and r["role"] == "assistant"]
            assert len(parent_tree) == 1
            assert parent_tree[0]["parent_agent_name"] is None

            # -- cst_all_tool_invocations --
            cursor.execute(
                "SELECT * FROM cst_all_tool_invocations WHERE session_id = ?",
                ("test-child-roundtrip",),
            )
            tool_rows = [dict(r) for r in cursor.fetchall()]
            assert len(tool_rows) == 1
            assert tool_rows[0]["name"] == "grep"
            assert tool_rows[0]["agent_display_name"] == "Explore Agent: Find files"
            assert tool_rows[0]["agent_nesting_level"] == 1
            assert tool_rows[0]["parent_message_id"] is not None

            # -- cst_subagent_summary --
            cursor.execute(
                "SELECT * FROM cst_subagent_summary WHERE session_id = ?",
                ("test-child-roundtrip",),
            )
            summary_rows = [dict(r) for r in cursor.fetchall()]
            assert len(summary_rows) == 1
            assert summary_rows[0]["agent_display_name"] == "Explore Agent: Find files"
            assert summary_rows[0]["tool_count"] == 1
            assert summary_rows[0]["command_count"] == 1
            assert summary_rows[0]["content_block_count"] == 1

    def test_grandchild_message_roundtrip(self, tmp_path):
        """Test recursive child→grandchild nesting survives insert and retrieval."""
        db = Database(tmp_path / "grandchild_msg.db")

        # Create grandchild message (nesting_level=2)
        grandchild_msg = ChatMessage(
            role="assistant",
            content="Grandchild result",
            agent_display_name="Deep Agent",
            agent_nesting_level=2,
            content_blocks=[ContentBlock(kind="text", content="Grandchild result")],
        )

        # Create child message containing grandchild
        child_msg = ChatMessage(
            role="assistant",
            content="Child delegated work",
            agent_display_name="Mid Agent",
            agent_nesting_level=1,
            content_blocks=[
                ContentBlock(kind="text", content="Child delegated work"),
                ContentBlock(
                    kind="subagent",
                    content="Grandchild result",
                    description="Deep Agent",
                    child_message=grandchild_msg,
                ),
            ],
            children=[grandchild_msg],
        )

        # Create parent message containing child
        parent_msg = ChatMessage(
            role="assistant",
            content="Starting search",
            content_blocks=[
                ContentBlock(kind="text", content="Starting search"),
                ContentBlock(
                    kind="subagent",
                    content="Child delegated work",
                    description="Mid Agent",
                    child_message=child_msg,
                ),
            ],
            children=[child_msg],
        )

        session = ChatSession(
            session_id="test-grandchild",
            workspace_name="test",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="Go deep"),
                parent_msg,
            ],
            created_at="2025-06-01T10:00:00Z",
            vscode_edition="stable",
        )
        db.add_session(session)

        # Retrieve and verify the full tree
        retrieved = db.get_session("test-grandchild")
        assert retrieved is not None
        assert len(retrieved.messages) == 2  # user + parent

        parent = retrieved.messages[1]
        assert len(parent.children) == 1

        child = parent.children[0]
        assert child.agent_display_name == "Mid Agent"
        assert child.agent_nesting_level == 1
        assert len(child.children) == 1

        grandchild = child.children[0]
        assert grandchild.agent_display_name == "Deep Agent"
        assert grandchild.agent_nesting_level == 2
        assert grandchild.content == "Grandchild result"

        # Verify content block linkage at each level
        child_block = parent.content_blocks[1]
        assert child_block.kind == "subagent"
        assert child_block.child_message is not None
        assert child_block.child_message.agent_display_name == "Mid Agent"

        grandchild_block = child_block.child_message.content_blocks[1]
        assert grandchild_block.kind == "subagent"
        assert grandchild_block.child_message is not None
        assert grandchild_block.child_message.agent_display_name == "Deep Agent"

        # Verify DB row count: user + parent + child + grandchild = 4
        with sqlite3.connect(str(db.db_path)) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM cst_messages WHERE session_id = 'test-grandchild'")
            assert cursor.fetchone()[0] == 4
