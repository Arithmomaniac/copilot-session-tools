"""Tests for the markdown exporter module."""

import pytest

from copilot_session_tools import (
    ChatMessage,
    ChatSession,
    CommandRun,
    ContentBlock,
    FileChange,
    ToolInvocation,
    export_session_to_file,
    generate_session_filename,
    session_to_markdown,
)
from copilot_session_tools.markdown_exporter import (
    _format_command_runs_summary,
    _format_timestamp,
    _format_tool_summary,
    _had_thinking_content,
)
from copilot_session_tools.utils import sanitize_filename as _sanitize_filename


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
        ],
        created_at="1704067200000",  # 2024-01-01 00:00:00
        updated_at="1704067260000",
        source_file="/path/to/session.json",
        vscode_edition="stable",
    )


@pytest.fixture
def session_with_thinking():
    """Create a session with thinking blocks."""
    return ChatSession(
        session_id="thinking-session-456",
        workspace_name="thinking-project",
        workspace_path="/home/user/thinking-project",
        messages=[
            ChatMessage(role="user", content="Think about this problem"),
            ChatMessage(
                role="assistant",
                content="Here's my answer after thinking.",
                content_blocks=[
                    ContentBlock(kind="thinking", content="Let me think about this..."),
                    ContentBlock(kind="text", content="Here's my answer after thinking."),
                ],
            ),
        ],
        created_at="1704067200000",
        vscode_edition="stable",
    )


@pytest.fixture
def session_with_tools():
    """Create a session with tool invocations."""
    return ChatSession(
        session_id="tools-session-789",
        workspace_name="tools-project",
        workspace_path="/home/user/tools-project",
        messages=[
            ChatMessage(role="user", content="Create a file"),
            ChatMessage(
                role="assistant",
                content="I've created the file.",
                tool_invocations=[
                    ToolInvocation(name="file_creator", input="test.py", result="Created"),
                    ToolInvocation(name="code_editor", input="edit test.py", result="Edited"),
                ],
                file_changes=[
                    FileChange(path="test.py", diff="+# New file"),
                ],
            ),
        ],
        created_at="1704067200000",
        vscode_edition="stable",
    )


class TestFormatTimestamp:
    """Tests for timestamp formatting."""

    def test_format_milliseconds_timestamp(self):
        """Test formatting a milliseconds timestamp."""
        result = _format_timestamp("1704067200000")
        assert "2024-01-01" in result

    def test_format_seconds_timestamp(self):
        """Test formatting a seconds timestamp."""
        result = _format_timestamp(1704067200)
        assert "2024-01-01" in result

    def test_format_none_timestamp(self):
        """Test formatting None timestamp."""
        result = _format_timestamp(None)
        assert result == "Unknown"

    def test_format_invalid_timestamp(self):
        """Test formatting invalid timestamp."""
        result = _format_timestamp("not-a-timestamp")
        assert result == "not-a-timestamp"


class TestSessionToMarkdown:
    """Tests for session_to_markdown function."""

    def test_basic_export(self, sample_session):
        """Test basic markdown export."""
        markdown = session_to_markdown(sample_session)

        assert "# Chat Session" in markdown
        assert "my-project" in markdown
        assert "test-session-123" in markdown
        assert "## Message 1: **USER**" in markdown
        assert "## Message 2: **ASSISTANT**" in markdown
        assert "How do I create a Python function?" in markdown
        assert "```python" in markdown

    def test_metadata_section(self, sample_session):
        """Test that metadata section is included."""
        markdown = session_to_markdown(sample_session)

        assert "## Metadata" in markdown
        assert "**Session ID:**" in markdown
        assert "**Workspace:**" in markdown
        assert "**Created:**" in markdown
        assert "**Edition:**" in markdown
        assert "**Messages:**" in markdown

    def test_horizontal_rules(self, sample_session):
        """Test that messages are separated by horizontal rules."""
        markdown = session_to_markdown(sample_session)

        # Count horizontal rules
        rule_count = markdown.count("\n---\n")
        # Should have at least 2: after metadata, after each message
        assert rule_count >= 2

    def test_thinking_blocks_omitted(self, session_with_thinking):
        """Test that thinking block content is completely omitted."""
        markdown = session_to_markdown(session_with_thinking)

        # Thinking content should be omitted
        assert "Let me think about this" not in markdown

        # Should NOT have any trace of thinking
        assert "*[Was thinking...]*" not in markdown
        assert "💭" not in markdown

        # Non-thinking content should be present
        assert "Here's my answer after thinking." in markdown

    def test_thinking_blocks_included_when_requested(self, session_with_thinking):
        """Test that thinking blocks are included when include_thinking=True."""
        markdown = session_to_markdown(session_with_thinking, content_set={"thinking", "agent-details", "tools", "commands", "file-changes"})

        # Thinking content should be included
        assert "Let me think about this" in markdown

        # Should be in a collapsible details block
        assert "<details>" in markdown
        assert "💭 Thinking" in markdown

        # Should NOT have the old blockquote format
        assert "> **Thinking:**" not in markdown

        # Non-thinking content should also be present
        assert "Here's my answer after thinking." in markdown

    def test_tool_summary_in_italics(self, session_with_tools):
        """Test that tool summaries are in italics with emoji."""
        markdown = session_to_markdown(session_with_tools)

        # Tool summary should have emoji prefix
        assert "*🔧" in markdown
        assert "file_creator" in markdown

    def test_file_changes_summary(self, session_with_tools):
        """Test that file changes are summarized."""
        markdown = session_to_markdown(session_with_tools)

        # File changes should be summarized with emoji prefix
        assert "*📄 Changed:" in markdown
        assert "test.py" in markdown

    def test_custom_title_shown(self):
        """Test that custom title is shown when available."""
        session = ChatSession(
            session_id="custom-title-session",
            workspace_name="workspace",
            workspace_path="/path",
            messages=[ChatMessage(role="user", content="Hello")],
            custom_title="My Custom Title",
        )
        markdown = session_to_markdown(session)

        assert "**Title:** My Custom Title" in markdown


class TestExportSessionToFile:
    """Tests for export_session_to_file function."""

    def test_export_to_file(self, sample_session, tmp_path):
        """Test exporting to a file."""
        output_path = tmp_path / "test_export.md"
        export_session_to_file(sample_session, output_path)

        assert output_path.exists()
        content = output_path.read_text()
        assert "# Chat Session" in content
        assert "my-project" in content


class TestGenerateSessionFilename:
    """Tests for generate_session_filename function."""

    def test_filename_with_custom_title(self):
        """Test filename generation with custom title."""
        session = ChatSession(
            session_id="test-123",
            workspace_name="workspace",
            workspace_path="/path",
            messages=[],
            custom_title="My Custom Session",
            created_at="1704067200000",
        )
        filename = generate_session_filename(session)

        assert filename.endswith(".md")
        assert "My_Custom_Session" in filename
        assert "test-123" in filename

    def test_filename_with_workspace(self):
        """Test filename generation with workspace name."""
        session = ChatSession(
            session_id="test-456",
            workspace_name="my-project",
            workspace_path="/path",
            messages=[],
            created_at="1704067200000",
        )
        filename = generate_session_filename(session)

        assert filename.endswith(".md")
        assert "my-project" in filename

    def test_filename_with_date(self):
        """Test filename includes date when available."""
        session = ChatSession(
            session_id="test-789",
            workspace_name="project",
            workspace_path="/path",
            messages=[],
            created_at="1704067200000",  # 2024-01-01
        )
        filename = generate_session_filename(session)

        assert "20240101" in filename

    def test_filename_sanitization(self):
        """Test that unsafe characters are removed from filename."""
        session = ChatSession(
            session_id="test-bad-chars",
            workspace_name="path/to/project:name",
            workspace_path="/path",
            messages=[],
        )
        filename = generate_session_filename(session)

        assert "/" not in filename
        assert ":" not in filename
        assert filename.endswith(".md")


class TestHadThinkingContent:
    """Tests for _had_thinking_content helper."""

    def test_no_content_blocks(self):
        """Test message with no content blocks."""
        message = ChatMessage(role="assistant", content="Hello")
        assert _had_thinking_content(message) is False

    def test_no_thinking_blocks(self):
        """Test message with only text blocks."""
        message = ChatMessage(
            role="assistant",
            content="Hello",
            content_blocks=[ContentBlock(kind="text", content="Hello")],
        )
        assert _had_thinking_content(message) is False

    def test_has_thinking_blocks(self):
        """Test message with thinking blocks."""
        message = ChatMessage(
            role="assistant",
            content="Hello",
            content_blocks=[
                ContentBlock(kind="thinking", content="Thinking..."),
                ContentBlock(kind="text", content="Hello"),
            ],
        )
        assert _had_thinking_content(message) is True


class TestFormatToolSummary:
    """Tests for _format_tool_summary helper."""

    def test_no_tools(self):
        """Test message with no tools."""
        message = ChatMessage(role="assistant", content="Hello")
        assert _format_tool_summary(message) == ""

    def test_single_tool(self):
        """Test message with single tool."""
        message = ChatMessage(
            role="assistant",
            content="Hello",
            tool_invocations=[ToolInvocation(name="my_tool")],
        )
        result = _format_tool_summary(message)
        assert "*🔧 Tool: my_tool*" in result

    def test_multiple_tools(self):
        """Test message with multiple tools."""
        message = ChatMessage(
            role="assistant",
            content="Hello",
            tool_invocations=[
                ToolInvocation(name="tool1"),
                ToolInvocation(name="tool2"),
            ],
        )
        result = _format_tool_summary(message)
        assert "*🔧 Tools:" in result
        assert "tool1" in result
        assert "tool2" in result

    def test_tool_with_input_included(self):
        """Test that tool inputs are included when flag is set."""
        message = ChatMessage(
            role="assistant",
            content="Hello",
            tool_invocations=[
                ToolInvocation(name="run_in_terminal", input="npm run test"),
            ],
        )
        result = _format_tool_summary(message, include_inputs=True)
        assert "*🔧 Tool: run_in_terminal*" in result
        assert "**run_in_terminal input:**" in result
        assert "```" in result
        assert "npm run test" in result

    def test_tool_without_input_when_not_included(self):
        """Test that tool inputs are not included when flag is False."""
        message = ChatMessage(
            role="assistant",
            content="Hello",
            tool_invocations=[
                ToolInvocation(name="run_in_terminal", input="npm run test"),
            ],
        )
        result = _format_tool_summary(message, include_inputs=False)
        assert "*🔧 Tool: run_in_terminal*" in result
        assert "npm run test" not in result
        assert "```" not in result

    def test_json_input_gets_language_tag(self):
        """Test that JSON tool inputs get ```json language tag and are prettified."""
        message = ChatMessage(
            role="assistant",
            content="Hello",
            tool_invocations=[
                ToolInvocation(name="grep", input='{"pattern":"TODO","path":"src/"}'),
            ],
        )
        result = _format_tool_summary(message, include_inputs=True)
        assert "```json" in result
        assert '"pattern": "TODO"' in result  # prettified


class TestFormatFileChangesSummary:
    """Tests for _format_file_changes_summary helper with diffs."""

    def test_no_file_changes(self):
        """Test message with no file changes."""
        from copilot_session_tools.markdown_exporter import _format_file_changes_summary

        message = ChatMessage(role="assistant", content="Hello")
        assert _format_file_changes_summary(message) == ""

    def test_file_changes_with_diff_included(self):
        """Test that file diffs are included when flag is set."""
        from copilot_session_tools.markdown_exporter import _format_file_changes_summary

        message = ChatMessage(
            role="assistant",
            content="Hello",
            file_changes=[
                FileChange(path="test.py", diff="+ def test():\n+     pass"),
            ],
        )
        result = _format_file_changes_summary(message, include_diffs=True)
        assert "*📄 Changed: test.py*" in result
        assert "**test.py:**" in result
        assert "```diff" in result
        assert "+ def test():" in result

    def test_file_changes_without_diff_when_not_included(self):
        """Test that file diffs are not included when flag is False."""
        from copilot_session_tools.markdown_exporter import _format_file_changes_summary

        message = ChatMessage(
            role="assistant",
            content="Hello",
            file_changes=[
                FileChange(path="test.py", diff="+ def test():\n+     pass"),
            ],
        )
        result = _format_file_changes_summary(message, include_diffs=False)
        assert "*📄 Changed: test.py*" in result
        assert "```diff" not in result
        assert "def test():" not in result


class TestSessionToMarkdownWithOptions:
    """Tests for session_to_markdown with include_diffs and include_tool_inputs options."""

    def test_with_tool_inputs(self):
        """Test markdown export with tool inputs included."""
        session = ChatSession(
            session_id="test-session",
            workspace_name="test",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="Run tests"),
                ChatMessage(
                    role="assistant",
                    content="Running tests...",
                    tool_invocations=[
                        ToolInvocation(name="run_in_terminal", input="pytest"),
                    ],
                ),
            ],
        )
        markdown = session_to_markdown(session, content_set={"tool-inputs", "tools", "commands", "file-changes", "agent-details"})
        assert "pytest" in markdown
        assert "```" in markdown

    def test_with_file_diffs(self):
        """Test markdown export with file diffs included."""
        session = ChatSession(
            session_id="test-session",
            workspace_name="test",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="Create a file"),
                ChatMessage(
                    role="assistant",
                    content="Creating file...",
                    file_changes=[
                        FileChange(path="new_file.py", diff="+ # New file\n+ print('hello')"),
                    ],
                ),
            ],
        )
        markdown = session_to_markdown(session, content_set={"diffs", "tools", "commands", "file-changes", "agent-details"})
        assert "```diff" in markdown
        assert "print('hello')" in markdown

    def test_without_options_no_details(self):
        """Test markdown export without options doesn't include details."""
        session = ChatSession(
            session_id="test-session",
            workspace_name="test",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="Do something"),
                ChatMessage(
                    role="assistant",
                    content="Done!",
                    tool_invocations=[
                        ToolInvocation(name="run_in_terminal", input="ls -la"),
                    ],
                    file_changes=[
                        FileChange(path="file.py", diff="+ new content"),
                    ],
                ),
            ],
        )
        # Default content_set includes tools/file-changes but not diffs/tool-inputs
        markdown = session_to_markdown(session)
        assert "ls -la" not in markdown
        assert "new content" not in markdown
        # But summaries should still be there
        assert "*🔧 Tool:" in markdown
        assert "*📄 Changed:" in markdown


class TestSanitizeFilename:
    """Tests for _sanitize_filename helper."""

    def test_safe_characters_unchanged(self):
        """Test that safe characters are unchanged."""
        result = _sanitize_filename("my-project_v1.0")
        assert result == "my-project_v1.0"

    def test_unsafe_characters_replaced(self):
        """Test that unsafe characters are replaced with underscores."""
        result = _sanitize_filename("path/to:project name")
        assert "/" not in result
        assert ":" not in result
        assert " " not in result
        assert "_" in result

    def test_max_length_enforced(self):
        """Test that max length is enforced."""
        long_name = "a" * 100
        result = _sanitize_filename(long_name, max_length=50)
        assert len(result) == 50

    def test_custom_max_length(self):
        """Test custom max length."""
        result = _sanitize_filename("test-project", max_length=5)
        assert len(result) == 5
        assert result == "test-"

    def test_empty_string(self):
        """Test empty string input."""
        result = _sanitize_filename("")
        assert result == ""

    def test_all_unsafe_characters(self):
        """Test string with all unsafe characters."""
        result = _sanitize_filename("!@#$%^&*()")
        # All characters should be replaced with underscores
        assert all(c == "_" for c in result)


class TestCommandRunDescriptionInMarkdown:
    """Tests for command run description/title display in markdown export."""

    def test_command_run_summary_uses_title_when_available(self):
        """Test that _format_command_runs_summary uses title instead of raw command."""
        message = ChatMessage(
            role="assistant",
            content="Running a search.",
            command_runs=[
                CommandRun(
                    command="cd C:\\_SRC\\repo && uv run copilot-chat-archive search 'query' --full --limit 10",
                    title="Search tenant query build SPN",
                    status="success",
                ),
            ],
        )
        result = _format_command_runs_summary(message)
        assert "Search tenant query build SPN" in result
        assert "copilot-chat-archive" not in result

    def test_command_run_summary_falls_back_to_command_without_title(self):
        """Test that _format_command_runs_summary falls back to truncated command."""
        message = ChatMessage(
            role="assistant",
            content="Running a command.",
            command_runs=[
                CommandRun(
                    command="git status",
                    title=None,
                    status="success",
                ),
            ],
        )
        result = _format_command_runs_summary(message)
        assert "git status" in result

    def test_inline_toolinvocation_block_uses_description_for_commands(self):
        """Test that toolInvocation content blocks show description for command runs."""
        session = ChatSession(
            session_id="cmd-desc-test",
            workspace_name="test",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="Search for something"),
                ChatMessage(
                    role="assistant",
                    content="Searching...",
                    content_blocks=[
                        ContentBlock(
                            kind="toolInvocation",
                            content="$ cd /repo && uv run search 'query'",
                            description="Search ARG test account access",
                        ),
                    ],
                ),
            ],
        )
        md = session_to_markdown(session)
        assert "Search ARG test account access" in md
        # Raw command should NOT appear since description is available
        assert "$ cd /repo" not in md

    def test_inline_toolinvocation_block_uses_content_for_regular_tools(self):
        """Test that toolInvocation content blocks show content for non-command tools."""
        session = ChatSession(
            session_id="tool-desc-test",
            workspace_name="test",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="Read a file"),
                ChatMessage(
                    role="assistant",
                    content="Reading...",
                    content_blocks=[
                        ContentBlock(
                            kind="toolInvocation",
                            content="Searching for `pattern` in `path`",
                            description="grep",
                        ),
                    ],
                ),
            ],
        )
        md = session_to_markdown(session)
        # Should use the pretty content, not the raw tool name
        assert "Searching for `pattern` in `path`" in md


class TestAgentBlockquotes:
    """Test agent subagent content block rendering in markdown."""

    @pytest.mark.parametrize("include_system", [False, True])
    def test_delegated_stream_content_and_visibility(self, include_system):
        prompt = ChatMessage(role="user", content="Inspect the workflow")
        system = ChatMessage(role="system", content="Runtime context")
        answer = ChatMessage(
            role="assistant",
            content="Verified workflow output",
            content_blocks=[ContentBlock(kind="thinking", content="Private analysis"), ContentBlock(kind="text", content="Verified workflow output")],
        )
        envelope = ChatMessage(role="assistant", content="", children=[prompt, system, answer])
        parent = ChatMessage(role="assistant", content="", content_blocks=[ContentBlock(kind="subagent", content="(no output)", child_message=envelope)])
        session = ChatSession(session_id="workflow-export", workspace_name=None, workspace_path=None, messages=[parent])
        content_set = {"agent-details"} | ({"system-messages"} if include_system else set())
        markdown = session_to_markdown(session, content_set=content_set)
        assert prompt.content in markdown and answer.content in markdown
        assert (system.content in markdown) == include_system
        assert "Private analysis" not in markdown
        assert "(no output)" not in markdown
        summary = session_to_markdown(session, content_set=set())
        assert answer.content not in summary

    def test_subagent_block_rendered_as_details(self):
        """ContentBlock(kind='subagent') should render as a collapsible details block."""
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                content_blocks=[
                    ContentBlock(kind="subagent", content="Found auth patterns in 3 files", description="Explore Agent"),
                ],
            ),
        ]
        session = ChatSession(
            session_id="test-123",
            workspace_name="test",
            workspace_path="/test",
            messages=messages,
        )
        md = session_to_markdown(session)
        assert "<details>" in md
        assert "Explore Agent" in md
        assert "> Found auth patterns in 3 files" in md

    def test_subagent_block_with_agent_number(self):
        """Subagent block with agent number in description renders correctly."""
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                content_blocks=[
                    ContentBlock(kind="subagent", content="Test results: all passed", description="Task Agent: Run tests"),
                ],
            ),
        ]
        session = ChatSession(
            session_id="test-numbered",
            workspace_name="test",
            workspace_path="/test",
            messages=messages,
        )
        md = session_to_markdown(session)
        assert "<details>" in md
        assert "Task Agent: Run tests" in md
        assert "> Test results: all passed" in md

    def test_non_agent_messages_not_blockquoted(self):
        """Top-level messages should not be blockquoted."""
        messages = [
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="World"),
        ]
        session = ChatSession(
            session_id="test-123",
            workspace_name="test",
            workspace_path="/test",
            messages=messages,
        )
        md = session_to_markdown(session)
        lines = md.split("\n")
        # Content lines shouldn't start with > (except empty lines or header)
        content_lines = [line for line in lines if line.strip() and not line.startswith("#") and not line.startswith("-") and not line.startswith("*")]
        blockquoted = [line for line in content_lines if line.startswith(">")]
        assert len(blockquoted) == 0, f"Non-agent messages shouldn't be blockquoted, got: {blockquoted}"

    def test_subagent_block_summary_includes_completed(self):
        """The details summary should include the completed label."""
        messages = [
            ChatMessage(
                role="assistant",
                content="",
                content_blocks=[
                    ContentBlock(kind="subagent", content="Result text", description="Code Review Agent"),
                ],
            ),
        ]
        session = ChatSession(
            session_id="test-header",
            workspace_name="test",
            workspace_path="/test",
            messages=messages,
        )
        md = session_to_markdown(session)
        assert "Code Review Agent" in md
        assert "completed" in md


# ── content_set tests ─────────────────────────────────────────────────


def _rich_session() -> ChatSession:
    """Session with all content types for content_set testing."""
    return ChatSession(
        session_id="content-set-test",
        workspace_name="test-ws",
        workspace_path="/test",
        messages=[
            ChatMessage(role="user", content="Do everything"),
            ChatMessage(
                role="assistant",
                content="Done.",
                content_blocks=[
                    ContentBlock(kind="thinking", content="Reasoning about the task..."),
                    ContentBlock(kind="text", content="Here is my answer."),
                ],
                tool_invocations=[
                    ToolInvocation(name="grep", input="pattern", result="match"),
                ],
                command_runs=[
                    CommandRun(command="npm test", title="Run tests", status="success"),
                ],
                file_changes=[
                    FileChange(path="src/app.py", diff="+new line"),
                ],
            ),
        ],
        created_at="1704067200000",
        vscode_edition="stable",
    )


def _system_message_session() -> ChatSession:
    """Session containing a system-role message for include/exclude testing."""
    return ChatSession(
        session_id="system-md-test",
        workspace_name="test-ws",
        workspace_path="/test",
        messages=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="system", content="Hidden by default"),
            ChatMessage(role="assistant", content="Done"),
        ],
        created_at="1704067200000",
        vscode_edition="stable",
    )


class TestEmojiPrefixes:
    """Emoji prefixes appear on tool, file, and command summary lines."""

    def test_tool_summary_has_emoji(self):
        """Tool summary lines include 🔧 emoji."""
        msg = ChatMessage(
            role="assistant",
            content="Done.",
            tool_invocations=[ToolInvocation(name="grep", input="x", result="y")],
        )
        result = _format_tool_summary(msg)
        assert "🔧" in result

    def test_file_summary_has_emoji(self):
        """File change summary lines include 📄 emoji."""
        session = _rich_session()
        md = session_to_markdown(session, content_set={"file-changes"})
        assert "📄" in md

    def test_command_summary_has_emoji(self):
        """Command run summary lines include ⚡ emoji."""
        msg = ChatMessage(
            role="assistant",
            content="Running.",
            command_runs=[CommandRun(command="npm test", title="Run tests", status="success")],
        )
        result = _format_command_runs_summary(msg)
        assert "⚡" in result


class TestThinkingBlocks:
    """Thinking block inclusion/exclusion via content_set."""

    def test_thinking_included_renders_as_details(self):
        """When thinking is in content_set, renders as <details> block."""
        session = _rich_session()
        md = session_to_markdown(session, content_set={"thinking", "agent-details"})
        assert "<details>" in md
        assert "💭 Thinking" in md
        assert "Reasoning about the task..." in md

    def test_thinking_excluded_fully_omitted(self):
        """When thinking is NOT in content_set, no trace in output."""
        session = _rich_session()
        md = session_to_markdown(session, content_set={"agent-details", "tools", "commands", "file-changes"})
        assert "thinking" not in md.lower().replace("here's my answer after thinking.", "")
        assert "[Was thinking]" not in md
        assert "💭" not in md


class TestContentSetSuppression:
    """T5/T6/T7 suppression via content_set."""

    def test_exclude_tools_omits_tool_lines(self):
        """Excluding 'tools' removes all tool-related lines."""
        session = _rich_session()
        md = session_to_markdown(session, content_set={"commands", "file-changes"})
        assert "🔧" not in md
        assert "grep" not in md

    def test_exclude_commands_omits_command_lines(self):
        """Excluding 'commands' removes all command-related lines."""
        session = _rich_session()
        md = session_to_markdown(session, content_set={"tools", "file-changes"})
        assert "⚡" not in md
        assert "Run tests" not in md

    def test_exclude_file_changes_omits_file_lines(self):
        """Excluding 'file-changes' removes all file-related lines."""
        session = _rich_session()
        md = session_to_markdown(session, content_set={"tools", "commands"})
        assert "📄" not in md
        assert "src/app.py" not in md


class TestContentSetEdgeCases:
    """Edge cases for content_set handling."""

    def test_all_content_excluded(self):
        """When everything is excluded, only message headers and prose remain."""
        session = _rich_session()
        md = session_to_markdown(session, content_set=set())
        # Header is still present
        assert "## Message" in md
        # Prose text from the text content block still renders
        assert "Here is my answer." in md
        # None of the optional content types appear
        assert "🔧" not in md
        assert "📄" not in md
        assert "⚡" not in md
        assert "💭" not in md

    def test_all_content_included(self):
        """When everything is included, all content renders."""
        session = _rich_session()
        full_set = {"thinking", "diffs", "tool-inputs", "agent-details", "tools", "commands", "file-changes"}
        md = session_to_markdown(session, content_set=full_set)
        assert "🔧" in md
        assert "📄" in md
        assert "⚡" in md
        assert "💭" in md

    def test_content_set_none_uses_defaults(self):
        """content_set=None uses DEFAULT_INCLUDES (tools, commands, file-changes, agent-details)."""
        from copilot_session_tools.content_types import DEFAULT_INCLUDES

        session = _rich_session()
        md_none = session_to_markdown(session, content_set=None)
        md_explicit = session_to_markdown(session, content_set=DEFAULT_INCLUDES.copy())
        assert md_none == md_explicit


class TestSystemMessagesContentSet:
    """System-role message inclusion/exclusion in markdown export."""

    def test_system_messages_omitted_by_default(self):
        """System messages are omitted when content_set uses defaults."""
        md = session_to_markdown(_system_message_session())
        assert "Hidden by default" not in md
        assert "## Message 2: **SYSTEM**" not in md

    def test_system_messages_included_when_requested(self):
        """Adding system-messages includes system-role content."""
        md = session_to_markdown(_system_message_session(), content_set={"agent-details", "tools", "commands", "file-changes", "system-messages"})
        assert "Hidden by default" in md
        assert "## Message 2: **SYSTEM**" in md

    def test_system_messages_excluded_when_not_in_content_set(self):
        """System messages are omitted when explicitly excluded from a full set."""
        full_set = {"thinking", "diffs", "tool-inputs", "agent-details", "tools", "commands", "file-changes", "system-messages"}
        md = session_to_markdown(_system_message_session(), content_set=full_set - {"system-messages"})
        assert "Hidden by default" not in md
