"""Tests for the HTML exporter module."""

import pytest

from copilot_session_tools import (
    ChatMessage,
    ChatSession,
    ContentBlock,
    ToolInvocation,
    export_session_to_html_file,
    generate_session_html_filename,
    session_to_html,
)


@pytest.fixture
def sample_session():
    """Create a sample chat session for testing."""
    return ChatSession(
        session_id="html-test-session-123",
        workspace_name="my-project",
        workspace_path="/home/user/projects/my-project",
        messages=[
            ChatMessage(role="user", content="Hello, can you help me?"),
            ChatMessage(role="assistant", content="Sure, I can help!"),
        ],
        created_at="1700000000000",
        updated_at="1700001000000",
        vscode_edition="stable",
    )


@pytest.fixture
def session_with_tools():
    """Create a session with tool invocations and content blocks."""
    return ChatSession(
        session_id="tools-test-session",
        workspace_name="tools-project",
        workspace_path="/home/user/tools-project",
        messages=[
            ChatMessage(
                role="user",
                content="Fix this bug",
            ),
            ChatMessage(
                role="assistant",
                content="Let me look at the code.",
                tool_invocations=[
                    ToolInvocation(
                        name="copilot_readFile",
                        input='{"path": "src/main.py"}',
                        result="def main(): pass",
                        invocation_message="Reading src/main.py",
                    ),
                ],
                content_blocks=[
                    ContentBlock(kind="text", content="Let me look at the code."),
                    ContentBlock(kind="toolInvocation", content="Reading `copilot_readFile`"),
                ],
            ),
        ],
        vscode_edition="stable",
    )


@pytest.fixture
def session_with_thinking():
    """Create a session with thinking blocks and subagent content."""
    return ChatSession(
        session_id="thinking-test-session",
        workspace_name="thinking-project",
        workspace_path="/home/user/thinking-project",
        messages=[
            ChatMessage(role="user", content="Fix the bug"),
            ChatMessage(
                role="assistant",
                content="I'll investigate the issue.",
                content_blocks=[
                    ContentBlock(kind="thinking", content="Let me analyze the error carefully."),
                    ContentBlock(kind="text", content="I'll investigate the issue."),
                    ContentBlock(
                        kind="subagent",
                        content="Agent explored the codebase.",
                        description="explorer agent",
                        content_blocks=[
                            ContentBlock(kind="text", content="Found the root cause in the parser."),
                            ContentBlock(kind="toolInvocation", content="Reading file"),
                        ],
                        tool_invocations=[
                            ToolInvocation(
                                name="readFile",
                                input='{"path": "bug.py"}',
                                result="buggy code",
                            ),
                        ],
                    ),
                ],
                tool_invocations=[
                    ToolInvocation(
                        name="readFile",
                        input='{"path": "main.py"}',
                        result="def main(): pass",
                        source_type="mcp",
                    ),
                ],
            ),
        ],
        vscode_edition="stable",
    )


@pytest.fixture
def session_with_system_message():
    """Create a session containing a system-role message."""
    return ChatSession(
        session_id="system-test-session",
        workspace_name="system-project",
        workspace_path="/home/user/system-project",
        messages=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="system", content="System instructions go here"),
            ChatMessage(role="assistant", content="Got it"),
        ],
        vscode_edition="stable",
    )


class TestSessionToHtml:
    """Tests for session_to_html function."""

    def test_returns_complete_html_document(self, sample_session):
        """Test that output is a complete HTML document."""
        html = session_to_html(sample_session)
        assert "<!DOCTYPE html>" in html
        assert "<html lang=" in html
        assert "</html>" in html
        assert "<head>" in html
        assert "<body" in html

    def test_contains_session_metadata(self, sample_session):
        """Test that session metadata is included."""
        html = session_to_html(sample_session)
        assert "my-project" in html
        assert "html-test-session-123" in html

    def test_contains_message_content(self, sample_session):
        """Test that message content appears in the output."""
        html = session_to_html(sample_session)
        assert "Hello, can you help me?" in html
        assert "Sure, I can help!" in html

    def test_top_level_messages_are_collapsible_open_by_default(self, sample_session):
        """User and assistant messages render as open-by-default details."""
        html = session_to_html(sample_session)
        assert html.count('<details class="message-details" open>') == 2
        assert '<summary class="message-header">' in html
        assert '<span class="collapsible-icon" aria-hidden="true">▼</span>' in html

    def test_fork_status_escaped_session_name_is_not_rendered_as_html(self):
        """Test that escaped fork link labels remain text when rendered as Markdown."""
        session_id = "11111111-1111-1111-1111-111111111111"
        session = ChatSession(
            session_id="fork-xss-test",
            workspace_name="fork-project",
            workspace_path="/home/user/fork-project",
            messages=[
                ChatMessage(
                    role="assistant",
                    content="",
                    content_blocks=[
                        ContentBlock(
                            kind="status",
                            content=f"Forked as [&lt;img src=x onerror=alert(1)&gt;](/session/{session_id}).",
                            description="fork",
                        )
                    ],
                )
            ],
        )

        html = session_to_html(session)
        assert '<a href="/session/11111111-1111-1111-1111-111111111111">&lt;img src=x onerror=alert(1)&gt;</a>' in html
        assert "<img src=x" not in html

    def test_static_mode_no_toolbar(self, sample_session):
        """Test that static HTML has no copy toolbar HTML element."""
        html = session_to_html(sample_session)
        assert '<div class="copy-markdown-toolbar">' not in html

    def test_static_mode_no_copy_buttons(self, sample_session):
        """Test that static HTML has no per-message copy buttons."""
        html = session_to_html(sample_session)
        assert 'class="message-copy-btn"' not in html

    def test_static_mode_no_font_awesome(self, sample_session):
        """Test that static HTML has no Font Awesome CDN link."""
        html = session_to_html(sample_session)
        assert "cdnjs.cloudflare.com" not in html

    def test_static_mode_no_ajax(self, sample_session):
        """Test that static HTML has no AJAX/fetch code."""
        html = session_to_html(sample_session)
        assert "buildMarkdownParams" not in html
        assert "api/markdown" not in html

    def test_static_mode_no_back_link(self, sample_session):
        """Test that static HTML has no back link."""
        html = session_to_html(sample_session)
        assert "Back to all sessions" not in html

    def test_static_mode_no_header(self, sample_session):
        """Test that static HTML has no sticky header."""
        html = session_to_html(sample_session)
        # The <header> element should not be present
        assert "<header>" not in html

    def test_static_mode_full_width(self, sample_session):
        """Test that static HTML has full-width container."""
        html = session_to_html(sample_session)
        assert "--container-max-width: none" in html

    def test_static_mode_larger_scroll_heights(self, sample_session):
        """Test that static HTML has unlimited pre max-heights."""
        html = session_to_html(sample_session)
        assert "--pre-max-height: none" in html
        assert "--cmd-max-height: none" in html

    def test_contains_shared_js(self, sample_session):
        """Test that shared JS (toggle, code-collapse) is still present."""
        html = session_to_html(sample_session)
        assert "collapsible-icon" in html
        assert "code-collapse-wrapper" in html

    def test_contains_dark_mode_css(self, sample_session):
        """Test that dark mode CSS variables are present."""
        html = session_to_html(sample_session)
        assert "prefers-color-scheme: dark" in html

    def test_with_tool_invocations(self, session_with_tools):
        """Test that tool invocations render correctly."""
        html = session_to_html(session_with_tools)
        assert "Let me look at the code" in html

    def test_message_anchors_present(self, sample_session):
        """Test that message anchors are present for navigation."""
        html = session_to_html(sample_session)
        assert 'id="msg-1"' in html
        assert 'id="msg-2"' in html

    def test_session_to_html_with_content_set(self, session_with_thinking):
        """content_set parameter controls which content types appear in HTML."""
        html = session_to_html(session_with_thinking, content_set={"agent-details"})
        # Agent details should be included (full subagent content)
        assert "Found the root cause" in html
        # Thinking blocks should NOT be rendered (not in content_set)
        assert '<details class="thinking-block" open>' not in html

    def test_session_to_html_content_set_none_uses_defaults(self, session_with_thinking):
        """content_set=None uses default includes."""
        html = session_to_html(session_with_thinking, content_set=None)
        assert "<!DOCTYPE html>" in html
        # Default includes agent-details, so subagent content should be present
        assert "Found the root cause" in html
        # Default does NOT include 'thinking', so thinking elements not rendered
        assert '<details class="thinking-block" open>' not in html

    def test_session_to_html_excludes_thinking(self, session_with_thinking):
        """Thinking blocks excluded when 'thinking' not in content_set."""
        html = session_to_html(session_with_thinking, content_set={"agent-details"})
        assert '<details class="thinking-block" open>' not in html
        # But message text should still be present
        assert "investigate the issue" in html

    def test_session_to_html_includes_thinking_when_requested(self, session_with_thinking):
        """Thinking blocks included when 'thinking' IS in content_set."""
        html = session_to_html(session_with_thinking, content_set={"agent-details", "thinking"})
        assert '<details class="thinking-block" open>' in html
        assert "analyze the error" in html

    def test_session_to_html_excludes_agent_details(self, session_with_thinking):
        """Agent content stripped when 'agent-details' not in content_set."""
        html = session_to_html(session_with_thinking, content_set={"thinking"})
        # Without agent-details, subagent shows result-only mode
        assert "result-block" in html
        # Text result from subagent should still appear
        assert "Found the root cause" in html

    def test_session_to_html_static_has_no_alpine(self, sample_session):
        """Static HTML export does not include Alpine.js."""
        html = session_to_html(sample_session)
        assert "alpinejs" not in html
        assert 'x-data="' not in html
        assert '<button class="settings-btn"' not in html

    def test_system_messages_omitted_by_default_in_static_export(self, session_with_system_message):
        """System-role messages are excluded from default static HTML exports."""
        html = session_to_html(session_with_system_message)
        assert '<div class="message system"' not in html
        assert "System instructions go here" not in html

    def test_system_messages_included_open_by_default(self, session_with_system_message):
        """Including system-messages renders open collapsible details in static export."""
        html = session_to_html(session_with_system_message, content_set={"agent-details", "tools", "commands", "file-changes", "system-messages"})
        assert 'class="message-details system-message-details" open' in html
        assert '<span class="collapsible-icon" aria-hidden="true">▼</span>' in html
        assert "System instructions go here" in html
        assert "expand-system-messages" not in html

    @pytest.mark.parametrize("include_system", [False, True])
    def test_nested_system_messages_respect_content_set(self, sample_session, include_system):
        system = ChatMessage(role="system", content="Delegated system instructions")
        assistant = ChatMessage(role="assistant", content="Delegated answer")
        nested_blocks = [ContentBlock(kind="subagent", description=message.role.title(), content=message.content, child_message=message) for message in (system, assistant)]
        sample_session.messages[1].content_blocks = [
            ContentBlock(
                kind="subagent",
                content="",
                description="Delegated task",
                content_blocks=nested_blocks,
                child_message=ChatMessage(
                    role="assistant",
                    content="",
                    content_blocks=nested_blocks,
                ),
            )
        ]
        content_set = {"agent-details"}
        if include_system:
            content_set.add("system-messages")

        html = session_to_html(sample_session, content_set=content_set)

        assert (system.content in html) is include_system
        assert assistant.content in html

    def test_auxiliary_sections_are_collapsible_open_by_default(self, session_with_thinking):
        """Representative auxiliary HTML sections render as open-by-default details."""
        html = session_to_html(session_with_thinking, content_set={"agent-details", "thinking", "tools"})
        assert '<details class="thinking-block" open>' in html
        assert '<details class="subagent-block" open>' in html
        assert '<details class="tool-invocation-wrapper" open>' in html


class TestExportSessionToHtmlFile:
    """Tests for export_session_to_html_file function."""

    def test_creates_html_file(self, sample_session, tmp_path):
        """Test that an HTML file is created."""
        output_path = tmp_path / "test_session.html"
        export_session_to_html_file(sample_session, output_path)
        assert output_path.exists()

    def test_file_contains_valid_html(self, sample_session, tmp_path):
        """Test that the created file contains valid HTML."""
        output_path = tmp_path / "test_session.html"
        export_session_to_html_file(sample_session, output_path)
        content = output_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "</html>" in content

    def test_file_is_utf8(self, tmp_path):
        """Test that the file is UTF-8 encoded with special characters."""
        session = ChatSession(
            session_id="utf8-test",
            workspace_name="utf8-project",
            workspace_path="/tmp/test",
            messages=[
                ChatMessage(role="user", content="Héllo wörld 日本語"),
            ],
            vscode_edition="stable",
        )
        output_path = tmp_path / "utf8_test.html"
        export_session_to_html_file(session, output_path)
        content = output_path.read_text(encoding="utf-8")
        assert "Héllo wörld 日本語" in content


class TestGenerateSessionHtmlFilename:
    """Tests for generate_session_html_filename function."""

    def test_html_extension(self, sample_session):
        """Test that filename has .html extension."""
        filename = generate_session_html_filename(sample_session)
        assert filename.endswith(".html")

    def test_contains_workspace_name(self, sample_session):
        """Test that filename contains workspace name."""
        filename = generate_session_html_filename(sample_session)
        assert "my-project" in filename or "my_project" in filename

    def test_mirrors_markdown_filename_structure(self, sample_session):
        """Test that HTML filename mirrors markdown filename structure."""
        from copilot_session_tools import generate_session_filename

        md_filename = generate_session_filename(sample_session)
        html_filename = generate_session_html_filename(sample_session)

        # Should be the same except for extension
        assert md_filename.replace(".md", ".html") == html_filename
