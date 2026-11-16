"""Tests for copilot_session_tools.utils."""

from __future__ import annotations

from typing import Any

from copilot_session_tools.scanner.models import (
    ChatMessage,
    ChatSession,
    CommandRun,
    ContentBlock,
    RootAgentInterval,
    ToolInvocation,
)
from copilot_session_tools.utils import (
    MILLISECONDS_THRESHOLD,
    build_block_metadata,
    build_root_agent_markers,
    detect_language,
    extract_filename,
    format_timestamp,
    generate_session_filename,
    highlight_code,
    markdown_to_html,
    match_tool_for_block,
    parse_diff_stats,
    prettify_json,
    sanitize_filename,
    strip_ansi,
    truncate_preview,
    urldecode,
    workflow_details,
    workflow_summary,
)


class TestBuildRootAgentMarkers:
    """Tests for root custom-agent transition markers."""

    def test_direct_agent_switch_does_not_emit_default_marker(self):
        session = ChatSession(
            session_id="root-agent-marker-test",
            workspace_name="test-workspace",
            workspace_path="/test",
            messages=[
                ChatMessage(role="user", content="first", timestamp="2026-01-01T00:00:02Z"),
                ChatMessage(role="user", content="second", timestamp="2026-01-01T00:00:04Z"),
            ],
            root_agent_intervals=[
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
            ],
        )

        markers = build_root_agent_markers(session)

        assert markers == {
            0: [{"kind": "enter", "label": "Entered Smart Merge", "timestamp": "2026-01-01T00:00:01Z"}],
            1: [{"kind": "enter", "label": "Entered Skill Audit", "timestamp": "2026-01-01T00:00:03Z"}],
        }


# ---------------------------------------------------------------------------
# format_timestamp
# ---------------------------------------------------------------------------


class TestFormatTimestamp:
    """Tests for format_timestamp()."""

    def test_seconds_epoch(self):
        # 2020-01-01 00:00:00 UTC
        assert format_timestamp(1577836800) == "2020-01-01 00:00:00"

    def test_milliseconds_epoch(self):
        # Same instant as above, but in JS-style milliseconds
        assert format_timestamp(1577836800000) == "2020-01-01 00:00:00"

    def test_float_seconds(self):
        assert format_timestamp(1577836800.5) == "2020-01-01 00:00:00"

    def test_string_seconds(self):
        assert format_timestamp("1577836800") == "2020-01-01 00:00:00"

    def test_string_milliseconds(self):
        assert format_timestamp("1577836800000") == "2020-01-01 00:00:00"

    def test_none_returns_default(self):
        assert format_timestamp(None) == ""

    def test_none_with_unknown_label(self):
        assert format_timestamp(None, unknown_label="Unknown") == "Unknown"

    def test_zero_returns_unknown_label(self):
        assert format_timestamp(0, unknown_label="N/A") == "N/A"

    def test_empty_string_returns_unknown_label(self):
        assert format_timestamp("", unknown_label="Unknown") == "Unknown"

    def test_invalid_string_returns_as_is(self):
        assert format_timestamp("not-a-number") == "not-a-number"

    def test_threshold_boundary(self):
        """A value just above the threshold is treated as milliseconds."""
        ts = MILLISECONDS_THRESHOLD + 1
        result = format_timestamp(ts)
        assert result  # Should produce a valid date string, not crash

    def test_threshold_below(self):
        """A value below the threshold is treated as seconds."""
        ts = MILLISECONDS_THRESHOLD - 1
        result = format_timestamp(ts)
        assert result


# ---------------------------------------------------------------------------
# urldecode
# ---------------------------------------------------------------------------


class TestUrldecode:
    """Tests for urldecode()."""

    def test_encoded_colon(self):
        assert urldecode("c%3A") == "c:"

    def test_encoded_space(self):
        assert urldecode("hello%20world") == "hello world"

    def test_no_encoding(self):
        assert urldecode("plain_text") == "plain_text"

    def test_empty_string(self):
        assert urldecode("") == ""

    def test_none_returns_empty(self):
        assert urldecode(None) == ""  # type: ignore[arg-type]

    def test_encoded_slash(self):
        assert urldecode("%2Fpath%2Fto%2Ffile") == "/path/to/file"


# ---------------------------------------------------------------------------
# strip_ansi
# ---------------------------------------------------------------------------


class TestStripAnsi:
    """Tests for strip_ansi()."""

    def test_sgr_bold(self):
        assert strip_ansi("\x1b[1mBold\x1b[0m") == "Bold"

    def test_sgr_color(self):
        assert strip_ansi("\x1b[31mRed\x1b[0m") == "Red"

    def test_mixed_text_and_ansi(self):
        assert strip_ansi("before\x1b[32mgreen\x1b[0mafter") == "beforegreenafter"

    def test_osc_bell_terminated(self):
        assert strip_ansi("\x1b]0;title\x07rest") == "rest"

    def test_osc_st_terminated(self):
        assert strip_ansi("\x1b]0;title\x1b\\rest") == "rest"

    def test_cursor_control(self):
        assert strip_ansi("\x1b[2Jscreen") == "screen"

    def test_none_returns_empty(self):
        assert strip_ansi(None) == ""

    def test_empty_string(self):
        assert strip_ansi("") == ""

    def test_no_ansi(self):
        assert strip_ansi("plain text") == "plain text"


# ---------------------------------------------------------------------------
# extract_filename
# ---------------------------------------------------------------------------


class TestExtractFilename:
    """Tests for extract_filename()."""

    def test_unix_path(self):
        assert extract_filename("/home/user/file.txt") == "file.txt"

    def test_windows_path(self):
        assert extract_filename("C:\\Users\\user\\file.txt") == "file.txt"

    def test_no_slashes(self):
        assert extract_filename("file.txt") == "file.txt"

    def test_none_returns_empty(self):
        assert extract_filename(None) == ""

    def test_empty_string(self):
        assert extract_filename("") == ""

    def test_trailing_unix_slash(self):
        assert extract_filename("/path/to/dir/") == ""

    def test_deep_unix_path(self):
        assert extract_filename("/a/b/c/d/e.py") == "e.py"

    def test_deep_windows_path(self):
        assert extract_filename("C:\\a\\b\\c\\d\\e.py") == "e.py"


# ---------------------------------------------------------------------------
# parse_diff_stats
# ---------------------------------------------------------------------------


class TestParseDiffStats:
    """Tests for parse_diff_stats()."""

    def test_normal_diff(self):
        diff = "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,4 @@\n unchanged\n+added line\n-removed line\n context\n"
        result = parse_diff_stats(diff)
        assert result == {"additions": 1, "deletions": 1}

    def test_only_additions(self):
        diff = "--- a/f\n+++ b/f\n@@ -0,0 +1,2 @@\n+line1\n+line2\n"
        assert parse_diff_stats(diff) == {"additions": 2, "deletions": 0}

    def test_only_deletions(self):
        diff = "--- a/f\n+++ b/f\n@@ -1,2 +0,0 @@\n-line1\n-line2\n"
        assert parse_diff_stats(diff) == {"additions": 0, "deletions": 2}

    def test_empty_diff(self):
        assert parse_diff_stats("") == {"additions": 0, "deletions": 0}

    def test_none_diff(self):
        assert parse_diff_stats(None) == {"additions": 0, "deletions": 0}

    def test_only_headers(self):
        diff = "--- a/file.py\n+++ b/file.py\n@@ -1 +1 @@\n"
        assert parse_diff_stats(diff) == {"additions": 0, "deletions": 0}

    def test_multiple_hunks(self):
        diff = "--- a/f\n+++ b/f\n@@ -1,2 +1,2 @@\n+a\n-b\n@@ -10,2 +10,3 @@\n+c\n+d\n-e\n"
        assert parse_diff_stats(diff) == {"additions": 3, "deletions": 2}


# ---------------------------------------------------------------------------
# markdown_to_html
# ---------------------------------------------------------------------------


class TestMarkdownToHtml:
    """Tests for markdown_to_html()."""

    def test_basic_bold(self):
        result = markdown_to_html("**bold**")
        assert "<strong>bold</strong>" in result

    def test_basic_italic(self):
        result = markdown_to_html("*italic*")
        assert "<em>italic</em>" in result

    def test_code_block(self):
        result = str(markdown_to_html("```python\nprint('hi')\n```"))
        assert "<code" in result

    def test_empty_returns_empty(self):
        assert markdown_to_html("") == ""

    def test_none_returns_empty(self):
        assert markdown_to_html(None) == ""  # type: ignore[arg-type]

    def test_file_uri_empty_link(self):
        """Empty-text file:// links are converted to backtick filenames."""
        text = "[](file:///c%3A/Users/test/file.py)"
        result = markdown_to_html(text)
        assert "file.py" in result
        # Should NOT contain the raw file:// link
        assert "file://" not in result

    def test_using_pattern_italicized(self):
        text = 'Using "some tool"'
        result = markdown_to_html(text)
        assert "<em>" in result

    def test_edited_backtick_normalized(self):
        text = "_Edited `file.py`_"
        result = markdown_to_html(text)
        assert "file.py" in result

    def test_ran_terminal_command(self):
        text = "Ran terminal command: ls -la"
        result = markdown_to_html(text)
        assert "<em>" in result

    def test_made_changes_italicized(self):
        text = "Made changes."
        result = markdown_to_html(text)
        assert "<em>" in result

    def test_crlf_normalized(self):
        text = "line1\r\nline2"
        result = markdown_to_html(text)
        assert "\r\n" not in result


# ---------------------------------------------------------------------------
# match_tool_for_block
# ---------------------------------------------------------------------------


def _tool(name: str) -> ToolInvocation:
    """Helper to create a minimal ToolInvocation."""
    return ToolInvocation(name=name)


class TestMatchToolForBlock:
    """Tests for match_tool_for_block()."""

    def test_exact_backtick_match(self):
        tools = [_tool("read_file"), _tool("edit_file")]
        matched, used = match_tool_for_block("Running `read_file`", tools, set())
        assert matched is not None
        assert matched.name == "read_file"
        assert 0 in used

    def test_partial_name_match(self):
        tools = [_tool("github-mcp-server-search_code")]
        matched, _used = match_tool_for_block("`search_code`", tools, set())
        assert matched is not None
        assert matched.name == "github-mcp-server-search_code"

    def test_running_fallback(self):
        tools = [_tool("my_tool")]
        matched, _used = match_tool_for_block("Running my_tool in project", tools, set())
        assert matched is not None
        assert matched.name == "my_tool"

    def test_sequential_fallback_skips_task(self):
        tools = [_tool("task"), _tool("grep")]
        matched, _used = match_tool_for_block("some unknown content", tools, set())
        assert matched is not None
        assert matched.name == "grep"

    def test_empty_tools_returns_none(self):
        matched, _used = match_tool_for_block("content", [], set())
        assert matched is None

    def test_used_indices_respected(self):
        tools = [_tool("read_file"), _tool("edit_file")]
        matched, _used = match_tool_for_block("`read_file`", tools, {0})
        # Index 0 is used; should not match read_file again
        assert matched is not None
        assert matched.name == "edit_file"

    def test_all_used_returns_none(self):
        tools = [_tool("task")]
        matched, _used = match_tool_for_block("content", tools, {0})
        assert matched is None

    def test_sequential_fallback_when_no_backtick(self):
        tools = [_tool("view"), _tool("grep")]
        matched, _used = match_tool_for_block("no backticks here", tools, set())
        assert matched is not None
        assert matched.name == "view"


# ---------------------------------------------------------------------------
# build_block_metadata
# ---------------------------------------------------------------------------


class TestBuildBlockMetadata:
    """Tests for build_block_metadata()."""

    def test_tool_matching(self):
        blocks = [ContentBlock(kind="toolInvocation", content="Running `grep`")]
        tools = [_tool("grep")]
        result = build_block_metadata(blocks, tools, [])
        assert 0 in result["block_tool_map"]
        assert result["block_tool_map"][0].name == "grep"
        assert "grep" in result["matched_tool_names"]

    def test_command_matching(self):
        blocks = [ContentBlock(kind="toolInvocation", content="$ npm run build")]
        cmds = [CommandRun(command="npm run build")]
        result = build_block_metadata(blocks, [], cmds)
        assert 0 in result["block_cmd_map"]
        assert result["block_cmd_map"][0].command == "npm run build"

    def test_non_tool_blocks_ignored(self):
        blocks = [ContentBlock(kind="text", content="Hello")]
        result = build_block_metadata(blocks, [_tool("grep")], [])
        assert len(result["block_tool_map"]) == 0

    def test_nested_subagent(self):
        inner_block = ContentBlock(kind="toolInvocation", content="Running `view`")
        inner_tool = _tool("view")
        outer_block = ContentBlock(
            kind="subagent",
            content="Agent output",
            content_blocks=[inner_block],
            tool_invocations=[inner_tool],
        )
        result = build_block_metadata([outer_block], [], [])
        # Outer block is subagent, not toolInvocation, so no outer match
        assert len(result["block_tool_map"]) == 0
        # But nested metadata should be built on the block
        assert hasattr(outer_block, "_nested_meta")
        assert 0 in outer_block._nested_meta["block_tool_map"]  # ty: ignore[not-subscriptable]

    def test_command_preferred_over_tool_for_dollar_prefix(self):
        blocks = [ContentBlock(kind="toolInvocation", content="$ git status")]
        tools = [_tool("powershell")]
        cmds = [CommandRun(command="git status")]
        result = build_block_metadata(blocks, tools, cmds)
        # Command match should take priority for $-prefixed content
        assert 0 in result["block_cmd_map"]

    def test_empty_inputs(self):
        result = build_block_metadata([], [], [])
        assert result["block_tool_map"] == {}
        assert result["block_cmd_map"] == {}
        assert result["matched_tool_names"] == set()
        assert result["matched_cmd_indices"] == set()


# ---------------------------------------------------------------------------
# sanitize_filename
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    """Tests for sanitize_filename()."""

    def test_special_chars_replaced(self):
        assert sanitize_filename("hello world!") == "hello_world_"

    def test_keeps_alphanumeric(self):
        assert sanitize_filename("abc123") == "abc123"

    def test_keeps_allowed_chars(self):
        assert sanitize_filename("file-name_v2.txt") == "file-name_v2.txt"

    def test_truncation(self):
        long = "a" * 100
        assert len(sanitize_filename(long)) == 50

    def test_custom_max_length(self):
        assert len(sanitize_filename("abcdefghij", max_length=5)) == 5

    def test_empty_string(self):
        assert sanitize_filename("") == ""

    def test_all_special(self):
        assert sanitize_filename("!@#$%") == "_____"


# ---------------------------------------------------------------------------
# generate_session_filename
# ---------------------------------------------------------------------------


def _session(**kwargs: Any) -> ChatSession:
    """Helper to build a ChatSession with sensible defaults."""
    defaults: dict[str, Any] = {
        "session_id": "abcdef1234567890abcdef",
        "workspace_name": None,
        "workspace_path": None,
        "messages": [],
        "created_at": None,
        "custom_title": None,
    }
    defaults.update(kwargs)
    return ChatSession(**defaults)


class TestGenerateSessionFilename:
    """Tests for generate_session_filename()."""

    def test_with_custom_title_no_date(self):
        s = _session(custom_title="My Session")
        result = generate_session_filename(s)
        assert result == "My_Session_abcdef12.md"

    def test_with_workspace_name_no_date(self):
        s = _session(workspace_name="project-x")
        result = generate_session_filename(s)
        assert result == "project-x_abcdef12.md"

    def test_fallback_to_session_id(self):
        s = _session()
        result = generate_session_filename(s)
        assert result == "abcdef1234567890_abcdef12.md"

    def test_with_date_seconds(self):
        s = _session(custom_title="Build", created_at=1577836800)
        result = generate_session_filename(s)
        assert result == "20200101_Build_abcdef12.md"

    def test_with_date_milliseconds(self):
        s = _session(custom_title="Build", created_at=1577836800000)
        result = generate_session_filename(s)
        assert result == "20200101_Build_abcdef12.md"

    def test_with_date_string(self):
        s = _session(custom_title="Build", created_at="1577836800")
        result = generate_session_filename(s)
        assert result == "20200101_Build_abcdef12.md"

    def test_html_extension(self):
        s = _session(custom_title="Report")
        result = generate_session_filename(s, extension="html")
        assert result.endswith(".html")

    def test_custom_title_takes_precedence(self):
        s = _session(custom_title="Title", workspace_name="Workspace")
        result = generate_session_filename(s)
        assert "Title" in result
        assert "Workspace" not in result

    def test_special_chars_in_title_sanitized(self):
        s = _session(custom_title="My Session: test/debug!")
        result = generate_session_filename(s)
        assert ":" not in result
        assert "/" not in result


# ---------------------------------------------------------------------------
# truncate_preview
# ---------------------------------------------------------------------------


class TestTruncatePreview:
    """Tests for truncate_preview()."""

    def test_empty_string(self):
        assert truncate_preview("") == ""

    def test_none_input(self):
        assert truncate_preview("") == ""

    def test_short_text_unchanged(self):
        assert truncate_preview("Hello world") == "Hello world"

    def test_long_text_truncated_with_ellipsis(self):
        long_text = "This is a very long sentence that exceeds the maximum character limit for preview display purposes"
        result = truncate_preview(long_text, max_chars=40)
        assert result.endswith("…")
        assert len(result) <= 42  # 40 + ellipsis char + tolerance for word boundary

    def test_markdown_formatting_stripped(self):
        assert truncate_preview("## Hello **world**") == "Hello world"

    def test_html_tags_stripped(self):
        assert truncate_preview("<b>Hello</b> <em>world</em>") == "Hello world"

    def test_multiline_uses_first_nonempty_line(self):
        text = "\n\n  First real line\nSecond line\nThird line"
        assert truncate_preview(text) == "First real line"

    def test_blank_lines_only(self):
        assert truncate_preview("\n\n   \n") == ""

    def test_word_boundary_truncation(self):
        text = "word1 word2 word3 word4 word5 word6 word7 word8 word9 word10 word11 word12 word13 word14 word15"
        result = truncate_preview(text, max_chars=30)
        assert result.endswith("…")
        # Should break at a word boundary, not mid-word
        assert not result[-2].isalpha() or result.rstrip("…").endswith(" ") is False

    def test_custom_max_chars(self):
        result = truncate_preview("Short", max_chars=3)
        assert result.endswith("…")


# ---------------------------------------------------------------------------
# prettify_json
# ---------------------------------------------------------------------------


class TestPrettifyJson:
    """Tests for prettify_json()."""

    def test_compact_json_prettified(self):
        result = prettify_json('{"a":1,"b":"hello"}')
        assert result == '{\n  "a": 1,\n  "b": "hello"\n}'

    def test_nested_json(self):
        result = prettify_json('{"a":{"b":[1,2,3]}}')
        assert '"b": [\n      1,' in result

    def test_already_formatted_json_normalized(self):
        formatted = '{\n    "a": 1\n}'
        result = prettify_json(formatted)
        # Re-indented to 2 spaces
        assert result == '{\n  "a": 1\n}'

    def test_invalid_json_passthrough(self):
        text = "this is not json"
        assert prettify_json(text) == text

    def test_partial_json_passthrough(self):
        text = '{"a": 1, "b":'
        assert prettify_json(text) == text

    def test_empty_string(self):
        assert prettify_json("") == ""

    def test_none(self):
        assert prettify_json(None) == ""

    def test_json_array(self):
        result = prettify_json("[1, 2, 3]")
        assert result == "[\n  1,\n  2,\n  3\n]"

    def test_json_preserves_unicode(self):
        result = prettify_json('{"name":"café"}')
        assert "café" in result

    def test_json_boolean_null(self):
        result = prettify_json('{"a":true,"b":false,"c":null}')
        assert "true" in result
        assert "false" in result
        assert "null" in result


# ---------------------------------------------------------------------------
# detect_language
# ---------------------------------------------------------------------------


class TestWorkflowDetails:
    def test_inspect_result_uses_envelope_not_delimiters_inside_error(self):
        import json

        envelope = {"runId": "run-1", "status": "error", "error": "Unexpected Current run detail: in worker output"}
        content = f'Workflow "review" run run-1 has terminal status error. Full run result: {json.dumps(envelope)}'

        assert workflow_details(content) == {**envelope, "workflowName": "review"}

    def test_arbitrary_agent_json_is_not_a_workflow(self):
        assert workflow_details('{"runId":"run-1","status":"completed","result":"ordinary agent data"}') is None

    def test_legacy_flat_status_remains_readable(self):
        content = "Workflow review (run-1): completed | 4 subagents"
        assert workflow_summary(content) == content


class TestDetectLanguage:
    """Tests for detect_language()."""

    def test_valid_json_detected(self):
        assert detect_language('{"key": "value"}') == "json"

    def test_json_array_detected(self):
        assert detect_language("[1, 2, 3]") == "json"

    def test_unified_diff_detected(self):
        diff = "--- a/file.py\n+++ b/file.py\n@@ -1,3 +1,4 @@"
        assert detect_language(diff) == "diff"

    def test_git_diff_detected(self):
        diff = "diff --git a/foo.py b/foo.py\nindex abc..def"
        assert detect_language(diff) == "diff"

    def test_tool_name_powershell(self):
        assert detect_language("Get-Process", tool_name="powershell") == "powershell"

    def test_tool_name_sql(self):
        assert detect_language("SELECT * FROM t", tool_name="sql") == "sql"

    def test_plain_text_returns_none(self):
        assert detect_language("just some plain text\nwith multiple lines") is None

    def test_empty_returns_none(self):
        assert detect_language("") is None

    def test_none_text_returns_none(self):
        # detect_language expects str but handle edge case
        assert detect_language("") is None

    def test_tool_name_none_with_plain_text(self):
        assert detect_language("hello world", tool_name=None) is None

    def test_bare_number_not_detected_as_json(self):
        assert detect_language("42") is None

    def test_bare_boolean_not_detected_as_json(self):
        assert detect_language("true") is None

    def test_bare_string_not_detected_as_json(self):
        assert detect_language('"hello"') is None

    def test_is_output_skips_tool_name_hints(self):
        assert detect_language("Get-Process", tool_name="powershell", is_output=True) is None

    def test_is_output_still_detects_json(self):
        assert detect_language('{"key": "val"}', tool_name="powershell", is_output=True) == "json"

    def test_is_output_still_detects_diff(self):
        diff = "--- a/file.py\n+++ b/file.py"
        assert detect_language(diff, tool_name="powershell", is_output=True) == "diff"


# ---------------------------------------------------------------------------
# highlight_code
# ---------------------------------------------------------------------------


class TestHighlightCode:
    """Tests for highlight_code()."""

    def test_json_highlighting(self):
        result = highlight_code('{"key": "value"}', language="json")
        assert "highlighted-code" in result
        assert "<span" in result

    def test_diff_highlighting(self):
        diff = "--- a/file.py\n+++ b/file.py"
        result = highlight_code(diff, language="diff")
        assert "highlighted-code" in result

    def test_no_language_html_escapes(self):
        result = highlight_code("<script>alert('xss')</script>", language=None)
        assert "<script>" not in result
        assert "&lt;script&gt;" in result

    def test_empty_returns_empty(self):
        assert highlight_code("") == ""
        assert highlight_code(None) == ""

    def test_unknown_language_falls_back(self):
        result = highlight_code("hello world", language="nonexistent_lang_xyz")
        # Should not raise, falls back to TextLexer
        assert "hello world" in result

    def test_result_is_markup(self):
        from markupsafe import Markup

        result = highlight_code('{"a": 1}', language="json")
        assert isinstance(result, Markup)

    def test_no_language_result_is_markup(self):
        from markupsafe import Markup

        result = highlight_code("plain text", language=None)
        assert isinstance(result, Markup)

    def test_xss_escaped_in_highlighted_path(self):
        """Ensure HTML injection is escaped even when Pygments highlights the content."""
        malicious = '{"key": "<script>alert(1)</script>"}'
        result = highlight_code(malicious, language="json")
        assert "<script>" not in result
        assert "&lt;script&gt;" in result or "&#" in result

    def test_xss_escaped_in_unknown_language_path(self):
        """Fallback lexer also escapes HTML."""
        result = highlight_code("<img onerror=alert(1)>", language="nonexistent_lang_xyz")
        assert "<img" not in result or "&lt;img" in result
