"""Markdown exporter for Copilot chat sessions.

Exports chat sessions to markdown format with:
- Header block with metadata (session ID, workspace, dates)
- Messages separated by horizontal rules
- Message numbers and roles as bold headers
- Tool call summaries with emoji prefixes
- Thinking blocks in collapsible details (when included)
"""

from pathlib import Path

from .scanner import ChatMessage, ChatSession
from .utils import (
    detect_language as _detect_language,
)
from .utils import (
    format_timestamp as _format_timestamp_raw,
)
from .utils import (
    generate_session_filename as _generate_session_filename,
)
from .utils import (
    prettify_json as _prettify_json,
)
from .utils import (
    urldecode as _urldecode,
)


def _format_timestamp(value: str | int | None) -> str:
    """Markdown-specific timestamp wrapper (returns "Unknown" for missing values)."""
    return _format_timestamp_raw(value, unknown_label="Unknown")


def _format_tool_summary(message: ChatMessage, include_inputs: bool = False) -> str:
    """Format tool invocations as an italicized summary line.

    Args:
        message: The message containing tool invocations.
        include_inputs: If True, include tool inputs as code blocks.
    """
    if not message.tool_invocations:
        return ""

    tool_names = [tool.name for tool in message.tool_invocations]
    count = len(tool_names)

    if count == 1:
        summary = f"\n\n*🔧 Tool: {tool_names[0]}*"
    elif count <= 3:
        summary = f"\n\n*🔧 Tools: {', '.join(tool_names)}*"
    else:
        summary = f"\n\n*🔧 Tools ({count}): {', '.join(tool_names[:3])}, ...*"

    if include_inputs:
        for tool in message.tool_invocations:
            if tool.input:
                lang = _detect_language(tool.input, tool.name) or ""
                pretty = _prettify_json(tool.input) if lang == "json" else tool.input
                summary += f"\n\n**{tool.name} input:**\n```{lang}\n{pretty}\n```"

    return summary


def _format_file_changes_summary(message: ChatMessage, include_diffs: bool = False) -> str:
    """Format file changes as an italicized summary line.

    Args:
        message: The message containing file changes.
        include_diffs: If True, include file diffs as code blocks.
    """
    if not message.file_changes:
        return ""

    paths = [change.path for change in message.file_changes]
    count = len(paths)

    if count == 1:
        summary = f"\n\n*📄 Changed: {paths[0]}*"
    elif count <= 3:
        summary = f"\n\n*📄 Changed: {', '.join(paths)}*"
    else:
        summary = f"\n\n*📄 Changed ({count}): {', '.join(paths[:3])}, ...*"

    if include_diffs:
        for change in message.file_changes:
            if change.diff:
                summary += f"\n\n**{change.path}:**\n```diff\n{change.diff}\n```"

    return summary


def _format_command_runs_summary(message: ChatMessage) -> str:
    """Format command runs as an italicized summary line."""
    if not message.command_runs:
        return ""

    count = len(message.command_runs)

    if count == 1:
        cmd = message.command_runs[0]
        if cmd.title:
            cmd_display = cmd.title
        else:
            cmd_display = cmd.command[:50] + "..." if len(cmd.command) > 50 else cmd.command
        return f"\n\n*⚡ Command: `{cmd_display}`*"
    else:
        cmd_names = []
        for cmd in message.command_runs:
            if cmd.title:
                cmd_names.append(f"`{cmd.title}`")
            else:
                truncated = cmd.command[:50] + "..." if len(cmd.command) > 50 else cmd.command
                cmd_names.append(f"`{truncated}`")
        return f"\n\n*⚡ Commands ({count}): {', '.join(cmd_names)}*"


def _had_thinking_content(message: ChatMessage) -> bool:
    """Check if the message had any thinking blocks."""
    if not message.content_blocks:
        return False
    return any(block.kind == "thinking" for block in message.content_blocks)


def _has_inline_tool_blocks(message: ChatMessage) -> bool:
    """Check if the message has inline tool invocation blocks.

    When tools are rendered inline (VSCode-style), we don't need to add
    a separate tool summary at the end of the message.
    """
    if not message.content_blocks:
        return False
    return any(block.kind == "toolInvocation" for block in message.content_blocks)


def _format_message_content(
    message: ChatMessage,
    include_thinking: bool = False,
    include_agent_details: bool = True,
    include_tools: bool = True,
) -> str:
    """Format message content, optionally including thinking blocks and agent details.

    Args:
        message: The ChatMessage to format.
        include_thinking: If True, include thinking block content in collapsible blocks.
                         If False, completely omit thinking blocks.
        include_agent_details: If True, include full agent content in collapsible blocks.
                              If False, show only a summary line for each agent.
        include_tools: If True, include inline tool invocation blocks.

    Returns:
        Formatted message content as a string.
    """
    parts = []

    if message.content_blocks:
        # Use structured content blocks
        for block in message.content_blocks:
            if block.kind == "thinking":
                if include_thinking:
                    # Include the actual thinking content in a collapsible block
                    parts.append(f"<details>\n<summary>💭 Thinking</summary>\n\n{block.content}\n\n</details>")
                # When not including, completely omit — no trace of thinking
                continue
            elif block.kind in ("subagent", "subagent_failed", "subagent_incomplete"):
                agent_name = block.description or "Agent"
                status = "failed" if block.kind == "subagent_failed" else "incomplete" if block.kind == "subagent_incomplete" else "completed"
                if not include_agent_details:
                    # Summary only
                    parts.append(f"*🤖 {agent_name} — {status}*")
                    continue
                # Full content in a collapsible details block
                agent_content = block.content.replace(chr(10), chr(10) + "> ")
                parts.append(f"<details>\n<summary>{agent_name} — {status}</summary>\n\n> {agent_content}\n\n</details>")
                continue
            elif block.kind == "status" and block.description == "task-complete":
                tc_content = block.content.replace(chr(10), chr(10) + "> ")
                parts.append(f"<details>\n<summary>✅ Task complete</summary>\n\n> {tc_content}\n\n</details>")
                continue
            elif block.kind == "toolInvocation":
                if not include_tools:
                    continue
                # For command runs, description holds the human-readable title
                # (content starts with "$ " and is the raw command)
                # For regular tools, content is already the pretty invocation message
                if block.description and block.content.startswith("$ "):
                    display = block.description
                else:
                    display = block.content.strip()
                if display:
                    parts.append(f"*🔧 {display}*")
            else:
                # Only add non-empty text blocks
                if block.content.strip():
                    parts.append(block.content)
    else:
        # Fall back to flat content
        if message.content.strip():
            parts.append(message.content)

    content = "\n\n".join(parts)

    # Post-process to normalize formatting patterns
    import re

    # "*Creating [](file://...)*" -> "*Creating filename*" (extract leaf name, keep italics, remove link)
    content = re.sub(r"\*Creating \[\]\(file://[^)]+/([^/)]+)\)\*", r"*Creating \1*", content)

    # "*Reading [](file://...)*" -> "*Reading filename*" (extract leaf name, keep italics, remove link)
    content = re.sub(r"\*Reading \[\]\(file://[^)]+/([^/)]+)\)\*", r"*Reading \1*", content)

    # "*Edited `filename`*" -> "*Edited filename*" (remove backticks within italics)
    content = re.sub(r"\*Edited `([^`]+)`\*", r"*Edited \1*", content)

    return content


def session_to_markdown(
    session: ChatSession,
    content_set: set[str] | None = None,
) -> str:
    """Convert a chat session to markdown format.

    Args:
        session: The ChatSession to convert.
        content_set: Controls which content types to include.
            Supported keys: "diffs", "tool-inputs", "thinking", "agent-details",
            "tools", "commands", "file-changes", "system-messages".
            If None, uses DEFAULT_INCLUDES from content_types module.

    Returns:
        Markdown string representation of the session.
    """
    from .content_types import DEFAULT_INCLUDES

    if content_set is None:
        content_set = DEFAULT_INCLUDES.copy()

    include_diffs = "diffs" in content_set
    include_tool_inputs = "tool-inputs" in content_set
    include_thinking = "thinking" in content_set
    include_agent_details = "agent-details" in content_set
    include_tools = "tools" in content_set
    include_commands = "commands" in content_set
    include_file_changes = "file-changes" in content_set
    include_system_messages = "system-messages" in content_set
    lines = []

    # Header block with metadata
    lines.append("# Chat Session")
    lines.append("")

    # Session title/name
    if session.custom_title:
        lines.append(f"**Title:** {session.custom_title}")
    elif session.workspace_name:
        lines.append(f"**Workspace:** {session.workspace_name}")
    else:
        lines.append(f"**Session:** {session.session_id[:8]}...")

    lines.append("")

    # Metadata in a clear format
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Session ID:** `{session.session_id}`")

    if session.workspace_name:
        lines.append(f"- **Workspace:** {session.workspace_name}")

    if session.workspace_path:
        decoded_path = _urldecode(session.workspace_path)
        lines.append(f"- **Path:** `{decoded_path}`")

    if session.repository_url:
        lines.append(f"- **Repository:** {session.repository_url}")

    if session.created_at:
        lines.append(f"- **Created:** {_format_timestamp(session.created_at)}")

    if session.updated_at:
        lines.append(f"- **Updated:** {_format_timestamp(session.updated_at)}")

    lines.append(f"- **Edition:** `{session.vscode_edition}`")
    if session.source_name:
        lines.append(f"- **Source:** {session.source_name}")
    lines.append(f"- **Messages:** {len(session.messages)}")

    if session.requester_username:
        lines.append(f"- **User:** {session.requester_username}")

    if session.responder_username:
        lines.append(f"- **Assistant:** {session.responder_username}")

    lines.append("")
    lines.append("---")
    lines.append("")

    # Messages
    for i, message in enumerate(session.messages, 1):
        if message.role == "system" and not include_system_messages:
            continue
        msg_md = message_to_markdown(
            message,
            message_number=i,
            include_diffs=include_diffs,
            include_tool_inputs=include_tool_inputs,
            include_thinking=include_thinking,
            include_agent_details=include_agent_details,
            include_tools=include_tools,
            include_commands=include_commands,
            include_file_changes=include_file_changes,
        )
        lines.append(msg_md)

    return "\n".join(lines)


def message_to_markdown(
    message: ChatMessage,
    message_number: int = 0,
    include_diffs: bool = False,
    include_tool_inputs: bool = False,
    include_thinking: bool = False,
    include_agent_details: bool = True,
    include_tools: bool = True,
    include_commands: bool = True,
    include_file_changes: bool = True,
) -> str:
    """Convert a single message to markdown format.

    Args:
        message: The ChatMessage to convert.
        message_number: The 1-based message number (0 means don't include header).
        include_diffs: If True, include file diffs as code blocks.
        include_tool_inputs: If True, include tool inputs as code blocks.
        include_thinking: If True, include thinking block content.
        include_tools: If True, include tool summaries and inline tool invocations.
        include_commands: If True, include command run summaries.
        include_file_changes: If True, include file change summaries.

    Returns:
        Markdown string representation of the message.
    """
    lines = []

    # Message header: number and role (if message_number > 0)
    if message_number > 0:
        role_display = message.role.upper()
        lines.append(f"## Message {message_number}: **{role_display}**")
        lines.append("")

    # Timestamp if available
    if message.timestamp:
        lines.append(f"*{_format_timestamp(message.timestamp)}*")
        lines.append("")

    # Content (optionally including thinking blocks and inline tools)
    content = _format_message_content(
        message,
        include_thinking=include_thinking,
        include_agent_details=include_agent_details,
        include_tools=include_tools,
    )
    lines.append(content)

    # Check if tools are rendered inline via content blocks
    # If so, skip the separate tool/command summaries to avoid duplication
    has_inline_tools = _has_inline_tool_blocks(message)

    if not has_inline_tools:
        # Tool invocations summary (in italics, with optional inputs)
        if include_tools:
            tool_summary = _format_tool_summary(message, include_inputs=include_tool_inputs)
            if tool_summary:
                lines.append(tool_summary)

        # Command runs summary (in italics)
        if include_commands:
            cmd_summary = _format_command_runs_summary(message)
            if cmd_summary:
                lines.append(cmd_summary)

    # File changes summary (in italics, with optional diffs)
    if include_file_changes:
        file_summary = _format_file_changes_summary(message, include_diffs=include_diffs)
        if file_summary:
            lines.append(file_summary)

    lines.append("")
    lines.append("---")
    lines.append("")

    return "\n".join(lines)


def export_session_to_file(
    session: ChatSession,
    output_path: Path | str,
    content_set: set[str] | None = None,
) -> None:
    """Export a single session to a markdown file.

    Args:
        session: The ChatSession to export.
        output_path: Path to the output markdown file.
        content_set: Controls which content types to include.
            If None, uses DEFAULT_INCLUDES from content_types module.
    """
    markdown = session_to_markdown(
        session,
        content_set=content_set,
    )
    Path(output_path).write_text(markdown, encoding="utf-8")


def generate_session_filename(session: ChatSession) -> str:
    """Generate a filename for a session's markdown export.

    Args:
        session: The ChatSession to generate a filename for.

    Returns:
        A safe filename string with .md extension.
    """
    return _generate_session_filename(session, extension="md")
