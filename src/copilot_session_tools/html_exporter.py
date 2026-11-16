"""HTML exporter for Copilot chat sessions.

Renders chat sessions as self-contained static HTML files using the same
Jinja2 template as the web viewer, but with interactive elements (toolbar,
AJAX, copy buttons) stripped out via the `static=True` flag.
"""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .scanner import ChatSession
from .utils import (
    build_block_metadata,
    build_root_agent_markers,
    compact_agent_label,
    detect_language,
    extract_filename,
    format_timestamp,
    highlight_code,
    markdown_to_html,
    match_tool_for_block,
    parse_diff_stats,
    prettify_json,
    strip_ansi,
    truncate_preview,
    urldecode,
    workflow_details,
)
from .utils import (
    generate_session_filename as _generate_filename,
)


def _get_jinja_env() -> Environment:
    """Create a standalone Jinja2 environment pointing at the web templates."""
    templates_dir = Path(__file__).parent / "web" / "templates"
    env = Environment(
        loader=FileSystemLoader(str(templates_dir)),
        autoescape=select_autoescape(["html"]),
    )
    env.filters["markdown"] = markdown_to_html
    env.filters["urldecode"] = urldecode
    env.filters["format_timestamp"] = format_timestamp
    env.filters["parse_diff_stats"] = parse_diff_stats
    env.filters["extract_filename"] = extract_filename
    env.filters["strip_ansi"] = strip_ansi
    env.filters["truncate_preview"] = truncate_preview
    env.filters["prettify_json"] = prettify_json
    env.filters["workflow_details"] = workflow_details
    env.filters["compact_agent_label"] = compact_agent_label
    env.globals["detect_language"] = detect_language
    env.globals["highlight_code"] = highlight_code
    env.globals["match_tool_for_block"] = match_tool_for_block
    env.globals["get_nested_meta"] = lambda block: getattr(block, "_nested_meta", {})
    return env


def _preprocess_messages(session: ChatSession) -> tuple[str | None, dict[int, dict]]:
    """Pre-process messages to match tool invocations with content blocks.

    Same logic as the webapp session_view route. Returns first user prompt
    and a dict mapping message index to its block metadata.
    """
    first_user_prompt = None
    message_metadata: dict[int, dict] = {}
    for msg_idx, message in enumerate(session.messages):
        if message.role == "user" and first_user_prompt is None:
            first_user_prompt = message.content

        message_metadata[msg_idx] = build_block_metadata(message.content_blocks, message.tool_invocations, message.command_runs)

    return first_user_prompt, message_metadata


def session_to_html(session: ChatSession, content_set: set[str] | None = None) -> str:
    """Convert a chat session to a self-contained static HTML string.

    Args:
        session: The ChatSession to convert.
        content_set: Controls which content types to include.
            If None, uses DEFAULT_INCLUDES from content_types module.

    Returns:
        Complete HTML document as a string.
    """
    from .content_types import DEFAULT_INCLUDES

    if content_set is None:
        content_set = DEFAULT_INCLUDES.copy()

    include_agent_details = "agent-details" in content_set
    include_thinking = "thinking" in content_set
    include_system_messages = "system-messages" in content_set

    # Compute CSS hide classes for static export (no Alpine.js)
    static_hide_classes_list: list[str] = []
    if "thinking" not in content_set:
        static_hide_classes_list.append("hide-thinking")
    if "diffs" not in content_set:
        static_hide_classes_list.append("hide-diffs")
    if "tool-inputs" not in content_set:
        static_hide_classes_list.append("hide-tool-inputs")
    if "agent-details" not in content_set:
        static_hide_classes_list.append("hide-agent-details")
    if "tools" not in content_set:
        static_hide_classes_list.append("hide-tools")
    if "commands" not in content_set:
        static_hide_classes_list.append("hide-commands")
    if "file-changes" not in content_set:
        static_hide_classes_list.append("hide-file-changes")
    first_user_prompt, message_metadata = _preprocess_messages(session)
    env = _get_jinja_env()
    template = env.get_template("session.html")
    return template.render(
        title=session.custom_title or session.workspace_name or f"Session {session.session_id[:8]}",
        session=session,
        message_count=len(session.messages),
        first_user_prompt=first_user_prompt,
        message_metadata=message_metadata,
        root_agent_markers=build_root_agent_markers(session),
        static=True,
        is_enriched=True,
        include_agent_details=include_agent_details,
        include_thinking=include_thinking,
        include_system_messages=include_system_messages,
        static_hide_classes=" ".join(static_hide_classes_list),
    )


def export_session_to_html_file(
    session: ChatSession,
    output_path: Path | str,
    content_set: set[str] | None = None,
) -> None:
    """Export a single session to a static HTML file.

    Args:
        session: The ChatSession to export.
        output_path: Path to the output HTML file.
        content_set: Controls which content types to include.
            If None, uses DEFAULT_INCLUDES from content_types module.
    """
    html = session_to_html(session, content_set=content_set)
    Path(output_path).write_text(html, encoding="utf-8")


def generate_session_html_filename(session: ChatSession) -> str:
    """Generate a filename for a session's HTML export.

    Args:
        session: The ChatSession to generate a filename for.

    Returns:
        A safe filename string with .html extension.
    """
    return _generate_filename(session, extension="html")
