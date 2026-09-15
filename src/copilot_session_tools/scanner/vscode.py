"""VS Code chat session parsing (JSON, VSCDB, JSONL formats)."""

import sqlite3
from pathlib import Path
from typing import cast

import ssrjson

from copilot_session_tools.scanner import PARSER_VERSION

from .content import (
    _extract_edit_group_text,
    _extract_inline_reference_name,
    _get_file_metadata,
    _get_first_truthy_value,
    _merge_content_blocks,
)
from .diff import _extract_file_content_from_tool, _parse_text_edit_group
from .git import detect_repository_url
from .models import (
    ChatMessage,
    ChatSession,
    CommandRun,
    ContentBlock,
    FileChange,
    ToolInvocation,
)
from .shared import (
    SHELL_TOOL_NAMES,
    extract_command_run,
    normalize_invocation_message,
    normalize_tool_status,
    strip_ansi_control_sequences,
)

RawBlock = (
    tuple[str, str]
    | tuple[str, str, str | None]
    | tuple[
        str,
        str,
        str | None,
        list[ContentBlock],
        list[ToolInvocation],
        list[FileChange],
        list[CommandRun],
    ]
    | tuple[
        str,
        str,
        str | None,
        list[ContentBlock],
        list[ToolInvocation],
        list[FileChange],
        list[CommandRun],
        str,  # prompt
    ]
)


def _response_text(value: object) -> str:
    """Extract sanitized text from a VS Code string or MarkdownString."""
    if isinstance(value, str):
        return strip_ansi_control_sequences(value)
    if isinstance(value, dict):
        nested = cast("dict[str, object]", value).get("value")
        if isinstance(nested, str):
            return strip_ansi_control_sequences(nested)
    return ""


def _format_question_carousel(item: dict) -> str:
    parts = []
    message = _response_text(item.get("message"))
    if message:
        parts.append(message)

    answers = item.get("data")
    answer_map = answers if isinstance(answers, dict) else {}
    questions = item.get("questions")
    if not isinstance(questions, list):
        questions = []

    for index, question in enumerate(questions, start=1):
        if not isinstance(question, dict):
            continue
        title = _response_text(question.get("title")) or f"Question {index}"
        prompt = _response_text(question.get("message"))
        line = f"{index}. **{title}**"
        if prompt and prompt != title:
            line += f": {prompt}"
        options = question.get("options")
        if isinstance(options, list):
            labels = [_response_text(option.get("label")) for option in options if isinstance(option, dict)]
            labels = [label for label in labels if label]
            if labels:
                line += f"\n   Options: {', '.join(labels)}"
        question_id = question.get("id")
        if isinstance(question_id, str) and question_id in answer_map:
            answer = answer_map[question_id]
            if isinstance(answer, dict):
                answer = answer.get("selectedValue") or answer.get("selectedValues") or answer.get("freeformValue")
            if isinstance(answer, list):
                answer = ", ".join(str(value) for value in answer)
            if answer not in (None, ""):
                line += f"\n   Answer: {answer}"
        parts.append(line)

    if item.get("isUsed") and not answer_map:
        parts.append("Answered")
    return "\n\n".join(parts)


def _format_interactive_response(item: dict) -> str:
    parts = []
    title = _response_text(item.get("title"))
    message = _response_text(item.get("message"))
    content = _response_text(item.get("content"))
    if title:
        parts.append(f"**{title}**")
    if message:
        parts.append(message)
    if content:
        parts.append(content)

    buttons = item.get("buttons")
    if isinstance(buttons, list):
        labels = [str(button) for button in buttons if button]
        if labels:
            parts.append(f"Options: {', '.join(labels)}")

    state = _response_text(item.get("state"))
    data = item.get("data")
    if state:
        parts.append(f"Response: {state}")
    elif isinstance(data, dict):
        action = _response_text(data.get("action") or data.get("actionId"))
        if data.get("rejected"):
            parts.append("Response: rejected")
        elif action:
            parts.append(f"Response: {action}")
    return "\n\n".join(parts)


def _parse_tool_invocation_serialized(item: dict) -> ToolInvocation | None:
    """Parse a single toolInvocationSerialized item from VS Code response.

    VS Code stores tool invocations as individual response items with kind=toolInvocationSerialized.
    Structure: {kind, toolId, invocationMessage, toolSpecificData, isComplete, resultDetails, ...}

    Tool types and their data:
    - Terminal tools: toolSpecificData.commandLine (can be string or {original, toolEdited})
    - File tools: toolSpecificData.file.uri
    - MCP tools: resultDetails.input (string) and resultDetails.output (array)
    """
    if not isinstance(item, dict):
        return None

    tool_id = item.get("toolId", "unknown")
    invocation_msg = item.get("invocationMessage", "")
    # invocationMessage can be a dict with 'value' field or a simple string
    if isinstance(invocation_msg, dict) and "value" in invocation_msg:
        invocation_msg = invocation_msg["value"]
    tool_data = item.get("toolSpecificData", {})
    result_details = item.get("resultDetails", {})
    subagent_invocation_id = item.get("subAgentInvocationId")

    # Extract input from toolSpecificData based on tool kind
    input_data = None
    result_data = None

    if isinstance(tool_data, dict):
        # Terminal command data - commandLine can be string or object
        if "commandLine" in tool_data:
            cmd_line = tool_data.get("commandLine")
            if isinstance(cmd_line, dict):
                # Use toolEdited if available (modified by AI), otherwise original
                input_data = cmd_line.get("toolEdited") or cmd_line.get("original")
            else:
                input_data = str(cmd_line) if cmd_line else None

        # File tool data - extract file URI
        elif "file" in tool_data and isinstance(tool_data.get("file"), dict):
            file_info = tool_data["file"]
            file_uri = file_info.get("uri", {})
            if isinstance(file_uri, dict):
                # Extract path from VS Code URI object
                file_path = file_uri.get("fsPath") or file_uri.get("path") or ""
                if file_path:
                    input_data = file_path
            elif isinstance(file_uri, str):
                input_data = file_uri

        # Other tool types might have different keys
        elif "input" in tool_data:
            val = tool_data.get("input")
            input_data = str(val) if val is not None else None

    # D6 (issue #59): Also try toolSpecificData.input when no input found yet
    # (above elif skips it when commandLine or file keys exist)
    if not input_data and isinstance(tool_data, dict) and "input" in tool_data:
        val = tool_data.get("input")
        if val is not None:
            input_data = str(val)

    # Extract MCP tool results if available
    if isinstance(result_details, dict):
        # MCP tools store input/output in resultDetails
        if "input" in result_details:
            mcp_input = result_details.get("input")
            if mcp_input and not input_data:
                input_data = str(mcp_input)

        # Extract output for MCP tools
        if "output" in result_details:
            outputs = result_details.get("output", [])
            if isinstance(outputs, list):
                output_parts = []
                for out in outputs:
                    if isinstance(out, dict) and out.get("value"):
                        output_parts.append(str(out["value"]))
                if output_parts:
                    result_data = "\n".join(output_parts)

    # Status: normalise to CLI canonical values (D1 — issue #55)
    is_complete = item.get("isComplete", False)
    # Detect error conditions: isComplete=false with error indicators in result
    has_error = False
    if not is_complete and isinstance(result_details, dict):
        error_msg = result_details.get("errorMessage") or result_details.get("error")
        if error_msg:
            has_error = True
    status = normalize_tool_status(None, is_complete=is_complete, has_error=has_error)

    # Extract source type (mcp vs internal)
    source = item.get("source", {})
    source_type = source.get("type") if isinstance(source, dict) else None

    # For terminal tools, also extract command output if available
    if isinstance(tool_data, dict) and tool_data.get("kind") == "terminal":
        terminal_output = tool_data.get("terminalCommandOutput", {})
        if isinstance(terminal_output, dict) and not result_data:
            text = terminal_output.get("text")
            if text:
                result_data = text

    # Handle sub-agent tool invocations
    if isinstance(tool_data, dict) and tool_data.get("kind") == "subagent":
        agent_name = tool_data.get("agentName", "Agent")
        description = tool_data.get("description", "")
        subagent_result = tool_data.get("result")
        if subagent_result and not result_data:
            result_data = str(subagent_result)
        if not (isinstance(invocation_msg, str) and invocation_msg.strip()):
            invocation_msg = f"\U0001f916 Agent ({agent_name}): {description}"

    # Normalise invocation message for built-in tools (D4 — issue #58)
    if isinstance(invocation_msg, str):
        invocation_msg = normalize_invocation_message(tool_id, tool_data, invocation_msg)

    return ToolInvocation(
        name=str(tool_id) if tool_id else "unknown",
        input=input_data,
        result=result_data,
        status=status,
        start_time=None,
        end_time=None,
        source_type=source_type,
        invocation_message=invocation_msg if isinstance(invocation_msg, str) else None,
        subagent_invocation_id=subagent_invocation_id,
    )


def _parse_tool_invocations(raw_invocations: list) -> list[ToolInvocation]:
    """Parse tool invocations from raw data (legacy format)."""
    invocations = []
    for inv in raw_invocations:
        if isinstance(inv, dict):
            invocations.append(
                ToolInvocation(
                    name=inv.get("name") or inv.get("toolName") or "unknown",
                    input=_get_first_truthy_value(inv.get("input"), inv.get("arguments")),
                    result=_get_first_truthy_value(inv.get("result"), inv.get("output")),
                    status=inv.get("status"),
                    start_time=inv.get("startTime"),
                    end_time=inv.get("endTime"),
                )
            )
    return invocations


def _parse_file_changes(raw_changes: list) -> list[FileChange]:
    """Parse file changes from raw data."""
    changes = []
    for change in raw_changes:
        if isinstance(change, dict):
            path = change.get("path") or change.get("uri") or ""
            path = path.removeprefix("file://")
            changes.append(
                FileChange(
                    path=path,
                    diff=change.get("diff"),
                    content=change.get("content"),
                    explanation=change.get("explanation"),
                    language_id=change.get("languageId"),
                )
            )
    return changes


def _parse_command_runs(raw_commands: list) -> list[CommandRun]:
    """Parse command runs from raw data."""
    commands = []
    for cmd in raw_commands:
        if isinstance(cmd, dict):
            result_val = cmd.get("result")
            commands.append(
                CommandRun(
                    command=cmd.get("command") or "unknown",
                    title=cmd.get("title"),
                    result=str(result_val) if result_val is not None else None,
                    status=cmd.get("status"),
                    output=cmd.get("output"),
                    timestamp=cmd.get("timestamp"),
                )
            )
    return commands


def _process_response_items(
    response_items: list,
    file_contents_cache: dict[str, str] | None = None,
) -> tuple[list[str], list[RawBlock], list[ToolInvocation], list[FileChange], list[CommandRun]]:
    """Process VS Code Copilot response items into structured data.

    Handles all response item kinds (toolInvocationSerialized, text, inlineReference,
    textEditGroup, etc.) and extracts content, tool invocations, file changes, and commands.

    Args:
        response_items: List of response item dicts from a VS Code Copilot response.
        file_contents_cache: Optional pre-built cache of file path -> content from readFile tools.
            If None, the cache will be built from readFile tool invocations in the response.

    Returns:
        Tuple of (response_content, raw_blocks, tool_invocations, file_changes, command_runs)
    """
    response_content: list[str] = []
    raw_blocks: list[RawBlock] = []
    tool_invocations: list[ToolInvocation] = []
    file_changes: list[FileChange] = []
    command_runs: list[CommandRun] = []

    # Build file contents cache if not provided
    if file_contents_cache is None:
        file_contents_cache = {}
        for item in response_items:
            if isinstance(item, dict) and item.get("kind") == "toolInvocationSerialized":
                file_content = _extract_file_content_from_tool(item)
                if file_content:
                    cached_path, cached_content = file_content
                    file_contents_cache[cached_path] = cached_content

    # Pre-scan: identify the latest parent subagent entry for each toolCallId
    # and dedupe child tool updates by their toolCallId.
    subagent_parents: dict[str, dict] = {}
    subagent_children: dict[str, dict[str, dict]] = {}
    for item in response_items:
        if isinstance(item, dict) and item.get("kind") == "toolInvocationSerialized":
            tool_data = item.get("toolSpecificData", {})
            if isinstance(tool_data, dict) and tool_data.get("kind") == "subagent":
                tcid = item.get("toolCallId", "")
                if tcid:
                    # Keep the latest entry so we render a single subagent block
                    # after the tool has accumulated its final result.
                    subagent_parents[tcid] = item
            said = item.get("subAgentInvocationId")
            if said:
                child_items = subagent_children.setdefault(said, {})
                child_tcid = item.get("toolCallId")
                if isinstance(child_tcid, str) and child_tcid:
                    child_items[child_tcid] = item
                else:
                    child_items[f"__child_{len(child_items)}"] = item

    # Process all response items
    for item in response_items:
        if isinstance(item, dict):
            kind = item.get("kind")

            # Handle tool invocations (current VS Code format)
            if kind == "toolInvocationSerialized":
                # Skip child tools of a subagent — they're absorbed into the parent's block
                said = item.get("subAgentInvocationId")
                if said and said in subagent_parents:
                    continue

                tool_data = item.get("toolSpecificData", {})
                tool_data_kind = tool_data.get("kind") if isinstance(tool_data, dict) else None
                tcid = item.get("toolCallId", "")

                # VS Code can emit multiple parent subagent updates with the same toolCallId.
                # Only render the latest one so we don't duplicate the agent block.
                if tool_data_kind == "subagent" and tcid and subagent_parents.get(tcid) is not item:
                    continue

                tool_inv = _parse_tool_invocation_serialized(item)
                if tool_inv:
                    tool_invocations.append(tool_inv)

                # D2 (issue #56): Extract CommandRun for terminal tools
                tool_id_str = item.get("toolId", "")
                if (tool_inv and tool_id_str in SHELL_TOOL_NAMES) or tool_data_kind == "terminal":
                    cmd = extract_command_run(
                        tool_name=tool_inv.name if tool_inv else tool_id_str,
                        command=tool_inv.input if tool_inv else None,
                        output=tool_inv.result if tool_inv else None,
                        status=tool_inv.status if tool_inv else None,
                    )
                    if cmd:
                        command_runs.append(cmd)

                # Check for sub-agent tool invocations (parent)
                if tool_data_kind == "subagent":
                    agent_name = tool_data.get("agentName", "Agent")
                    description = tool_data.get("description", "")
                    result_text = tool_data.get("result", "")
                    prompt_text = tool_data.get("prompt", "")
                    # Build content: child tool summaries + result
                    parts: list[str] = []
                    # Build structured content_blocks for nested rendering
                    nested_blocks: list[ContentBlock] = []
                    nested_tool_invocations: list[ToolInvocation] = []
                    nested_file_changes: list[FileChange] = []
                    nested_command_runs: list[CommandRun] = []
                    for child in subagent_children.get(tcid, {}).values():
                        child_inv = _parse_tool_invocation_serialized(child)
                        if child_inv and child_inv.invocation_message:
                            parts.append(f"*{child_inv.invocation_message}*")
                            nested_tool_invocations.append(child_inv)
                            nested_blocks.append(ContentBlock(kind="toolInvocation", content=child_inv.invocation_message))
                        # Extract file changes from child textEditGroup items
                        child_kind = child.get("kind")
                        if child_kind == "textEditGroup":
                            fc = _parse_text_edit_group(child, file_contents_cache)
                            if fc:
                                nested_file_changes.append(fc)
                        # D2 (issue #56): Extract CommandRun for child terminal tools
                        if child_inv and child_inv.name in SHELL_TOOL_NAMES:
                            child_cmd = extract_command_run(
                                tool_name=child_inv.name,
                                command=child_inv.input,
                                output=child_inv.result,
                                status=child_inv.status,
                            )
                            if child_cmd:
                                nested_command_runs.append(child_cmd)
                                # Only add a $ block if no invocation_message already covers this child
                                if not child_inv.invocation_message:
                                    cmd_content = f"$ {child_cmd.command}" if child_cmd.command else child_inv.name
                                    nested_blocks.append(ContentBlock(kind="toolInvocation", content=cmd_content))

                    if result_text:
                        parts.append(str(result_text))
                        nested_blocks.append(ContentBlock(kind="text", content=str(result_text)))
                    content = "\n\n".join(parts) if parts else "(no output)"
                    title = f"{agent_name}: {description}" if description else agent_name
                    # Store as structured tuple: (kind, content, title, nested_blocks, nested_tool_invocations, nested_file_changes, nested_command_runs, prompt)
                    raw_blocks.append(("subagent", content, title, nested_blocks, nested_tool_invocations, nested_file_changes, nested_command_runs, prompt_text or ""))
                    # Subagent content lives in the child ChatMessage; exclude from parent's flat content for FTS
                # Also extract the invocation message as content (non-subagent)
                elif item.get("invocationMessage"):
                    msg_text = item["invocationMessage"]
                    # invocationMessage can be a dict with 'value' field
                    if isinstance(msg_text, dict) and "value" in msg_text:
                        msg_text = msg_text["value"]
                    msg_text = str(msg_text)
                    response_content.append(msg_text)
                    # D3 (issue #57): Populate description with toolId
                    raw_blocks.append(("toolInvocation", msg_text, tool_id_str or None))
                elif tool_inv and tool_inv.name in SHELL_TOOL_NAMES:
                    command = tool_inv.input or ""
                    inline_text = f"$ {command}" if command else "run_in_terminal"
                    response_content.append(inline_text)
                    raw_blocks.append(("toolInvocation", inline_text, None))

            elif kind == "questionCarousel":
                content = _format_question_carousel(item)
                if content:
                    response_content.append(content)
                    raw_blocks.append(("ask_user", content, "questions"))
            elif kind in ("elicitationSerialized", "confirmation", "planReview"):
                content = _format_interactive_response(item)
                if content:
                    response_content.append(content)
                    raw_blocks.append(("ask_user", content, kind))
            elif kind in ("progressMessage", "systemNotification", "warning", "info"):
                content = _response_text(item.get("content"))
                if content:
                    response_content.append(content)
                    raw_blocks.append(("status", content, kind))
            elif kind == "hook":
                hook_type = _response_text(item.get("hookType"))
                message = _response_text(item.get("stopReason") or item.get("systemMessage"))
                if message:
                    label = f"{hook_type}: {message}" if hook_type else message
                    response_content.append(label)
                    raw_blocks.append(("status", label, "hook"))
            elif kind == "autoModeResolution":
                resolved = item.get("resolved")
                model = _response_text(resolved.get("name") or resolved.get("id")) if isinstance(resolved, dict) else ""
                label = f"Auto selected {model}" if model else "Auto is selecting a model"
                response_content.append(label)
                raw_blocks.append(("status", label, "auto-mode"))
            elif kind == "disabledClaudeHooks":
                label = "Claude hooks are disabled"
                response_content.append(label)
                raw_blocks.append(("status", label, "warning"))
            # Extract text content with kind info
            elif item.get("value"):
                value = item["value"]
                # If value is a dict with nested 'value', extract the string
                if isinstance(value, dict) and "value" in value:
                    value = value["value"]
                # Convert to string if not already
                if not isinstance(value, str):
                    value = str(value)
                kind = kind or "text"
                response_content.append(value)
                # For thinking blocks, extract the generatedTitle as description
                description = None
                if kind == "thinking":
                    description = item.get("generatedTitle")
                raw_blocks.append((kind, value, description))
            # Handle inline file references (VS Code Copilot Chat format)
            elif kind == "inlineReference":
                ref_name = _extract_inline_reference_name(item)
                if ref_name:
                    response_content.append(ref_name)
                    raw_blocks.append(("text", ref_name, None))
            # Handle file edit indicators (textEditGroup, notebookEditGroup, codeblockUri)
            elif kind == "textEditGroup":
                edit_text = _extract_edit_group_text(item, "Edited")
                if edit_text:
                    response_content.append(edit_text)
                    raw_blocks.append(("toolInvocation", edit_text, None))
                # Parse the actual edits as FileChange with diff content
                # Pass file contents cache for better diff generation
                file_change = _parse_text_edit_group(item, file_contents_cache)
                if file_change:
                    file_changes.append(file_change)
            elif kind == "notebookEditGroup":
                edit_text = _extract_edit_group_text(item, "Edited notebook")
                if edit_text:
                    response_content.append(edit_text)
                    raw_blocks.append(("toolInvocation", edit_text, None))
            elif kind == "codeblockUri":
                edit_text = _extract_edit_group_text(item, "Editing")
                if edit_text:
                    response_content.append(edit_text)
                    raw_blocks.append(("toolInvocation", edit_text, None))
            # Handle progress indicators
            elif kind == "progressTaskSerialized":
                content = item.get("content", {})
                progress_text = content.get("value", "") if isinstance(content, dict) else str(content)
                if progress_text and progress_text.strip():
                    response_content.append(progress_text)
                    raw_blocks.append(("status", progress_text.strip(), "progress"))
            # Skip internal/metadata kinds (no user-visible content)
            elif kind in ("prepareToolInvocation", "mcpServersStarting", "undoStop"):
                pass  # These are internal markers, skip them

            # Extract tool invocations (legacy format - nested array)
            if item.get("toolInvocations"):
                tool_invocations.extend(_parse_tool_invocations(item["toolInvocations"]))

            # Extract file changes
            for key in ("fileChanges", "fileEdits", "files"):
                if item.get(key):
                    file_changes.extend(_parse_file_changes(item[key]))

            # Extract command runs
            if item.get("commandRuns"):
                command_runs.extend(_parse_command_runs(item["commandRuns"]))

    return response_content, raw_blocks, tool_invocations, file_changes, command_runs


def _parse_chat_session_file(file_path: Path, workspace_name: str | None, workspace_path: str | None, edition: str) -> ChatSession | None:
    """Parse a single chat session JSON file.

    Supports multiple formats including:
    - Standard messages array format
    - VS Code Copilot Chat "requests" format (from Arbuzov/copilot-chat-history)
    """
    try:
        with file_path.open("rb") as f:
            raw_json_bytes = f.read()
            data = ssrjson.loads(raw_json_bytes)
    except (ssrjson.JSONDecodeError, OSError):
        return None

    messages = []

    # Try to extract messages from various possible structures
    # The "requests" format is from VS Code Copilot Chat (Arbuzov/copilot-chat-history)
    # Check each key explicitly to avoid the issue where empty list [] is falsy
    raw_messages = None
    for key in ("requests", "messages", "exchanges"):
        val = data.get(key)
        if val:  # Only use if non-empty
            raw_messages = val
            break
    raw_messages = raw_messages or []

    for msg in raw_messages:
        if isinstance(msg, dict):
            # Handle "requests" format from VS Code Copilot Chat
            # Each request has message.text (user) and response[] (assistant)
            if "message" in msg and isinstance(msg.get("message"), dict):
                # User message
                user_text = msg["message"].get("text", "")
                if user_text:
                    messages.append(
                        ChatMessage(
                            role="user",
                            content=user_text,
                            timestamp=str(msg.get("timestamp")) if msg.get("timestamp") else None,
                        )
                    )

                # Assistant response with tool invocations, file changes, etc.
                response_items = msg.get("response", [])
                if response_items:
                    response_content, raw_blocks, tool_invocations, file_changes, command_runs = _process_response_items(response_items)

                    # Also check top-level of the request
                    if msg.get("toolInvocations"):
                        tool_invocations.extend(_parse_tool_invocations(msg["toolInvocations"]))
                    if msg.get("commandRuns"):
                        command_runs.extend(_parse_command_runs(msg["commandRuns"]))
                    if msg.get("fileChanges"):
                        file_changes.extend(_parse_file_changes(msg["fileChanges"]))

                    if response_content or raw_blocks or tool_invocations or file_changes or command_runs:
                        # Merge consecutive text blocks for better markdown rendering
                        content_blocks = _merge_content_blocks(raw_blocks)
                        # Collect child messages from subagent content blocks
                        children = [cb.child_message for cb in content_blocks if cb.kind == "subagent" and cb.child_message is not None]
                        assistant_msg = ChatMessage(
                            role="assistant",
                            content="".join(response_content),
                            tool_invocations=tool_invocations,
                            file_changes=file_changes,
                            command_runs=command_runs,
                            content_blocks=content_blocks,
                            children=children,
                        )
                        messages.append(assistant_msg)
            else:
                # Standard message format
                role = msg.get("role", msg.get("type", "unknown"))
                if role in ("human", "user"):
                    role = "user"
                elif role in ("assistant", "copilot", "ai"):
                    role = "assistant"

                content = msg.get("content", msg.get("text", msg.get("message", "")))
                if isinstance(content, list):
                    content = "\n".join(str(c.get("text", c) if isinstance(c, dict) else c) for c in content)

                timestamp = msg.get("timestamp", msg.get("createdAt"))

                # Parse tool invocations and file changes from standard format
                tool_invocations = _parse_tool_invocations(msg.get("toolInvocations", []))
                file_changes = _parse_file_changes(msg.get("fileChanges", []) or msg.get("fileEdits", []))
                command_runs = _parse_command_runs(msg.get("commandRuns", []))

                messages.append(
                    ChatMessage(
                        role=role,
                        content=str(content),
                        timestamp=str(timestamp) if timestamp else None,
                        tool_invocations=tool_invocations,
                        file_changes=file_changes,
                        command_runs=command_runs,
                    )
                )

    if not messages:
        return None

    session_id = data.get("sessionId", data.get("id", file_path.stem))
    created_at = data.get("createdAt", data.get("created", data.get("creationDate")))
    updated_at = data.get("updatedAt", data.get("lastModified", data.get("lastMessageDate")))

    # Capture file metadata for incremental refresh
    source_file_mtime, source_file_size = _get_file_metadata(file_path)

    # Detect repository URL from workspace path
    repository_url = detect_repository_url(workspace_path)

    session = ChatSession(
        session_id=str(session_id),
        workspace_name=workspace_name,
        workspace_path=workspace_path,
        messages=messages,
        created_at=str(created_at) if created_at else None,
        updated_at=str(updated_at) if updated_at else None,
        source_file=str(file_path),
        vscode_edition=edition,
        custom_title=data.get("customTitle"),
        requester_username=data.get("requesterUsername"),
        responder_username=data.get("responderUsername"),
        source_file_mtime=source_file_mtime,
        source_file_size=source_file_size,
        repository_url=repository_url,
    )
    session.parser_version = PARSER_VERSION
    return session


def _parse_vscdb_file(file_path: Path, workspace_name: str | None, workspace_path: str | None, edition: str) -> list[ChatSession]:
    """Parse a VS Code SQLite database file for chat sessions.

    VS Code stores extension state in SQLite databases with .vscdb extension.
    """
    sessions = []
    try:
        conn = sqlite3.connect(str(file_path))
        cursor = conn.cursor()

        # VS Code stores key-value pairs in the ItemTable
        cursor.execute("SELECT key, value FROM ItemTable WHERE key LIKE '%copilot%chat%' OR key LIKE '%sessions%'")
        rows = cursor.fetchall()

        for _key, value in rows:
            if value:
                try:
                    data = ssrjson.loads(value)
                    # Try to parse as session data
                    if isinstance(data, dict):
                        session = _extract_session_from_dict(data, workspace_name, workspace_path, edition, str(file_path))
                        if session:
                            sessions.append(session)
                    elif isinstance(data, list):
                        for item in data:
                            if isinstance(item, dict):
                                session = _extract_session_from_dict(item, workspace_name, workspace_path, edition, str(file_path))
                                if session:
                                    sessions.append(session)
                except (ssrjson.JSONDecodeError, TypeError):
                    pass

        conn.close()
    except (sqlite3.DatabaseError, sqlite3.OperationalError, OSError):
        # SQLite database might not have expected structure or might be corrupted
        pass

    return sessions


def _extract_session_from_dict(data: dict, workspace_name: str | None, workspace_path: str | None, edition: str, source_file: str | None) -> ChatSession | None:
    """Extract a chat session from a dictionary structure.

    Supports the VS Code Copilot Chat format with requests, tool invocations, etc.
    """
    messages = []

    # Look for messages in various formats - "requests" is the VS Code Copilot format
    # Check each key explicitly to avoid the issue where empty list [] is falsy
    raw_messages = None
    for key in ("requests", "messages", "exchanges", "history"):
        val = data.get(key)
        if val:  # Only use if non-empty
            raw_messages = val
            break

    if not raw_messages:
        return None

    for msg in raw_messages:
        if isinstance(msg, dict):
            # Handle "requests" format from VS Code Copilot Chat
            if "message" in msg and isinstance(msg.get("message"), dict):
                user_text = msg["message"].get("text", "")
                if user_text:
                    messages.append(
                        ChatMessage(
                            role="user",
                            content=user_text,
                            timestamp=str(msg.get("timestamp")) if msg.get("timestamp") else None,
                        )
                    )

                response_items = msg.get("response", [])
                if response_items:
                    response_content, raw_blocks, tool_invocations, file_changes, command_runs = _process_response_items(response_items)

                    if response_content or raw_blocks or tool_invocations or file_changes or command_runs:
                        # Merge consecutive text blocks for better markdown rendering
                        content_blocks = _merge_content_blocks(raw_blocks)
                        # Collect child messages from subagent content blocks
                        children = [cb.child_message for cb in content_blocks if cb.kind == "subagent" and cb.child_message is not None]
                        assistant_msg = ChatMessage(
                            role="assistant",
                            content="".join(response_content),
                            tool_invocations=tool_invocations,
                            file_changes=file_changes,
                            command_runs=command_runs,
                            content_blocks=content_blocks,
                            children=children,
                        )
                        messages.append(assistant_msg)
            else:
                # Standard format
                role = msg.get("role", msg.get("type", "unknown"))
                if role in ("human", "user"):
                    role = "user"
                elif role in ("assistant", "copilot", "ai"):
                    role = "assistant"

                content = msg.get("content", msg.get("text", msg.get("message", "")))
                if isinstance(content, list):
                    content = "\n".join(str(c.get("text", c) if isinstance(c, dict) else c) for c in content)

                timestamp = msg.get("timestamp", msg.get("createdAt"))
                tool_invocations = _parse_tool_invocations(msg.get("toolInvocations", []))
                file_changes = _parse_file_changes(msg.get("fileChanges", []) or msg.get("fileEdits", []))
                command_runs = _parse_command_runs(msg.get("commandRuns", []))

                messages.append(
                    ChatMessage(
                        role=role,
                        content=str(content),
                        timestamp=str(timestamp) if timestamp else None,
                        tool_invocations=tool_invocations,
                        file_changes=file_changes,
                        command_runs=command_runs,
                    )
                )

    if not messages:
        return None

    session_id = data.get("sessionId", data.get("id", str(hash(source_file))))
    created_at = data.get("createdAt", data.get("created", data.get("creationDate")))
    updated_at = data.get("updatedAt", data.get("lastModified", data.get("lastMessageDate")))

    # Capture file metadata for incremental refresh
    source_file_mtime, source_file_size = _get_file_metadata(source_file)

    # Detect repository URL from workspace path
    repository_url = detect_repository_url(workspace_path)

    session = ChatSession(
        session_id=str(session_id),
        workspace_name=workspace_name,
        workspace_path=workspace_path,
        messages=messages,
        created_at=str(created_at) if created_at else None,
        updated_at=str(updated_at) if updated_at else None,
        source_file=source_file,
        vscode_edition=edition,
        custom_title=data.get("customTitle"),
        requester_username=data.get("requesterUsername"),
        responder_username=data.get("responderUsername"),
        source_file_mtime=source_file_mtime,
        source_file_size=source_file_size,
        repository_url=repository_url,
    )
    session.parser_version = PARSER_VERSION
    return session


def _apply_jsonl_operations(base: dict, operations: list[dict]) -> dict:
    """Apply JSONL append-log operations (kind=1 set, kind=2 push) to a base snapshot.

    Args:
        base: The base session dict from kind=0 snapshot.
        operations: List of operation dicts with kind=1 or kind=2.

    Returns:
        The mutated base dict with all operations applied.
    """
    for op in operations:
        kind = op.get("kind")
        path = op.get("k", [])
        value = op.get("v")

        if not path:
            continue

        # Navigate to the parent of the target
        target = base
        for segment in path[:-1]:
            if isinstance(target, dict) and isinstance(segment, str):
                target = target.get(segment)
            elif isinstance(target, list) and isinstance(segment, int) and 0 <= segment < len(target):
                target = target[segment]
            else:
                target = None
                break

        if target is None:
            continue

        last_key = path[-1]
        if kind == 1:
            # Set value at path
            if (isinstance(target, dict) and isinstance(last_key, str)) or (isinstance(target, list) and isinstance(last_key, int) and 0 <= last_key < len(target)):
                target[last_key] = value
        elif kind == 2:
            # Push value(s) to array at path
            if isinstance(target, dict) and isinstance(last_key, str):
                arr = target.get(last_key)
                if isinstance(arr, list) and isinstance(value, list):
                    arr.extend(value)
            elif isinstance(target, list) and isinstance(last_key, int) and 0 <= last_key < len(target):
                arr = target[last_key]
                if isinstance(arr, list) and isinstance(value, list):
                    arr.extend(value)

    return base


def _parse_vscode_jsonl_file(file_path: Path, workspace_name: str | None, workspace_path: str | None, edition: str) -> ChatSession | None:
    """Parse a VS Code JSONL append-log chat session file.

    VS Code >= Jan 2026 stores chat sessions as JSONL append-only operation logs:
    - kind=0: Full session snapshot (same structure as old .json format)
    - kind=1: Set a property at a path
    - kind=2: Push to an array at a path

    The kind=0 line's 'v' field is identical to the old .json format (ISerializableChatData v3),
    so we can extract it and pass to _extract_session_from_dict().
    """
    try:
        with file_path.open("rb") as f:
            raw_bytes = f.read()
    except OSError:
        return None

    lines = raw_bytes.split(b"\n")
    base_data = None
    operations = []

    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            entry = ssrjson.loads(line)
        except ssrjson.JSONDecodeError:
            continue

        kind = entry.get("kind")
        if kind == 0 and base_data is None:
            # Full session snapshot — 'v' has the same shape as old .json format
            base_data = entry.get("v")
        elif kind in (1, 2):
            operations.append(entry)

    if not base_data or not isinstance(base_data, dict):
        return None

    # Apply incremental operations to the base snapshot
    if operations:
        base_data = _apply_jsonl_operations(base_data, operations)

    return _extract_session_from_dict(
        base_data,
        workspace_name,
        workspace_path,
        edition,
        source_file=str(file_path),
    )
