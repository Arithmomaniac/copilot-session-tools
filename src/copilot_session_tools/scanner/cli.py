"""GitHub Copilot CLI session parsing."""

import html
import json
import re
from pathlib import Path
from typing import cast

import ssrjson

from copilot_session_tools.scanner import PARSER_VERSION

from .content import _format_tool_display_message, _get_file_metadata
from .diff import CliFileOp, cli_file_op_from_tool, consolidate_cli_file_ops, strip_view_line_numbers
from .git import _normalize_git_url, detect_repository_url
from .models import (
    ChatMessage,
    ChatSession,
    CommandRun,
    ContentBlock,
    RootAgentInterval,
    SessionContextEntry,
    ShellIOEntry,
    ToolInvocation,
)
from .shared import strip_ansi_control_sequences

_SESSION_ID_PATTERN = r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}"
_SESSION_REFERENCE_PATTERN = re.compile(
    rf'(?:"(?P<name>[^"\n]+)" \((?P<named_id>{_SESSION_ID_PATTERN})\)|(?P<bare_id>{_SESSION_ID_PATTERN}))',
    re.IGNORECASE,
)
_FORK_BOUNDARY_PATTERN = re.compile(rf" before event (?P<event_id>{_SESSION_ID_PATTERN})", re.IGNORECASE)


def _as_tool_argument_dict(arguments: object) -> dict[str, object]:
    """Return mapping-style tool arguments; freeform tools may store a string."""
    return {str(key): value for key, value in arguments.items()} if isinstance(arguments, dict) else {}


def _serialize_tool_arguments(arguments: object) -> str | None:
    """Serialize tool arguments for storage without losing freeform string input."""
    if not arguments:
        return None
    if isinstance(arguments, str):
        return arguments
    try:
        return json.dumps(arguments)
    except (TypeError, ValueError):
        return str(arguments)


def _get_tool_arg_str(arguments: dict[str, object], key: str, default: str = "") -> str:
    """Get a string-valued tool argument."""
    value = arguments.get(key)
    return value if isinstance(value, str) else default


def _stringify_event_value(value: object) -> str:
    """Return a compact display string for structured event payload values."""
    if value is None:
        return ""
    if isinstance(value, str):
        return strip_ansi_control_sequences(value)
    if isinstance(value, bool | int | float):
        return str(value)
    if isinstance(value, list):
        value_list = cast("list[object]", value)
        parts = [_stringify_event_value(item) for item in value_list[:5]]
        text = ", ".join(part for part in parts if part)
        if len(value_list) > 5:
            text += f", ... (+{len(value_list) - 5} more)" if text else f"+{len(value_list) - 5} more"
        return text
    if isinstance(value, dict):
        value_dict = cast("dict[str, object]", value)
        for key in (
            "message",
            "content",
            "text",
            "value",
            "answer",
            "command",
            "commandName",
            "toolName",
            "serverName",
            "url",
            "path",
            "name",
            "kind",
            "action",
            "operation",
            "reason",
            "error",
        ):
            if key in value_dict:
                text = _stringify_event_value(value_dict.get(key))
                if text:
                    return text
        return ", ".join(f"{key}={_stringify_event_value(val)}" for key, val in value_dict.items() if _stringify_event_value(val))
    return str(value).strip()


def _format_permission_request(event_data: dict) -> str:
    request = event_data.get("permissionRequest")
    prompt = event_data.get("promptRequest")
    request_dict = request if isinstance(request, dict) else {}
    prompt_dict = prompt if isinstance(prompt, dict) else {}

    kind = request_dict.get("kind") or prompt_dict.get("kind") or "permission"
    details = _stringify_event_value(request_dict.get("command") or request_dict.get("path") or request_dict.get("url") or request_dict.get("toolName"))
    if not details:
        details = _stringify_event_value(prompt_dict.get("commands") or prompt_dict.get("path") or prompt_dict.get("url"))
    content = f"Permission requested: {kind}"
    return f"{content} - {details}" if details else content


def _format_permission_result(event_data: dict) -> str:
    result = event_data.get("result")
    result_dict = result if isinstance(result, dict) else {}
    verdict = result_dict.get("kind") or result_dict.get("decision") or result_dict.get("type") or _stringify_event_value(result)
    return f"Permission {verdict}" if verdict else "Permission completed"


def _format_fork_session_reference(match: re.Match[str], current_session_id: str) -> str:
    session_id = match.group("named_id") or match.group("bare_id")
    display = match.group("name") or session_id[:8]
    safe_display = _escape_markdown_text(display)
    if session_id.lower() == current_session_id.lower():
        return safe_display
    return f"[{safe_display}](/session/{session_id})"


def _escape_markdown_text(text: str) -> str:
    escaped = html.escape(text, quote=False)
    return escaped.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _format_fork_info_message(message: str, current_session_id: str) -> str:
    references = list(_SESSION_REFERENCE_PATTERN.finditer(message))
    if not references:
        return _escape_markdown_text(message)

    boundary_match = _FORK_BOUNDARY_PATTERN.search(message)
    boundary = f" before event {boundary_match.group('event_id')}" if boundary_match else ""

    if message.startswith("Forked this session into "):
        return f"Forked as {_format_fork_session_reference(references[0], current_session_id)}{boundary}."

    if message.startswith("Forked from "):
        return f"Forked from {_format_fork_session_reference(references[0], current_session_id)}{boundary}."

    parts: list[str] = []
    position = 0
    for reference in references:
        parts.append(_escape_markdown_text(message[position : reference.start()]))
        parts.append(_format_fork_session_reference(reference, current_session_id))
        position = reference.end()
    parts.append(_escape_markdown_text(message[position:]))
    return "".join(parts)


def _format_session_event_status(event_type: str, event_data: dict) -> tuple[str, str] | None:
    """Format newer CLI event types that carry user-visible status."""
    if event_type == "auto_mode_switch.requested":
        code = event_data.get("errorCode")
        retry = event_data.get("retryAfterSeconds")
        details = f" ({code})" if code else ""
        if retry:
            details += f", retry after {retry}s"
        return f"Auto mode switch requested{details}", "auto-mode"
    if event_type == "auto_mode_switch.completed":
        response = _stringify_event_value(event_data.get("response"))
        return f"Auto mode switch completed: {response}" if response else "Auto mode switch completed", "auto-mode"
    if event_type == "session.auto_mode_resolved":
        model = _stringify_event_value(event_data.get("chosenModel"))
        bucket = _stringify_event_value(event_data.get("reasoningBucket"))
        details = f" ({bucket} reasoning)" if bucket else ""
        return f"Auto selected {model}{details}" if model else "Auto model resolved", "auto-mode"
    if event_type == "command.queued":
        command = _stringify_event_value(event_data.get("command"))
        return f"Command queued: {command}" if command else "Command queued", "command"
    if event_type == "command.execute":
        command = _stringify_event_value(event_data.get("command") or event_data.get("commandName"))
        return f"Command executed: {command}" if command else "Command executed", "command"
    if event_type == "command.completed":
        return "Command completed", "command"
    if event_type == "hook.end" and not event_data.get("success", True):
        hook_type = _stringify_event_value(event_data.get("hookType"))
        error = _stringify_event_value(event_data.get("error"))
        label = f"Hook failed: {hook_type}" if hook_type else "Hook failed"
        return f"{label} - {error}" if error else label, "hook-error"
    if event_type == "hook.progress":
        message = _stringify_event_value(event_data.get("message"))
        return f"Hook progress: {message}" if message else "Hook progress", "hook-progress"
    if event_type == "mcp_app.tool_call_complete":
        server = _stringify_event_value(event_data.get("serverName"))
        tool = _stringify_event_value(event_data.get("toolName"))
        success = event_data.get("success")
        label = "MCP app tool completed"
        if server or tool:
            label += f": {server}/{tool}" if server and tool else f": {server or tool}"
        if success is False:
            error = _stringify_event_value(event_data.get("error"))
            label += f" - failed: {error}" if error else " - failed"
        return label, "mcp-app"
    if event_type == "mcp.oauth_required":
        server = _stringify_event_value(event_data.get("serverName") or event_data.get("serverUrl"))
        return f"MCP OAuth required: {server}" if server else "MCP OAuth required", "mcp-oauth"
    if event_type == "mcp.oauth_completed":
        return "MCP OAuth completed", "mcp-oauth"
    if event_type == "model.call_failure":
        source = _stringify_event_value(event_data.get("source"))
        error = _stringify_event_value(event_data.get("error") or event_data.get("message"))
        label = f"Model call failed: {source}" if source else "Model call failed"
        return f"{label} - {error}" if error else label, "error"
    if event_type == "permission.requested":
        return _format_permission_request(event_data), "permission"
    if event_type == "permission.completed":
        return _format_permission_result(event_data), "permission"
    if event_type == "session.compaction_start":
        tokens = [event_data.get("systemTokens"), event_data.get("conversationTokens"), event_data.get("toolDefinitionsTokens")]
        token_text = ", ".join(f"{int(token):,}" for token in tokens if isinstance(token, int | float))
        return f"Compaction started ({token_text} tokens)" if token_text else "Compaction started", "compaction"
    if event_type == "session.schedule_created":
        prompt = _stringify_event_value(event_data.get("prompt"))
        interval = _stringify_event_value(event_data.get("interval"))
        details = " | ".join(part for part in (interval, prompt) if part)
        return f"Schedule created: {details}" if details else "Schedule created", "schedule"
    if event_type == "session.schedule_cancelled":
        schedule_id = _stringify_event_value(event_data.get("id") or event_data.get("scheduleId"))
        return f"Schedule cancelled: {schedule_id}" if schedule_id else "Schedule cancelled", "schedule"
    if event_type == "session.truncation":
        reason = _stringify_event_value(event_data.get("reason"))
        return f"Context truncated: {reason}" if reason else "Context truncated", "truncation"
    if event_type == "session.workspace_file_changed":
        operation = _stringify_event_value(event_data.get("operation"))
        path = _stringify_event_value(event_data.get("path"))
        details = " ".join(part for part in (operation, path) if part)
        return f"Workspace file changed: {details}" if details else "Workspace file changed", "workspace-file"
    if event_type == "system.notification":
        content = _stringify_event_value(event_data.get("content"))
        kind = _stringify_event_value(event_data.get("kind"))
        if content:
            return content, "notification"
        return f"System notification: {kind}" if kind else "System notification", "notification"
    if event_type == "tool.execution_progress":
        progress = _stringify_event_value(event_data.get("progressMessage"))
        return f"Tool progress: {progress}" if progress else "Tool progress", "tool-progress"
    if event_type == "tool.execution_partial_result":
        partial = _stringify_event_value(event_data.get("partialOutput"))
        return f"Tool partial result: {partial}" if partial else "Tool partial result", "tool-progress"
    return None


_FUSION_PHASE_LABELS = {
    "primary": "Drafted answer",
    "draft": "Drafted",
    "judge": "Checked",
    "repair": "Repaired",
    "critic": "Reviewed",
    "revision": "Revised",
    "follow_up": "Continued",
}


def _build_fusion_statuses(events: list[dict]) -> dict[str, str]:
    """Reduce durable HydraFusion events to one safe transcript status per run."""
    summaries: dict[str, dict[str, object]] = {}

    for event in events:
        event_type = event.get("type")
        data = event.get("data", {})
        if not isinstance(data, dict):
            continue
        fusion_id = data.get("fusionId")
        if not isinstance(fusion_id, str) or not fusion_id:
            continue

        summary = summaries.setdefault(fusion_id, {"milestones": []})
        milestones = cast("list[str]", summary["milestones"])

        if event_type == "session.fusion_resolved":
            summary["pattern"] = data.get("pattern")
            summary["primaryModel"] = data.get("primaryModel")
            summary["secondaryModel"] = data.get("secondaryModel")
        elif event_type == "assistant.fusion_phase_completed":
            phase_kind = _stringify_event_value(data.get("phaseKind"))
            label = _FUSION_PHASE_LABELS.get(phase_kind, phase_kind.replace("_", " ").title())
            verdict = _stringify_event_value(data.get("verdict"))
            if phase_kind == "judge" and verdict == "accept":
                label = "Accepted"
            elif phase_kind == "judge" and verdict == "reject":
                label = "Changes requested"
            if label and label not in milestones:
                milestones.append(label)
        elif event_type == "assistant.fusion_phase_failed":
            phase_kind = _stringify_event_value(data.get("phaseKind"))
            label = _FUSION_PHASE_LABELS.get(phase_kind, phase_kind.replace("_", " ").title())
            failure = f"Warning: {label or 'Fusion phase'} failed"
            if data.get("degradedToPhaseId"):
                failure += "; continuing with reduced Fusion coverage"
            milestones.append(failure)
        elif event_type == "session.fusion_completed":
            degraded = data.get("outcome") == "degraded" or bool(data.get("degradedReason"))
            milestones.append("Warning: final response prepared with reduced Fusion coverage" if degraded else "Final response prepared")
            summary["completed"] = True

    statuses: dict[str, str] = {}
    for fusion_id, summary in summaries.items():
        pattern = _stringify_event_value(summary.get("pattern"))
        pattern_label = pattern.title() if pattern else "Workflow"
        models = [
            _stringify_event_value(summary.get("primaryModel")),
            _stringify_event_value(summary.get("secondaryModel")),
        ]
        model_text = " + ".join(model for model in models if model)
        header = f"Fusion · {pattern_label}"
        if model_text:
            header += f" · {model_text}"
        milestones = cast("list[str]", summary["milestones"])
        statuses[fusion_id] = "; ".join([header, *milestones])

    return statuses


def _parse_workspace_yaml(session_dir: Path) -> dict[str, str]:
    """Parse a workspace.yaml file from a CLI session directory.

    The workspace.yaml is a simple key-value YAML file maintained by the Copilot CLI:
        id: <session-uuid>
        cwd: <working-directory>
        summary: <session-title>
        created_at: <timestamp>
        ...

    We parse it manually to avoid adding a PyYAML dependency.

    Args:
        session_dir: Path to the CLI session directory containing workspace.yaml.

    Returns:
        Dictionary of key-value pairs from the file, or empty dict on failure.
    """
    workspace_file = session_dir / "workspace.yaml"
    if not workspace_file.exists():
        return {}

    try:
        result: dict[str, str] = {}
        with workspace_file.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Split on first colon only
                if ":" in line:
                    key, _, value = line.partition(":")
                    result[key.strip()] = value.strip()
        return result
    except OSError:
        return {}


def _workspace_name_from_path(workspace_path: str | None) -> str | None:
    return Path(workspace_path).name if workspace_path else None


def _append_context_entry(
    entries: list[SessionContextEntry],
    *,
    workspace_path: str | None,
    repository_url: str | None,
    branch: str | None = None,
    timestamp: str | None = None,
    message_index: int = 0,
    source: str,
) -> None:
    entry = SessionContextEntry(
        workspace_name=_workspace_name_from_path(workspace_path),
        workspace_path=workspace_path,
        repository_url=repository_url,
        branch=branch,
        timestamp=timestamp,
        message_index=max(0, message_index),
        source=source,
    )
    key = (entry.workspace_name, entry.workspace_path, entry.repository_url, entry.branch)
    if entries:
        previous = entries[-1]
        previous_key = (previous.workspace_name, previous.workspace_path, previous.repository_url, previous.branch)
        if key == previous_key:
            return
    entries.append(entry)


class _CliSessionBuilder:
    """Accumulates CLI session events into ChatMessage objects.

    Manages state for building assistant messages from streaming events,
    combining consecutive assistant messages and interleaving tool invocations.
    """

    def __init__(
        self,
        tool_executions: dict,
        subagent_child_tools: dict[str, list[str]] | None = None,
        subagent_failures: dict[str, str] | None = None,
        bg_agent_results: dict[str, str] | None = None,
        bg_agent_id_map: dict[str, str] | None = None,
        subagent_completions: set[str] | None = None,
        subagent_child_structured: dict[str, list[tuple]] | None = None,
        agent_display_names: dict[str, str] | None = None,
    ) -> None:
        self.tool_executions = tool_executions
        self.subagent_child_tools = subagent_child_tools or {}
        self.subagent_failures = subagent_failures or {}
        self.bg_agent_results = bg_agent_results or {}
        self.bg_agent_id_map = bg_agent_id_map or {}
        self.subagent_completions = subagent_completions or set()
        self.subagent_child_structured = subagent_child_structured or {}
        self.agent_display_names = agent_display_names or {}
        self.shell_title_map: dict[str, str] = {}  # shellId → title for async shell backlinks
        self.view_base: dict[str, str] = {}  # path → last full-file view content (reconstruction base)
        self.messages: list[ChatMessage] = []
        self.current_assistant_content_blocks: list[ContentBlock] = []
        self.current_assistant_tool_invocations: list[ToolInvocation] = []
        self.current_assistant_command_runs: list[CommandRun] = []
        self.current_assistant_file_ops: list[CliFileOp] = []
        self.current_assistant_timestamp: str | None = None
        self.current_assistant_source_event_id: str | None = None
        self.pending_tool_requests: dict[str, dict] = {}

    def note_assistant_event(self, event_id: object) -> None:
        """Remember the first source event id that contributes to the current assistant message."""
        if self.current_assistant_source_event_id is None and isinstance(event_id, str) and event_id:
            self.current_assistant_source_event_id = event_id

    def flush_assistant_message(self) -> None:
        """Flush accumulated assistant content blocks into a single message."""
        has_content = self.current_assistant_content_blocks or self.current_assistant_tool_invocations or self.current_assistant_command_runs or self.current_assistant_file_ops
        if not has_content:
            return

        _SUBAGENT_KINDS = ("subagent", "subagent_failed", "subagent_incomplete")

        # Build flat content from content blocks — exclude subagent blocks
        # (their text lives in child_message.content instead)
        text_parts = []
        for block in self.current_assistant_content_blocks:
            if block.kind == "text" and block.content.strip():
                text_parts.append(block.content)
        flat_content = "\n\n".join(text_parts)

        # Collect child messages from subagent blocks
        children: list[ChatMessage] = []
        for block in self.current_assistant_content_blocks:
            if block.kind in _SUBAGENT_KINDS and block.child_message is not None:
                children.append(block.child_message)

        self.messages.append(
            ChatMessage(
                role="assistant",
                content=flat_content,
                timestamp=self.current_assistant_timestamp,
                source_event_id=self.current_assistant_source_event_id,
                tool_invocations=self.current_assistant_tool_invocations.copy(),
                command_runs=self.current_assistant_command_runs.copy(),
                content_blocks=self.current_assistant_content_blocks.copy(),
                file_changes=consolidate_cli_file_ops(self.current_assistant_file_ops, self.view_base),
                children=children,
            )
        )

        # Reset state
        self.current_assistant_content_blocks = []
        self.current_assistant_tool_invocations = []
        self.current_assistant_command_runs = []
        self.current_assistant_file_ops = []
        self.current_assistant_timestamp = None
        self.current_assistant_source_event_id = None

    def build_tool_invocation(self, tool_call_id: str, tool_name: str, arguments: object) -> tuple[ToolInvocation | None, CommandRun | None]:
        """Build a ToolInvocation or CommandRun from tool request data."""
        argument_dict = _as_tool_argument_dict(arguments)
        # Get execution result if available
        execution = self.tool_executions.get(tool_call_id, {})
        complete_event = execution.get("complete")
        start_event = execution.get("start")

        result = None
        status = None
        if complete_event:
            complete_data = complete_event.get("data", {})
            status = "success" if complete_data.get("success") else "error"
            result_obj = complete_data.get("result", {})
            if isinstance(result_obj, dict):
                result = result_obj.get("content", "")
            else:
                result = str(result_obj) if result_obj else None

        # Get description and MCP metadata from start event
        description = None
        mcp_server_name = None
        mcp_tool_name = None
        if start_event:
            start_data = start_event.get("data", {})
            start_args = _as_tool_argument_dict(start_data.get("arguments", {}))
            description = _get_tool_arg_str(start_args, "description") or None
            server_name = start_data.get("mcpServerName")
            tool_name_value = start_data.get("mcpToolName")
            mcp_server_name = server_name if isinstance(server_name, str) else None
            mcp_tool_name = tool_name_value if isinstance(tool_name_value, str) else None
        if not description:
            description = _get_tool_arg_str(argument_dict, "description") or None

        # Check if this is a shell/powershell command
        if tool_name in ("powershell", "bash", "shell", "run_command"):
            command = _get_tool_arg_str(argument_dict, "command")
            shell_mode = _get_tool_arg_str(argument_dict, "mode")
            shell_id = _get_tool_arg_str(argument_dict, "shellId")
            detach = argument_dict.get("detach", False)
            is_async = shell_mode in ("async", "background") or bool(detach)
            return None, CommandRun(
                command=command,
                title=description,
                result=result,
                status=status,
                output=result,
                shell_id=shell_id if is_async else None,
                is_async=is_async,
                is_detached=bool(detach),
            )
        else:
            # Regular tool invocation
            input_str = _serialize_tool_arguments(arguments)

            # Build invocation message for inline display
            invocation_message = _format_tool_display_message(
                tool_name,
                argument_dict,
                description,
                mcp_server_name=mcp_server_name,
                mcp_tool_name=mcp_tool_name,
            )

            return ToolInvocation(
                name=tool_name,
                input=input_str,
                result=result,
                status=status,
                invocation_message=invocation_message,
                source_type="mcp" if mcp_server_name else None,
            ), None

    def add_tool_inline(self, tool_call_id: str, tool_name: str, arguments: object, *, intention_summary: str | None = None, tool_title: str | None = None) -> None:
        """Add a tool invocation inline in the current assistant message."""
        argument_dict = _as_tool_argument_dict(arguments)
        # Handle special meta-tools with pretty formatting
        if tool_name == "report_intent":
            intent_text = _get_tool_arg_str(argument_dict, "intent") or _get_tool_arg_str(argument_dict, "description")
            if intent_text:
                self.current_assistant_content_blocks.append(
                    ContentBlock(
                        kind="intent",
                        content=intent_text,
                    )
                )
            return

        if tool_name == "skill":
            skill_name = _get_tool_arg_str(argument_dict, "name") or _get_tool_arg_str(argument_dict, "skill")
            if skill_name:
                self.current_assistant_content_blocks.append(
                    ContentBlock(
                        kind="skill",
                        content=skill_name,
                    )
                )
            return

        if tool_name == "ask_user":
            question = _get_tool_arg_str(argument_dict, "question") or _get_tool_arg_str(argument_dict, "message")
            choices_value = argument_dict.get("choices", [])
            choices = choices_value if isinstance(choices_value, list) else []
            if question:
                content = f"❓ {question}"
                if choices:
                    choices_text = ", ".join(str(c) for c in choices[:5])  # Limit to 5 choices
                    if len(choices) > 5:
                        choices_text += f", ... (+{len(choices) - 5} more)"
                    content += f"\n   Options: {choices_text}"
                # Look up the user's answer from the tool execution result
                execution = self.tool_executions.get(tool_call_id, {})
                complete_event = execution.get("complete")
                if complete_event:
                    complete_data = complete_event.get("data", {})
                    if complete_data.get("success"):
                        result_obj = complete_data.get("result", {})
                        answer = result_obj.get("content", "") if isinstance(result_obj, dict) else str(result_obj)
                        answer = answer.removeprefix("User responded: ")
                        if answer:
                            content += f"\n   ✅ **Answer:** {answer}"
                    else:
                        content += "\n   ⏭️ *Skipped*"
                self.current_assistant_content_blocks.append(
                    ContentBlock(
                        kind="ask_user",
                        content=content,
                        description="user-input",
                    )
                )
            return

        # Skip truly internal tools with no user-visible output
        internal_tools = {
            "read_bash",
        }
        if tool_name in internal_tools:
            return

        # Handle shell interaction tools as backlinks to the original async shell
        shell_interaction_tools = {
            "read_powershell": "read",
            "write_powershell": "write",
            "stop_powershell": "stop",
        }
        if tool_name in shell_interaction_tools:
            shell_id = str(argument_dict.get("shellId", ""))
            action = shell_interaction_tools[tool_name]
            shell_title = self.shell_title_map.get(shell_id, shell_id) if shell_id else ""
            label = f"↩ {shell_title} — {action}" if shell_title else f"↩ shell — {action}"
            self.current_assistant_content_blocks.append(ContentBlock(kind="toolInvocation", content=label))
            tool_inv, _ = self.build_tool_invocation(tool_call_id, tool_name, arguments)
            if tool_inv:
                tool_inv.invocation_message = label
                tool_inv.is_shell_backlink = True
                tool_inv.backlink_shell_id = shell_id or None
                self.current_assistant_tool_invocations.append(tool_inv)
            return

        # Resolve read_agent agent_id to display name
        if tool_name == "read_agent":
            agent_id = str(argument_dict.get("agent_id", ""))
            display = self.agent_display_names.get(agent_id, agent_id)
            resolved_msg = f"⏳ Checking agent {display}"
            self.current_assistant_content_blocks.append(ContentBlock(kind="toolInvocation", content=resolved_msg))
            tool_inv, _ = self.build_tool_invocation(tool_call_id, tool_name, arguments)
            if tool_inv:
                tool_inv.invocation_message = resolved_msg
                tool_inv.is_agent_backlink = True
                tool_inv.backlink_agent_id = agent_id
                self.current_assistant_tool_invocations.append(tool_inv)
            return

        # Capture full-file view results as reconstruction bases for later
        # edit-only sequences. Only whole-file reads (no view_range) are trusted;
        # the result is line-number-prefixed, so strip those to recover content.
        if tool_name == "view" and not argument_dict.get("view_range"):
            view_path = _get_tool_arg_str(argument_dict, "path")
            if view_path:
                execution = self.tool_executions.get(tool_call_id, {})
                complete = execution.get("complete")
                if complete and complete.get("data", {}).get("success"):
                    result_obj = complete["data"].get("result", {})
                    text = result_obj.get("content", "") if isinstance(result_obj, dict) else ""
                    if text:
                        self.view_base[view_path] = strip_view_line_numbers(text)

        tool_inv, cmd_run = self.build_tool_invocation(tool_call_id, tool_name, arguments)

        if cmd_run:
            # Add command run inline as a content block
            # Use intentionSummary/toolTitle as title if available
            if (intention_summary or tool_title) and not cmd_run.title:
                cmd_run.title = intention_summary or tool_title
            cmd_display = cmd_run.title or cmd_run.command
            if len(cmd_display) > 60:
                cmd_display = cmd_display[:57] + "..."
            self.current_assistant_content_blocks.append(
                ContentBlock(
                    kind="toolInvocation",
                    content=f"$ {cmd_run.command}" if cmd_run.command else cmd_display,
                    description=intention_summary or tool_title or cmd_run.title,
                )
            )
            self.current_assistant_command_runs.append(cmd_run)
            # Track async shell titles for backlink resolution
            if cmd_run.is_async and cmd_run.shell_id and cmd_run.title:
                self.shell_title_map[cmd_run.shell_id] = cmd_run.title

        elif tool_inv:
            if tool_name == "task":
                # Don't render task tool inline — the subagent block replaces it
                self.current_assistant_tool_invocations.append(tool_inv)
                return
            # Reshape successful edit/create into a CLI file op; multiple ops on
            # the same file in this turn are consolidated into one FileChange at
            # flush (VS Code parity). Gated on success like the runtime hook.
            if tool_name in ("edit", "create") and tool_inv.status == "success":
                file_op = cli_file_op_from_tool(tool_name, argument_dict)
                if file_op is not None:
                    self.current_assistant_file_ops.append(file_op)
            # Add tool invocation inline as a content block
            display_text = tool_inv.invocation_message or tool_inv.name
            # Use intentionSummary or toolTitle as enhanced description
            enhanced_desc = intention_summary or tool_title or tool_inv.name
            self.current_assistant_content_blocks.append(
                ContentBlock(
                    kind="toolInvocation",
                    content=display_text,
                    description=enhanced_desc,
                )
            )
            self.current_assistant_tool_invocations.append(tool_inv)

    def add_external_tool_inline(self, tool_call_id: str, tool_name: str, arguments: object) -> None:
        """Add an external (host-app) tool invocation inline in the current assistant message.

        External tools come from ``external_tool.requested`` events. The request carries the
        tool name and its ``arguments`` (parameters), but the paired ``external_tool.completed``
        event is a bare acknowledgement with no result payload — the host application executes
        these tools and does not record a response back into the session. We therefore render the
        parameters (previously discarded) and leave ``result`` unset; the template surfaces a note
        explaining that no response was recorded.
        """
        argument_dict = _as_tool_argument_dict(arguments)
        description = _get_tool_arg_str(argument_dict, "description") or None
        invocation_message = _format_tool_display_message(tool_name, argument_dict, description)
        tool_inv = ToolInvocation(
            name=tool_name,
            input=_serialize_tool_arguments(arguments),
            result=None,
            status=None,
            invocation_message=invocation_message,
            source_type="external",
        )
        self.current_assistant_content_blocks.append(
            ContentBlock(
                kind="toolInvocation",
                content=invocation_message or tool_name,
                description=description or tool_name,
            )
        )
        self.current_assistant_tool_invocations.append(tool_inv)


def _parse_cli_jsonl_file(file_path: Path) -> ChatSession | None:
    """Parse a GitHub Copilot CLI JSONL session file.

    CLI sessions are stored as JSONL (JSON Lines) where each line is a JSON object
    representing an event. The event-based format uses types like:
    - session.start: Session initialization with sessionId, copilotVersion, etc.
    - session.info: Info messages (authentication, mcp, folder_trust)
    - session.model_change: Model switching (newModel)
    - session.error: Error events (errorType, message)
    - session.truncation: Context window management events
    - user.message: User prompts with content and attachments
    - system.message: System-level messages
    - assistant.message: Assistant responses with content and toolRequests
    - assistant.turn_start/end: Turn boundaries
    - tool.execution_start/complete: Tool invocation lifecycle
    - tool.user_requested: User-requested tool executions
    - abort: Session/turn abort events

    This function renders CLI sessions similarly to how VS Code renders background
    chats:
    - Consecutive assistant messages are combined into one
    - Tool calls are displayed inline within the assistant message content

    Args:
        file_path: Path to the JSONL file.

    Returns:
        ChatSession object or None if parsing fails.
    """
    try:
        events = []

        with file_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = ssrjson.loads(line)
                    events.append(data)
                except ssrjson.JSONDecodeError:
                    continue

        if not events:
            return None

        fusion_statuses = _build_fusion_statuses(events)

        # Extract session metadata from session.start event
        session_id = None
        created_at = None
        session_start_context: dict = {}
        session_title_from_event: str | None = None

        for event in events:
            event_type = event.get("type", "")
            event_data = event.get("data", {})

            if event_type == "session.start":
                session_id = event_data.get("sessionId")
                created_at = event_data.get("startTime") or event.get("timestamp")
                # Extract context for workspace info
                context = event_data.get("context", {})
                session_start_context = context
                break  # Only need the first session.start event

        # If no session.start, use file stem as session ID
        if not session_id:
            session_id = file_path.stem

        # Extract workspace from session.start context or folder_trust event
        workspace_path = session_start_context.get("cwd") or session_start_context.get("gitRoot")
        workspace_name = _workspace_name_from_path(workspace_path)
        requester_username = None
        session_repository = session_start_context.get("repository")  # e.g. "owner/repo"

        for event in events:
            if event.get("type") == "session.info":
                event_data = event.get("data", {})
                info_type = event_data.get("infoType")
                message = event_data.get("message", "")

                if info_type == "folder_trust" and not workspace_path:
                    # Parse "Folder C:\_SRC\ZTS has been added to trusted folders."
                    if message.startswith("Folder ") and " has been added" in message:
                        folder_path = message[7 : message.find(" has been added")]
                        workspace_path = folder_path
                        workspace_name = _workspace_name_from_path(folder_path)

                elif info_type == "authentication" and not requester_username and "as user: " in message:
                    # Parse "Logged in with gh as user: Arithmomaniac"
                    requester_username = message.split("as user: ")[-1].strip()

        host_type = session_start_context.get("hostType")  # "github" or "ado"
        repository_url = None
        if session_repository and host_type != "ado":
            repository_url = _normalize_git_url(f"https://github.com/{session_repository}")
        if not repository_url and host_type != "ado":
            repository_url = detect_repository_url(workspace_path)

        context_entries: list[SessionContextEntry] = []
        _append_context_entry(
            context_entries,
            workspace_path=workspace_path,
            repository_url=repository_url,
            branch=session_start_context.get("branch"),
            timestamp=created_at,
            message_index=0,
            source="initial",
        )

        # Build tool execution map: toolCallId -> (start_data, complete_data, user_requested)
        tool_executions: dict = {}
        root_agent_intervals: list[RootAgentInterval] = []
        active_root_agent: RootAgentInterval | None = None
        # Map parent agent toolCallId -> list of child tool display names
        subagent_child_tools: dict[str, list[str]] = {}
        # Map parent agent toolCallId -> list of (ToolInvocation | None, CommandRun | None, content_blocks)
        subagent_child_structured: dict[str, list[tuple]] = {}
        # Track which subagents failed (toolCallId -> error message)
        subagent_failures: dict[str, str] = {}
        # Track which subagents completed successfully
        subagent_completions: set[str] = set()
        # Track all subagent.started toolCallIds (for inferring completion)
        subagent_started_ids: set[str] = set()
        # Map agent-N -> best result text from read_agent (for background agents)
        bg_agent_results: dict[str, str] = {}
        # Map task toolCallId -> agent-N id (parsed from background placeholder)
        bg_agent_id_map: dict[str, str] = {}
        # Map agent-N -> display title (e.g., "Explore Agent: Find auth")
        agent_display_names: dict[str, str] = {}
        for event in events:
            event_type = event.get("type", "")
            event_data = event.get("data", {})
            timestamp = event.get("timestamp")

            if event_type == "tool.execution_start":
                tool_call_id = event_data.get("toolCallId")
                if tool_call_id:
                    if tool_call_id not in tool_executions:
                        tool_executions[tool_call_id] = {"start": None, "complete": None, "user_requested": False}
                    tool_executions[tool_call_id]["start"] = event
                # Track child tools of subagents
                parent_tcid = event_data.get("parentToolCallId")
                if parent_tcid:
                    tool_name = event_data.get("toolName", "")
                    # Skip internal/meta tools from child tool display
                    if tool_name not in ("report_intent", "read_agent", "read_powershell", "read_bash", "skill", "task"):
                        arguments = event_data.get("arguments", {})
                        display = _format_tool_display_message(tool_name, arguments)
                        if display:
                            subagent_child_tools.setdefault(parent_tcid, []).append(display)
                        # Also store structured data for child tool (toolCallId, toolName, arguments)
                        subagent_child_structured.setdefault(parent_tcid, []).append((tool_call_id, tool_name, arguments))

            elif event_type == "tool.user_requested":
                # User explicitly requested this tool execution
                tool_call_id = event_data.get("toolCallId")
                if tool_call_id:
                    if tool_call_id not in tool_executions:
                        tool_executions[tool_call_id] = {"start": None, "complete": None, "user_requested": True}
                    else:
                        tool_executions[tool_call_id]["user_requested"] = True

            elif event_type == "tool.execution_complete":
                tool_call_id = event_data.get("toolCallId")
                if tool_call_id:
                    if tool_call_id not in tool_executions:
                        tool_executions[tool_call_id] = {"start": None, "complete": None, "user_requested": False}
                    tool_executions[tool_call_id]["complete"] = event
                    # Map task toolCallId -> agent_id from background placeholder
                    result_obj = event_data.get("result", {})
                    result_content = result_obj.get("content", "") if isinstance(result_obj, dict) else ""
                    if result_content.startswith("Agent started in background with agent_id: "):
                        bg_id = result_content.split("agent_id: ", 1)[1].split(".")[0].strip()
                        bg_agent_id_map[tool_call_id] = bg_id
                        # Build display name from task tool arguments
                        start_info = tool_executions[tool_call_id].get("start")
                        if start_info:
                            args = _as_tool_argument_dict(start_info.get("data", {}).get("arguments", {}))
                            agent_type = _get_tool_arg_str(args, "agent_type", "agent")
                            desc = _get_tool_arg_str(args, "description")
                            agent_display_names[bg_id] = f"{agent_type}: {desc}" if desc else agent_type
                    # Collect read_agent results for background agents
                    start_info = tool_executions[tool_call_id].get("start")
                    if start_info and start_info.get("data", {}).get("toolName") == "read_agent" and isinstance(result_obj, dict):
                        content = result_content
                        detailed = result_obj.get("detailedContent", "")
                        # Extract result text from content after "Result:\n" header
                        result_from_content = ""
                        if content and "Result:\n" in content:
                            result_from_content = content.split("Result:\n", 1)[-1].strip()
                        # Use the longer of content-after-Result vs detailedContent
                        best = result_from_content if len(result_from_content) > len(detailed) else detailed
                        if best:
                            args = _as_tool_argument_dict(start_info["data"].get("arguments", {}))
                            agent_id = str(args.get("agent_id", ""))
                            # Only keep the best (longest) result per agent
                            if len(best) > len(bg_agent_results.get(agent_id, "")):
                                bg_agent_results[agent_id] = best

            elif event_type == "subagent.failed":
                tool_call_id = event_data.get("toolCallId", "")
                error = event_data.get("error") or ""
                if tool_call_id:
                    subagent_failures[tool_call_id] = error

            elif event_type == "subagent.started":
                tool_call_id = event_data.get("toolCallId", "")
                if tool_call_id:
                    subagent_started_ids.add(tool_call_id)

            elif event_type == "subagent.completed":
                tool_call_id = event_data.get("toolCallId", "")
                if tool_call_id:
                    subagent_completions.add(tool_call_id)

            elif event_type == "subagent.selected":
                if timestamp is None:
                    continue
                if active_root_agent is not None:
                    active_root_agent.end_timestamp = timestamp
                    root_agent_intervals.append(active_root_agent)
                agent_name = event_data.get("agentName") or event_data.get("agentDisplayName") or "unknown"
                agent_display_name = event_data.get("agentDisplayName") or agent_name
                tools = event_data.get("tools")
                active_root_agent = RootAgentInterval(
                    agent_name=agent_name,
                    agent_display_name=agent_display_name,
                    start_timestamp=timestamp,
                    tools=tools if isinstance(tools, list) else None,
                )

            elif event_type == "subagent.deselected":
                if active_root_agent is not None:
                    active_root_agent.end_timestamp = timestamp
                    root_agent_intervals.append(active_root_agent)
                    active_root_agent = None

        if active_root_agent is not None:
            root_agent_intervals.append(active_root_agent)

        # Infer subagent completion from tool.execution_complete when no
        # explicit subagent.completed event was emitted (common in older CLI versions)
        for tcid in subagent_started_ids:
            if tcid not in subagent_completions and tcid not in subagent_failures:
                exec_info = tool_executions.get(tcid, {})
                if exec_info.get("complete") is not None:
                    subagent_completions.add(tcid)

        # Build messages using VSCode-style rendering:
        # - Process events in order
        # - Combine consecutive assistant messages
        # - Interleave tool invocations inline with content
        builder = _CliSessionBuilder(
            tool_executions,
            subagent_child_tools,
            subagent_failures,
            bg_agent_results,
            bg_agent_id_map,
            subagent_completions,
            subagent_child_structured,
            agent_display_names,
        )

        for event in events:
            event_type = event.get("type", "")
            event_data = event.get("data", {})
            timestamp = event.get("timestamp")

            # Skip events belonging to a subagent — they have parentToolCallId
            if event_data.get("parentToolCallId"):
                continue

            if event_type == "user.message":
                # Flush any pending assistant content before user message
                builder.flush_assistant_message()
                builder.pending_tool_requests.clear()

                content = event_data.get("content", "")
                builder.messages.append(
                    ChatMessage(
                        role="user",
                        content=content,
                        timestamp=timestamp,
                        source_event_id=event.get("id"),
                    )
                )

            elif event_type == "system.message":
                # Flush pending assistant content
                builder.flush_assistant_message()
                builder.pending_tool_requests.clear()

                content = event_data.get("content", "")
                if content:
                    builder.messages.append(
                        ChatMessage(
                            role="system",
                            content=content,
                            timestamp=timestamp,
                            source_event_id=event.get("id"),
                        )
                    )

            elif event_type in ("assistant.turn_start", "assistant.turn_end"):
                # Turn boundaries are internal to a single user interaction.
                # Do NOT flush or create separate messages - all assistant turns
                # between user messages should be combined into a single message.
                # Just continue accumulating content.
                pass

            elif event_type == "assistant.message":
                # Set timestamp from first assistant message in the sequence
                if builder.current_assistant_timestamp is None:
                    builder.current_assistant_timestamp = timestamp
                builder.note_assistant_event(event.get("id"))

                content = event_data.get("content", "")
                tool_requests = event_data.get("toolRequests", [])

                # Add reasoning/thinking content before text (inline format)
                reasoning_text = event_data.get("reasoningText", "")
                if reasoning_text and reasoning_text.strip():
                    builder.current_assistant_content_blocks.append(
                        ContentBlock(
                            kind="thinking",
                            content=reasoning_text.strip(),
                            description="reasoning",
                        )
                    )

                # Add any text content first
                if content and content.strip():
                    builder.current_assistant_content_blocks.append(
                        ContentBlock(
                            kind="text",
                            content=content.strip(),
                        )
                    )

                # Store tool requests for processing when execution starts/completes
                for req in tool_requests:
                    tool_call_id = req.get("toolCallId")
                    if tool_call_id:
                        builder.pending_tool_requests[tool_call_id] = req

            elif event_type == "tool.execution_start":
                builder.note_assistant_event(event.get("id"))
                # Add the tool invocation inline when execution starts
                tool_call_id = event_data.get("toolCallId")
                tool_name = event_data.get("toolName", "unknown")
                arguments = event_data.get("arguments", {})

                # Use stored request data if available, otherwise use start event data
                req = builder.pending_tool_requests.get(tool_call_id, {})
                if not arguments and req:
                    arguments = req.get("arguments", {})
                if tool_name == "unknown" and req:
                    tool_name = req.get("name", tool_name)

                # Extract intentionSummary/toolTitle from the tool request
                intention_summary = req.get("intentionSummary") if req else None
                tool_title = req.get("toolTitle") if req else None

                builder.add_tool_inline(tool_call_id, tool_name, arguments, intention_summary=intention_summary, tool_title=tool_title)

            elif event_type == "abort":
                builder.note_assistant_event(event.get("id"))
                # Session or turn was aborted - add as status block
                abort_reason = event_data.get("reason", "unknown")
                builder.current_assistant_content_blocks.append(
                    ContentBlock(
                        kind="status",
                        content=f"Aborted: {abort_reason}",
                        description="abort",
                    )
                )

            elif event_type == "session.error":
                builder.note_assistant_event(event.get("id"))
                # Session encountered an error - add as status block
                error_type = event_data.get("errorType", "unknown")
                error_message = event_data.get("message", "")
                builder.current_assistant_content_blocks.append(
                    ContentBlock(
                        kind="status",
                        content=f"Error: {error_message}" if error_message else f"Error: {error_type}",
                        description="error",
                    )
                )

            elif event_type == "session.model_change":
                builder.note_assistant_event(event.get("id"))
                # Model was changed during session
                new_model = event_data.get("newModel", "unknown")
                builder.current_assistant_content_blocks.append(
                    ContentBlock(
                        kind="status",
                        content=f"Switched to {new_model}",
                        description="model-change",  # hyphenated for CSS class
                    )
                )

            elif event_type == "assistant.reasoning":
                builder.note_assistant_event(event.get("id"))
                # Reasoning content - similar to VS Code thinking blocks
                reasoning_content = event_data.get("content", "")
                if reasoning_content and reasoning_content.strip():
                    builder.current_assistant_content_blocks.append(
                        ContentBlock(
                            kind="thinking",  # Use existing kind for consistency
                            content=reasoning_content.strip(),
                            description="reasoning",
                        )
                    )

            elif event_type == "assistant.intent":
                builder.note_assistant_event(event.get("id"))
                intent_text = (event_data.get("intent") or "").strip()
                if intent_text:
                    builder.current_assistant_content_blocks.append(
                        ContentBlock(
                            kind="intent",
                            content=intent_text,
                        )
                    )

            elif event_type == "skill.invoked":
                builder.note_assistant_event(event.get("id"))
                # Skill was loaded - show name and content summary
                skill_name = event_data.get("name", "unknown")
                skill_content = event_data.get("content", "")
                # Extract description from YAML frontmatter if present
                skill_desc = None
                if skill_content and "description:" in skill_content:
                    for line in skill_content.split("\n"):
                        if line.strip().startswith("description:"):
                            skill_desc = line.split("description:", 1)[1].strip()
                            break
                builder.current_assistant_content_blocks.append(
                    ContentBlock(
                        kind="skill",
                        content=f"Loaded skill: {skill_name}",
                        description=skill_desc,
                    )
                )

            elif event_type == "session.compaction_complete":
                builder.note_assistant_event(event.get("id"))
                # Session was compacted - show checkpoint info
                checkpoint_num = event_data.get("checkpointNumber", 0)
                summary = event_data.get("summaryContent", "")
                # Extract overview section if present
                overview = None
                if "<overview>" in summary and "</overview>" in summary:
                    overview = summary.split("<overview>")[1].split("</overview>")[0].strip()
                    if len(overview) > 200:
                        overview = overview[:197] + "..."
                if not overview:
                    overview = f"Session compacted to checkpoint {checkpoint_num}"
                builder.current_assistant_content_blocks.append(
                    ContentBlock(
                        kind="status",
                        content=overview,
                        description="compaction",
                    )
                )

            # --- Subagent lifecycle events ---
            elif event_type == "subagent.started":
                builder.note_assistant_event(event.get("id"))
                display_name = event_data.get("agentDisplayName") or event_data.get("agentName", "unknown")
                tool_call_id = event_data.get("toolCallId", "")
                req = builder.pending_tool_requests.get(tool_call_id, {})
                req_args = _as_tool_argument_dict(req.get("arguments", {}))
                description = _get_tool_arg_str(req_args, "description")
                failed = tool_call_id in builder.subagent_failures
                completed = tool_call_id in builder.subagent_completions
                # Title includes status indicator for template
                title = f"{display_name}: {description}" if description else display_name
                if failed:
                    title = f"❌ {title}"
                # Build content: child tool summaries + result or error
                parts: list[str] = []
                for child_display in builder.subagent_child_tools.get(tool_call_id, []):
                    parts.append(f"*{child_display}*")
                # Detect background agent (even for failed ones)
                is_bg = tool_call_id in builder.bg_agent_id_map
                bg_id = builder.bg_agent_id_map.get(tool_call_id, "")
                result_text = ""
                if failed:
                    error = builder.subagent_failures[tool_call_id]
                    parts.append(f"**Error:** {error}" if error else "**Error:** Unknown error")
                else:
                    # Get the agent's result — check for background agent placeholder
                    execution = builder.tool_executions.get(tool_call_id, {})
                    complete_event = execution.get("complete")
                    if complete_event:
                        complete_data = complete_event.get("data", {})
                        result_obj = complete_data.get("result", {})
                        if isinstance(result_obj, dict):
                            result_text = result_obj.get("content", "")
                        elif result_obj:
                            result_text = str(result_obj)
                    # Background agents return a placeholder — use read_agent result instead
                    if not result_text or result_text.startswith("Agent started in background"):
                        is_bg = True
                        if bg_id:
                            result_text = builder.bg_agent_results.get(bg_id, "")
                        else:
                            result_text = ""
                    if result_text:
                        parts.append(result_text)
                content = "\n\n".join(parts) if parts else "(no output)"
                if failed:
                    kind = "subagent_failed"
                elif completed:
                    kind = "subagent"
                else:
                    kind = "subagent_incomplete"

                # Build structured content_blocks for the subagent interior
                nested_blocks: list[ContentBlock] = []
                nested_tool_invocations: list[ToolInvocation] = []
                nested_command_runs: list[CommandRun] = []
                nested_file_ops: list[CliFileOp] = []
                for child_tcid, child_tool_name, child_args in builder.subagent_child_structured.get(tool_call_id, []):
                    child_args_dict = _as_tool_argument_dict(child_args)
                    # Capture subagent full-file views so its own edit-only sequences can
                    # reconstruct a net diff (parity with the main-pass capture).
                    if child_tool_name == "view" and not child_args_dict.get("view_range"):
                        vp = _get_tool_arg_str(child_args_dict, "path")
                        complete = builder.tool_executions.get(child_tcid, {}).get("complete")
                        if vp and complete and complete.get("data", {}).get("success"):
                            r = complete["data"].get("result", {})
                            vt = r.get("content", "") if isinstance(r, dict) else ""
                            if vt:
                                builder.view_base[vp] = strip_view_line_numbers(vt)
                    tool_inv, cmd_run = builder.build_tool_invocation(child_tcid, child_tool_name, child_args)
                    display = _format_tool_display_message(child_tool_name, child_args)
                    if tool_inv:
                        nested_tool_invocations.append(tool_inv)
                        nested_blocks.append(ContentBlock(kind="toolInvocation", content=display or child_tool_name))
                        # Reshape successful child edit/create into a file op (VS Code parity)
                        if child_tool_name in ("edit", "create") and tool_inv.status == "success":
                            child_op = cli_file_op_from_tool(child_tool_name, child_args_dict)
                            if child_op is not None:
                                nested_file_ops.append(child_op)
                    elif cmd_run:
                        nested_command_runs.append(cmd_run)
                        # Use "$ command" format so metadata matching connects them inline
                        cmd_content = f"$ {cmd_run.command}" if cmd_run.command else (display or cmd_run.command)
                        nested_blocks.append(
                            ContentBlock(
                                kind="toolInvocation",
                                content=cmd_content,
                                description=cmd_run.title,
                            )
                        )
                # Add result text as a text block
                if result_text:
                    nested_blocks.append(ContentBlock(kind="text", content=result_text))
                elif failed:
                    error = builder.subagent_failures.get(tool_call_id, "Unknown error")
                    nested_blocks.append(ContentBlock(kind="text", content=f"**Error:** {error}" if error else "**Error:** Unknown error"))

                # Create child ChatMessage for the subagent
                nested_consolidated_changes = consolidate_cli_file_ops(nested_file_ops, builder.view_base)
                child_msg = ChatMessage(
                    role="assistant",
                    content=content,
                    agent_display_name=title,
                    # TODO: For multi-level nesting, derive from parent's nesting level
                    agent_nesting_level=1,
                    source_event_id=builder.current_assistant_source_event_id,
                    tool_invocations=nested_tool_invocations,
                    command_runs=nested_command_runs,
                    content_blocks=nested_blocks,
                    file_changes=nested_consolidated_changes,
                )

                block = ContentBlock(
                    kind=kind,
                    content=content,
                    description=title,
                    child_message=child_msg,
                    prompt=_get_tool_arg_str(req_args, "prompt"),
                    is_background=is_bg,
                    agent_id=bg_id if is_bg else "",
                )
                # Also set deprecated fields for backward compat during transition
                block.content_blocks = nested_blocks
                block.tool_invocations = nested_tool_invocations
                block.command_runs = nested_command_runs
                block.file_changes = nested_consolidated_changes
                builder.current_assistant_content_blocks.append(block)

            elif event_type == "subagent.completed":
                builder.note_assistant_event(event.get("id"))
                display_name = event_data.get("agentDisplayName") or event_data.get("agentName", "unknown")
                tool_call_id = event_data.get("toolCallId", "")
                req = builder.pending_tool_requests.get(tool_call_id, {})
                req_args = _as_tool_argument_dict(req.get("arguments", {}))
                description = _get_tool_arg_str(req_args, "description")
                label = f"{display_name}: {description} — completed" if description else f"{display_name} — completed"
                builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=label, description="subagent"))

            elif event_type == "subagent.failed":
                builder.note_assistant_event(event.get("id"))
                display_name = event_data.get("agentDisplayName") or event_data.get("agentName", "unknown")
                tool_call_id = event_data.get("toolCallId", "")
                req = builder.pending_tool_requests.get(tool_call_id, {})
                req_args = _as_tool_argument_dict(req.get("arguments", {}))
                description = _get_tool_arg_str(req_args, "description")
                label = f"{display_name}: {description} — failed" if description else f"{display_name} — failed"
                builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=label, description="subagent-error"))

            # --- Session lifecycle events ---
            elif event_type == "session.info" and event_data.get("infoType") == "fork":
                message = _stringify_event_value(event_data.get("message"))
                if message:
                    fork_message = _format_fork_info_message(message, session_id)
                    builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=fork_message, description="fork"))

            elif event_type == "session.fusion_resolved":
                fusion_id = _stringify_event_value(event_data.get("fusionId"))
                status = fusion_statuses.get(fusion_id)
                if status:
                    builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=status, description="fusion"))

            elif event_type == "session.handoff":
                source_type = event_data.get("sourceType") or "unknown"
                repo = event_data.get("repository") or {}
                owner = repo.get("owner") or ""
                name = repo.get("name") or ""
                branch = repo.get("branch") or ""
                if owner and name:
                    repo_info = f" ({owner}/{name} @ {branch})" if branch else f" ({owner}/{name})"
                else:
                    repo_info = ""
                builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=f"🔄 Session handoff from {source_type}{repo_info}", description="handoff"))

            elif event_type == "session.warning":
                message = (event_data.get("message") or "").strip()
                if message:
                    builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=f"⚠️ {message}", description="warning"))

            elif event_type == "session.mode_changed":
                prev_mode = event_data.get("previousMode") or "unknown"
                new_mode = event_data.get("newMode") or "unknown"
                builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=f"Mode changed: {prev_mode} → {new_mode}", description="mode-change"))

            elif event_type == "session.context_changed":
                cwd = event_data.get("cwd") or ""
                branch = event_data.get("branch") or ""
                branch_info = f" ({branch})" if branch else ""
                changed_repository_url = detect_repository_url(cwd) if host_type != "ado" else None
                _append_context_entry(
                    context_entries,
                    workspace_path=cwd or None,
                    repository_url=changed_repository_url,
                    branch=branch or None,
                    timestamp=timestamp,
                    message_index=len(builder.messages),
                    source="context_changed",
                )
                builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=f"Working directory changed: {cwd}{branch_info}", description="context-change"))

            elif event_type == "session.remote_steerable_changed":
                remote_steerable = event_data.get("remoteSteerable")
                state = "enabled" if remote_steerable else "disabled"
                builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=f"Remote steering {state}", description="remote-steering"))

            elif event_type == "session.plan_changed":
                operation = event_data.get("operation") or "changed"
                builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=f"Plan {operation}", description="plan-change"))

            elif event_type == "session.task_complete":
                summary = (event_data.get("summary") or "").strip()
                if summary:
                    builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=summary, description="task-complete"))

            elif event_type == "session.title_changed":
                title = (event_data.get("title") or "").strip()
                if title:
                    session_title_from_event = title
                    builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=f'📝 Session title: "{title}"', description="title-changed"))

            elif event_type == "assistant.usage":
                model = event_data.get("model", "")
                input_tokens = event_data.get("inputTokens") or 0
                output_tokens = event_data.get("outputTokens") or 0
                cost = event_data.get("cost")
                duration = event_data.get("duration")
                parts = [model] if model else []
                if input_tokens or output_tokens:
                    parts.append(f"{int(input_tokens):,} in → {int(output_tokens):,} out")
                if cost is not None:
                    parts.append(f"cost {cost}")
                if duration is not None:
                    parts.append(f"{duration / 1000:.1f}s")
                effort = event_data.get("reasoningEffort")
                if effort:
                    parts.append(f"effort={effort}")
                if parts:
                    builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content="📊 " + " · ".join(parts), description="usage"))

            elif event_type == "session.shutdown":
                shutdown_type = event_data.get("shutdownType") or "unknown"
                code_changes = event_data.get("codeChanges") or {}
                lines_added = code_changes.get("linesAdded", 0)
                lines_removed = code_changes.get("linesRemoved", 0)
                files_modified = code_changes.get("filesModified") or []
                parts = [f"Session ended ({shutdown_type})"]
                if lines_added or lines_removed:
                    parts.append(f"+{lines_added}/-{lines_removed} lines across {len(files_modified)} files")
                model_metrics = event_data.get("modelMetrics") or {}
                for model_name, metrics in model_metrics.items():
                    metric_obj = metrics or {}
                    requests = (metric_obj.get("requests") or {}).get("count", 0)
                    cost = (metric_obj.get("requests") or {}).get("cost", 0)
                    if requests:
                        parts.append(f"{model_name}: {requests} requests, cost {cost}")
                builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=" · ".join(parts), description="shutdown"))

            elif event_type == "elicitation.requested":
                message = _stringify_event_value(event_data.get("message"))
                if message:
                    mode = _stringify_event_value(event_data.get("mode"))
                    content = f"? {message}"
                    if mode:
                        content += f"\n   Mode: {mode}"
                    builder.current_assistant_content_blocks.append(ContentBlock(kind="ask_user", content=content, description="elicitation"))

            elif event_type == "elicitation.completed":
                action = _stringify_event_value(event_data.get("action"))
                content = _stringify_event_value(event_data.get("content"))
                if action or content:
                    label = f"Elicitation {action}" if action else "Elicitation completed"
                    if content:
                        label += f": {content}"
                    builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=label, description="elicitation"))

            elif event_type == "user_input.requested":
                question = _stringify_event_value(event_data.get("question"))
                if question:
                    choices_value = event_data.get("choices")
                    choices = choices_value if isinstance(choices_value, list) else []
                    content = f"? {question}"
                    if choices:
                        content += f"\n   Options: {_stringify_event_value(choices)}"
                    if event_data.get("allowFreeform"):
                        content += "\n   Freeform answer allowed"
                    builder.current_assistant_content_blocks.append(ContentBlock(kind="ask_user", content=content, description="user-input"))

            elif event_type == "user_input.completed":
                answer = _stringify_event_value(event_data.get("answer"))
                if answer:
                    builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=f"User answered: {answer}", description="user-input"))

            elif event_type == "external_tool.requested":
                # Host-app tool invoked outside the CLI. The request carries the parameters;
                # render them as a proper tool block instead of a bare status breadcrumb.
                ext_tool_name = _stringify_event_value(event_data.get("toolName"))
                ext_tool_call_id = _stringify_event_value(event_data.get("toolCallId"))
                builder.add_external_tool_inline(ext_tool_call_id, ext_tool_name, event_data.get("arguments", {}))

            elif event_type == "external_tool.completed":
                # Bare acknowledgement (only carries requestId) — no result payload to render.
                pass

            elif formatted_status := _format_session_event_status(event_type, event_data):
                content, description = formatted_status
                if content:
                    builder.current_assistant_content_blocks.append(ContentBlock(kind="status", content=content, description=description))

            # Skip internal/metadata events (already parsed in metadata extraction or no user content)
            elif event_type in (
                "session.start",  # Parsed above for sessionId, startTime, context
                "session.info",  # Informational (e.g., model confirmations)
                "session.error",  # Handled above
                # Internal turn boundary markers
                "assistant.turn_start",
                "assistant.turn_end",
                # Internal lifecycle
                "session.resume",
                # Hook starts are noisy; failed hook endings are rendered above.
                "hook.start",
                # Streaming deltas (partial content, final is in assistant.message)
                "assistant.message_start",
                "assistant.message_delta",
                "assistant.reasoning_delta",
                "assistant.streaming_delta",
                "assistant.tool_call_delta",
                "assistant.server_tool_progress",
                "assistant.fusion_phase_activity",
                "assistant.fusion_phase_started",
                "assistant.fusion_phase_completed",
                "assistant.fusion_phase_failed",
                "assistant.turn_retry",
                # Plan mode lifecycle
                "exit_plan_mode.requested",
                "exit_plan_mode.completed",
                # Internal state tracking
                "pending_messages.modified",
                # Ephemeral SDK/client dispatch events. Slash-command chat content
                # persists through user.message/session.mode_changed instead.
                "capabilities.changed",
                "commands.changed",
                "sampling.completed",
                "sampling.requested",
                "session.background_tasks_changed",
                "session.canvas.opened",
                "session.canvas.closed",
                "session.canvas.registry_changed",
                "session.canvas.recorded",
                "session.canvas.removed",
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
                "session.usage_checkpoint",
                "session.context_cleared",
                "session.fusion_commit_started",
                "session.fusion_completed",
                "session.fusion_handoff",
                "session.fusion_route_started",
                "session.fusion_route_failed",
                "subagent.configured",
                "skill.context_delivered",
                "skill.context_delivered_ref",
                "skill.invoked_ref",
                "model.call_start",
                "model.call_finished",
                "permission.carriedForward",
                "permission.messageAuthorization",
                "permission.messageAuthorizationDegraded",
                "permission.messageAuthorizationRead",
                "prompt_cache_break",
                "sandbox.decision",
                "session.completion_receipt",
                "session.session_limits_changed",
                "tool_search.activated",
                "factory.run_started",
                "factory.run_updated",
                "factory.run_settled",
            ):
                pass

        # Flush any remaining assistant content
        builder.flush_assistant_message()

        messages = builder.messages
        if not messages:
            return None

        # Post-process: collect shell IO entries and attach to parent CommandRun
        # Build shellId → list of (msg_idx, CommandRun) for temporal scoping
        shell_cmd_runs: dict[str, list[tuple[int, CommandRun]]] = {}
        for msg_idx, msg in enumerate(messages):
            for cmd in msg.command_runs:
                if cmd.is_async and cmd.shell_id:
                    shell_cmd_runs.setdefault(cmd.shell_id, []).append((msg_idx, cmd))

        # Walk all tool invocations, create IO entries for shell backlinks
        # For duplicate shellIds, attach to the most recent CommandRun that appeared before the backlink
        io_counters: dict[str, int] = {}  # (shellId, cmd_msg_idx, action) → counter for unique IDs
        for msg_idx, msg in enumerate(messages):
            for tool in msg.tool_invocations:
                if not tool.is_shell_backlink or not tool.backlink_shell_id:
                    continue
                runs = shell_cmd_runs.get(tool.backlink_shell_id)
                if not runs:
                    continue
                # Find the most recent CommandRun at or before this message
                parent_cmd = None
                for run_msg_idx, cmd in reversed(runs):
                    if run_msg_idx <= msg_idx:
                        parent_cmd = cmd
                        break
                if not parent_cmd:
                    parent_cmd = runs[0][1]  # fallback to first
                action = tool.name.replace("_powershell", "")
                # Generate unique IDs scoped to the specific CommandRun instance
                counter_key = f"{tool.backlink_shell_id}-{id(parent_cmd)}-{action}"
                io_counters[counter_key] = io_counters.get(counter_key, 0) + 1
                count = io_counters[counter_key]
                anchor_id = f"io-{tool.backlink_shell_id}-{action}-{count}"
                pill_id = f"pill-{tool.backlink_shell_id}-{action}-{count}"
                # Extract write input from tool arguments
                input_text = None
                if action == "write" and tool.input:
                    try:
                        args = json.loads(tool.input)
                        input_text = args.get("input", "")
                    except (json.JSONDecodeError, TypeError):
                        pass
                entry = ShellIOEntry(
                    action=action,
                    input_text=input_text,
                    result=tool.result,
                    anchor_id=anchor_id,
                    pill_id=pill_id,
                )
                parent_cmd.io_entries.append(entry)
                # Store IDs on the ToolInvocation so the template can render pill with correct anchors
                tool.shell_pill_id = pill_id
                tool.shell_anchor_id = anchor_id

        # Get file metadata for incremental refresh
        source_file_mtime, source_file_size = _get_file_metadata(file_path)

        # Get updated_at from last event timestamp
        updated_at = events[-1].get("timestamp") if events else None

        # Determine session title: prefer session.title_changed event, then workspace.yaml, then first intent
        custom_title = None
        if session_title_from_event:
            custom_title = session_title_from_event
        if not custom_title:
            workspace_meta = _parse_workspace_yaml(file_path.parent)
            if workspace_meta.get("summary"):
                custom_title = workspace_meta["summary"]
        if not custom_title:
            # Fall back to first report_intent content block
            for msg in messages:
                for block in msg.content_blocks:
                    if block.kind == "intent" and block.content:
                        custom_title = block.content
                        break
                if custom_title:
                    break

        session = ChatSession(
            session_id=session_id,
            workspace_name=workspace_name,
            workspace_path=workspace_path,
            messages=messages,
            created_at=created_at,
            updated_at=updated_at,
            source_file=str(file_path),
            vscode_edition="cli",  # CLI edition badge
            custom_title=custom_title,
            requester_username=requester_username,
            responder_username=None,
            source_file_mtime=source_file_mtime,
            source_file_size=source_file_size,
            type="cli",
            repository_url=repository_url,
            root_agent_intervals=root_agent_intervals,
            context_entries=context_entries,
        )
        session.parser_version = PARSER_VERSION
        return session
    except (OSError, Exception):
        return None
