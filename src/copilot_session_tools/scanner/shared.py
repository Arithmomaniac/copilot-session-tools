"""Shared scanner helpers used by both CLI and VS Code parsers.

Provides normalizations that ensure both scanners produce compatible output
for the same logical concepts (status values, command runs, invocation messages).
"""

import re

from .models import CommandRun

_ANSI_ESCAPE_PATTERN = re.compile(r"\x1b(?:\[[0-9;]*[A-Za-z]|\][^\x07]*\x07|\][^\x1b]*\x1b\\)")
_CONTROL_CHARACTER_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def strip_ansi_control_sequences(text: str) -> str:
    """Remove terminal escape sequences and non-printing control characters."""
    return _CONTROL_CHARACTER_PATTERN.sub("", _ANSI_ESCAPE_PATTERN.sub("", text)).strip()


# ---------------------------------------------------------------------------
# Status normalisation  (Issue #55 — D1)
# ---------------------------------------------------------------------------

# VS Code uses "completed" / "pending"; CLI uses "success" / "error" / None.
# The HTML template and CSS already handle "success", "error", "pending".
_VSCODE_STATUS_MAP = {
    "completed": "success",
    "complete": "success",
}


def normalize_tool_status(
    raw_status: str | None,
    *,
    is_complete: bool | None = None,
    has_error: bool = False,
) -> str | None:
    """Normalise tool-invocation status to the CLI canonical values.

    Args:
        raw_status: The status string from the scanner (e.g. ``"completed"``).
        is_complete: VS-Code-style boolean completeness flag.
        has_error: Whether an error condition was detected.

    Returns:
        ``"success"``, ``"error"``, ``"pending"``, or ``None``.
    """
    if has_error:
        return "error"
    if raw_status:
        mapped = _VSCODE_STATUS_MAP.get(raw_status.lower())
        if mapped:
            return mapped
        return raw_status
    if is_complete is not None:
        return "success" if is_complete else "pending"
    return None


# ---------------------------------------------------------------------------
# Command-run extraction  (Issue #56 — D2)
# ---------------------------------------------------------------------------

# Tool names that map to shell/terminal commands in the CLI scanner.
SHELL_TOOL_NAMES = frozenset(
    {
        "powershell",
        "bash",
        "shell",
        "run_command",
        "run_in_terminal",
        "copilot_runInTerminal",
    }
)


def extract_command_run(
    tool_name: str,
    command: str | None,
    output: str | None = None,
    title: str | None = None,
    status: str | None = None,
) -> CommandRun | None:
    """Create a ``CommandRun`` when *tool_name* represents a shell command.

    Returns ``None`` if *tool_name* is not a shell tool or *command* is empty.
    """
    if tool_name not in SHELL_TOOL_NAMES:
        return None
    if not command:
        return None
    return CommandRun(
        command=command,
        title=title,
        result=output,
        status=status,
        output=output,
    )


# ---------------------------------------------------------------------------
# Invocation-message normalisation  (Issue #58 — D4)
# ---------------------------------------------------------------------------

# Mapping from VS Code built-in tool IDs to human-readable prefixes.
_VSCODE_TOOL_MESSAGE_MAP: dict[str, str] = {
    "copilot_readFile": "Viewing",
    "copilot_replaceString": "Edited",
    "copilot_multiReplaceString": "Edited",
    "copilot_applyPatch": "Edited",
    "copilot_createFile": "Created",
    "copilot_listDirectory": "Listing",
}


def normalize_invocation_message(
    tool_id: str,
    tool_data: dict | None,
    raw_message: str | None,
) -> str | None:
    """Generate a CLI-style invocation message for VS Code built-in tools.

    Only overrides when the original message is generic (``Using "…"``) or
    contains ``file:///`` URIs.  MCP tool messages are left untouched.
    """
    if not raw_message:
        return raw_message

    # Don't override MCP or already-specific messages
    is_generic = raw_message.startswith("Using ") or "file:///" in raw_message
    if not is_generic:
        return raw_message

    prefix = _VSCODE_TOOL_MESSAGE_MAP.get(tool_id)
    if not prefix:
        return raw_message

    # Try to extract a filename from toolSpecificData
    filename = _extract_file_path_from_tool_data(tool_data)
    if filename:
        return f"{prefix} `{filename}`"

    return raw_message


def _extract_file_path_from_tool_data(tool_data: dict | None) -> str | None:
    """Pull the shortest useful file path from VS Code ``toolSpecificData``."""
    if not isinstance(tool_data, dict):
        return None

    # File tools: toolSpecificData.file.uri
    file_info = tool_data.get("file")
    if isinstance(file_info, dict):
        uri = file_info.get("uri", {})
        if isinstance(uri, dict):
            path = uri.get("fsPath") or uri.get("path")
        elif isinstance(uri, str):
            path = uri
        else:
            path = None
        if path:
            # Return leaf filename for brevity
            if "/" in path:
                return path.split("/")[-1]
            if "\\" in path:
                return path.split("\\")[-1]
            return path

    return None
