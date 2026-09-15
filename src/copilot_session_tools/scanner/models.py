"""Data models for Copilot chat session scanning."""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ToolInvocation:
    """Represents a tool invocation in a chat response.

    Based on ChatToolInvocation from Arbuzov/copilot-chat-history.
    """

    name: str
    input: str | None = None
    result: str | None = None
    status: str | None = None
    start_time: int | None = None
    end_time: int | None = None
    source_type: str | None = None  # 'mcp' or 'internal'
    invocation_message: str | None = None  # Pretty display message (e.g., "Reading file.txt, lines 1 to 100")
    subagent_invocation_id: str | None = None  # VS Code: links child tools to parent sub-agent
    is_agent_backlink: bool = False  # True for read_agent calls that should render as back-links
    backlink_agent_id: str | None = None  # The agent-N id this back-link points to
    is_shell_backlink: bool = False  # True for read/write/stop_powershell calls linked to an async shell
    backlink_shell_id: str | None = None  # The shellId this back-link points to
    shell_pill_id: str | None = None  # Unique pill id for bidirectional linking with IO entry
    shell_anchor_id: str | None = None  # Corresponding IO entry anchor id to link to


@dataclass
class FileChange:
    """Represents a file change in a chat response.

    Based on ChatFileChange from Arbuzov/copilot-chat-history.
    """

    path: str
    diff: str | None = None
    content: str | None = None
    explanation: str | None = None
    language_id: str | None = None


@dataclass
class ShellIOEntry:
    """Represents a single read/write/stop interaction with an async shell."""

    action: str  # 'read', 'write', 'stop'
    input_text: str | None = None  # For write: the text sent to the shell
    result: str | None = None  # The response/output from the interaction
    anchor_id: str | None = None  # Unique id for bidirectional linking (e.g., "io-poll2-write-1")
    pill_id: str | None = None  # Corresponding pill id in conversation flow


@dataclass
class CommandRun:
    """Represents a command execution in a chat response.

    Based on ChatCommandRun from Arbuzov/copilot-chat-history.
    """

    command: str
    title: str | None = None
    result: str | None = None
    status: str | None = None
    output: str | None = None
    timestamp: int | None = None
    shell_id: str | None = None  # shellId for async/detached shells
    is_async: bool = False  # True when mode="async" (interactive shell session)
    is_detached: bool = False  # True when detach=true (persists after session shutdown)
    io_entries: list[ShellIOEntry] = field(default_factory=list)  # Collected read/write/stop interactions


@dataclass
class ContentBlock:
    """Represents a content block in an assistant response.

    Each block has a kind (e.g., 'text', 'thinking', 'tool') and content.
    This allows differentiation between thinking/reasoning and regular output.

    For subagent blocks (kind='subagent'), use child_message to access the
    structured child ChatMessage containing content_blocks, tool_invocations,
    etc. The 'content' field serves as a markdown fallback for legacy data.

    Deprecated fields (use child_message instead):
        content_blocks, tool_invocations, file_changes, command_runs
    """

    kind: str  # 'text', 'thinking', 'tool', 'promptFile', etc.
    content: str
    description: str | None = None  # Optional description (e.g., generatedTitle for thinking blocks)
    child_message: "ChatMessage | None" = None  # For subagent blocks: the child message containing structured data
    prompt: str | None = None  # For subagent blocks: the prompt text passed to the agent
    is_background: bool = False  # True for background (async) agents, False for sync
    agent_id: str | None = None  # For background agents: the agent-N id (e.g., "agent-0")
    # Deprecated: use child_message instead. Kept for backward compatibility.
    content_blocks: list["ContentBlock"] = field(default_factory=list)
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    file_changes: list[FileChange] = field(default_factory=list)
    command_runs: list[CommandRun] = field(default_factory=list)


@dataclass
class ChatMessage:
    """Represents a single message in a chat session.

    Enhanced based on ChatMessage/ChatResponseItem from Arbuzov/copilot-chat-history.
    """

    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str | None = None
    tool_invocations: list[ToolInvocation] = field(default_factory=list)
    file_changes: list[FileChange] = field(default_factory=list)
    command_runs: list[CommandRun] = field(default_factory=list)
    content_blocks: list[ContentBlock] = field(default_factory=list)  # Structured content with kind
    cached_markdown: str | None = None  # Pre-computed markdown for this message
    source_event_id: str | None = None  # CLI event id used to identify copied fork history
    agent_id: str | None = None  # toolCallId linking to parent task tool invocation
    agent_display_name: str | None = None  # "General Purpose Agent", "Explore Agent", etc.
    agent_nesting_level: int = 0  # 0=top-level, 1=inside agent, 2=nested agent, etc.
    child_index: int | None = None  # Sibling ordering within parent (None for top-level messages)
    original_content: str | None = None  # Pre-cleanup content (None = not cleaned)
    cleanup_model: str | None = None  # LLM model used for cleanup
    children: list["ChatMessage"] = field(default_factory=list)  # Child subagent messages


@dataclass
class RootAgentInterval:
    """Represents a root custom-agent selection interval in a CLI session."""

    agent_name: str
    agent_display_name: str
    start_timestamp: str
    end_timestamp: str | None = None
    tools: list[str] | None = None


@dataclass
class SessionContextEntry:
    """Represents the workspace/repository context active during part of a session."""

    workspace_name: str | None = None
    workspace_path: str | None = None
    repository_url: str | None = None
    branch: str | None = None
    timestamp: str | None = None
    message_index: int = 0
    source: str = "initial"


@dataclass
class ChatSession:
    """Represents a Copilot chat session.

    Based on ChatSession/ChatSessionData from Arbuzov/copilot-chat-history.
    """

    session_id: str
    workspace_name: str | None
    workspace_path: str | None
    messages: list[ChatMessage]
    created_at: str | None = None
    updated_at: str | None = None
    source_file: str | None = None
    vscode_edition: str = "stable"  # 'stable' or 'insider'
    custom_title: str | None = None
    requester_username: str | None = None
    responder_username: str | None = None
    source_file_mtime: float | None = None  # File modification time for incremental refresh
    source_file_size: int | None = None  # File size in bytes for incremental refresh
    type: str = "vscode"  # 'vscode' or 'cli'
    repository_url: str | None = None  # Git remote URL for repository-scoped memories
    parser_version: int = 1
    source_format: str | None = None  # 'cli', 'json', 'jsonl', 'vscdb'
    root_agent_intervals: list[RootAgentInterval] = field(default_factory=list)
    context_entries: list[SessionContextEntry] = field(default_factory=list)
    source_id: str | None = None
    native_session_id: str | None = None
    source_name: str | None = None

    def effective_context_entries(self) -> list[SessionContextEntry]:
        """Return context entries, falling back to the legacy scalar metadata."""
        entries = list(self.context_entries)
        if not entries and (self.workspace_name or self.workspace_path or self.repository_url):
            entries.append(
                SessionContextEntry(
                    workspace_name=self.workspace_name,
                    workspace_path=self.workspace_path,
                    repository_url=self.repository_url,
                    message_index=0,
                    source="initial",
                )
            )

        deduped: list[SessionContextEntry] = []
        for entry in entries:
            key = (entry.workspace_name, entry.workspace_path, entry.repository_url, entry.branch)
            if deduped:
                previous = deduped[-1]
                previous_key = (previous.workspace_name, previous.workspace_path, previous.repository_url, previous.branch)
                if key == previous_key:
                    continue
            deduped.append(entry)
        return deduped


@dataclass
class SessionFileInfo:
    """Lightweight metadata about a session file for incremental scanning.

    This allows checking mtime/size before expensive parsing.
    """

    file_path: Path
    file_type: str  # 'json', 'vscdb', 'jsonl'
    session_type: str  # 'vscode' or 'cli'
    vscode_edition: str  # 'stable', 'insider', or 'cli'
    mtime: float
    size: int
    workspace_name: str | None = None
    workspace_path: str | None = None
