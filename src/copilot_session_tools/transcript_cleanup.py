"""Transcript cleanup for voice-dictated user messages.

Uses a two-stage approach:
1. Heuristic pre-filter (pure Python, no deps) to identify likely voice-dictated messages
2. Batch LLM call via LiteLLM to classify and clean messages in a single pass

Uses structured output with auto-dispatch per model type:
- GPT on /chat/completions: response_format=Pydantic (native json_schema)
- GPT 5.4 family: litellm.aresponses() (Responses API)
- Claude/Gemini/other: tool_choice (forced function call matching schema)
"""

from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .scanner import ChatMessage

if TYPE_CHECKING:
    from .database import Database
    from .scanner import ChatSession

# ---------------------------------------------------------------------------
# Heuristic pre-filter (pure Python, no external deps)
# ---------------------------------------------------------------------------

# Common speech-to-text filler words
_FILLER_WORDS = frozenset(
    {
        "like",
        "um",
        "uh",
        "basically",
        "actually",
        "so",
        "literally",
        "right",
        "okay",
        "well",
    }
)


def compute_voice_score(text: str) -> float:
    """Score 0.0 (definitely typed) to 1.0 (definitely voice-dictated).

    Uses pure-Python heuristics — regex and string operations only.
    No external NLP libraries or GPU required.
    """
    if not text or not text.strip():
        return 0.0

    words = text.split()
    word_count = len(words)

    if word_count < 3:
        return 0.0

    score = 0.0

    # Repeated word sequences (e.g., "by the by the by the")
    repeated = re.findall(r"\b(\w+(?:\s+\w+){0,2})\s+\1\b", text.lower())
    if repeated:
        score += min(0.35, len(repeated) * 0.15)

    # Filler word density
    filler_count = sum(1 for w in words if w.lower().strip(".,!?") in _FILLER_WORDS)
    filler_ratio = filler_count / word_count
    if filler_ratio > 0.08:
        score += min(0.25, filler_ratio * 2)

    # Missing terminal punctuation on longer text
    stripped = text.strip()
    if stripped and stripped[-1] not in ".!?\"')" and word_count > 8:
        score += 0.15

    # Run-on: long text with very few sentence breaks
    sentence_enders = len(re.findall(r"[.!?]", text))
    if word_count > 30 and sentence_enders < 2:
        score += 0.2

    # Low punctuation density
    punct_count = len(re.findall(r"[,;:.!?]", text))
    punct_ratio = punct_count / word_count if word_count > 0 else 0
    if word_count > 15 and punct_ratio < 0.03:
        score += 0.1

    return min(1.0, score)


def session_needs_cleanup(session: ChatSession, threshold: float = 0.3) -> list[int]:
    """Return message indices of user messages likely needing cleanup.

    If no messages exceed the threshold, returns an empty list
    (meaning the LLM call can be skipped entirely).
    """
    candidates = []
    for i, msg in enumerate(session.messages):
        if msg.role == "user" and msg.content:
            score = compute_voice_score(msg.content)
            if score >= threshold:
                candidates.append(i)
    return candidates


# ---------------------------------------------------------------------------
# Batch LLM cleanup
# ---------------------------------------------------------------------------

# Gemini 3.8 Flash medium keeps this high-volume structured cleanup role bounded
# while replacing the retiring Gemini 3.5 default.
DEFAULT_MODEL = "github_copilot/gemini-3.8-flash"
DEFAULT_REASONING_EFFORT = "medium"

# Models that require the /responses API (not available on /chat/completions)
_RESPONSES_ONLY_MODELS = frozenset({"gpt-5.4-mini", "gpt-5.4"})

_SYSTEM_PROMPT = """\
You are a transcript cleanup assistant for Copilot chat sessions.
You receive a batch of user messages, some of which may be voice-dictated
(with speech-to-text errors) while others may be cleanly typed.

For each message, you are given context: the AI assistant's response and
reasoning, which show what the user actually meant.

For each message, determine:
1. Whether it appears to be voice-dictated (contains speech artifacts) or typed
2. If voice-dictated, provide a cleaned version

Rules for cleaning:
- Preserve the user's intent and technical meaning exactly
- Fix speech-to-text artifacts: repeated words, homophones, filler words,
  broken sentence structure
- Add paragraph breaks where the user shifts topic or makes a distinct point —
  voice dictation produces walls of text with no structure
- Use bullet points or numbered lists when the user is clearly enumerating items,
  but be conservative — only use lists when the structure is unambiguous
- Maintain the user's natural voice and style — don't make it overly formal
- Do NOT add information that wasn't in the original message
- Short messages like "yes", "go ahead", "looks good" are typed — don't touch them
- If typed, set cleaned to the original text unchanged"""


def _needs_responses_api(model: str) -> bool:
    """Check if this model needs the /responses API instead of /chat/completions."""
    short = model.rsplit("/", 1)[-1] if "/" in model else model
    return short in _RESPONSES_ONLY_MODELS


def _extract_responses_text(output: list) -> str:
    """Extract text content from a Responses API output list."""
    for item in output:
        if hasattr(item, "content") and item.content:
            for c in item.content:
                if hasattr(c, "text"):
                    return c.text
    return ""


def _structured_completion(
    model: str,
    messages: list[dict],
    schema: dict,
    schema_name: str,
    **kwargs,
) -> dict:
    """Structured output with auto-dispatch per model type.

    Three strategies (matching litellm-copilot skill guidance):
    1. GPT on /chat/completions: response_format with json_schema (native)
    2. GPT 5.4 family on /responses: text_format with Pydantic model (native)
    3. Claude/Gemini/other: tool_choice (forced function call matching schema)

    Returns parsed dict matching the schema.
    """
    import litellm

    model_lower = model.lower()

    if _needs_responses_api(model):
        # Build a dynamic Pydantic model from the schema for text_format
        pydantic_model = _schema_to_pydantic(schema, schema_name)
        # Responses API accepts 'instructions' for system prompt and 'input' for user content
        system_content = ""
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
        resp = asyncio.run(
            litellm.aresponses(
                model=model,
                input=user_content,
                instructions=system_content or None,
                text_format=pydantic_model,
                max_output_tokens=kwargs.pop("max_tokens", 4096),
                timeout=kwargs.pop("timeout", 60.0),
            )
        )
        text = _extract_responses_text(resp.output) if resp.output else ""
        if not text and hasattr(resp, "output_text") and resp.output_text:
            text = resp.output_text
        if not text:
            msg = f"Empty response from Responses API for model {model}"
            raise ValueError(msg)
        return json.loads(text)

    use_native = "gpt" in model_lower or "o3" in model_lower or "o4" in model_lower

    if use_native:
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "schema": schema,
                "strict": True,
            },
        }
        resp = litellm.completion(
            model=model,
            messages=messages,
            response_format=response_format,
            **kwargs,
        )
        return json.loads(resp.choices[0].message.content)
    else:
        # Tool-call mode for Claude, Gemini, etc.
        tool_schema = {
            "type": "function",
            "function": {
                "name": schema_name,
                "description": f"Return a {schema_name} object.",
                "parameters": schema,
            },
        }
        resp = litellm.completion(
            model=model,
            messages=messages,
            tools=[tool_schema],
            tool_choice={"type": "function", "function": {"name": schema_name}},
            **kwargs,
        )
        tool_call = resp.choices[0].message.tool_calls[0]
        raw = tool_call.function.arguments
        return json.loads(raw) if isinstance(raw, str) else raw


def _schema_to_pydantic(schema: dict, name: str):
    """Build a Pydantic model from a JSON schema dict for text_format.

    Handles the cleanup_results schema: { messages: [{ index, is_voice, cleaned }] }
    """
    from pydantic import create_model

    # For our specific cleanup schema, build directly
    props = schema.get("properties", {})
    if "messages" in props and props["messages"].get("type") == "array":
        item_props = props["messages"]["items"]["properties"]
        # Build the item model
        fields = {}
        for key, spec in item_props.items():
            py_type = {"integer": int, "boolean": bool, "string": str}.get(spec["type"], str)
            fields[key] = (py_type, ...)

        ItemModel = create_model(f"{name}_Item", **fields)

        return create_model(name, messages=(list[ItemModel], ...))

    # Fallback: flat object
    fields = {}
    for key, spec in props.items():
        py_type = {"integer": int, "boolean": bool, "string": str}.get(spec.get("type", "string"), str)
        fields[key] = (py_type, ...)
    return create_model(name, **fields)


def _get_assistant_context(session: ChatSession, user_msg_index: int) -> str:
    """Get the assistant's response following a user message (truncated)."""
    next_idx = user_msg_index + 1
    if next_idx < len(session.messages) and session.messages[next_idx].role == "assistant":
        content = session.messages[next_idx].content or ""
        return content[:500]
    return ""


def _get_thinking_traces(session: ChatSession, user_msg_index: int) -> str:
    """Get thinking/reasoning traces from the assistant's response."""
    next_idx = user_msg_index + 1
    if next_idx >= len(session.messages):
        return ""

    assistant_msg = session.messages[next_idx]
    traces = []
    for block in assistant_msg.content_blocks:
        if block.kind == "thinking" and block.content:
            traces.append(block.content[:200])
    return " | ".join(traces)[:500] if traces else ""


def _get_tool_summary(session: ChatSession, user_msg_index: int) -> str:
    """Summarize tool invocations from the assistant's response."""
    next_idx = user_msg_index + 1
    if next_idx >= len(session.messages):
        return ""

    assistant_msg = session.messages[next_idx]
    summaries = []
    for tool in assistant_msg.tool_invocations[:5]:
        input_preview = (tool.input or "")[:100]
        summaries.append(f"{tool.name}({input_preview})")
    return "; ".join(summaries) if summaries else ""


def build_batch_prompt(
    session: ChatSession,
    message_indices: list[int],
) -> list[dict]:
    """Build a single LLM prompt containing all target user messages with context.

    Returns the messages list for litellm.completion().
    """
    user_content_parts = []
    for i, msg_idx in enumerate(message_indices):
        msg = session.messages[msg_idx]
        assistant_response = _get_assistant_context(session, msg_idx)
        thinking = _get_thinking_traces(session, msg_idx)
        tools = _get_tool_summary(session, msg_idx)

        part = f'Message {i} (index {msg_idx}):\n  User: "{msg.content}"\n'
        if assistant_response:
            part += f'  Assistant response (first 500 chars): "{assistant_response}"\n'
        if thinking:
            part += f'  Thinking: "{thinking}"\n'
        if tools:
            part += f'  Tools called: "{tools}"\n'
        user_content_parts.append(part)

    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": "\n".join(user_content_parts)},
    ]


@dataclass
class MessageCleanupResult:
    """Result of cleaning a single message."""

    message_index: int
    is_voice: bool
    original: str
    cleaned: str


@dataclass
class SessionCleanupResult:
    """Result of cleaning an entire session."""

    session_id: str
    total_user_messages: int
    detected_voice: int
    cleaned: int
    skipped_clean: int
    failed: int
    results: list[MessageCleanupResult]


def call_cleanup_llm(
    prompt_messages: list[dict],
    model: str,
    expected_count: int,
) -> list[dict]:
    """Call the LLM with structured output and return parsed results.

    Auto-dispatches structured output per model type:
    - GPT on /chat/completions: native json_schema
    - GPT 5.4 family: /responses API with prompt-based JSON
    - Claude/Gemini/other: forced tool_choice

    Raises ImportError if litellm is not installed.
    Raises ValueError if the response doesn't match expected structure.
    """
    try:
        import litellm  # noqa: F401
    except ImportError as e:
        raise ImportError("litellm is required for transcript cleanup. Install with: pip install copilot-session-tools[llm]") from e

    cleanup_schema = {
        "type": "object",
        "properties": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "is_voice": {"type": "boolean"},
                        "cleaned": {"type": "string"},
                    },
                    "required": ["index", "is_voice", "cleaned"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["messages"],
        "additionalProperties": False,
    }

    # GPT-5.4 models don't support temperature=0; omit for responses API
    kwargs: dict = {}
    if not _needs_responses_api(model):
        kwargs["temperature"] = 0.0
    if model == DEFAULT_MODEL:
        kwargs["reasoning_effort"] = DEFAULT_REASONING_EFFORT

    parsed = _structured_completion(
        model=model,
        messages=prompt_messages,
        schema=cleanup_schema,
        schema_name="cleanup_results",
        **kwargs,
    )

    results = parsed.get("messages", [])
    if len(results) != expected_count:
        msg = f"Expected {expected_count} results, got {len(results)}"
        raise ValueError(msg)

    return results


def cleanup_session(
    db: Database,
    session_id: str,
    model: str = DEFAULT_MODEL,
    all_messages: bool = False,
    message_index: int | None = None,
    force: bool = False,
    threshold: float = 0.3,
    dry_run: bool = False,
) -> SessionCleanupResult:
    """Orchestrate transcript cleanup for a session.

    Args:
        db: Database instance
        session_id: Session to clean
        model: LiteLLM model identifier
        all_messages: Clean all user messages (skip heuristic detection)
        message_index: Clean only this specific message index
        force: Re-clean already-cleaned messages
        threshold: Voice score threshold for heuristic detection
        dry_run: Preview changes without writing to DB

    Returns:
        SessionCleanupResult with stats and per-message results
    """
    from .markdown_exporter import message_to_markdown

    session = db.get_session(session_id)
    if not session:
        msg = f"Session not found: {session_id}"
        raise ValueError(msg)

    # Reject cleanup on unenriched sessions (writes go to cst_messages which won't exist)
    _require_enriched(db, session_id)

    # Determine which messages to process
    if message_index is not None:
        if message_index >= len(session.messages):
            msg = f"Message index {message_index} out of range (session has {len(session.messages)} messages)"
            raise ValueError(msg)
        if session.messages[message_index].role != "user":
            msg = f"Message {message_index} is not a user message"
            raise ValueError(msg)
        target_indices = [message_index]
    elif all_messages:
        target_indices = [i for i, m in enumerate(session.messages) if m.role == "user" and m.content]
    else:
        target_indices = session_needs_cleanup(session, threshold)

    # Filter out already-cleaned messages unless force is set
    if not force:
        target_indices = [i for i in target_indices if session.messages[i].original_content is None]

    total_user = sum(1 for m in session.messages if m.role == "user")

    if not target_indices:
        return SessionCleanupResult(
            session_id=session_id,
            total_user_messages=total_user,
            detected_voice=0,
            cleaned=0,
            skipped_clean=0,
            failed=0,
            results=[],
        )

    # Build batch prompt and call LLM, chunking if needed to stay within token limits
    CHUNK_SIZE = 10  # Max messages per LLM call
    all_llm_results = []
    failed_count = 0

    for chunk_start in range(0, len(target_indices), CHUNK_SIZE):
        chunk_indices = target_indices[chunk_start : chunk_start + CHUNK_SIZE]
        prompt = build_batch_prompt(session, chunk_indices)
        try:
            chunk_results = call_cleanup_llm(prompt, model, len(chunk_indices))
            # Validate returned indices match requested chunk
            returned_indices = {r["index"] for r in chunk_results}
            expected_indices = set(chunk_indices)
            if returned_indices != expected_indices:
                # Model returned wrong indices — remap by position
                for i, r in enumerate(chunk_results):
                    r["index"] = chunk_indices[i]
            all_llm_results.extend(chunk_results)
        except Exception as e:
            failed_count += len(chunk_indices)
            all_llm_results.extend({"index": i, "is_voice": True, "cleaned": None, "_error": str(e)} for i in chunk_indices)

    llm_results = all_llm_results

    # Process results
    cleaned_count = 0
    skipped_count = 0
    message_results = []

    for llm_result in llm_results:
        msg_idx = llm_result["index"]
        is_voice = llm_result["is_voice"]
        cleaned_text = llm_result["cleaned"]
        original_msg = session.messages[msg_idx]

        # Skip error results from failed chunks
        if "_error" in llm_result:
            message_results.append(
                MessageCleanupResult(
                    message_index=msg_idx,
                    is_voice=is_voice,
                    original=original_msg.content,
                    cleaned=f"ERROR: {llm_result['_error']}",
                )
            )
            continue

        result = MessageCleanupResult(
            message_index=msg_idx,
            is_voice=is_voice,
            original=original_msg.content,
            cleaned=cleaned_text,
        )
        message_results.append(result)

        if not is_voice or cleaned_text == original_msg.content:
            skipped_count += 1
            continue

        if not dry_run:
            # Regenerate cached markdown with cleaned content
            cleaned_msg = ChatMessage(
                role=original_msg.role,
                content=cleaned_text,
                timestamp=original_msg.timestamp,
                tool_invocations=original_msg.tool_invocations,
                file_changes=original_msg.file_changes,
                command_runs=original_msg.command_runs,
                content_blocks=original_msg.content_blocks,
                agent_id=original_msg.agent_id,
                agent_display_name=original_msg.agent_display_name,
                agent_nesting_level=original_msg.agent_nesting_level,
            )
            new_cached_md = message_to_markdown(
                cleaned_msg,
                message_number=msg_idx + 1,
                include_diffs=True,
                include_tool_inputs=True,
            )

            _update_message_content(
                db,
                session_id=session_id,
                message_index=msg_idx,
                new_content=cleaned_text,
                original_content=original_msg.original_content or original_msg.content,
                cleanup_model=model,
                cached_markdown=new_cached_md,
            )

        cleaned_count += 1

    detected_voice = sum(1 for r in message_results if r.is_voice)

    return SessionCleanupResult(
        session_id=session_id,
        total_user_messages=total_user,
        detected_voice=detected_voice,
        cleaned=cleaned_count,
        skipped_clean=skipped_count,
        failed=failed_count,
        results=message_results,
    )


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------


def _require_enriched(db: Database, session_id: str) -> None:
    """Raise ValueError if the session is not in the enriched cst_* tables."""
    if not db.has_cst_tables():
        msg = "No enriched sessions available. Run 'scan' first."
        raise ValueError(msg)
    with db._get_connection() as conn:
        row = conn.execute("SELECT 1 FROM cst_sessions WHERE session_id = ?", (session_id,)).fetchone()
        if not row:
            msg = f"Session {session_id} is not enriched. Run 'scan' or 'enrich' first."
            raise ValueError(msg)


def _update_message_content(
    db: Database,
    session_id: str,
    message_index: int,
    new_content: str,
    original_content: str,
    cleanup_model: str,
    cached_markdown: str,
) -> None:
    """Update a message's content in the database, preserving the original."""
    with db._get_connection() as conn:
        conn.execute(
            """
            UPDATE cst_messages
            SET content = ?,
                original_content = ?,
                cleanup_model = ?,
                cached_markdown = ?
            WHERE session_id = ? AND message_index = ?
            """,
            (new_content, original_content, cleanup_model, cached_markdown, session_id, message_index),
        )


def revert_message(db: Database, session_id: str, message_index: int) -> bool:
    """Revert a single message to its original content.

    Returns True if the message was reverted, False if it had no original content.
    """
    from .markdown_exporter import message_to_markdown

    session = db.get_session(session_id)
    if not session:
        msg = f"Session not found: {session_id}"
        raise ValueError(msg)

    if message_index >= len(session.messages):
        msg = f"Message index {message_index} out of range"
        raise ValueError(msg)

    message = session.messages[message_index]
    if message.original_content is None:
        return False

    # Regenerate cached markdown with original content
    reverted_msg = ChatMessage(
        role=message.role,
        content=message.original_content,
        timestamp=message.timestamp,
        tool_invocations=message.tool_invocations,
        file_changes=message.file_changes,
        command_runs=message.command_runs,
        content_blocks=message.content_blocks,
        agent_id=message.agent_id,
        agent_display_name=message.agent_display_name,
        agent_nesting_level=message.agent_nesting_level,
    )
    new_cached_md = message_to_markdown(
        reverted_msg,
        message_number=message_index + 1,
        include_diffs=True,
        include_tool_inputs=True,
    )

    with db._get_connection() as conn:
        conn.execute(
            """
            UPDATE cst_messages
            SET content = original_content,
                original_content = NULL,
                cleanup_model = NULL,
                cached_markdown = ?
            WHERE session_id = ? AND message_index = ?
            """,
            (new_cached_md, session_id, message_index),
        )

    return True


def revert_session(db: Database, session_id: str) -> int:
    """Revert all cleaned messages in a session to their originals.

    Returns the number of messages reverted.
    """
    session = db.get_session(session_id)
    if not session:
        msg = f"Session not found: {session_id}"
        raise ValueError(msg)

    reverted = 0
    for i, msg in enumerate(session.messages):
        if msg.original_content is not None and revert_message(db, session_id, i):
            reverted += 1
    return reverted
