"""Tests for transcript cleanup functionality."""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from copilot_session_tools.scanner.models import ChatMessage, ChatSession, ContentBlock, ToolInvocation


def _make_test_session(**kwargs: Any) -> ChatSession:
    """Helper to create ChatSession with sensible defaults for tests."""
    defaults: dict[str, Any] = {
        "session_id": "test-session",
        "workspace_name": "test-workspace",
        "workspace_path": "/test/workspace",
        "messages": [],
    }
    defaults.update(kwargs)
    return ChatSession(**defaults)


from copilot_session_tools.transcript_cleanup import (
    build_batch_prompt,
    call_cleanup_llm,
    cleanup_session,
    compute_voice_score,
    revert_message,
    revert_session,
    session_needs_cleanup,
)

# ---------------------------------------------------------------------------
# compute_voice_score tests
# ---------------------------------------------------------------------------


class TestComputeVoiceScore:
    """Test the heuristic voice detection scoring."""

    def test_empty_string(self):
        assert compute_voice_score("") == 0.0

    def test_short_typed_message(self):
        assert compute_voice_score("yes") == 0.0
        assert compute_voice_score("go ahead") == 0.0

    def test_clean_typed_message(self):
        score = compute_voice_score("Please add a button to the settings page that allows users to export their data.")
        assert score < 0.3

    def test_garbled_repeated_words(self):
        score = compute_voice_score("So I want to I want to add a button the the the settings page")
        assert score >= 0.3

    def test_garbled_filler_words(self):
        score = compute_voice_score("so basically like I want to um actually add like a button to basically the settings page")
        assert score >= 0.3

    def test_garbled_missing_punctuation_runon(self):
        score = compute_voice_score(
            "I want the web UI to be a part of the work actually I imagine you're going to need "
            "a button to potentially scan an entire chat you'll need a button for individual "
            "messages you'll need a button to revert a message you'll need a toggle for back and forth"
        )
        assert score >= 0.3

    def test_code_snippet_not_detected(self):
        score = compute_voice_score("def hello():\n    print('Hello, world!')\n    return True")
        assert score < 0.3

    def test_mixed_signals_real_voice_sample(self):
        score = compute_voice_score(
            "For the pre shootout Rely on a heavier LLM model Instead of doing a manual labeling "
            "I don't have time for that really and maybe even get a consensus among the try reviewers"
        )
        assert score >= 0.2  # Should show some signal even if borderline


# ---------------------------------------------------------------------------
# session_needs_cleanup tests
# ---------------------------------------------------------------------------


class TestSessionNeedsCleanup:
    def _make_session(self, messages: list[tuple[str, str]]) -> ChatSession:
        return _make_test_session(
            messages=[ChatMessage(role=role, content=content) for role, content in messages],
        )

    def test_no_user_messages(self):
        session = self._make_session([("assistant", "Hello!")])
        assert session_needs_cleanup(session) == []

    def test_clean_session(self):
        session = self._make_session(
            [
                ("user", "Add a button to the settings page."),
                ("assistant", "Done!"),
            ]
        )
        assert session_needs_cleanup(session) == []

    def test_garbled_session(self):
        session = self._make_session(
            [
                ("user", "so basically like I want to um actually add like a button to basically the settings page you know"),
                ("assistant", "I'll add the button."),
            ]
        )
        result = session_needs_cleanup(session, threshold=0.3)
        assert 0 in result

    def test_custom_threshold(self):
        session = self._make_session(
            [
                ("user", "add a button"),
                ("assistant", "Done!"),
            ]
        )
        # Very low threshold should catch more, but 2-word messages still score 0 (< 3 words)
        result = session_needs_cleanup(session, threshold=0.0)
        assert 0 in result  # "add a button" scores > 0.0 with threshold=0.0


# ---------------------------------------------------------------------------
# build_batch_prompt tests
# ---------------------------------------------------------------------------


class TestBuildBatchPrompt:
    def _make_session_with_context(self) -> ChatSession:
        return _make_test_session(
            messages=[
                ChatMessage(role="user", content="garbled text here"),
                ChatMessage(
                    role="assistant",
                    content="I understand you want to add a button.",
                    content_blocks=[
                        ContentBlock(kind="thinking", content="The user wants a button on the settings page."),
                    ],
                    tool_invocations=[
                        ToolInvocation(name="edit", input='{"path": "settings.py"}'),
                    ],
                ),
                ChatMessage(role="user", content="yes go ahead"),
                ChatMessage(role="assistant", content="Done!"),
            ],
        )

    def test_includes_system_prompt(self):
        session = self._make_session_with_context()
        prompt = build_batch_prompt(session, [0])
        assert prompt[0]["role"] == "system"
        assert "transcript cleanup" in prompt[0]["content"].lower()

    def test_includes_user_message(self):
        session = self._make_session_with_context()
        prompt = build_batch_prompt(session, [0])
        assert "garbled text here" in prompt[1]["content"]

    def test_includes_assistant_context(self):
        session = self._make_session_with_context()
        prompt = build_batch_prompt(session, [0])
        content = prompt[1]["content"]
        assert "add a button" in content

    def test_includes_thinking_traces(self):
        session = self._make_session_with_context()
        prompt = build_batch_prompt(session, [0])
        content = prompt[1]["content"]
        assert "settings page" in content

    def test_includes_tool_summary(self):
        session = self._make_session_with_context()
        prompt = build_batch_prompt(session, [0])
        content = prompt[1]["content"]
        assert "edit" in content

    def test_multiple_messages_in_batch(self):
        session = self._make_session_with_context()
        prompt = build_batch_prompt(session, [0, 2])
        content = prompt[1]["content"]
        assert "Message 0" in content
        assert "Message 1" in content
        assert "garbled text here" in content
        assert "yes go ahead" in content


# ---------------------------------------------------------------------------
# call_cleanup_llm tests (mocked)
# ---------------------------------------------------------------------------


class TestCallCleanupLLM:
    def test_basic_call(self):
        mock_result = {"messages": [{"index": 0, "is_voice": True, "cleaned": "cleaned text"}]}

        with patch(
            "copilot_session_tools.transcript_cleanup._structured_completion",
            return_value=mock_result,
        ) as structured_completion:
            results = call_cleanup_llm(
                [{"role": "user", "content": "test"}],
                model="github_copilot/gemini-3.8-flash",
                expected_count=1,
            )

        assert len(results) == 1
        assert results[0]["is_voice"] is True
        assert results[0]["cleaned"] == "cleaned text"
        assert structured_completion.call_args.kwargs["reasoning_effort"] == "medium"

    def test_wrong_count_raises(self):
        """If the LLM returns wrong number of results, raise ValueError."""
        mock_result = {"messages": [{"index": 0, "is_voice": True, "cleaned": "text"}]}

        with (
            patch("copilot_session_tools.transcript_cleanup._structured_completion", return_value=mock_result),
            pytest.raises(ValueError, match="Expected 2 results"),
        ):
            call_cleanup_llm(
                [{"role": "user", "content": "test"}],
                model="github_copilot/gemini-3.8-flash",
                expected_count=2,
            )


# ---------------------------------------------------------------------------
# cleanup_session integration tests (mocked LLM)
# ---------------------------------------------------------------------------


class TestCleanupSession:
    @pytest.fixture
    def db_with_session(self, tmp_path):
        """Create a temporary database with a test session."""
        from copilot_session_tools.database import Database

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        session = _make_test_session(
            session_id="test-session-123",
            custom_title="Test Session",
            messages=[
                ChatMessage(
                    role="user",
                    content="so basically like I want to um actually add like a button",
                ),
                ChatMessage(
                    role="assistant",
                    content="I'll add a button to the settings page.",
                    content_blocks=[
                        ContentBlock(kind="thinking", content="User wants a button"),
                    ],
                ),
                ChatMessage(role="user", content="yes go ahead"),
                ChatMessage(role="assistant", content="Done!"),
            ],
        )

        db.add_session(session)
        return db

    def _mock_llm_result(self, results: list[dict]) -> dict:
        return {"messages": results}

    def test_dry_run_no_changes(self, db_with_session):
        mock_result = self._mock_llm_result(
            [
                {"index": 0, "is_voice": True, "cleaned": "I want to add a button"},
            ]
        )

        with patch("copilot_session_tools.transcript_cleanup._structured_completion", return_value=mock_result):
            result = cleanup_session(
                db=db_with_session,
                session_id="test-session-123",
                all_messages=True,
                dry_run=True,
            )

        # Dry run should detect but not change
        assert result.detected_voice >= 1

        # Verify original content unchanged
        session = db_with_session.get_session("test-session-123")
        assert session.messages[0].original_content is None

    def test_cleanup_updates_content(self, db_with_session):
        mock_result = self._mock_llm_result(
            [
                {"index": 0, "is_voice": True, "cleaned": "I want to add a button"},
                {"index": 2, "is_voice": False, "cleaned": "yes go ahead"},
            ]
        )

        with patch("copilot_session_tools.transcript_cleanup._structured_completion", return_value=mock_result):
            result = cleanup_session(
                db=db_with_session,
                session_id="test-session-123",
                all_messages=True,
            )

        assert result.cleaned == 1
        assert result.skipped_clean == 1

        session = db_with_session.get_session("test-session-123")
        # Message 0 should be cleaned
        assert session.messages[0].content == "I want to add a button"
        assert "basically" in session.messages[0].original_content
        assert session.messages[0].cleanup_model == "github_copilot/gemini-3.8-flash"

        # Message 2 should be unchanged (not voice)
        assert session.messages[2].content == "yes go ahead"
        assert session.messages[2].original_content is None

    def test_revert_message(self, db_with_session):
        mock_result = self._mock_llm_result(
            [
                {"index": 0, "is_voice": True, "cleaned": "I want to add a button"},
            ]
        )

        with patch("copilot_session_tools.transcript_cleanup._structured_completion", return_value=mock_result):
            cleanup_session(
                db=db_with_session,
                session_id="test-session-123",
                message_index=0,
            )

        # Verify cleaned
        session = db_with_session.get_session("test-session-123")
        assert session.messages[0].content == "I want to add a button"

        # Revert
        reverted = revert_message(db_with_session, "test-session-123", 0)
        assert reverted is True

        # Verify reverted
        session = db_with_session.get_session("test-session-123")
        assert "basically" in session.messages[0].content
        assert session.messages[0].original_content is None
        assert session.messages[0].cleanup_model is None

    def test_revert_session(self, db_with_session):
        mock_result = self._mock_llm_result(
            [
                {"index": 0, "is_voice": True, "cleaned": "I want to add a button"},
                {"index": 2, "is_voice": True, "cleaned": "Yes, go ahead"},
            ]
        )

        with patch("copilot_session_tools.transcript_cleanup._structured_completion", return_value=mock_result):
            cleanup_session(
                db=db_with_session,
                session_id="test-session-123",
                all_messages=True,
            )

        count = revert_session(db_with_session, "test-session-123")
        assert count == 2

        session = db_with_session.get_session("test-session-123")
        assert session.messages[0].original_content is None
        assert session.messages[2].original_content is None

    def test_force_reclean(self, db_with_session):
        mock_result = self._mock_llm_result(
            [
                {"index": 0, "is_voice": True, "cleaned": "First cleanup"},
            ]
        )

        with patch("copilot_session_tools.transcript_cleanup._structured_completion", return_value=mock_result):
            cleanup_session(
                db=db_with_session,
                session_id="test-session-123",
                message_index=0,
            )

        # Without force, should skip already-cleaned
        with patch("copilot_session_tools.transcript_cleanup._structured_completion", return_value=self._mock_llm_result([])):
            result = cleanup_session(
                db=db_with_session,
                session_id="test-session-123",
                message_index=0,
                force=False,
            )
        assert result.cleaned == 0

        # With force, should re-clean
        mock_result2 = self._mock_llm_result(
            [
                {"index": 0, "is_voice": True, "cleaned": "Second cleanup"},
            ]
        )
        with patch("copilot_session_tools.transcript_cleanup._structured_completion", return_value=mock_result2):
            result = cleanup_session(
                db=db_with_session,
                session_id="test-session-123",
                message_index=0,
                force=True,
            )
        assert result.cleaned == 1

        session = db_with_session.get_session("test-session-123")
        assert session.messages[0].content == "Second cleanup"

    def test_fts_sync_after_cleanup(self, db_with_session):
        """Verify FTS5 index is updated after cleanup via UPDATE trigger."""
        mock_result = self._mock_llm_result(
            [
                {"index": 0, "is_voice": True, "cleaned": "I want to add a button"},
            ]
        )

        with patch("copilot_session_tools.transcript_cleanup._structured_completion", return_value=mock_result):
            cleanup_session(
                db=db_with_session,
                session_id="test-session-123",
                message_index=0,
            )

        # Search for the cleaned text in FTS
        with db_with_session._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT content FROM cst_messages_fts WHERE cst_messages_fts MATCH ?",
                ("button",),
            )
            results = cursor.fetchall()
            assert any("I want to add a button" in r[0] for r in results)

    def test_session_not_found(self, db_with_session):
        with pytest.raises(ValueError, match="Session not found"):
            cleanup_session(db_with_session, "nonexistent-session")

    def test_message_index_out_of_range(self, db_with_session):
        with pytest.raises(ValueError, match="out of range"):
            cleanup_session(db_with_session, "test-session-123", message_index=999)

    def test_message_not_user(self, db_with_session):
        with pytest.raises(ValueError, match="not a user message"):
            cleanup_session(db_with_session, "test-session-123", message_index=1)


# ---------------------------------------------------------------------------
# Schema migration test
# ---------------------------------------------------------------------------


class TestSchemaMigration:
    def test_v5_columns_exist(self, tmp_path):
        """Verify schema v5 adds original_content and cleanup_model columns."""
        from copilot_session_tools.database import Database

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(cst_messages)")
            columns = {row[1] for row in cursor.fetchall()}

        assert "original_content" in columns
        assert "cleanup_model" in columns

    def test_fts_update_trigger_exists(self, tmp_path):
        """Verify the FTS UPDATE trigger exists for content sync."""
        from copilot_session_tools.database import Database

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))

        with db._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='trigger' AND name='cst_messages_au'")
            result = cursor.fetchone()

        assert result is not None, "FTS UPDATE trigger cst_messages_au should exist"


# ---------------------------------------------------------------------------
# Gherkin-style user flow coverage
# ---------------------------------------------------------------------------


class TestForceRecleanPreservesOriginal:
    """Given a message cleaned once, when force re-cleaned, original voice text is preserved."""

    @pytest.fixture
    def db_with_session(self, tmp_path):
        from copilot_session_tools.database import Database

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))
        session = _make_test_session(
            session_id="force-test",
            messages=[
                ChatMessage(role="user", content="the the garbled original text"),
                ChatMessage(role="assistant", content="I understand"),
            ],
        )
        db.add_session(session)
        return db

    def test_second_cleanup_preserves_true_original(self, db_with_session):
        # First cleanup
        mock1 = {"messages": [{"index": 0, "is_voice": True, "cleaned": "First cleanup"}]}
        with patch("copilot_session_tools.transcript_cleanup._structured_completion", return_value=mock1):
            cleanup_session(db_with_session, "force-test", message_index=0)

        # Second cleanup with force
        mock2 = {"messages": [{"index": 0, "is_voice": True, "cleaned": "Second cleanup"}]}
        with patch("copilot_session_tools.transcript_cleanup._structured_completion", return_value=mock2):
            cleanup_session(db_with_session, "force-test", message_index=0, force=True)

        session = db_with_session.get_session("force-test")
        assert session.messages[0].content == "Second cleanup"
        # The TRUE original (garbled) must be preserved, not "First cleanup"
        assert session.messages[0].original_content == "the the garbled original text"

    def test_revert_after_double_cleanup_restores_true_original(self, db_with_session):
        # Clean twice
        for text in ["First", "Second"]:
            mock_result = {"messages": [{"index": 0, "is_voice": True, "cleaned": text}]}
            with patch("copilot_session_tools.transcript_cleanup._structured_completion", return_value=mock_result):
                cleanup_session(db_with_session, "force-test", message_index=0, force=True)

        # Revert should restore the true original
        revert_message(db_with_session, "force-test", 0)
        session = db_with_session.get_session("force-test")
        assert session.messages[0].content == "the the garbled original text"
        assert session.messages[0].original_content is None


class TestPartialChunkFailure:
    """Given a large session, when one LLM chunk fails, partial results are reported correctly."""

    @pytest.fixture
    def db_with_large_session(self, tmp_path):
        from copilot_session_tools.database import Database

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))
        # Create 15 user messages (will be split into 2 chunks of 10+5)
        messages = []
        for i in range(15):
            messages.append(ChatMessage(role="user", content=f"so um basically message {i}"))
            messages.append(ChatMessage(role="assistant", content=f"Response {i}"))
        session = _make_test_session(session_id="chunk-test", messages=messages)
        db.add_session(session)
        return db

    def test_partial_failure_reports_failed_count(self, db_with_large_session):
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First chunk succeeds
                return {"messages": [{"index": i * 2, "is_voice": True, "cleaned": f"Cleaned {i}"} for i in range(10)]}
            # Second chunk fails
            raise ConnectionError("Network error")

        with patch("copilot_session_tools.transcript_cleanup._structured_completion", side_effect=side_effect):
            result = cleanup_session(db_with_large_session, "chunk-test", all_messages=True)

        assert result.cleaned > 0
        assert result.failed > 0
        assert result.cleaned + result.failed + result.skipped_clean <= 15


class TestCleanupOnUnenrichedSession:
    """Given a session that hasn't been enriched, cleanup should raise an error."""

    def test_cleanup_rejects_unenriched(self, tmp_path):
        from copilot_session_tools.database import Database

        db_path = tmp_path / "test.db"
        db = Database(str(db_path))
        # Don't add any session — empty cst_sessions table
        with pytest.raises(ValueError, match=r"not found|not enriched"):
            cleanup_session(db, "nonexistent-session")
