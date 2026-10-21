"""Benchmark script for transcript cleanup: pre-filter strategies and LLM models.

Two dimensions:
  A) Pre-filter accuracy: compare hand-rolled heuristics vs textstat-enhanced vs no-filter
     using tri-review model consensus as ground truth labels.
  B) LLM model comparison: compare cheap/fast models for runtime cleanup quality.

Usage:
    uv run python scripts/benchmark_cleanup.py
    uv run python scripts/benchmark_cleanup.py --part=prefilter   # only pre-filter benchmark
    uv run python scripts/benchmark_cleanup.py --part=models      # only model comparison
    uv run python scripts/benchmark_cleanup.py --db=path/to/db    # use specific database
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import litellm

# ---------------------------------------------------------------------------
# Corpus: sample messages for model comparison (Part B)
# Each has garbled input, context, and expected cleaned output
# ---------------------------------------------------------------------------

GARBLED_SAMPLES = [
    {
        "garbled": (
            "So it happens not infrequently now that I'm using voice dictation that I wish that "
            "the index that was actually being used by copilot was and the and the view itself "
            "actually did some sort of attempts at not text to speech or whatever So what I would "
            "like is I mean we have text to speech but I would like it cleaned up so that it matches "
            "so that the actual transcript matches what I have in mind"
        ),
        "context": "The assistant understood this as a request to add LLM-based transcript cleanup for voice-dictated messages in the copilot-session-tools project.",
        "expected": (
            "It happens not infrequently now that I'm using voice dictation that I wish that "
            "the index being used by Copilot and the view itself actually did some sort of "
            "text cleanup. What I would like is for the transcript to be cleaned up so that "
            "it matches what I actually have in mind."
        ),
    },
    {
        "garbled": (
            "I want the web UI to be a part of the work actually I imagine you're going to nee"
            "Need a button to potentially scan an entire chat you'll need a button to for individual "
            "messages you'll need a button to revert a message you'll need a toggle for back and forth "
            "and a given message for the raw for the system message for the raw view versus the cleaned up view"
        ),
        "context": "The assistant understood this as a request to add cleanup/revert buttons and a raw/cleaned toggle per message in the web UI.",
        "expected": (
            "I want the web UI to be a part of the work. You're going to need a button to "
            "scan an entire chat, a button for individual messages, a button to revert a message, "
            "and a toggle for switching between the raw view versus the cleaned up view on a given message."
        ),
    },
    {
        "garbled": (
            "For the pre shootout Rely on a heavier LLM model Instead of doing a manual labeling "
            "I don't have time for that really and maybe even get a consensus among the try reviewers"
        ),
        "context": "The assistant understood this as: use heavy LLM models (tri-review models) instead of manual labeling for ground truth in the pre-filter benchmark.",
        "expected": (
            "For the pre-filter shootout, rely on a heavier LLM model instead of doing manual labeling. "
            "I don't have time for that. Maybe even get a consensus among the tri-reviewers."
        ),
    },
    {
        "garbled": "yes go ahead looks good",
        "context": "The assistant proceeded with the proposed plan.",
        "expected": "yes go ahead looks good",  # Should remain unchanged - this is typed
    },
    {
        "garbled": ("And I don't think we need FTS original search we can get rid of it"),
        "context": "The assistant removed the FTS original search feature from the plan.",
        "expected": "I don't think we need FTS original search. We can get rid of it.",
    },
]


# ---------------------------------------------------------------------------
# Part A: Pre-filter heuristics
# ---------------------------------------------------------------------------


def compute_voice_score(text: str) -> float:
    """Score 0.0 (typed) to 1.0 (voice-dictated) using pure-Python heuristics."""
    if not text or not text.strip():
        return 0.0

    score = 0.0
    words = text.split()
    word_count = len(words)

    if word_count < 3:
        return 0.0  # Very short messages are almost certainly typed

    # Repeated word sequences (e.g., "by the by the by the")
    repeated = re.findall(r"\b(\w+(?:\s+\w+){0,2})\s+\1\b", text.lower())
    if repeated:
        score += min(0.35, len(repeated) * 0.15)

    # Filler word density
    fillers = {"like", "um", "uh", "basically", "actually", "so", "you know", "I mean"}
    filler_count = sum(1 for w in words if w.lower().strip(".,!?") in fillers)
    filler_ratio = filler_count / word_count
    if filler_ratio > 0.08:
        score += min(0.25, filler_ratio * 2)

    # Missing terminal punctuation
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


def compute_voice_score_with_textstat(text: str) -> float:
    """Enhanced voice score using textstat readability metrics."""
    try:
        import textstat
    except ImportError:
        print("WARNING: textstat not installed, falling back to basic heuristics")
        return compute_voice_score(text)

    base_score = compute_voice_score(text)

    if not text.strip() or len(text.split()) < 5:
        return base_score

    # Abnormal readability = likely garbled
    flesch = textstat.flesch_reading_ease(text)
    # Very high Flesch (> 90) with long text = simple/repetitive = likely voice
    if flesch > 90 and len(text.split()) > 20:
        base_score += 0.1
    # Very low Flesch (< 10) = hard to read = possibly garbled
    if flesch < 10:
        base_score += 0.1

    # Gunning Fog: garbled text often has weird fog index
    fog = textstat.gunning_fog(text)
    if fog > 20:  # Extremely complex = likely broken sentence structure
        base_score += 0.1

    return min(1.0, base_score)


# ---------------------------------------------------------------------------
# Part A: Ground truth labeling via tri-review consensus
# ---------------------------------------------------------------------------

TRI_REVIEW_MODELS = [
    ("github_copilot/claude-opus-5.5", "high"),
    ("github_copilot/gpt-6-astra", "high"),
    ("github_copilot/gemini-3.8-flash", "medium"),
]

LABELING_PROMPT = """You are a text analysis expert. For each user message below, determine whether it was:
- **voice-dictated** (speech-to-text with artifacts like repeated words, filler words, missing punctuation, broken grammar)
- **typed** (cleanly written, proper punctuation, no speech artifacts)

Respond with a JSON array. Each element has:
- "index": the message index (0-based)
- "label": "voice" or "typed"
- "confidence": 0.0 to 1.0

Messages:
"""

LABELING_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "labeling_result",
        "schema": {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "label": {"type": "string", "enum": ["voice", "typed"]},
                            "confidence": {"type": "number"},
                        },
                        "required": ["index", "label", "confidence"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["results"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


def label_messages_with_model(
    messages: list[str],
    model: str,
    reasoning_effort: str,
) -> list[dict]:
    """Ask a single model to label messages as voice/typed."""
    prompt = LABELING_PROMPT
    for i, msg in enumerate(messages):
        prompt += f"\n[{i}]: {msg!r}\n"

    prompt += '\nRespond with JSON only: {"results": [{"index": N, "label": "voice"|"typed", "confidence": 0.0-1.0}, ...]}'

    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        reasoning_effort=reasoning_effort,
    )
    content = response.choices[0].message.content
    # Strip markdown code fences if present
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1] if "\n" in content else content[3:]
        content = content.removesuffix("```")
        content = content.strip()
    parsed = json.loads(content)
    return parsed["results"]


def get_consensus_labels(messages: list[str]) -> list[dict]:
    """Get voice/typed labels via majority vote across tri-review models."""
    all_labels: dict[int, list[str]] = {i: [] for i in range(len(messages))}
    model_results = {}

    for model, reasoning_effort in TRI_REVIEW_MODELS:
        print(f"  Labeling with {model} ({reasoning_effort})...")
        try:
            results = label_messages_with_model(messages, model, reasoning_effort)
            model_results[model] = results
            for r in results:
                idx = r["index"]
                if idx in all_labels:
                    all_labels[idx].append(r["label"])
        except Exception as e:
            print(f"  ERROR with {model}: {e}")

    # Majority vote
    consensus = []
    for i in range(len(messages)):
        votes = all_labels[i]
        if not votes:
            consensus.append({"index": i, "label": "unknown", "agreement": 0})
            continue
        voice_count = sum(1 for v in votes if v == "voice")
        typed_count = sum(1 for v in votes if v == "typed")
        label = "voice" if voice_count > typed_count else "typed"
        agreement = max(voice_count, typed_count) / len(votes)
        consensus.append({"index": i, "label": label, "agreement": agreement})

    return consensus


# ---------------------------------------------------------------------------
# Part B: LLM model comparison for cleanup quality
# ---------------------------------------------------------------------------

CLEANUP_MODELS = [
    "github_copilot/gemini-3.8-flash",
    "github_copilot/gpt-6-luna",
    "github_copilot/gpt-6-sol",
    "github_copilot/claude-haiku-4.5",
    "github_copilot/claude-haiku-5.5",
    "github_copilot/gemini-4-argon",
]

CLEANUP_REASONING_EFFORTS = {
    "github_copilot/gemini-3.8-flash": "medium",
    "github_copilot/gpt-6-luna": "high",
    "github_copilot/gpt-6-sol": "medium",
    "github_copilot/claude-haiku-5.5": "high",
    "github_copilot/gemini-4-argon": "high",
}

CLEANUP_SYSTEM_PROMPT = """You are a transcript cleanup assistant. You receive voice-dictated text \
that may contain speech-to-text errors. Using the provided context, rewrite the message to \
accurately reflect the user's intended meaning.

Rules:
- Preserve the user's intent and technical meaning exactly
- Fix speech-to-text artifacts: repeated words, homophones, filler words, broken sentence structure
- Add paragraph breaks where the user shifts topic or makes a distinct point
- Use bullet points or numbered lists when the user is clearly enumerating items,
  but be conservative — only use lists when the structure is unambiguous
- Maintain the user's natural voice and style — don't make it overly formal
- Do NOT add information that wasn't in the original message
- If the text appears already clean, return it unchanged"""

CLEANUP_RESPONSE_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "cleanup_result",
        "schema": {
            "type": "object",
            "properties": {
                "is_voice": {"type": "boolean"},
                "cleaned": {"type": "string"},
            },
            "required": ["is_voice", "cleaned"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}


@dataclass
class CleanupBenchmarkResult:
    model: str
    sample_index: int
    cleaned: str
    is_voice: bool
    latency_s: float
    prompt_tokens: int
    completion_tokens: int
    error: str | None = None


@dataclass
class PrefilterMetrics:
    strategy: str
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    messages_sent_to_llm: int = 0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def cleanup_single(garbled: str, context: str, model: str) -> CleanupBenchmarkResult:
    """Run a single cleanup through a model and measure results."""
    user_content = (
        f'User message to clean:\n{garbled}\n\nContext -- what the AI understood:\n{context}\n\nRespond with JSON only: {{"is_voice": true/false, "cleaned": "cleaned text"}}'
    )

    start = time.perf_counter()
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": CLEANUP_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
        }
        reasoning_effort = CLEANUP_REASONING_EFFORTS.get(model)
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        response = litellm.completion(
            **kwargs,
        )
        elapsed = time.perf_counter() - start
        content = response.choices[0].message.content
        # Strip markdown code fences if present
        content = content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            content = content.removesuffix("```")
            content = content.strip()
        parsed = json.loads(content)
        usage = response.usage
        return CleanupBenchmarkResult(
            model=model,
            sample_index=-1,
            cleaned=parsed["cleaned"],
            is_voice=parsed["is_voice"],
            latency_s=elapsed,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
        )
    except Exception as e:
        elapsed = time.perf_counter() - start
        return CleanupBenchmarkResult(
            model=model,
            sample_index=-1,
            cleaned="",
            is_voice=False,
            latency_s=elapsed,
            prompt_tokens=0,
            completion_tokens=0,
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Part A: Run pre-filter benchmark
# ---------------------------------------------------------------------------


def load_messages_from_db(db_path: str | None, count: int = 20) -> list[str]:
    """Load real user messages from the database for labeling."""
    if db_path is None:
        default = Path.home() / ".copilot-session-tools" / "copilot_chats.db"
        if default.exists():
            db_path = str(default)
        else:
            print("No database found. Using built-in garbled samples for pre-filter benchmark.")
            return [s["garbled"] for s in GARBLED_SAMPLES]

    # Import here to avoid circular dependency
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    from copilot_session_tools.database import Database

    db = Database(db_path)
    messages = []

    try:
        with db._get_connection() as conn:
            cursor = conn.cursor()
            # Get a diverse sample: mix of short and long user messages from different sessions
            cursor.execute(
                """
                SELECT content FROM cst_messages
                WHERE role = 'user'
                  AND content IS NOT NULL
                  AND length(content) > 10
                  AND agent_nesting_level = 0
                ORDER BY RANDOM()
                LIMIT ?
                """,
                (count,),
            )
            messages = [row[0] for row in cursor.fetchall()]
    except Exception as e:
        print(f"Error loading from DB: {e}")

    if not messages:
        print("No messages found in DB. Using built-in samples.")
        return [s["garbled"] for s in GARBLED_SAMPLES]

    print(f"Loaded {len(messages)} real user messages from database.")
    return messages


def run_prefilter_benchmark(messages: list[str]) -> None:
    """Run Part A: compare pre-filter strategies against consensus labels."""
    print("\n" + "=" * 70)
    print("PART A: Pre-filter Strategy Comparison")
    print("=" * 70)

    # Step 1: Get ground truth via tri-review consensus
    print("\nStep 1: Getting ground truth labels via tri-review consensus...")
    consensus = get_consensus_labels(messages)

    print("\nGround truth labels:")
    for c in consensus:
        snippet = messages[c["index"]][:60].replace("\n", " ")
        print(f"  [{c['index']}] {c['label']:6s} (agreement: {c['agreement']:.0%}) | {snippet}...")

    # Step 2: Evaluate pre-filter strategies
    threshold = 0.3
    strategies = {
        "hand-rolled": compute_voice_score,
        "textstat-enhanced": compute_voice_score_with_textstat,
        "no-filter (baseline)": lambda _t: 1.0,  # Always sends to LLM
    }

    print(f"\nStep 2: Evaluating pre-filter strategies (threshold={threshold})...\n")

    all_metrics: list[PrefilterMetrics] = []

    for strategy_name, score_fn in strategies.items():
        metrics = PrefilterMetrics(strategy=strategy_name)

        for c in consensus:
            idx = c["index"]
            text = messages[idx]
            score = score_fn(text)
            predicted_voice = score >= threshold
            actual_voice = c["label"] == "voice"

            if predicted_voice:
                metrics.messages_sent_to_llm += 1
                if actual_voice:
                    metrics.true_positives += 1
                else:
                    metrics.false_positives += 1
            else:
                if actual_voice:
                    metrics.false_negatives += 1
                else:
                    metrics.true_negatives += 1

        all_metrics.append(metrics)

    # Print results table
    print(f"{'Strategy':<25s} {'TP':>4s} {'FP':>4s} {'TN':>4s} {'FN':>4s} {'Prec':>6s} {'Recall':>6s} {'F1':>6s} {'->LLM':>6s}")
    print("-" * 75)
    for m in all_metrics:
        print(
            f"{m.strategy:<25s} {m.true_positives:>4d} {m.false_positives:>4d} "
            f"{m.true_negatives:>4d} {m.false_negatives:>4d} "
            f"{m.precision:>6.2f} {m.recall:>6.2f} {m.f1:>6.2f} {m.messages_sent_to_llm:>5d}"
        )

    print(f"\nTotal messages: {len(messages)}")
    voice_count = sum(1 for c in consensus if c["label"] == "voice")
    print(f"Voice-dictated (ground truth): {voice_count}")
    print(f"Typed (ground truth): {len(messages) - voice_count}")


# ---------------------------------------------------------------------------
# Part B: Run model comparison
# ---------------------------------------------------------------------------


def run_model_comparison() -> None:
    """Run Part B: compare cleanup models on garbled samples."""
    print("\n" + "=" * 70)
    print("PART B: LLM Model Comparison (Cleanup Quality)")
    print("=" * 70)

    results_by_model: dict[str, list[CleanupBenchmarkResult]] = {}

    for model in CLEANUP_MODELS:
        print(f"\n--- {model} ---")
        results_by_model[model] = []

        for i, sample in enumerate(GARBLED_SAMPLES):
            print(f"  Sample {i}: ", end="", flush=True)
            result = cleanup_single(sample["garbled"], sample["context"], model)
            result.sample_index = i
            results_by_model[model].append(result)

            if result.error:
                print(f"ERROR: {result.error}")
            else:
                print(f"{result.latency_s:.2f}s, {result.prompt_tokens}+{result.completion_tokens} tokens, voice={result.is_voice}")

    # Print detailed comparison
    print("\n" + "-" * 70)
    print("DETAILED COMPARISON")
    print("-" * 70)

    for i, sample in enumerate(GARBLED_SAMPLES):
        print(f"\n{'=' * 60}")
        print(f"Sample {i}:")
        print(f"  INPUT:    {sample['garbled'][:100]}...")
        print(f"  EXPECTED: {sample['expected'][:100]}...")
        for model in CLEANUP_MODELS:
            r = results_by_model[model][i]
            if r.error:
                print(f"  {model}: ERROR - {r.error}")
            else:
                print(f"  {model}:")
                print(f"    is_voice: {r.is_voice}")
                print(f"    cleaned:  {r.cleaned[:100]}...")
                print(f"    latency:  {r.latency_s:.2f}s | tokens: {r.prompt_tokens}+{r.completion_tokens}")

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n{'Model':<40s} {'Avg Latency':>12s} {'Avg Prompt':>11s} {'Avg Compl':>10s} {'Errors':>7s}")
    print("-" * 80)
    for model in CLEANUP_MODELS:
        results = results_by_model[model]
        ok = [r for r in results if not r.error]
        if ok:
            avg_lat = sum(r.latency_s for r in ok) / len(ok)
            avg_prompt = sum(r.prompt_tokens for r in ok) / len(ok)
            avg_compl = sum(r.completion_tokens for r in ok) / len(ok)
        else:
            avg_lat = avg_prompt = avg_compl = 0
        errors = sum(1 for r in results if r.error)
        print(f"{model:<40s} {avg_lat:>10.2f}s {avg_prompt:>10.0f} {avg_compl:>9.0f} {errors:>7d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Benchmark transcript cleanup strategies and models")
    parser.add_argument("--part", choices=["prefilter", "models", "all"], default="all", help="Which benchmark to run")
    parser.add_argument("--db", type=str, default=None, help="Path to copilot_chats.db (for loading real messages)")
    parser.add_argument("--count", type=int, default=20, help="Number of messages to load from DB for pre-filter benchmark")
    args = parser.parse_args()

    print("Transcript Cleanup Benchmark")

    if args.part in ("prefilter", "all"):
        messages = load_messages_from_db(args.db, args.count)
        run_prefilter_benchmark(messages)

    if args.part in ("models", "all"):
        run_model_comparison()

    print("\nDone!")


if __name__ == "__main__":
    main()
