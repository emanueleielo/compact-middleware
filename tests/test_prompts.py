"""Tests for prompt construction and summary formatting."""

from compact_middleware.prompts import (
    format_compact_summary,
    get_compact_prompt,
    get_compact_user_summary_message,
    get_partial_compact_prompt,
)


def test_get_compact_prompt_contains_sections() -> None:
    prompt = get_compact_prompt()
    assert "Primary Request and Intent" in prompt
    assert "Files and Code Sections" in prompt
    assert "CRITICAL: Respond with TEXT ONLY" in prompt
    assert "REMINDER: Do NOT call any tools" in prompt


def test_get_compact_prompt_with_custom_instructions() -> None:
    prompt = get_compact_prompt("Focus on TypeScript changes")
    assert "<additional_instructions>" in prompt
    assert "Focus on TypeScript changes" in prompt
    assert "</additional_instructions>" in prompt


def test_get_partial_compact_prompt_from() -> None:
    prompt = get_partial_compact_prompt(direction="from")
    assert "RECENT" in prompt


def test_get_partial_compact_prompt_up_to() -> None:
    prompt = get_partial_compact_prompt(direction="up_to")
    assert "Context for Continuing Work" in prompt


def test_format_compact_summary_strips_analysis() -> None:
    raw = "<analysis>thinking...</analysis>\n\n<summary>The result.</summary>"
    formatted = format_compact_summary(raw)
    assert "thinking" not in formatted
    assert "The result." in formatted


def test_format_compact_summary_no_tags() -> None:
    raw = "Just a plain summary with no XML tags."
    formatted = format_compact_summary(raw)
    assert formatted == raw


def test_get_compact_user_summary_message_basic() -> None:
    msg = get_compact_user_summary_message("<summary>Test</summary>")
    assert "continued from a previous conversation" in msg
    assert "Test" in msg


def test_get_compact_user_summary_message_with_transcript() -> None:
    msg = get_compact_user_summary_message(
        "<summary>Test</summary>",
        transcript_path="/history/session.md",
    )
    assert "/history/session.md" in msg


def test_get_compact_user_summary_message_suppress_followup() -> None:
    msg = get_compact_user_summary_message(
        "<summary>Test</summary>",
        suppress_follow_up_questions=True,
    )
    assert "without asking" in msg
