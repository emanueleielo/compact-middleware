"""Tests for token estimation and budget enforcement."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from compact_middleware.tokens import (
    enforce_message_budget,
    enforce_tool_result_budget,
    estimate_message_tokens,
    estimate_tokens,
    get_token_count_from_usage,
    get_token_usage,
    rough_token_count,
    rough_token_count_json,
    token_count_with_estimation,
)


def test_rough_token_count_basic() -> None:
    assert rough_token_count("hello world") == max(1, len("hello world") // 4)


def test_rough_token_count_empty() -> None:
    assert rough_token_count("") == 1


def test_rough_token_count_json_dict() -> None:
    data = {"key": "value", "number": 42}
    result = rough_token_count_json(data)
    assert result > 0


def test_estimate_message_tokens_human() -> None:
    msg = HumanMessage(content="Tell me about Python")
    tokens = estimate_message_tokens(msg)
    assert tokens > 0


def test_estimate_message_tokens_ai_with_tool_calls() -> None:
    msg = AIMessage(
        content="Let me check that.",
        tool_calls=[{"name": "read_file", "args": {"path": "/foo.py"}, "id": "tc1"}],
    )
    tokens = estimate_message_tokens(msg)
    assert tokens > estimate_message_tokens(AIMessage(content="Let me check that."))


def test_estimate_tokens_with_padding() -> None:
    msgs = [HumanMessage(content="x" * 400)]
    padded = estimate_tokens(msgs, pad=True)
    unpadded = estimate_tokens(msgs, pad=False)
    assert padded > unpadded


# ---------------------------------------------------------------------------
# Hybrid token counting
# ---------------------------------------------------------------------------


def test_get_token_usage_anthropic_format() -> None:
    msg = AIMessage(
        content="hello",
        response_metadata={
            "usage": {"input_tokens": 100, "output_tokens": 50},
        },
    )
    usage = get_token_usage(msg)
    assert usage is not None
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 50


def test_get_token_usage_openai_format() -> None:
    msg = AIMessage(
        content="hello",
        response_metadata={
            "token_usage": {"prompt_tokens": 200, "completion_tokens": 80},
        },
    )
    usage = get_token_usage(msg)
    assert usage is not None
    assert usage["input_tokens"] == 200
    assert usage["output_tokens"] == 80


def test_get_token_usage_none_for_human() -> None:
    assert get_token_usage(HumanMessage(content="hi")) is None


def test_get_token_usage_none_when_no_metadata() -> None:
    assert get_token_usage(AIMessage(content="hi")) is None


def test_get_token_count_from_usage() -> None:
    usage = {
        "input_tokens": 1000,
        "output_tokens": 200,
        "cache_creation_input_tokens": 50,
        "cache_read_input_tokens": 30,
    }
    assert get_token_count_from_usage(usage) == 1280


def test_token_count_with_estimation_uses_real_usage() -> None:
    """When an AIMessage has real usage, it should anchor the count."""
    msgs = [
        HumanMessage(content="What is Python?"),
        AIMessage(
            content="Python is a programming language.",
            response_metadata={
                "usage": {"input_tokens": 500, "output_tokens": 100},
            },
        ),
        HumanMessage(content="Tell me more"),  # only this gets heuristic
    ]
    hybrid = token_count_with_estimation(msgs)
    # Real: 500 + 100 = 600, plus heuristic for "Tell me more"
    # Should be close to 600 + small heuristic, NOT a full heuristic of all 3 msgs
    assert hybrid >= 600
    assert hybrid < 1000  # pure heuristic would be much less than 600


def test_token_count_with_estimation_fallback_no_usage() -> None:
    """Without any API usage, falls back to pure heuristic."""
    msgs = [HumanMessage(content="x" * 400)]
    hybrid = token_count_with_estimation(msgs)
    pure = estimate_tokens(msgs, pad=True)
    assert hybrid == pure


def test_token_count_with_estimation_no_pad() -> None:
    msgs = [
        AIMessage(
            content="response",
            response_metadata={
                "usage": {"input_tokens": 300, "output_tokens": 50},
            },
        ),
    ]
    padded = token_count_with_estimation(msgs, pad=True)
    unpadded = token_count_with_estimation(msgs, pad=False)
    # No tail messages, so both should equal the real count (350)
    assert padded == 350
    assert unpadded == 350


def test_token_count_with_estimation_multiple_api_responses() -> None:
    """Should use the LAST AIMessage with usage, not the first."""
    msgs = [
        AIMessage(
            content="old",
            response_metadata={"usage": {"input_tokens": 100, "output_tokens": 20}},
        ),
        HumanMessage(content="middle"),
        AIMessage(
            content="new",
            response_metadata={"usage": {"input_tokens": 5000, "output_tokens": 500}},
        ),
        HumanMessage(content="final question"),
    ]
    hybrid = token_count_with_estimation(msgs)
    # Should anchor on the LAST usage (5000 + 500 = 5500) + heuristic for "final question"
    assert hybrid >= 5500
    # Should NOT be anchored on the first (100 + 20 = 120)
    assert hybrid > 1000


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------


def test_enforce_tool_result_budget_within() -> None:
    content = "short"
    result, was_truncated = enforce_tool_result_budget(content, per_tool_chars=100)
    assert result == content
    assert not was_truncated


def test_enforce_tool_result_budget_over() -> None:
    content = "x" * 10_000
    result, was_truncated = enforce_tool_result_budget(content, per_tool_chars=1_000)
    assert was_truncated
    assert len(result) < len(content)
    assert "truncated" in result


def test_enforce_message_budget_per_tool() -> None:
    msgs = [
        ToolMessage(content="x" * 100_000, tool_call_id="tc1"),
    ]
    result = enforce_message_budget(msgs, per_tool_chars=1_000)
    assert len(result[0].content) < 100_000


def test_enforce_message_budget_aggregate() -> None:
    msgs = [
        ToolMessage(content="x" * 80_000, tool_call_id="tc1"),
        ToolMessage(content="y" * 80_000, tool_call_id="tc2"),
        ToolMessage(content="z" * 80_000, tool_call_id="tc3"),
    ]
    result = enforce_message_budget(msgs, per_message_chars=100_000, per_tool_chars=100_000)
    total_chars = sum(len(m.content) for m in result)
    # Should be capped around per_message_chars
    assert total_chars <= 110_000  # some slack for the truncation message
