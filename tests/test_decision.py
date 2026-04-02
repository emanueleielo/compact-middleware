"""Tests for the decision cascade engine."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from compact_middleware.config import CompactionConfig
from compact_middleware.decision import (
    CompactionLevel,
    evaluate,
    should_trigger,
    strip_images,
    truncate_head_for_ptl_retry,
)


def test_should_trigger_tokens() -> None:
    msgs = [HumanMessage(content="hi")]
    assert should_trigger(msgs, 100_000, ("tokens", 50_000), None)
    assert not should_trigger(msgs, 10_000, ("tokens", 50_000), None)


def test_should_trigger_messages() -> None:
    msgs = [HumanMessage(content="hi")] * 30
    assert should_trigger(msgs, 0, ("messages", 20), None)
    assert not should_trigger(msgs, 0, ("messages", 50), None)


def test_should_trigger_fraction() -> None:
    msgs = [HumanMessage(content="hi")]
    assert should_trigger(msgs, 90_000, ("fraction", 0.8), 100_000)
    assert not should_trigger(msgs, 70_000, ("fraction", 0.8), 100_000)


def test_should_trigger_none() -> None:
    assert not should_trigger([], 0, None, None)


def test_evaluate_no_trigger() -> None:
    config = CompactionConfig(trigger=("tokens", 999_999))
    msgs = [HumanMessage(content="short")]
    result = evaluate(msgs, None, None, config, None)
    assert result.level == CompactionLevel.NONE
    assert not result.needs_full_compaction


def test_evaluate_circuit_breaker() -> None:
    config = CompactionConfig(trigger=("messages", 1), max_consecutive_failures=3)
    msgs = [HumanMessage(content="hi")] * 10
    result = evaluate(msgs, None, None, config, None, consecutive_failures=3)
    assert result.level == CompactionLevel.NONE


def test_truncate_head_preserves_recent() -> None:
    msgs = [HumanMessage(content=f"msg {i}") for i in range(10)]
    result = truncate_head_for_ptl_retry(msgs, fraction=0.5)
    assert len(result) < len(msgs)
    assert len(result) > 0


def test_truncate_head_respects_tool_boundaries() -> None:
    msgs = [
        HumanMessage(content="start"),
        AIMessage(content="", tool_calls=[{"name": "read", "args": {}, "id": "tc1"}]),
        ToolMessage(content="result", tool_call_id="tc1"),
        HumanMessage(content="end"),
    ]
    result = truncate_head_for_ptl_retry(msgs, fraction=0.25)
    # Should not leave orphaned ToolMessage at start
    if result:
        assert not isinstance(result[0], ToolMessage)


def test_strip_images() -> None:
    msgs = [
        HumanMessage(content=[
            {"type": "text", "text": "Look at this:"},
            {"type": "image", "source": {"data": "base64..."}},
        ]),
        HumanMessage(content="plain text"),
    ]
    result = strip_images(msgs)
    # First message should have image removed
    assert len(result[0].content) == 1
    assert result[0].content[0]["type"] == "text"
    # Second message unchanged
    assert result[1].content == "plain text"
