"""Tests for the decision cascade engine."""

from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from compact_middleware.config import (
    CollapseConfig,
    CompactionConfig,
    MicrocompactConfig,
)
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


def test_evaluate_no_trigger_no_llm() -> None:
    """Even below token threshold, lightweight levels run but LLM compaction doesn't."""
    config = CompactionConfig(trigger=("tokens", 999_999))
    msgs = [HumanMessage(content="short")]
    result = evaluate(msgs, None, None, config, None)
    assert not result.needs_full_compaction
    assert not result.needs_partial_compaction


def test_evaluate_lightweight_levels_run_below_threshold() -> None:
    """Microcompact fires on time-gap even when tokens are well below the global trigger."""
    old_ts = (datetime.now(UTC) - timedelta(minutes=120)).isoformat()
    msgs = [
        HumanMessage(content="start"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {}, "id": "tc1"},
                {"name": "read_file", "args": {}, "id": "tc2"},
            ],
            additional_kwargs={"timestamp": old_ts},
        ),
        ToolMessage(content="big result " * 200, tool_call_id="tc1"),
        ToolMessage(content="big result " * 200, tool_call_id="tc2"),
        HumanMessage(content="end"),
    ]
    config = CompactionConfig(
        trigger=("tokens", 999_999),  # very high — global trigger won't fire
        microcompact=MicrocompactConfig(
            enabled=True,
            gap_threshold_minutes=60,
            keep_recent=1,
            compactable_tools={"read_file"},
        ),
    )
    result = evaluate(msgs, None, None, config, None)
    # Microcompact should have fired (time gap 120min > 60min threshold)
    assert result.microcompact_event is not None
    assert result.microcompact_event["tokens_saved"] > 0
    # But LLM compaction should NOT be needed (tokens below global trigger)
    assert not result.needs_full_compaction
    assert not result.needs_partial_compaction


def test_evaluate_circuit_breaker_still_runs_lightweight() -> None:
    """Circuit breaker blocks LLM compaction but lightweight levels still run."""
    old_ts = (datetime.now(UTC) - timedelta(minutes=120)).isoformat()
    msgs = [
        HumanMessage(content="start"),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read_file", "args": {}, "id": "tc1"},
                {"name": "read_file", "args": {}, "id": "tc2"},
            ],
            additional_kwargs={"timestamp": old_ts},
        ),
        ToolMessage(content="big result " * 200, tool_call_id="tc1"),
        ToolMessage(content="big result " * 200, tool_call_id="tc2"),
        HumanMessage(content="end"),
    ]
    config = CompactionConfig(
        trigger=("messages", 1),
        max_consecutive_failures=3,
        microcompact=MicrocompactConfig(
            enabled=True,
            gap_threshold_minutes=60,
            keep_recent=1,
            compactable_tools={"read_file"},
        ),
    )
    result = evaluate(msgs, None, None, config, None, consecutive_failures=3)
    # Circuit breaker prevents LLM compaction
    assert not result.needs_full_compaction
    assert not result.needs_partial_compaction
    # But microcompact still ran
    assert result.microcompact_event is not None


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
