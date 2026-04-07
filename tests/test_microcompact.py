"""Tests for time-based microcompaction."""

from datetime import UTC, datetime, timedelta

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from compact_middleware.config import MicrocompactConfig
from compact_middleware.microcompact import (
    evaluate_time_based_trigger,
    microcompact_messages,
)


def _make_ai_with_timestamp(ts: datetime, tool_calls: list | None = None) -> AIMessage:
    return AIMessage(
        content="response",
        tool_calls=tool_calls or [],
        additional_kwargs={"timestamp": ts.isoformat()},
    )


def test_trigger_fires_on_old_gap() -> None:
    config = MicrocompactConfig(enabled=True, gap_threshold_minutes=60)
    old = datetime.now(UTC) - timedelta(hours=2)
    msgs = [_make_ai_with_timestamp(old)]
    gap = evaluate_time_based_trigger(msgs, config)
    assert gap is not None
    assert gap >= 120


def test_trigger_does_not_fire_on_recent() -> None:
    config = MicrocompactConfig(enabled=True, gap_threshold_minutes=60)
    recent = datetime.now(UTC) - timedelta(minutes=5)
    msgs = [_make_ai_with_timestamp(recent)]
    assert evaluate_time_based_trigger(msgs, config) is None


def test_trigger_disabled() -> None:
    config = MicrocompactConfig(enabled=False)
    old = datetime.now(UTC) - timedelta(hours=2)
    msgs = [_make_ai_with_timestamp(old)]
    assert evaluate_time_based_trigger(msgs, config) is None


def test_microcompact_clears_old_results() -> None:
    config = MicrocompactConfig(
        enabled=True,
        gap_threshold_minutes=60,
        keep_recent=1,
        compactable_tools={"read_file"},
    )
    old = datetime.now(UTC) - timedelta(hours=2)
    msgs = [
        _make_ai_with_timestamp(old, [{"name": "read_file", "args": {}, "id": "tc1"}]),
        ToolMessage(content="file content 1", tool_call_id="tc1"),
        _make_ai_with_timestamp(old, [{"name": "read_file", "args": {}, "id": "tc2"}]),
        ToolMessage(content="file content 2", tool_call_id="tc2"),
    ]
    result, event = microcompact_messages(msgs, config)
    assert event is not None
    # tc1 should be cleared, tc2 should be kept (most recent)
    assert result[1].content == config.cleared_message
    assert result[3].content == "file content 2"


def test_microcompact_skips_empty_ids() -> None:
    """Regression: empty-string IDs should not cause unintended clearing."""
    config = MicrocompactConfig(
        enabled=True,
        gap_threshold_minutes=60,
        keep_recent=1,
        compactable_tools={"read_file"},
    )
    old = datetime.now(UTC) - timedelta(hours=2)
    msgs = [
        _make_ai_with_timestamp(old, [{"name": "read_file", "args": {}, "id": ""}]),  # empty "id"
        ToolMessage(content="should not be cleared", tool_call_id=""),
    ]
    result, event = microcompact_messages(msgs, config)
    # No valid IDs collected, so nothing should be cleared
    assert event is None
    assert result[1].content == "should not be cleared"
