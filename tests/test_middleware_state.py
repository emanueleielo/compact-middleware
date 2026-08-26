"""Tests for compaction state persistence and cutoff bookkeeping.

These cover the two things that were previously untested and wrong:

* compaction state lived on the middleware instance, so it was lost whenever
  a new instance was created (per-request agents, serverless, multi-process);
* the cutoff was computed against the *collapsed* message list but persisted
  as an index into the raw state list, so it drifted by however many messages
  collapse had removed.
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from compact_middleware.collapse import collapse_messages
from compact_middleware.config import CollapseConfig, CompactionConfig
from compact_middleware.decision import CompactionLevel, DecisionResult, evaluate
from compact_middleware.middleware import (
    CompactionMiddleware,
    _read_event,
    _read_failures,
)
from compact_middleware.state import CompactionEvent


def _read_pair(idx: int, tool: str = "read_file") -> list:
    """One (AIMessage with a tool call, ToolMessage) pair."""
    call_id = f"c{idx}"
    return [
        AIMessage(
            content="",
            tool_calls=[{"name": tool, "args": {"path": f"f{idx}.py"}, "id": call_id}],
        ),
        ToolMessage(content=f"contents of f{idx}.py", tool_call_id=call_id),
    ]


def _summary(text: str = "Summary:\nprior work") -> HumanMessage:
    return HumanMessage(content=text, additional_kwargs={"lc_source": "compaction"})


def _event(cutoff: int) -> CompactionEvent:
    return CompactionEvent(
        cutoff_index=cutoff,
        summary_message=_summary(),
        file_path=None,
        strategy="full",
        tokens_before=0,
        tokens_after=0,
    )


# ---------------------------------------------------------------------------
# State accessors
# ---------------------------------------------------------------------------


def test_read_event_handles_missing_and_malformed_state():
    assert _read_event(None) is None
    assert _read_event({}) is None
    assert _read_event({"_compaction_event": None}) is None
    assert _read_event({"_compaction_event": "nonsense"}) is None

    event = _event(4)
    assert _read_event({"_compaction_event": event}) == event


def test_read_failures_defaults_to_zero():
    assert _read_failures(None) == 0
    assert _read_failures({}) == 0
    assert _read_failures({"_compaction_failures": "nope"}) == 0
    assert _read_failures({"_compaction_failures": 3}) == 3


def test_state_update_carries_event_and_resets_breaker():
    command = CompactionMiddleware._state_update(_event(7), 0)
    assert command.update["_compaction_event"]["cutoff_index"] == 7
    assert command.update["_compaction_failures"] == 0


def test_state_update_without_event_only_bumps_breaker():
    """A failed compaction must not clobber the last good event."""
    command = CompactionMiddleware._state_update(None, 2)
    assert "_compaction_event" not in command.update
    assert command.update["_compaction_failures"] == 2


# ---------------------------------------------------------------------------
# Effective view reconstruction
# ---------------------------------------------------------------------------


def test_apply_event_rebuilds_view_from_persisted_state():
    messages = [HumanMessage(content=f"m{i}") for i in range(6)]
    result = CompactionMiddleware._apply_event_to_messages(messages, _event(4))

    assert CompactionMiddleware._is_summary_message(result[0])
    assert [m.content for m in result[1:]] == ["m4", "m5"]


def test_apply_event_without_state_returns_raw_history():
    messages = [HumanMessage(content=f"m{i}") for i in range(3)]
    assert CompactionMiddleware._apply_event_to_messages(messages, None) == messages


def test_state_cutoff_round_trips_across_successive_compactions():
    """Two compactions in a row must keep pointing at the same raw message."""
    raw = [HumanMessage(content=f"m{i}") for i in range(20)]

    first = CompactionMiddleware._compute_state_cutoff(None, 12)
    assert first == 12

    effective = CompactionMiddleware._apply_event_to_messages(raw, _event(first))
    # effective == [summary, m12 .. m19]; keep the last three.
    second = CompactionMiddleware._compute_state_cutoff(_event(first), 6)

    assert effective[6].content == raw[second].content
    assert [m.content for m in effective[6:]] == [m.content for m in raw[second:]]


# ---------------------------------------------------------------------------
# Cutoff translation through collapse (the drift bug)
# ---------------------------------------------------------------------------


def test_collapse_returns_identity_source_map_when_nothing_collapses():
    messages = [HumanMessage(content="hi"), AIMessage(content="hello")]
    result, event, source_map = collapse_messages(messages, CollapseConfig())

    assert event is None
    assert result == messages
    assert source_map == [0, 1]


def test_collapse_source_map_tracks_original_indices():
    messages = [HumanMessage(content="go")]
    for i in range(4):
        messages += _read_pair(i)
    messages.append(AIMessage(content="done"))

    collapsed, event, source_map = collapse_messages(
        messages, CollapseConfig(min_group_size=2)
    )

    assert event is not None
    assert len(collapsed) < len(messages)
    assert len(source_map) == len(collapsed)
    # Strictly increasing, and the last entry accounts for the last message.
    assert source_map == sorted(source_map)
    assert source_map[-1] == len(messages) - 1


def test_to_source_index_undoes_collapse_shortening():
    messages = [HumanMessage(content="go")]
    for i in range(4):
        messages += _read_pair(i)
    messages.append(AIMessage(content="done"))

    collapsed, _, source_map = collapse_messages(
        messages, CollapseConfig(min_group_size=2)
    )
    decision = DecisionResult(
        messages=collapsed,
        level=CompactionLevel.COLLAPSE,
        tokens_before=0,
        tokens_after=0,
        source_map=source_map,
    )

    # Summarizing the whole collapsed list must cover the whole source list.
    assert decision.to_source_index(len(collapsed)) == len(messages)
    assert decision.to_source_index(0) == 0

    # And any partial cutoff must map to at least as many source messages,
    # never fewer — that undercount was the drift.
    for cutoff in range(1, len(collapsed) + 1):
        assert decision.to_source_index(cutoff) >= cutoff


def test_to_source_index_is_identity_without_a_map():
    decision = DecisionResult(
        messages=[],
        level=CompactionLevel.NONE,
        tokens_before=0,
        tokens_after=0,
    )
    assert decision.to_source_index(5) == 5


def test_evaluate_reports_a_source_map_covering_its_messages():
    messages = [HumanMessage(content="go")]
    for i in range(4):
        messages += _read_pair(i)

    decision = evaluate(messages, None, None, CompactionConfig(), None)

    assert len(decision.source_map) == len(decision.messages)
    assert decision.to_source_index(len(decision.messages)) == len(messages)


def test_collapse_drift_regression():
    """The exact bug: a cutoff from the collapsed list, used raw, undercounts.

    Persisting the un-translated index made the next turn's effective view
    re-include messages the summary already covered.
    """
    messages = [HumanMessage(content="go")]
    for i in range(4):
        messages += _read_pair(i)
    messages.append(AIMessage(content="done"))

    collapsed, _, source_map = collapse_messages(
        messages, CollapseConfig(min_group_size=2)
    )
    decision = DecisionResult(
        messages=collapsed,
        level=CompactionLevel.COLLAPSE,
        tokens_before=0,
        tokens_after=0,
        source_map=source_map,
    )

    cutoff = len(collapsed) - 1
    naive = CompactionMiddleware._compute_state_cutoff(None, cutoff)
    fixed = CompactionMiddleware._compute_state_cutoff(
        None, decision.to_source_index(cutoff)
    )

    assert naive < fixed, "collapse must shorten the list for this to be a test"

    # Under the old arithmetic the tail re-included already-summarized turns.
    assert len(messages) - naive > len(messages) - fixed


# ---------------------------------------------------------------------------
# Tool-boundary safety
# ---------------------------------------------------------------------------


def test_cutoff_never_orphans_a_tool_message():
    messages = [HumanMessage(content="go"), *_read_pair(0), *_read_pair(1)]

    # Index 2 is the ToolMessage replying to the AIMessage at index 1.
    assert isinstance(messages[2], ToolMessage)
    aligned = CompactionMiddleware._align_cutoff_to_tool_boundary(messages, 2)

    assert not isinstance(messages[aligned], ToolMessage)
    assert aligned == 1


def test_cutoff_alignment_leaves_safe_boundaries_alone():
    messages = [HumanMessage(content="go"), *_read_pair(0), HumanMessage(content="next")]

    for cutoff in (0, 1, 3, len(messages)):
        assert (
            CompactionMiddleware._align_cutoff_to_tool_boundary(messages, cutoff)
            == cutoff
        )


@pytest.mark.parametrize("keep", [("messages", 1), ("messages", 2), ("messages", 3)])
def test_determined_cutoff_is_always_tool_safe(keep):
    messages = [HumanMessage(content="go")]
    for i in range(3):
        messages += _read_pair(i)

    mw = CompactionMiddleware.__new__(CompactionMiddleware)
    mw._config = CompactionConfig(keep=keep)
    mw._max_input_tokens = None

    cutoff = mw._determine_cutoff_index(messages)

    if 0 < cutoff < len(messages):
        assert not isinstance(messages[cutoff], ToolMessage)
