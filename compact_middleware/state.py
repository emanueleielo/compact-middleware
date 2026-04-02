"""State types for the Compaction Middleware.

Defines the compaction event structure and the state schema used to persist
compaction information across agent turns.
"""

from __future__ import annotations

from typing import Annotated, Any, NotRequired

from langchain.agents.middleware.types import AgentState, PrivateStateAttr
from langchain_core.messages import HumanMessage
from typing_extensions import TypedDict


class CompactionEvent(TypedDict):
    """Represents a compaction event persisted across turns.

    Attributes:
        cutoff_index: Absolute index in state messages where compaction occurred.
        summary_message: The HumanMessage containing the formatted summary.
        file_path: Backend path where conversation history was offloaded, or None.
        strategy: Which compaction strategy was used.
        tokens_before: Estimated tokens before compaction.
        tokens_after: Estimated tokens after compaction.
    """

    cutoff_index: int
    summary_message: HumanMessage
    file_path: str | None
    strategy: str
    tokens_before: int
    tokens_after: int


class MicrocompactEvent(TypedDict):
    """Tracks a microcompaction event (tool result clearing).

    Attributes:
        cleared_tool_ids: IDs of tool results that were cleared.
        tokens_saved: Approximate tokens freed.
        trigger: What triggered the microcompact ('time_based' or 'threshold').
    """

    cleared_tool_ids: list[str]
    tokens_saved: int
    trigger: str


class CollapseEvent(TypedDict):
    """Tracks a message collapse event.

    Attributes:
        groups_collapsed: Number of read/search groups collapsed.
        messages_reduced: How many messages were merged.
    """

    groups_collapsed: int
    messages_reduced: int


class CompactionState(AgentState):
    """State schema for the Compaction Middleware.

    Extends AgentState with private fields for tracking compaction events.
    All fields use ``PrivateStateAttr`` so they are not leaked to parent graphs.
    """

    _compaction_event: Annotated[NotRequired[CompactionEvent | None], PrivateStateAttr]
    """Most recent full/partial compaction event."""

    _microcompact_event: Annotated[NotRequired[MicrocompactEvent | None], PrivateStateAttr]
    """Most recent microcompaction event."""

    _collapse_event: Annotated[NotRequired[CollapseEvent | None], PrivateStateAttr]
    """Most recent collapse event."""

    _compaction_failures: Annotated[NotRequired[int], PrivateStateAttr]
    """Consecutive auto-compaction failures (circuit breaker counter)."""
