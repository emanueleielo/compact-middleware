"""Time-based microcompaction — tool result clearing without full summarization.

Ports Claude Code's ``microCompact.ts`` time-based path. When the gap since
the last assistant message exceeds a threshold (default: 60 minutes), old
tool results from compactable tools are replaced with a cleared-message
placeholder. The most recent N results are always preserved.

This is a lightweight, pre-summarization optimization that reduces token
usage without the cost of an LLM call.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from compact_middleware.config import MicrocompactConfig
from compact_middleware.state import MicrocompactEvent
from compact_middleware.tokens import rough_token_count, rough_token_count_json

logger = logging.getLogger(__name__)


def _get_last_assistant_timestamp(messages: list[AnyMessage]) -> datetime | None:
    """Find the timestamp of the last AIMessage.

    Looks for ``timestamp`` in ``additional_kwargs`` or ``response_metadata``.
    Returns None if no timestamp is found.
    """
    for msg in reversed(messages):
        if not isinstance(msg, AIMessage):
            continue
        # Check additional_kwargs first (common in LangChain)
        ts = msg.additional_kwargs.get("timestamp")
        if ts is None:
            ts = msg.response_metadata.get("timestamp") if hasattr(msg, "response_metadata") else None
        if ts is not None:
            if isinstance(ts, datetime):
                return ts
            if isinstance(ts, (int, float)):
                return datetime.fromtimestamp(ts, tz=UTC)
            if isinstance(ts, str):
                try:
                    return datetime.fromisoformat(ts)
                except ValueError:
                    pass
        # Fallback: check if the message itself has a created_at
        if hasattr(msg, "response_metadata") and isinstance(msg.response_metadata, dict):
            created = msg.response_metadata.get("created_at")
            if isinstance(created, str):
                try:
                    return datetime.fromisoformat(created)
                except ValueError:
                    pass
    return None


def _collect_compactable_tool_ids(
    messages: list[AnyMessage],
    compactable_tools: set[str],
) -> list[str]:
    """Collect tool_use IDs for compactable tools in encounter order.

    Walks AIMessages and collects IDs from tool_calls whose name is in
    the compactable set.
    """
    ids: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc.get("name", "") in compactable_tools:
                    tc_id = tc.get("id", "")
                    if tc_id:
                        ids.append(tc_id)
    return ids


def _estimate_tool_message_tokens(msg: ToolMessage) -> int:
    """Estimate tokens in a ToolMessage's content."""
    content = msg.content
    if isinstance(content, str):
        return rough_token_count(content)
    if isinstance(content, list):
        total = 0
        for block in content:
            if isinstance(block, str):
                total += rough_token_count(block)
            elif isinstance(block, dict):
                btype = block.get("type", "")
                if btype == "text":
                    total += rough_token_count(block.get("text", ""))
                elif btype in ("image", "document"):
                    total += 2000
                else:
                    total += rough_token_count_json(block)
        return total
    return rough_token_count_json(content)


def evaluate_time_based_trigger(
    messages: list[AnyMessage],
    config: MicrocompactConfig,
) -> float | None:
    """Check if time-based microcompaction should trigger.

    Args:
        messages: Current message history.
        config: Microcompact configuration.

    Returns:
        Gap in minutes if trigger fires, None otherwise.
    """
    if not config.enabled:
        return None

    last_ts = _get_last_assistant_timestamp(messages)
    if last_ts is None:
        return None

    now = datetime.now(UTC)
    if last_ts.tzinfo is None:
        last_ts = last_ts.replace(tzinfo=UTC)

    gap_minutes = (now - last_ts).total_seconds() / 60.0

    if gap_minutes < config.gap_threshold_minutes:
        return None

    return gap_minutes


def microcompact_messages(
    messages: list[AnyMessage],
    config: MicrocompactConfig,
) -> tuple[list[AnyMessage], MicrocompactEvent | None]:
    """Apply time-based microcompaction to messages.

    Replaces old tool result content with a cleared-message placeholder
    when the time gap since the last assistant message exceeds the threshold.
    Keeps the most recent N tool results intact.

    Args:
        messages: Current message history.
        config: Microcompact configuration.

    Returns:
        Tuple of (possibly modified messages, event or None).
    """
    gap_minutes = evaluate_time_based_trigger(messages, config)
    if gap_minutes is None:
        return messages, None

    compactable_ids = _collect_compactable_tool_ids(messages, config.compactable_tools)
    if not compactable_ids:
        return messages, None

    keep_recent = max(1, config.keep_recent)
    keep_set = set(compactable_ids[-keep_recent:])
    clear_set = set(id_ for id_ in compactable_ids if id_ not in keep_set)

    if not clear_set:
        return messages, None

    # Build a map of tool_call_id -> tool_call_id for ToolMessages
    tokens_saved = 0
    cleared_ids: list[str] = []
    result: list[AnyMessage] = []

    for msg in messages:
        if isinstance(msg, ToolMessage) and msg.tool_call_id in clear_set:
            if isinstance(msg.content, str) and msg.content != config.cleared_message:
                tokens_saved += rough_token_count(msg.content)
                cleared_ids.append(msg.tool_call_id)
                msg = msg.model_copy()
                msg.content = config.cleared_message
        result.append(msg)

    if tokens_saved == 0:
        return messages, None

    logger.info(
        "[TIME-BASED MC] gap %.0fmin > %.0fmin, cleared %d tool results (~%d tokens), kept last %d",
        gap_minutes,
        config.gap_threshold_minutes,
        len(cleared_ids),
        tokens_saved,
        len(keep_set),
    )

    event = MicrocompactEvent(
        cleared_tool_ids=cleared_ids,
        tokens_saved=tokens_saved,
        trigger="time_based",
    )

    return result, event
