"""Multi-level decision engine for compaction.

Implements Claude Code's cascading pipeline:

1. **Collapse** — group consecutive read/search tool calls
2. **Truncate** — shorten large tool-call arguments
3. **Microcompact** — clear old tool results (time-based)
4. **Full / Partial compaction** — LLM-based summarization

The engine evaluates each level in order and applies the first (or multiple)
that bring token usage below the threshold. Includes:

- Circuit breaker: stops auto-compaction after N consecutive failures
- PTL error recovery: retries with head truncation on prompt-too-long
- Image stripping: removes images when media errors occur
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import BaseTool

from compact_middleware.collapse import collapse_messages
from compact_middleware.config import CompactionConfig, ContextSize
from compact_middleware.microcompact import microcompact_messages
from compact_middleware.state import CollapseEvent, CompactionEvent, MicrocompactEvent
from compact_middleware.tokens import estimate_tokens, token_count_with_estimation
from compact_middleware.truncation import truncate_args

logger = logging.getLogger(__name__)


class CompactionLevel(Enum):
    """Compaction levels in the decision cascade."""
    NONE = "none"
    COLLAPSE = "collapse"
    TRUNCATE = "truncate"
    MICROCOMPACT = "microcompact"
    PARTIAL = "partial"
    FULL = "full"


@dataclass
class DecisionResult:
    """Result of the decision engine evaluation.

    Contains the modified messages and metadata about what was applied.
    """

    messages: list[AnyMessage]
    """Messages after applying decided compaction levels."""

    level: CompactionLevel
    """Highest compaction level that was applied."""

    tokens_before: int
    """Estimated tokens before any compaction."""

    tokens_after: int
    """Estimated tokens after compaction levels applied (before full/partial)."""

    collapse_event: CollapseEvent | None = None
    microcompact_event: MicrocompactEvent | None = None
    args_truncated: bool = False
    needs_full_compaction: bool = False
    needs_partial_compaction: bool = False


def get_max_input_tokens(model_profile: dict[str, Any] | None) -> int | None:
    """Extract max_input_tokens from a model profile dict."""
    if (
        model_profile is not None
        and isinstance(model_profile, dict)
        and "max_input_tokens" in model_profile
    ):
        val = model_profile["max_input_tokens"]
        if isinstance(val, int):
            return val
    return None


def compute_auto_compact_threshold(
    max_input_tokens: int,
    config: CompactionConfig,
) -> int:
    """Compute the auto-compaction threshold.

    threshold = context_window - max_output_for_summary - buffer

    Args:
        max_input_tokens: Model's max input tokens.
        config: Compaction config with buffer values.

    Returns:
        Token count threshold that triggers auto-compaction.
    """
    effective = max_input_tokens - config.max_output_tokens_for_summary
    return effective - config.autocompact_buffer_tokens


def should_trigger(
    messages: list[AnyMessage],
    total_tokens: int,
    trigger: ContextSize | list[ContextSize] | None,
    max_input_tokens: int | None,
) -> bool:
    """Check if any compaction trigger condition is met.

    Args:
        messages: Current messages.
        total_tokens: Current token estimate.
        trigger: Trigger condition(s).
        max_input_tokens: Model limit for fraction-based triggers.

    Returns:
        True if compaction should be considered.
    """
    if trigger is None:
        return False

    conditions = trigger if isinstance(trigger, list) else [trigger]

    for kind, value in conditions:
        if kind == "tokens" and total_tokens >= value:
            return True
        if kind == "messages" and len(messages) >= value:
            return True
        if kind == "fraction" and max_input_tokens is not None:
            threshold = int(max_input_tokens * value)
            if total_tokens >= max(1, threshold):
                return True

    return False


def evaluate(
    messages: list[AnyMessage],
    system_message: SystemMessage | None,
    tools: list[BaseTool | dict[str, Any]] | None,
    config: CompactionConfig,
    max_input_tokens: int | None,
    consecutive_failures: int = 0,
) -> DecisionResult:
    """Run the multi-level decision cascade.

    Evaluates each compaction level in order. Lightweight levels (collapse,
    truncate, microcompact) are always applied when their conditions are met.
    If token usage still exceeds the threshold after lightweight levels,
    the engine signals that full or partial compaction is needed.

    Args:
        messages: Current effective message list.
        system_message: System message (for token counting).
        tools: Tools list (for token counting).
        config: Full compaction configuration.
        max_input_tokens: Model's max input token limit.
        consecutive_failures: Current circuit breaker counter.

    Returns:
        DecisionResult with modified messages and metadata.
    """
    # Circuit breaker check
    if consecutive_failures >= config.max_consecutive_failures:
        logger.warning(
            "Circuit breaker: %d consecutive failures >= %d, skipping auto-compaction",
            consecutive_failures,
            config.max_consecutive_failures,
        )
        return DecisionResult(
            messages=messages,
            level=CompactionLevel.NONE,
            tokens_before=0,
            tokens_after=0,
        )

    # Estimate current tokens (hybrid: real API usage + heuristic for tail)
    counted = [system_message, *messages] if system_message is not None else list(messages)
    tokens_before = token_count_with_estimation(counted)

    # Check if any trigger fires
    if not should_trigger(messages, tokens_before, config.trigger, max_input_tokens):
        return DecisionResult(
            messages=messages,
            level=CompactionLevel.NONE,
            tokens_before=tokens_before,
            tokens_after=tokens_before,
        )

    current_messages = messages
    current_level = CompactionLevel.NONE
    collapse_event = None
    microcompact_event = None
    args_truncated = False

    # Level 1: Collapse
    collapsed, c_event = collapse_messages(current_messages, config.collapse)
    if c_event is not None:
        current_messages = collapsed
        current_level = CompactionLevel.COLLAPSE
        collapse_event = c_event

    # Level 2: Truncate args
    truncated, was_truncated = truncate_args(
        current_messages,
        system_message,
        tools,
        config.truncate_args,
        max_input_tokens,
    )
    if was_truncated:
        current_messages = truncated
        current_level = CompactionLevel.TRUNCATE
        args_truncated = True

    # Level 3: Microcompact
    microcompacted, mc_event = microcompact_messages(
        current_messages,
        config.microcompact,
    )
    if mc_event is not None:
        current_messages = microcompacted
        current_level = CompactionLevel.MICROCOMPACT
        microcompact_event = mc_event

    # Re-estimate after lightweight levels
    counted_after = [system_message, *current_messages] if system_message is not None else list(current_messages)
    tokens_after = token_count_with_estimation(counted_after)

    # Check if we still need full/partial compaction
    still_over = should_trigger(
        current_messages, tokens_after, config.trigger, max_input_tokens,
    )

    needs_partial = False
    needs_full = False

    if still_over:
        if config.prefer_partial and len(current_messages) > 10:
            needs_partial = True
            current_level = CompactionLevel.PARTIAL
        else:
            needs_full = True
            current_level = CompactionLevel.FULL

    return DecisionResult(
        messages=current_messages,
        level=current_level,
        tokens_before=tokens_before,
        tokens_after=tokens_after,
        collapse_event=collapse_event,
        microcompact_event=microcompact_event,
        args_truncated=args_truncated,
        needs_full_compaction=needs_full,
        needs_partial_compaction=needs_partial,
    )


# ---------------------------------------------------------------------------
# PTL error recovery
# ---------------------------------------------------------------------------

def truncate_head_for_ptl_retry(
    messages: list[AnyMessage],
    fraction: float = 0.5,
) -> list[AnyMessage]:
    """Truncate the first portion of messages for prompt-too-long retry.

    Removes the first ``fraction`` of messages (by count), respecting
    tool_call/ToolMessage boundaries.

    Args:
        messages: Current messages.
        fraction: Fraction of messages to remove from the head.

    Returns:
        Truncated message list.
    """
    if not messages:
        return messages

    cut_count = max(1, int(len(messages) * fraction))

    # Adjust cut point to not split a tool_call from its ToolMessage
    while cut_count < len(messages):
        msg = messages[cut_count]
        # If we'd start with a ToolMessage, include it in the cut
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Include the AIMessage and its following ToolMessages
            cut_count += 1
            while cut_count < len(messages) and isinstance(messages[cut_count], ToolMessage):
                cut_count += 1
            break
        elif isinstance(msg, ToolMessage):
            cut_count += 1
        else:
            break

    return messages[cut_count:]


def strip_images(messages: list[AnyMessage]) -> list[AnyMessage]:
    """Remove image content blocks from messages.

    Used as a recovery strategy when the API returns media-related errors.

    Args:
        messages: Messages to process.

    Returns:
        New message list with image blocks removed.
    """
    result: list[AnyMessage] = []
    for msg in messages:
        content = msg.content
        if isinstance(content, list):
            filtered = [
                block for block in content
                if not (isinstance(block, dict) and block.get("type") in ("image", "image_url"))
            ]
            if len(filtered) != len(content):
                msg = msg.model_copy()
                msg.content = filtered if filtered else "[ images removed ]"
        result.append(msg)
    return result
