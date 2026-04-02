"""Type-aware token estimation — hybrid real + heuristic.

Ports Claude Code's ``tokenCountWithEstimation()`` strategy:

1. Walk messages backwards to find the last ``AIMessage`` with real API usage
2. Use the **real** ``input_tokens + output_tokens + cache_*`` from that response
3. Estimate only the **new messages after it** with the chars-based heuristic

Heuristic fallback (when no API response is available):

- **Text**: ``len(text) / 4`` (rough char-to-token ratio)
- **JSON / structured data**: ``len(text) / 2`` (JSON is ~2 bytes/token)
- **Images / documents**: flat 2,000 tokens each
- **Tool use blocks**: name + serialized input
- **Thinking blocks**: count only the thinking text

The primary entry point is ``token_count_with_estimation()`` — it replaces
the older pure-heuristic ``estimate_tokens()`` in all decision paths.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Primitive estimators
# ---------------------------------------------------------------------------

IMAGE_TOKEN_SIZE = 2_000
"""Flat token estimate for any image or document block."""

CONSERVATIVE_PAD = 4 / 3
"""Pad rough estimates by 4/3 (same as Claude Code)."""


def rough_token_count(text: str) -> int:
    """Estimate tokens for a plain text string (chars / 4)."""
    return max(1, len(text) // 4)


def rough_token_count_json(data: Any) -> int:
    """Estimate tokens for JSON-serializable data (chars / 2)."""
    try:
        serialized = json.dumps(data, ensure_ascii=False)
    except (TypeError, ValueError):
        serialized = str(data)
    return max(1, len(serialized) // 2)


# ---------------------------------------------------------------------------
# Content block estimator
# ---------------------------------------------------------------------------

def _estimate_content_block_tokens(block: dict[str, Any] | str) -> int:
    """Estimate tokens for a single content block.

    Handles the LangChain content block format (dicts with ``type`` key)
    as well as plain strings.
    """
    if isinstance(block, str):
        return rough_token_count(block)

    block_type = block.get("type", "")

    if block_type == "text":
        return rough_token_count(block.get("text", ""))

    if block_type in ("image", "image_url", "document"):
        return IMAGE_TOKEN_SIZE

    if block_type == "tool_use":
        name = block.get("name", "")
        input_data = block.get("input", {})
        return rough_token_count(name) + rough_token_count_json(input_data)

    if block_type == "tool_result":
        content = block.get("content", "")
        if isinstance(content, str):
            return rough_token_count(content)
        if isinstance(content, list):
            return sum(_estimate_content_block_tokens(b) for b in content)
        return rough_token_count_json(content)

    if block_type == "thinking":
        return rough_token_count(block.get("thinking", ""))

    if block_type == "redacted_thinking":
        return rough_token_count(block.get("data", ""))

    # Fallback: serialize the whole block as JSON
    return rough_token_count_json(block)


# ---------------------------------------------------------------------------
# Message-level estimator
# ---------------------------------------------------------------------------

def estimate_message_tokens(message: AnyMessage) -> int:
    """Estimate tokens for a single LangChain message.

    Handles both string content and structured content blocks.
    Also accounts for tool_calls on AIMessages.
    """
    tokens = 0

    # Content
    content = message.content
    if isinstance(content, str):
        tokens += rough_token_count(content)
    elif isinstance(content, list):
        for block in content:
            tokens += _estimate_content_block_tokens(block)

    # Tool calls (on AIMessage)
    if isinstance(message, AIMessage) and message.tool_calls:
        for tc in message.tool_calls:
            tokens += rough_token_count(tc.get("name", ""))
            tokens += rough_token_count_json(tc.get("args", {}))

    return tokens


def estimate_tokens(
    messages: list[AnyMessage],
    *,
    pad: bool = True,
) -> int:
    """Estimate total tokens for a list of messages.

    Args:
        messages: LangChain messages to estimate.
        pad: If True, apply the conservative 4/3 padding factor.

    Returns:
        Estimated token count.
    """
    total = sum(estimate_message_tokens(msg) for msg in messages)
    if pad:
        total = int(total * CONSERVATIVE_PAD)
    return total


# ---------------------------------------------------------------------------
# Real API token usage extraction
# ---------------------------------------------------------------------------

def get_token_usage(message: AnyMessage) -> dict[str, int] | None:
    """Extract token usage from an AIMessage's response metadata.

    LangChain stores API usage in ``response_metadata`` under keys like
    ``token_usage``, ``usage``, or ``usage_metadata``. Supports Anthropic,
    OpenAI, and generic LangChain formats.

    Args:
        message: Any LangChain message (only AIMessages carry usage).

    Returns:
        Usage dict with ``input_tokens``, ``output_tokens``, etc., or None.
    """
    if not isinstance(message, AIMessage):
        return None

    meta = getattr(message, "response_metadata", None)
    if not isinstance(meta, dict):
        return None

    # Anthropic format: response_metadata.usage
    usage = meta.get("usage")
    if isinstance(usage, dict) and "input_tokens" in usage:
        return usage

    # OpenAI / LangChain format: response_metadata.token_usage
    usage = meta.get("token_usage")
    if isinstance(usage, dict):
        # Normalize OpenAI keys → Anthropic keys
        return {
            "input_tokens": usage.get("prompt_tokens", usage.get("input_tokens", 0)),
            "output_tokens": usage.get("completion_tokens", usage.get("output_tokens", 0)),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        }

    # usage_metadata (newer LangChain)
    usage = meta.get("usage_metadata")
    if isinstance(usage, dict) and ("input_tokens" in usage or "prompt_tokens" in usage):
        return {
            "input_tokens": usage.get("input_tokens", usage.get("prompt_tokens", 0)),
            "output_tokens": usage.get("output_tokens", usage.get("completion_tokens", 0)),
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        }

    return None


def get_token_count_from_usage(usage: dict[str, int]) -> int:
    """Sum all token fields from a usage dict.

    Matches Claude Code's ``getTokenCountFromUsage()``:
    ``input_tokens + cache_creation + cache_read + output_tokens``

    Args:
        usage: Usage dict from ``get_token_usage()``.

    Returns:
        Total token count.
    """
    return (
        usage.get("input_tokens", 0)
        + usage.get("cache_creation_input_tokens", 0)
        + usage.get("cache_read_input_tokens", 0)
        + usage.get("output_tokens", 0)
    )


def token_count_with_estimation(
    messages: list[AnyMessage],
    *,
    pad: bool = True,
) -> int:
    """Hybrid token count: real API usage + heuristic for new messages.

    Walks ``messages`` backwards to find the last ``AIMessage`` that carries
    real API token usage. Uses that as the anchor, then adds a heuristic
    estimate only for messages that came after it.

    Falls back to pure heuristic (``estimate_tokens``) when no API response
    with usage data is found.

    This is the **primary entry point** for all compaction threshold decisions.

    Args:
        messages: LangChain messages (conversation history).
        pad: If True, apply the conservative 4/3 padding to the heuristic
             portion only (the real count is already exact).

    Returns:
        Estimated total token count.
    """
    # Walk backwards to find the last message with real usage
    for i in range(len(messages) - 1, -1, -1):
        usage = get_token_usage(messages[i])
        if usage is not None:
            real_count = get_token_count_from_usage(usage)
            # Estimate only the messages after this one
            tail = messages[i + 1:]
            if tail:
                heuristic = sum(estimate_message_tokens(m) for m in tail)
                if pad:
                    heuristic = int(heuristic * CONSERVATIVE_PAD)
                return real_count + heuristic
            return real_count

    # No API response found — pure heuristic fallback
    logger.debug("No API usage found in %d messages, using pure heuristic", len(messages))
    return estimate_tokens(messages, pad=pad)


# ---------------------------------------------------------------------------
# Budget enforcement
# ---------------------------------------------------------------------------

def enforce_tool_result_budget(
    content: str,
    *,
    per_tool_chars: int = 50_000,
) -> tuple[str, bool]:
    """Truncate a tool result string to fit within budget.

    Args:
        content: The tool result text.
        per_tool_chars: Maximum characters allowed.

    Returns:
        Tuple of (possibly truncated content, was_truncated).
    """
    if len(content) <= per_tool_chars:
        return content, False

    # Keep beginning and end for context
    head = per_tool_chars * 4 // 5
    tail = per_tool_chars // 5
    truncated = (
        content[:head]
        + f"\n\n... [{len(content) - per_tool_chars:,} characters truncated] ...\n\n"
        + content[-tail:]
    )
    return truncated, True


def enforce_message_budget(
    messages: list[AnyMessage],
    *,
    per_message_chars: int = 200_000,
    per_tool_chars: int = 50_000,
) -> list[AnyMessage]:
    """Enforce per-tool and per-message character budgets on ToolMessages.

    Modifies ToolMessage content in-place when it exceeds budgets.
    Returns a new list (original messages are not mutated).

    Args:
        messages: Messages to process.
        per_message_chars: Aggregate character limit per logical user turn.
        per_tool_chars: Character limit per individual tool result.

    Returns:
        New message list with budgets enforced.
    """
    result = []
    # Track aggregate chars per logical user turn (consecutive ToolMessages)
    turn_chars = 0
    turn_start = -1

    for i, msg in enumerate(messages):
        if isinstance(msg, ToolMessage) and isinstance(msg.content, str):
            # Per-tool budget
            truncated, was_truncated = enforce_tool_result_budget(
                msg.content, per_tool_chars=per_tool_chars,
            )
            if was_truncated:
                msg = msg.model_copy()
                msg.content = truncated

            # Per-message aggregate budget
            if turn_start == -1:
                turn_start = len(result)
                turn_chars = 0
            turn_chars += len(msg.content)

            if turn_chars > per_message_chars:
                overflow = turn_chars - per_message_chars
                # Truncate this message's content to fit the budget
                allowed = max(0, len(msg.content) - overflow)
                if allowed < len(msg.content):
                    msg = msg.model_copy() if not was_truncated else msg
                    msg.content = msg.content[:allowed] + f"\n\n... [aggregate budget exceeded] ..."
                    turn_chars = per_message_chars
        else:
            # Reset turn tracking on non-ToolMessage
            turn_chars = 0
            turn_start = -1

        result.append(msg)
    return result
