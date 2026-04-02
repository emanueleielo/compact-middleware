"""Argument truncation and tool result budget enforcement.

Extends DeepAgents' existing argument truncation (which only targets
``write_file``/``edit_file``) to ALL tools, and adds Claude Code's per-tool
and per-message character budgets.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, SystemMessage
from langchain_core.tools import BaseTool

from compact_middleware.config import CompactionConfig, ContextSize, TruncateArgsConfig
from compact_middleware.tokens import token_count_with_estimation

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument truncation
# ---------------------------------------------------------------------------

def _truncate_tool_call(
    tool_call: dict[str, Any],
    max_length: int,
    truncation_text: str,
    truncate_all_tools: bool,
) -> tuple[dict[str, Any], bool]:
    """Truncate large arguments in a single tool call.

    Args:
        tool_call: Tool call dict with 'name', 'args', 'id' keys.
        max_length: Max character length per argument value.
        truncation_text: Suffix for truncated values.
        truncate_all_tools: If True, truncate args for all tools.

    Returns:
        Tuple of (possibly modified tool_call, was_modified).
    """
    # In original DeepAgents, only write_file and edit_file are truncated.
    # We extend this to all tools when truncate_all_tools is True.
    if not truncate_all_tools:
        if tool_call.get("name") not in ("write_file", "edit_file"):
            return tool_call, False

    args = tool_call.get("args", {})
    truncated_args = {}
    modified = False

    for key, value in args.items():
        if isinstance(value, str) and len(value) > max_length:
            truncated_args[key] = value[:max_length] + truncation_text
            modified = True
        else:
            truncated_args[key] = value

    if modified:
        return {**tool_call, "args": truncated_args}, True
    return tool_call, False


def _should_truncate_args(
    messages: list[AnyMessage],
    total_tokens: int,
    trigger: ContextSize | None,
    max_input_tokens: int | None,
) -> bool:
    """Check if argument truncation should be triggered.

    Args:
        messages: Current message list.
        total_tokens: Current total token count.
        trigger: Trigger threshold, or None to disable.
        max_input_tokens: Model's max input tokens (for fraction triggers).

    Returns:
        True if truncation should fire.
    """
    if trigger is None:
        return False

    trigger_type, trigger_value = trigger

    if trigger_type == "messages":
        return len(messages) >= trigger_value
    if trigger_type == "tokens":
        return total_tokens >= trigger_value
    if trigger_type == "fraction":
        if max_input_tokens is None:
            return False
        threshold = int(max_input_tokens * trigger_value)
        return total_tokens >= max(1, threshold)

    return False


def _determine_truncate_cutoff(
    messages: list[AnyMessage],
    keep: ContextSize,
    max_input_tokens: int | None,
) -> int:
    """Determine cutoff index: messages before this get truncated.

    Args:
        messages: Message list.
        keep: How many recent messages to keep intact.
        max_input_tokens: For fraction-based keep.

    Returns:
        Index; messages[i < cutoff] may be truncated.
    """
    keep_type, keep_value = keep

    if keep_type == "messages":
        n = int(keep_value)
        if len(messages) <= n:
            return len(messages)
        return len(messages) - n

    if keep_type == "fraction":
        if max_input_tokens is None:
            fallback = 20
            if len(messages) <= fallback:
                return len(messages)
            return len(messages) - fallback
        target = int(max_input_tokens * keep_value)
        tokens_kept = 0
        for i in range(len(messages) - 1, -1, -1):
            from compact_middleware.tokens import estimate_message_tokens
            msg_tokens = estimate_message_tokens(messages[i])
            if tokens_kept + msg_tokens > target:
                return i + 1
            tokens_kept += msg_tokens
        return 0

    if keep_type == "tokens":
        target = int(keep_value)
        tokens_kept = 0
        for i in range(len(messages) - 1, -1, -1):
            from compact_middleware.tokens import estimate_message_tokens
            msg_tokens = estimate_message_tokens(messages[i])
            if tokens_kept + msg_tokens > target:
                return i + 1
            tokens_kept += msg_tokens
        return 0

    return len(messages)


def truncate_args(
    messages: list[AnyMessage],
    system_message: SystemMessage | None,
    tools: list[BaseTool | dict[str, Any]] | None,
    config: TruncateArgsConfig,
    max_input_tokens: int | None = None,
) -> tuple[list[AnyMessage], bool]:
    """Truncate large tool call arguments in old messages.

    Args:
        messages: Messages to process.
        system_message: System message (for token counting).
        tools: Tools (for token counting).
        config: Truncation configuration.
        max_input_tokens: Model's max input tokens (for fraction triggers).

    Returns:
        Tuple of (possibly truncated messages, was_modified).
    """
    # Estimate total tokens
    counted = [system_message, *messages] if system_message is not None else list(messages)
    total_tokens = token_count_with_estimation(counted, pad=False)

    if not _should_truncate_args(messages, total_tokens, config.trigger, max_input_tokens):
        return messages, False

    cutoff = _determine_truncate_cutoff(messages, config.keep, max_input_tokens)
    if cutoff >= len(messages):
        return messages, False

    result: list[AnyMessage] = []
    modified = False

    for i, msg in enumerate(messages):
        if i < cutoff and isinstance(msg, AIMessage) and msg.tool_calls:
            truncated_calls = []
            msg_modified = False

            for tc in msg.tool_calls:
                new_tc, was_modified = _truncate_tool_call(
                    tc,
                    config.max_length,
                    config.truncation_text,
                    config.truncate_all_tools,
                )
                if was_modified:
                    msg_modified = True
                truncated_calls.append(new_tc)

            if msg_modified:
                new_msg = msg.model_copy()
                new_msg.tool_calls = truncated_calls
                result.append(new_msg)
                modified = True
            else:
                result.append(msg)
        else:
            result.append(msg)

    return result, modified
