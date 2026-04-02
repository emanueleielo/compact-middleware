"""Full and partial compaction — LLM-based summarization.

Implements the core compaction logic:
- **Full compaction**: summarizes the entire conversation using the 9-section prompt
- **Partial compaction**: summarizes either the prefix or suffix of the conversation

Both paths use the ``<analysis>`` scratchpad for quality, then strip it
from the final summary via ``format_compact_summary()``.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, SystemMessage

from compact_middleware.config import CompactionConfig, ContextSize
from compact_middleware.prompts import (
    format_compact_summary,
    get_compact_prompt,
    get_compact_user_summary_message,
    get_partial_compact_prompt,
)

if TYPE_CHECKING:
    from langchain.chat_models import BaseChatModel
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary generation
# ---------------------------------------------------------------------------

def _build_summarization_messages(
    messages_to_summarize: list[AnyMessage],
    prompt: str,
) -> list[AnyMessage]:
    """Build the message list for the summarization LLM call.

    The prompt is injected as a SystemMessage, and the messages to summarize
    follow as the conversation history.

    Args:
        messages_to_summarize: Messages that need to be summarized.
        prompt: The full compaction prompt (with no-tools preamble/trailer).

    Returns:
        Message list for the summarization model call.
    """
    return [
        SystemMessage(content=prompt),
        HumanMessage(content="Please summarize the following conversation:"),
        *messages_to_summarize,
        HumanMessage(content="Now provide your summary following the structure above."),
    ]


def generate_summary(
    model: BaseChatModel,
    messages_to_summarize: list[AnyMessage],
    custom_instructions: str | None = None,
) -> str:
    """Generate a full conversation summary using the 9-section prompt.

    Args:
        model: LLM to use for summarization.
        messages_to_summarize: Messages to summarize.
        custom_instructions: Optional extra instructions.

    Returns:
        Raw summary string (including ``<analysis>`` and ``<summary>`` tags).
    """
    prompt = get_compact_prompt(custom_instructions)
    llm_messages = _build_summarization_messages(messages_to_summarize, prompt)

    response = model.invoke(llm_messages)
    return _extract_text(response)


async def agenerate_summary(
    model: BaseChatModel,
    messages_to_summarize: list[AnyMessage],
    custom_instructions: str | None = None,
) -> str:
    """Async version of ``generate_summary``."""
    prompt = get_compact_prompt(custom_instructions)
    llm_messages = _build_summarization_messages(messages_to_summarize, prompt)

    response = await model.ainvoke(llm_messages)
    return _extract_text(response)


def generate_partial_summary(
    model: BaseChatModel,
    messages_to_summarize: list[AnyMessage],
    direction: str = "from",
    custom_instructions: str | None = None,
) -> str:
    """Generate a partial summary (for partial compaction).

    Args:
        model: LLM to use.
        messages_to_summarize: The subset of messages to summarize.
        direction: ``'from'`` = summarize recent messages after retained prefix.
                   ``'up_to'`` = summarize prefix before retained recent messages.
        custom_instructions: Optional extra instructions.

    Returns:
        Raw summary string.
    """
    prompt = get_partial_compact_prompt(custom_instructions, direction)
    llm_messages = _build_summarization_messages(messages_to_summarize, prompt)

    response = model.invoke(llm_messages)
    return _extract_text(response)


async def agenerate_partial_summary(
    model: BaseChatModel,
    messages_to_summarize: list[AnyMessage],
    direction: str = "from",
    custom_instructions: str | None = None,
) -> str:
    """Async version of ``generate_partial_summary``."""
    prompt = get_partial_compact_prompt(custom_instructions, direction)
    llm_messages = _build_summarization_messages(messages_to_summarize, prompt)

    response = await model.ainvoke(llm_messages)
    return _extract_text(response)


def _extract_text(response: BaseMessage) -> str:
    """Extract text content from an LLM response message."""
    content = response.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(parts)
    return str(content)


# ---------------------------------------------------------------------------
# Full compaction
# ---------------------------------------------------------------------------

def compact_conversation(
    messages: list[AnyMessage],
    model: BaseChatModel,
    config: CompactionConfig,
    *,
    cutoff_index: int,
) -> tuple[list[AnyMessage], str, str | None]:
    """Perform full conversation compaction.

    Summarizes messages before ``cutoff_index`` and builds the continuation
    message. Does NOT handle backend offloading (that's the middleware's job).

    Args:
        messages: Effective message list.
        model: LLM for summarization.
        config: Compaction configuration.
        cutoff_index: Index separating messages to summarize from preserved.

    Returns:
        Tuple of:
        - New message list (summary + preserved messages)
        - Raw summary string
        - Formatted summary (with analysis stripped)
    """
    to_summarize = messages[:cutoff_index]
    preserved = messages[cutoff_index:]

    summary = generate_summary(model, to_summarize, config.custom_instructions)
    formatted = format_compact_summary(summary)

    # Build continuation message
    user_msg_content = get_compact_user_summary_message(
        summary,
        suppress_follow_up_questions=config.suppress_follow_up_questions,
        recent_messages_preserved=bool(preserved),
    )

    summary_message = HumanMessage(
        content=user_msg_content,
        additional_kwargs={"lc_source": "compaction"},
    )

    new_messages: list[AnyMessage] = [summary_message, *preserved]
    return new_messages, summary, formatted


async def acompact_conversation(
    messages: list[AnyMessage],
    model: BaseChatModel,
    config: CompactionConfig,
    *,
    cutoff_index: int,
) -> tuple[list[AnyMessage], str, str | None]:
    """Async version of ``compact_conversation``."""
    to_summarize = messages[:cutoff_index]
    preserved = messages[cutoff_index:]

    summary = await agenerate_summary(model, to_summarize, config.custom_instructions)
    formatted = format_compact_summary(summary)

    user_msg_content = get_compact_user_summary_message(
        summary,
        suppress_follow_up_questions=config.suppress_follow_up_questions,
        recent_messages_preserved=bool(preserved),
    )

    summary_message = HumanMessage(
        content=user_msg_content,
        additional_kwargs={"lc_source": "compaction"},
    )

    new_messages: list[AnyMessage] = [summary_message, *preserved]
    return new_messages, summary, formatted


# ---------------------------------------------------------------------------
# Partial compaction
# ---------------------------------------------------------------------------

def partial_compact_conversation(
    messages: list[AnyMessage],
    model: BaseChatModel,
    config: CompactionConfig,
    *,
    cutoff_index: int,
    direction: str = "from",
) -> tuple[list[AnyMessage], str, str | None]:
    """Perform partial conversation compaction.

    For ``direction='from'``: summarize messages[cutoff_index:] (recent),
    keep messages[:cutoff_index] (earlier) intact.

    For ``direction='up_to'``: summarize messages[:cutoff_index] (earlier),
    keep messages[cutoff_index:] (recent) intact.

    Args:
        messages: Effective message list.
        model: LLM for summarization.
        config: Compaction configuration.
        cutoff_index: Split point.
        direction: Which part to summarize.

    Returns:
        Tuple of (new messages, raw summary, formatted summary).
    """
    if direction == "up_to":
        to_summarize = messages[:cutoff_index]
        preserved = messages[cutoff_index:]
    else:
        preserved = messages[:cutoff_index]
        to_summarize = messages[cutoff_index:]

    summary = generate_partial_summary(
        model, to_summarize, direction, config.custom_instructions,
    )
    formatted = format_compact_summary(summary)

    user_msg_content = get_compact_user_summary_message(
        summary,
        suppress_follow_up_questions=config.suppress_follow_up_questions,
        recent_messages_preserved=bool(preserved),
    )

    summary_message = HumanMessage(
        content=user_msg_content,
        additional_kwargs={"lc_source": "compaction"},
    )

    if direction == "up_to":
        new_messages: list[AnyMessage] = [summary_message, *preserved]
    else:
        new_messages = [*preserved, summary_message]

    return new_messages, summary, formatted


async def apartial_compact_conversation(
    messages: list[AnyMessage],
    model: BaseChatModel,
    config: CompactionConfig,
    *,
    cutoff_index: int,
    direction: str = "from",
) -> tuple[list[AnyMessage], str, str | None]:
    """Async version of ``partial_compact_conversation``."""
    if direction == "up_to":
        to_summarize = messages[:cutoff_index]
        preserved = messages[cutoff_index:]
    else:
        preserved = messages[:cutoff_index]
        to_summarize = messages[cutoff_index:]

    summary = await agenerate_partial_summary(
        model, to_summarize, direction, config.custom_instructions,
    )
    formatted = format_compact_summary(summary)

    user_msg_content = get_compact_user_summary_message(
        summary,
        suppress_follow_up_questions=config.suppress_follow_up_questions,
        recent_messages_preserved=bool(preserved),
    )

    summary_message = HumanMessage(
        content=user_msg_content,
        additional_kwargs={"lc_source": "compaction"},
    )

    if direction == "up_to":
        new_messages: list[AnyMessage] = [summary_message, *preserved]
    else:
        new_messages = [*preserved, summary_message]

    return new_messages, summary, formatted
