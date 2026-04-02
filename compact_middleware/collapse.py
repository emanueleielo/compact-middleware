"""Message collapsing — read/search grouping and duplicate detection.

Ports Claude Code's ``collapseReadSearch.ts`` and ``contextAnalysis.ts``.

When consecutive tool calls are all reads or searches, they can be collapsed
into a single summary message (e.g., "Searched 3 files, Read 5 files") to
reduce token usage without losing semantic information.

Also detects duplicate file reads so the model can avoid re-reading.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from compact_middleware.config import CollapseConfig
from compact_middleware.state import CollapseEvent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool classification
# ---------------------------------------------------------------------------

_READ_TOOLS = {"read_file"}
_SEARCH_TOOLS = {"grep", "glob", "web_search"}
_COLLAPSE_TOOLS = _READ_TOOLS | _SEARCH_TOOLS


def _is_collapsible_tool(tool_name: str, collapse_tools: set[str]) -> bool:
    """Check if a tool is eligible for collapsing."""
    return tool_name in collapse_tools


# ---------------------------------------------------------------------------
# Context analysis (duplicate detection)
# ---------------------------------------------------------------------------

def analyze_context(messages: list[AnyMessage]) -> dict[str, Any]:
    """Analyze message context for duplicate reads and search patterns.

    Scans through all messages and detects:
    - Files that were read multiple times
    - Search queries that were run multiple times
    - Tool call patterns

    Args:
        messages: Full message history to analyze.

    Returns:
        Dictionary with analysis results:
        - ``duplicate_reads``: dict of file_path -> count (only where count > 1)
        - ``duplicate_searches``: dict of query -> count (only where count > 1)
        - ``tool_call_counts``: Counter of tool_name -> total invocations
        - ``total_tool_calls``: total number of tool calls
    """
    read_files: Counter[str] = Counter()
    search_queries: Counter[str] = Counter()
    tool_counts: Counter[str] = Counter()

    for msg in messages:
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue

        for tc in msg.tool_calls:
            name = tc.get("name", "")
            args = tc.get("args", {})
            tool_counts[name] += 1

            if name == "read_file":
                path = args.get("file_path", args.get("path", ""))
                if path:
                    read_files[path] += 1

            elif name in ("grep", "glob"):
                pattern = args.get("pattern", "")
                if pattern:
                    search_queries[f"{name}:{pattern}"] += 1

            elif name == "web_search":
                query = args.get("query", "")
                if query:
                    search_queries[f"web_search:{query}"] += 1

    return {
        "duplicate_reads": {k: v for k, v in read_files.items() if v > 1},
        "duplicate_searches": {k: v for k, v in search_queries.items() if v > 1},
        "tool_call_counts": dict(tool_counts),
        "total_tool_calls": sum(tool_counts.values()),
    }


# ---------------------------------------------------------------------------
# Group detection
# ---------------------------------------------------------------------------

def _find_collapsible_groups(
    messages: list[AnyMessage],
    collapse_tools: set[str],
    min_group_size: int,
) -> list[tuple[int, int]]:
    """Find ranges of consecutive collapsible tool call/result pairs.

    A "group" is a sequence of (AIMessage with single tool_call, ToolMessage)
    pairs where all tool calls are to collapsible tools.

    Args:
        messages: Message list to scan.
        collapse_tools: Set of tool names eligible for collapsing.
        min_group_size: Minimum number of pairs to form a group.

    Returns:
        List of (start_index, end_index) tuples (exclusive end).
    """
    groups: list[tuple[int, int]] = []
    i = 0
    n = len(messages)

    while i < n - 1:
        # Look for start of a group: AIMessage with a single collapsible tool_call
        if (
            isinstance(messages[i], AIMessage)
            and messages[i].tool_calls
            and len(messages[i].tool_calls) == 1
            and _is_collapsible_tool(messages[i].tool_calls[0].get("name", ""), collapse_tools)
            and i + 1 < n
            and isinstance(messages[i + 1], ToolMessage)
        ):
            group_start = i
            group_count = 1
            j = i + 2

            # Extend the group
            while j < n - 1:
                if (
                    isinstance(messages[j], AIMessage)
                    and messages[j].tool_calls
                    and len(messages[j].tool_calls) == 1
                    and _is_collapsible_tool(messages[j].tool_calls[0].get("name", ""), collapse_tools)
                    and j + 1 < n
                    and isinstance(messages[j + 1], ToolMessage)
                ):
                    group_count += 1
                    j += 2
                else:
                    break

            if group_count >= min_group_size:
                groups.append((group_start, j))

            i = j
        else:
            i += 1

    return groups


def _build_collapse_badge(messages: list[AnyMessage], start: int, end: int) -> str:
    """Build a badge string summarizing a collapsed group.

    Example: "Searched 3 files, Read 5 files"
    """
    counts: Counter[str] = Counter()
    for k in range(start, end, 2):
        if isinstance(messages[k], AIMessage) and messages[k].tool_calls:
            name = messages[k].tool_calls[0].get("name", "unknown")
            if name in _READ_TOOLS:
                counts["Read"] += 1
            elif name in _SEARCH_TOOLS:
                counts["Searched"] += 1
            else:
                counts[name] += 1

    parts = []
    for action, count in counts.items():
        parts.append(f"{action} {count}")
    return ", ".join(parts) if parts else "Collapsed tools"


# ---------------------------------------------------------------------------
# Main collapse function
# ---------------------------------------------------------------------------

def collapse_messages(
    messages: list[AnyMessage],
    config: CollapseConfig,
) -> tuple[list[AnyMessage], CollapseEvent | None]:
    """Collapse consecutive read/search tool groups into summary messages.

    Each group of consecutive collapsible tool pairs (AIMessage + ToolMessage)
    is replaced by a single HumanMessage with a badge summarizing the group.

    Args:
        messages: Current message history.
        config: Collapse configuration.

    Returns:
        Tuple of (possibly collapsed messages, event or None).
    """
    if not config.enabled:
        return messages, None

    groups = _find_collapsible_groups(messages, config.collapse_tools, config.min_group_size)

    if not groups:
        return messages, None

    result: list[AnyMessage] = []
    total_collapsed = 0
    prev_end = 0

    for start, end in groups:
        # Add messages before this group
        result.extend(messages[prev_end:start])

        # Build collapsed summary
        badge = _build_collapse_badge(messages, start, end)
        group_size = (end - start) // 2

        # Keep just the last tool call and result from the group intact
        # (the model may need the most recent result), collapse the rest
        last_ai = messages[end - 2]
        last_tool = messages[end - 1]

        # Add a summary message for the collapsed portion
        if group_size > 1:
            result.append(HumanMessage(
                content=f"[{badge} — {group_size} tool calls collapsed to save context]",
                additional_kwargs={"lc_source": "compaction_collapse"},
            ))
            # Keep the last pair
            result.append(last_ai)
            result.append(last_tool)
            total_collapsed += group_size - 1
        else:
            # Group too small after filtering, keep as-is
            result.extend(messages[start:end])

        prev_end = end

    # Add remaining messages after last group
    result.extend(messages[prev_end:])

    if total_collapsed == 0:
        return messages, None

    event = CollapseEvent(
        groups_collapsed=len(groups),
        messages_reduced=total_collapsed * 2,  # Each collapsed pair = 2 messages
    )

    logger.info(
        "Collapsed %d groups (%d tool call pairs reduced)",
        len(groups),
        total_collapsed,
    )

    return result, event
