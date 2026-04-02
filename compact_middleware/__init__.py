"""Compaction Middleware for DeepAgents.

Combines Claude Code's advanced compaction techniques with the composable
middleware architecture of DeepAgents. Drop-in replacement for the built-in
``SummarizationMiddleware`` with:

- 9-section structured summarization prompts
- Microcompaction (time-based tool result clearing)
- Message collapsing (read/search grouping, duplicate detection)
- Multi-level decision cascade (collapse → truncate → micro → full/partial)
- Post-compaction file/plan/skill restoration
- Circuit breaker and PTL error recovery
- Partial compaction (earlier/later direction)
- Type-aware token estimation

Usage::

    from compact_middleware import CompactionMiddleware, CompactionToolMiddleware

    mw = CompactionMiddleware(model=model, backend=backend)
    tool_mw = CompactionToolMiddleware(mw)
    agent = create_deep_agent(middleware=[mw, tool_mw])
"""

from compact_middleware.config import CompactionConfig
from compact_middleware.middleware import (
    CompactionMiddleware,
    CompactionToolMiddleware,
    create_compaction_middleware,
    create_compaction_tool_middleware,
)
from compact_middleware.state import CompactionEvent, CompactionState

__all__ = [
    "CompactionConfig",
    "CompactionEvent",
    "CompactionMiddleware",
    "CompactionState",
    "CompactionToolMiddleware",
    "create_compaction_middleware",
    "create_compaction_tool_middleware",
]
