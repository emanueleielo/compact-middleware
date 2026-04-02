"""Post-compaction context restoration.

After compaction strips old messages, critical context is lost. This module
restores it by appending to the summary message:

- **File restoration**: re-reads the top N most recently accessed files
- **Plan restoration**: re-attaches the active plan state
- **Skill restoration**: re-attaches active skill descriptions

Ports Claude Code's post-compact restoration from ``compact.ts`` lines 1415–1560.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage

from compact_middleware.config import RestorationConfig

if TYPE_CHECKING:
    from deepagents.backends.protocol import BackendProtocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# File restoration
# ---------------------------------------------------------------------------

def _extract_recent_file_reads(
    messages: list[AnyMessage],
    max_files: int,
) -> list[str]:
    """Extract paths of the most recently read files from the conversation.

    Scans AIMessages backwards for ``read_file`` tool calls and collects
    unique file paths up to ``max_files``.

    Args:
        messages: Full message history (before compaction).
        max_files: Maximum number of files to collect.

    Returns:
        List of file paths, most recent first.
    """
    seen: set[str] = set()
    paths: list[str] = []

    for msg in reversed(messages):
        if len(paths) >= max_files:
            break
        if not isinstance(msg, AIMessage) or not msg.tool_calls:
            continue
        for tc in msg.tool_calls:
            if tc.get("name") == "read_file":
                path = tc.get("args", {}).get("file_path", "")
                if not path:
                    path = tc.get("args", {}).get("path", "")
                if path and path not in seen:
                    seen.add(path)
                    paths.append(path)
                    if len(paths) >= max_files:
                        break

    return paths


def restore_files(
    backend: BackendProtocol,
    messages_before_compaction: list[AnyMessage],
    config: RestorationConfig,
) -> str:
    """Read recent files and build a restoration section.

    Args:
        backend: Backend to read files from.
        messages_before_compaction: Full history before compaction.
        config: Restoration configuration.

    Returns:
        A string to append to the summary message, or empty string.
    """
    if not config.enabled:
        return ""

    paths = _extract_recent_file_reads(messages_before_compaction, config.max_files)
    if not paths:
        return ""

    sections: list[str] = []
    total_chars = 0

    for path in paths:
        if total_chars >= config.file_budget_chars:
            break
        try:
            result = backend.read(path)
            if result is None or getattr(result, "error", None):
                continue
            content = getattr(result, "content", None)
            if content is None:
                content = str(result)
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            # Truncate per-file
            if len(content) > config.per_file_chars:
                content = content[: config.per_file_chars] + "\n... [truncated]"
            total_chars += len(content)
            sections.append(f"### {path}\n```\n{content}\n```")
        except Exception:
            logger.debug("Failed to read %s for restoration", path, exc_info=True)
            continue

    if not sections:
        return ""

    return "\n\n## Recently Read Files (restored after compaction)\n\n" + "\n\n".join(sections)


async def arestore_files(
    backend: BackendProtocol,
    messages_before_compaction: list[AnyMessage],
    config: RestorationConfig,
) -> str:
    """Async version of ``restore_files``."""
    if not config.enabled:
        return ""

    paths = _extract_recent_file_reads(messages_before_compaction, config.max_files)
    if not paths:
        return ""

    sections: list[str] = []
    total_chars = 0

    for path in paths:
        if total_chars >= config.file_budget_chars:
            break
        try:
            result = await backend.aread(path)
            if result is None or getattr(result, "error", None):
                continue
            content = getattr(result, "content", None)
            if content is None:
                content = str(result)
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            if len(content) > config.per_file_chars:
                content = content[: config.per_file_chars] + "\n... [truncated]"
            total_chars += len(content)
            sections.append(f"### {path}\n```\n{content}\n```")
        except Exception:
            logger.debug("Failed to read %s for restoration", path, exc_info=True)
            continue

    if not sections:
        return ""

    return "\n\n## Recently Read Files (restored after compaction)\n\n" + "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Plan restoration
# ---------------------------------------------------------------------------

def restore_plan(state: dict[str, Any]) -> str:
    """Extract and format active plan from agent state.

    Looks for plan data in common state keys.

    Args:
        state: Agent state dictionary.

    Returns:
        Plan restoration section, or empty string.
    """
    # Check common plan state keys
    plan = state.get("_plan") or state.get("plan") or state.get("_active_plan")
    if not plan:
        return ""

    if isinstance(plan, str):
        plan_text = plan
    elif isinstance(plan, dict):
        plan_text = _format_plan_dict(plan)
    elif isinstance(plan, list):
        plan_text = "\n".join(f"- {item}" for item in plan)
    else:
        return ""

    if not plan_text.strip():
        return ""

    return f"\n\n## Active Plan (restored after compaction)\n\n{plan_text}"


def _format_plan_dict(plan: dict[str, Any]) -> str:
    """Format a plan dictionary into readable text."""
    parts = []
    if "title" in plan:
        parts.append(f"**{plan['title']}**")
    if "steps" in plan:
        for i, step in enumerate(plan["steps"], 1):
            if isinstance(step, str):
                parts.append(f"{i}. {step}")
            elif isinstance(step, dict):
                status = step.get("status", "pending")
                desc = step.get("description", step.get("text", ""))
                marker = "[x]" if status == "completed" else "[ ]"
                parts.append(f"{i}. {marker} {desc}")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Combined restoration
# ---------------------------------------------------------------------------

def build_restoration_context(
    backend: BackendProtocol | None,
    messages_before_compaction: list[AnyMessage],
    state: dict[str, Any],
    config: RestorationConfig,
) -> str:
    """Build the full restoration context to append to the summary.

    Args:
        backend: Backend for file reads (may be None).
        messages_before_compaction: Full history.
        state: Agent state.
        config: Restoration config.

    Returns:
        Combined restoration string.
    """
    parts: list[str] = []

    if backend is not None:
        file_section = restore_files(backend, messages_before_compaction, config)
        if file_section:
            parts.append(file_section)

    plan_section = restore_plan(state)
    if plan_section:
        parts.append(plan_section)

    return "".join(parts)


async def abuild_restoration_context(
    backend: BackendProtocol | None,
    messages_before_compaction: list[AnyMessage],
    state: dict[str, Any],
    config: RestorationConfig,
) -> str:
    """Async version of ``build_restoration_context``."""
    parts: list[str] = []

    if backend is not None:
        file_section = await arestore_files(backend, messages_before_compaction, config)
        if file_section:
            parts.append(file_section)

    plan_section = restore_plan(state)
    if plan_section:
        parts.append(plan_section)

    return "".join(parts)
