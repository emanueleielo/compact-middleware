"""Configuration for the Compaction Middleware.

All thresholds, budgets, and feature toggles in one place. Mirrors the
constants from Claude Code's ``autoCompact.ts``, ``microCompact.ts``,
``toolResultStorage.ts``, and ``compact.ts``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain.chat_models import BaseChatModel


# ---------------------------------------------------------------------------
# Context size specification (reuses DeepAgents convention)
# ---------------------------------------------------------------------------
# ("tokens", 170_000) | ("messages", 20) | ("fraction", 0.85)
ContextSize = tuple[str, int | float]


@dataclass
class TokenBudgetConfig:
    """Per-tool and per-message token budgets for result truncation."""

    per_tool_chars: int = 50_000
    """Max characters kept per single tool result (Claude Code default: 50k)."""

    per_message_chars: int = 200_000
    """Max aggregate characters per user message (Claude Code default: 200k)."""

    file_preview_chars: int = 5_000
    """Characters kept when persisting a large result to disk."""

    warning_token_threshold: int = 20_000
    """Token count remaining that triggers a warning."""

    error_token_threshold: int = 20_000
    """Token count remaining that triggers an error."""

    blocking_token_threshold: int = 3_000
    """Token count remaining that blocks further input."""


@dataclass
class MicrocompactConfig:
    """Settings for time-based microcompaction (tool result clearing)."""

    enabled: bool = True
    """Master switch for microcompaction."""

    gap_threshold_minutes: float = 60.0
    """Clear old results when gap since last assistant msg exceeds this."""

    keep_recent: int = 5
    """Always keep this many most-recent compactable tool results."""

    compactable_tools: set[str] = field(default_factory=lambda: {
        "read_file", "execute", "grep", "glob",
        "web_search", "web_fetch", "edit_file", "write_file",
    })
    """Tool names whose results can be cleared by microcompaction."""

    cleared_message: str = "[Old tool result content cleared]"
    """Replacement text for cleared tool results."""


@dataclass
class CollapseConfig:
    """Settings for read/search message collapsing."""

    enabled: bool = True
    """Master switch for message collapsing."""

    min_group_size: int = 2
    """Minimum consecutive read/search tools to form a collapsible group."""

    collapse_tools: set[str] = field(default_factory=lambda: {
        "read_file", "grep", "glob", "web_search",
    })
    """Tool names eligible for collapse grouping."""


@dataclass
class RestorationConfig:
    """Settings for post-compaction context restoration."""

    enabled: bool = True
    """Master switch for post-compaction restoration."""

    max_files: int = 5
    """Maximum number of recently-read files to restore."""

    file_budget_chars: int = 50_000
    """Total character budget for file restoration."""

    per_file_chars: int = 5_000
    """Max characters per restored file."""

    skill_budget_chars: int = 25_000
    """Total character budget for skill restoration."""

    per_skill_chars: int = 5_000
    """Max characters per restored skill."""

    restore_plans: bool = True
    """Whether to restore active plan state after compaction."""


@dataclass
class TruncateArgsConfig:
    """Settings for truncating large tool-call arguments before compaction."""

    trigger: ContextSize | None = None
    """Threshold that activates truncation. None disables it."""

    keep: ContextSize = ("messages", 20)
    """How many recent messages to leave untouched."""

    max_length: int = 2_000
    """Character limit per argument value before truncation."""

    truncation_text: str = "...(argument truncated)"
    """Suffix appended to truncated arguments."""

    truncate_all_tools: bool = True
    """If True, truncate args for ALL tools (not just write_file/edit_file)."""


@dataclass
class CompactionConfig:
    """Master configuration for the Compaction Middleware.

    Sensible defaults match Claude Code's production settings.
    """

    # --- Auto-compaction trigger ---
    trigger: ContextSize | list[ContextSize] | None = None
    """Threshold(s) that trigger auto-compaction. None = model-aware defaults."""

    keep: ContextSize = ("messages", 20)
    """Context retention policy after compaction."""

    # --- Buffer constants (from Claude Code autoCompact.ts) ---
    autocompact_buffer_tokens: int = 13_000
    """Buffer between effective context window and auto-compact threshold."""

    max_output_tokens_for_summary: int = 20_000
    """Tokens reserved for compaction summary output."""

    # --- Circuit breaker ---
    max_consecutive_failures: int = 3
    """Stop auto-compaction after this many consecutive failures."""

    # --- PTL error recovery ---
    ptl_max_retries: int = 3
    """Max retries on prompt-too-long errors."""

    # --- Sub-configs ---
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    microcompact: MicrocompactConfig = field(default_factory=MicrocompactConfig)
    collapse: CollapseConfig = field(default_factory=CollapseConfig)
    restoration: RestorationConfig = field(default_factory=RestorationConfig)
    truncate_args: TruncateArgsConfig = field(default_factory=TruncateArgsConfig)

    # --- Compaction strategy ---
    prefer_partial: bool = True
    """When True, prefer partial compaction over full when applicable."""

    custom_instructions: str | None = None
    """Additional instructions to include in the compaction prompt."""

    suppress_follow_up_questions: bool = True
    """Whether to suppress follow-up questions in the continuation prompt."""

    # --- Backend ---
    history_path_prefix: str = "/conversation_history"
    """Path prefix for storing conversation history on the backend."""


def compute_compaction_defaults(model: BaseChatModel) -> CompactionConfig:
    """Compute default compaction settings based on model profile.

    Args:
        model: A resolved chat model instance.

    Returns:
        CompactionConfig with model-aware defaults.
    """
    has_profile = (
        model.profile is not None
        and isinstance(model.profile, dict)
        and "max_input_tokens" in model.profile
        and isinstance(model.profile["max_input_tokens"], int)
    )

    if has_profile:
        return CompactionConfig(
            trigger=("fraction", 0.85),
            keep=("fraction", 0.10),
            truncate_args=TruncateArgsConfig(
                trigger=("fraction", 0.85),
                keep=("fraction", 0.10),
            ),
        )

    return CompactionConfig(
        trigger=("tokens", 170_000),
        keep=("messages", 6),
        truncate_args=TruncateArgsConfig(
            trigger=("messages", 20),
            keep=("messages", 20),
        ),
    )
