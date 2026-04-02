"""Basic usage examples for compact-middleware.

Shows zero-config, custom config, and full create_deep_agent integration.
"""

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model

from compact_middleware import (
    CompactionConfig,
    CompactionMiddleware,
    CompactionToolMiddleware,
)
from compact_middleware.config import (
    CollapseConfig,
    MicrocompactConfig,
    RestorationConfig,
    TruncateArgsConfig,
)


def zero_config() -> None:
    """Minimal setup — model-aware defaults handle everything."""

    backend = FilesystemBackend(root_dir="/data/workspace")

    mw = CompactionMiddleware(
        model="anthropic:claude-sonnet-4-6",
        backend=backend,
    )
    tool_mw = CompactionToolMiddleware(mw)

    agent = create_deep_agent(
        model="anthropic:claude-sonnet-4-6",
        system_prompt="You are a coding assistant.",
        backend=backend,
        middleware=[mw, tool_mw],
    )

    result = agent.invoke({"messages": [("human", "Refactor the auth module")]})
    print(result)


def custom_config() -> None:
    """Fine-tuned settings for aggressive compaction."""

    backend = FilesystemBackend(root_dir="/data/workspace")
    model = init_chat_model("anthropic:claude-sonnet-4-6")

    config = CompactionConfig(
        # Trigger earlier, keep less
        trigger=("fraction", 0.75),
        keep=("messages", 8),

        # Aggressive microcompaction
        microcompact=MicrocompactConfig(
            gap_threshold_minutes=30,
            keep_recent=3,
        ),

        # Tighter truncation
        truncate_args=TruncateArgsConfig(
            trigger=("fraction", 0.75),
            max_length=1_000,
        ),

        # Collapse with smaller groups
        collapse=CollapseConfig(
            min_group_size=2,
        ),

        # Restore fewer files
        restoration=RestorationConfig(
            max_files=3,
            file_budget_chars=30_000,
        ),

        # Custom summary focus
        custom_instructions=(
            "Focus on code changes and test results. "
            "Include full file paths and error messages verbatim."
        ),

        # More retries before circuit break
        max_consecutive_failures=5,
    )

    mw = CompactionMiddleware(model=model, backend=backend, config=config)
    tool_mw = CompactionToolMiddleware(mw)

    agent = create_deep_agent(
        model=model,
        system_prompt="You are a senior engineer.",
        backend=backend,
        middleware=[mw, tool_mw],
        memory=["/memory/AGENTS.md"],
        interrupt_on={"edit_file": True},
    )

    result = agent.invoke({"messages": [("human", "Add pagination to the API")]})
    print(result)


async def async_usage() -> None:
    """Async variant — offloading and summary run concurrently."""

    backend = FilesystemBackend(root_dir="/data/workspace")
    model = init_chat_model("anthropic:claude-sonnet-4-6")

    mw = CompactionMiddleware(model=model, backend=backend)

    agent = create_deep_agent(
        model=model,
        backend=backend,
        middleware=[mw, CompactionToolMiddleware(mw)],
    )

    result = await agent.ainvoke(
        {"messages": [("human", "Run the test suite and fix failures")]}
    )
    print(result)


if __name__ == "__main__":
    zero_config()
