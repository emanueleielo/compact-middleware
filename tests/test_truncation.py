"""Tests for argument truncation."""

from langchain_core.messages import AIMessage, HumanMessage

from compact_middleware.config import TruncateArgsConfig
from compact_middleware.truncation import truncate_args


def test_truncation_respects_max_length() -> None:
    """Regression: was truncating to 20 chars instead of max_length."""
    config = TruncateArgsConfig(
        trigger=("messages", 1),
        keep=("messages", 0),
        max_length=500,
        truncate_all_tools=True,
    )
    messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "write_file",
                "args": {"content": "x" * 1000, "path": "/foo.py"},
                "id": "tc1",
            }],
        ),
    ]
    result, modified = truncate_args(messages, None, None, config)
    assert modified
    ai = result[0]
    truncated_content = ai.tool_calls[0]["args"]["content"]
    # Should be ~500 + truncation suffix, NOT 20 + suffix
    assert len(truncated_content) > 100
    assert truncated_content.startswith("x" * 500)


def test_truncation_skips_short_args() -> None:
    config = TruncateArgsConfig(
        trigger=("messages", 1),
        keep=("messages", 0),
        max_length=2000,
        truncate_all_tools=True,
    )
    messages = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "read_file",
                "args": {"path": "/short.py"},
                "id": "tc1",
            }],
        ),
    ]
    result, modified = truncate_args(messages, None, None, config)
    assert not modified


def test_truncation_keeps_recent_messages() -> None:
    config = TruncateArgsConfig(
        trigger=("messages", 1),
        keep=("messages", 2),
        max_length=100,
        truncate_all_tools=True,
    )
    messages = [
        HumanMessage(content="old"),
        AIMessage(
            content="",
            tool_calls=[{"name": "write_file", "args": {"content": "x" * 500}, "id": "tc1"}],
        ),
    ]
    # Only 2 messages, keep=2, so nothing should be truncated
    result, modified = truncate_args(messages, None, None, config)
    assert not modified
