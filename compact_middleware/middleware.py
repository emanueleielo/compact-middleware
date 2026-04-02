"""Main middleware classes for compaction.

``CompactionMiddleware`` — automatic compaction via ``wrap_model_call()``.
``CompactionToolMiddleware`` — exposes ``compact_conversation`` tool for manual use.

These follow the DeepAgents ``AgentMiddleware`` protocol and are drop-in
replacements for the built-in ``SummarizationMiddleware``.

**Compatibility note**: LangChain 1.2.x's agent factory ignores
``Command(update={...})`` returned by middleware, so this implementation
stores compaction state on the middleware *instance* (keyed by thread ID)
rather than relying on LangGraph state updates. The LLM always sees the
compacted message view; the raw state keeps growing but is never sent to
the model.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

from langchain.agents.middleware.types import AgentMiddleware, AgentState
from langchain.tools import ToolRuntime

try:
    from langchain_core.exceptions import ContextOverflowError
except ImportError:

    class ContextOverflowError(Exception):  # type: ignore[no-redef]
        pass


from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    get_buffer_string,
)
from langgraph.config import get_config
from langgraph.types import Command
from pydantic import BaseModel

from compact_middleware.compaction import (
    acompact_conversation,
    apartial_compact_conversation,
    compact_conversation,
    partial_compact_conversation,
)
from compact_middleware.config import CompactionConfig, compute_compaction_defaults
from compact_middleware.decision import (
    CompactionLevel,
    DecisionResult,
    evaluate,
    get_max_input_tokens,
)
from compact_middleware.restoration import (
    abuild_restoration_context,
    build_restoration_context,
)
from compact_middleware.state import CompactionEvent, CompactionState
from compact_middleware.tokens import estimate_tokens, token_count_with_estimation

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from langchain.agents.middleware.types import ModelRequest, ModelResponse
    from langchain.chat_models import BaseChatModel
    from langchain_core.runnables.config import RunnableConfig
    from langchain_core.tools import BaseTool
    from langgraph.runtime import Runtime

    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Instance-level compaction state (bypasses broken Command(update={}))
# ---------------------------------------------------------------------------


@dataclass
class _ThreadCompactionState:
    """Per-thread compaction state stored on the middleware instance.

    LangChain 1.2.x does not process ``Command(update={...})`` returned
    from middleware, so we track compaction state here instead of in
    LangGraph's agent state.
    """

    event: CompactionEvent | None = None
    """Last compaction event (cutoff index + summary message)."""

    failures: int = 0
    """Consecutive auto-compaction failures (circuit breaker)."""


# ---------------------------------------------------------------------------
# System prompt for the compact tool
# ---------------------------------------------------------------------------

COMPACTION_SYSTEM_PROMPT = """\
## Compact Conversation Tool `compact_conversation`

You have access to a `compact_conversation` tool. This tool refreshes your \
context window to reduce context bloat and costs.

You should use the tool when:
- The user asks to move on to a completely new task for which previous \
context is likely irrelevant.
- You have finished extracting or synthesizing a result and previous \
working context is no longer needed.
- The conversation is getting long and you want to free up context window space.
"""


class CompactConversationSchema(BaseModel):
    """Input schema for the compact_conversation tool."""


# ---------------------------------------------------------------------------
# CompactionMiddleware — automatic compaction
# ---------------------------------------------------------------------------


class CompactionMiddleware(AgentMiddleware):
    """Advanced compaction middleware with multi-level decision cascade.

    Combines Claude Code's compaction techniques with DeepAgents' composable
    middleware architecture.  Replaces the built-in ``SummarizationMiddleware``.

    **State handling**: compaction metadata (cutoff, summary, failure count)
    is stored on the middleware *instance* rather than in LangGraph state.
    This is necessary because LangChain 1.2.x ignores ``Command(update=…)``
    returned from ``wrap_model_call``.  The raw message list in LangGraph
    state keeps growing, but the middleware always reconstructs the compacted
    "effective" view before passing messages to the LLM.

    Usage::

        mw = CompactionMiddleware(model=model, backend=backend)
        agent = create_deep_agent(middleware=[mw])
    """

    state_schema = CompactionState

    def __init__(
        self,
        model: str | BaseChatModel,
        *,
        backend: BACKEND_TYPES,
        config: CompactionConfig | None = None,
    ) -> None:
        if isinstance(model, str):
            from deepagents._models import resolve_model

            model = resolve_model(model)

        self._model: BaseChatModel = model
        self._backend = backend

        if config is None:
            config = compute_compaction_defaults(model)
        self._config = config

        self._max_input_tokens = get_max_input_tokens(
            getattr(model, "profile", None)
        )

        # Per-thread compaction state (keyed by thread_id).
        # Protected by a lock for safety across concurrent async tasks.
        self._lock = threading.Lock()
        self._thread_states: dict[str, _ThreadCompactionState] = {}

        # Stable fallback ID when no thread_id is in the LangGraph config.
        # Generated once at init so every call within the same middleware
        # instance maps to the same compaction state.
        self._fallback_thread_id = f"session_{uuid.uuid4().hex[:8]}"

    @property
    def config(self) -> CompactionConfig:
        return self._config

    # --- Per-thread state helpers ---

    def _get_thread_state(self, thread_id: str) -> _ThreadCompactionState:
        with self._lock:
            if thread_id not in self._thread_states:
                self._thread_states[thread_id] = _ThreadCompactionState()
            return self._thread_states[thread_id]

    def _save_compaction_event(
        self,
        thread_id: str,
        event: CompactionEvent,
    ) -> None:
        with self._lock:
            ts = self._thread_states.setdefault(
                thread_id, _ThreadCompactionState()
            )
            ts.event = event
            ts.failures = 0  # reset on success

    def _increment_failures(self, thread_id: str) -> None:
        with self._lock:
            ts = self._thread_states.setdefault(
                thread_id, _ThreadCompactionState()
            )
            ts.failures += 1

    # --- Backend resolution ---

    def _get_backend(
        self, state: AgentState, runtime: Runtime
    ) -> BackendProtocol:
        if callable(self._backend):
            config = cast("RunnableConfig", getattr(runtime, "config", {}))
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)
        return self._backend

    # --- Thread ID / history path ---

    def _get_thread_id(self) -> str:
        try:
            config = get_config()
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id is not None:
                return str(thread_id)
        except RuntimeError:
            pass
        # Stable fallback — same ID for every call on this middleware instance.
        # If users need multi-conversation support they must pass thread_id
        # in the LangGraph configurable.
        return self._fallback_thread_id

    def _get_history_path(self, thread_id: str) -> str:
        return f"{self._config.history_path_prefix}/{thread_id}.md"

    # --- Message utils ---

    @staticmethod
    def _is_summary_message(msg: AnyMessage) -> bool:
        if not isinstance(msg, HumanMessage):
            return False
        source = msg.additional_kwargs.get("lc_source", "")
        return source in ("compaction", "summarization")

    @staticmethod
    def _apply_event_to_messages(
        messages: list[AnyMessage],
        event: CompactionEvent | None,
    ) -> list[AnyMessage]:
        """Reconstruct effective messages from state + compaction event."""
        if event is None:
            return list(messages)

        try:
            summary_msg = event["summary_message"]
            cutoff_idx = event["cutoff_index"]
        except (KeyError, TypeError) as exc:
            logger.warning("Malformed compaction event: %s", exc)
            return list(messages)

        if cutoff_idx > len(messages):
            logger.warning(
                "Compaction cutoff %d > message count %d",
                cutoff_idx,
                len(messages),
            )
            return [summary_msg]

        return [summary_msg, *messages[cutoff_idx:]]

    @staticmethod
    def _compute_state_cutoff(
        event: CompactionEvent | None,
        effective_cutoff: int,
    ) -> int:
        """Translate effective-list cutoff to absolute state index."""
        if event is None:
            return effective_cutoff
        prior_cutoff = event.get("cutoff_index")
        if not isinstance(prior_cutoff, int):
            return effective_cutoff
        return prior_cutoff + effective_cutoff - 1

    # --- Backend offloading ---

    def _offload_to_backend(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
        thread_id: str,
    ) -> str | None:
        path = self._get_history_path(thread_id)
        filtered = [m for m in messages if not self._is_summary_message(m)]

        timestamp = datetime.now(UTC).isoformat()
        new_section = (
            f"## Compacted at {timestamp}\n\n"
            f"{get_buffer_string(filtered)}\n\n"
        )

        existing_content = ""
        try:
            responses = backend.download_files([path])
            if (
                responses
                and responses[0].content is not None
                and getattr(responses[0], "error", None) is None
            ):
                raw = responses[0].content
                existing_content = (
                    raw.decode("utf-8") if isinstance(raw, bytes) else raw
                )
        except Exception:
            logger.debug("No existing history at %s", path, exc_info=True)

        combined = existing_content + new_section
        try:
            result = (
                backend.edit(path, existing_content, combined)
                if existing_content
                else backend.write(path, combined)
            )
            if result is None or getattr(result, "error", None):
                logger.warning("Failed to offload to %s", path)
                return None
        except Exception:
            logger.warning("Exception offloading to %s", path, exc_info=True)
            return None

        logger.debug("Offloaded %d messages to %s", len(filtered), path)
        return path

    async def _aoffload_to_backend(
        self,
        backend: BackendProtocol,
        messages: list[AnyMessage],
        thread_id: str,
    ) -> str | None:
        path = self._get_history_path(thread_id)
        filtered = [m for m in messages if not self._is_summary_message(m)]

        timestamp = datetime.now(UTC).isoformat()
        new_section = (
            f"## Compacted at {timestamp}\n\n"
            f"{get_buffer_string(filtered)}\n\n"
        )

        existing_content = ""
        try:
            responses = await backend.adownload_files([path])
            if (
                responses
                and responses[0].content is not None
                and getattr(responses[0], "error", None) is None
            ):
                raw = responses[0].content
                existing_content = (
                    raw.decode("utf-8") if isinstance(raw, bytes) else raw
                )
        except Exception:
            logger.debug("No existing history at %s", path, exc_info=True)

        combined = existing_content + new_section
        try:
            result = (
                await backend.aedit(path, existing_content, combined)
                if existing_content
                else await backend.awrite(path, combined)
            )
            if result is None or getattr(result, "error", None):
                logger.warning("Failed to offload to %s", path)
                return None
        except Exception:
            logger.warning("Exception offloading to %s", path, exc_info=True)
            return None

        logger.debug("Offloaded %d messages to %s", len(filtered), path)
        return path

    # --- Cutoff determination ---

    def _determine_cutoff_index(self, messages: list[AnyMessage]) -> int:
        keep_type, keep_value = self._config.keep

        if keep_type == "messages":
            n = int(keep_value)
            return max(0, len(messages) - n)

        if keep_type == "fraction":
            if self._max_input_tokens is None:
                return max(0, len(messages) - 20)
            target = int(self._max_input_tokens * keep_value)
            tokens_kept = 0
            for i in range(len(messages) - 1, -1, -1):
                from compact_middleware.tokens import estimate_message_tokens

                if tokens_kept + estimate_message_tokens(messages[i]) > target:
                    return i + 1
                tokens_kept += estimate_message_tokens(messages[i])
            return 0

        if keep_type == "tokens":
            target = int(keep_value)
            tokens_kept = 0
            for i in range(len(messages) - 1, -1, -1):
                from compact_middleware.tokens import estimate_message_tokens

                if tokens_kept + estimate_message_tokens(messages[i]) > target:
                    return i + 1
                tokens_kept += estimate_message_tokens(messages[i])
            return 0

        return len(messages)

    # --- Enrichment helpers ---

    @staticmethod
    def _enrich_summary(
        new_messages: list[AnyMessage],
        restoration: str,
        file_path: str | None,
    ) -> list[AnyMessage]:
        """Append restoration context and transcript path to the summary."""
        if not new_messages or not isinstance(new_messages[0], HumanMessage):
            return new_messages

        content = new_messages[0].content
        kwargs = new_messages[0].additional_kwargs

        if restoration:
            content = content + restoration

        if file_path and isinstance(content, str) and "transcript at:" not in content:
            content = content + f"\n\nFull transcript at: {file_path}"

        if content != new_messages[0].content:
            new_messages[0] = HumanMessage(
                content=content, additional_kwargs=kwargs
            )

        return new_messages

    # ======================================================================
    # wrap_model_call — SYNC
    # ======================================================================

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Process messages before model call with multi-level compaction.

        1. Reconstruct effective messages from instance-level compaction state
        2. Run lightweight cascade (collapse, truncate, microcompact)
        3. If still over threshold: full/partial LLM compaction
        4. On ContextOverflowError: fallback to compaction
        5. Call handler with the (possibly compacted) messages
        """
        thread_id = self._get_thread_id()
        ts = self._get_thread_state(thread_id)

        effective = self._apply_event_to_messages(request.messages, ts.event)

        decision = evaluate(
            effective,
            request.system_message,
            request.tools,
            self._config,
            self._max_input_tokens,
            ts.failures,
        )

        # No full compaction needed — try the call with lightweight fixes
        if not decision.needs_full_compaction and not decision.needs_partial_compaction:
            try:
                return handler(request.override(messages=decision.messages))
            except ContextOverflowError:
                decision = DecisionResult(
                    messages=decision.messages,
                    level=CompactionLevel.FULL,
                    tokens_before=decision.tokens_before,
                    tokens_after=decision.tokens_after,
                    needs_full_compaction=True,
                )

        return self._perform_compaction_sync(
            request, handler, decision, ts.event, thread_id
        )

    # ======================================================================
    # awrap_model_call — ASYNC
    # ======================================================================

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of wrap_model_call."""
        thread_id = self._get_thread_id()
        ts = self._get_thread_state(thread_id)

        effective = self._apply_event_to_messages(request.messages, ts.event)

        decision = evaluate(
            effective,
            request.system_message,
            request.tools,
            self._config,
            self._max_input_tokens,
            ts.failures,
        )

        if not decision.needs_full_compaction and not decision.needs_partial_compaction:
            try:
                return await handler(
                    request.override(messages=decision.messages)
                )
            except ContextOverflowError:
                decision = DecisionResult(
                    messages=decision.messages,
                    level=CompactionLevel.FULL,
                    tokens_before=decision.tokens_before,
                    tokens_after=decision.tokens_after,
                    needs_full_compaction=True,
                )

        return await self._perform_compaction_async(
            request, handler, decision, ts.event, thread_id
        )

    # ======================================================================
    # Compaction execution
    # ======================================================================

    def _perform_compaction_sync(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
        decision: DecisionResult,
        prior_event: CompactionEvent | None,
        thread_id: str,
    ) -> ModelResponse:
        messages = decision.messages
        cutoff = self._determine_cutoff_index(messages)

        if cutoff <= 0:
            return handler(request.override(messages=messages))

        to_summarize = messages[:cutoff]

        # Offload to backend
        backend = self._get_backend(request.state, request.runtime)
        file_path = self._offload_to_backend(backend, to_summarize, thread_id)
        if file_path is None:
            warnings.warn(
                "Backend offload failed — older messages will not be recoverable.",
                stacklevel=2,
            )

        # Generate summary
        try:
            if decision.needs_partial_compaction:
                new_messages, _, _ = partial_compact_conversation(
                    messages,
                    self._model,
                    self._config,
                    cutoff_index=cutoff,
                    direction="up_to",
                )
                strategy = "partial"
            else:
                new_messages, _, _ = compact_conversation(
                    messages,
                    self._model,
                    self._config,
                    cutoff_index=cutoff,
                )
                strategy = "full"
        except Exception:
            logger.exception("Compaction failed")
            self._increment_failures(thread_id)
            return handler(request.override(messages=messages))

        # Post-compaction restoration
        restoration = build_restoration_context(
            backend,
            to_summarize,
            dict(request.state),
            self._config.restoration,
        )
        new_messages = self._enrich_summary(new_messages, restoration, file_path)

        # Save compaction state on the instance
        state_cutoff = self._compute_state_cutoff(prior_event, cutoff)
        self._save_compaction_event(
            thread_id,
            CompactionEvent(
                cutoff_index=state_cutoff,
                summary_message=(
                    new_messages[0]
                    if new_messages
                    else HumanMessage(content="[compaction failed]")
                ),
                file_path=file_path,
                strategy=strategy,
                tokens_before=decision.tokens_before,
                tokens_after=estimate_tokens(new_messages),
            ),
        )

        logger.info(
            "[COMPACTION] %s — %d msgs → %d msgs (thread %s)",
            strategy,
            len(messages),
            len(new_messages),
            thread_id,
        )

        return handler(request.override(messages=new_messages))

    async def _perform_compaction_async(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
        decision: DecisionResult,
        prior_event: CompactionEvent | None,
        thread_id: str,
    ) -> ModelResponse:
        messages = decision.messages
        cutoff = self._determine_cutoff_index(messages)

        if cutoff <= 0:
            return await handler(request.override(messages=messages))

        to_summarize = messages[:cutoff]
        backend = self._get_backend(request.state, request.runtime)

        # Offload + summarize concurrently
        try:
            if decision.needs_partial_compaction:
                file_path, (new_messages, _, _) = await asyncio.gather(
                    self._aoffload_to_backend(
                        backend, to_summarize, thread_id
                    ),
                    apartial_compact_conversation(
                        messages,
                        self._model,
                        self._config,
                        cutoff_index=cutoff,
                        direction="up_to",
                    ),
                )
                strategy = "partial"
            else:
                file_path, (new_messages, _, _) = await asyncio.gather(
                    self._aoffload_to_backend(
                        backend, to_summarize, thread_id
                    ),
                    acompact_conversation(
                        messages,
                        self._model,
                        self._config,
                        cutoff_index=cutoff,
                    ),
                )
                strategy = "full"
        except Exception:
            logger.exception("Compaction failed")
            self._increment_failures(thread_id)
            return await handler(request.override(messages=messages))

        if file_path is None:
            warnings.warn(
                "Backend offload failed — older messages will not be recoverable.",
                stacklevel=2,
            )

        # Async restoration
        restoration = await abuild_restoration_context(
            backend,
            to_summarize,
            dict(request.state),
            self._config.restoration,
        )
        new_messages = self._enrich_summary(new_messages, restoration, file_path)

        state_cutoff = self._compute_state_cutoff(prior_event, cutoff)
        self._save_compaction_event(
            thread_id,
            CompactionEvent(
                cutoff_index=state_cutoff,
                summary_message=(
                    new_messages[0]
                    if new_messages
                    else HumanMessage(content="[compaction failed]")
                ),
                file_path=file_path,
                strategy=strategy,
                tokens_before=decision.tokens_before,
                tokens_after=estimate_tokens(new_messages),
            ),
        )

        logger.info(
            "[COMPACTION] %s — %d msgs → %d msgs (thread %s)",
            strategy,
            len(messages),
            len(new_messages),
            thread_id,
        )

        return await handler(request.override(messages=new_messages))


# ---------------------------------------------------------------------------
# CompactionToolMiddleware — manual compact tool
# ---------------------------------------------------------------------------


class CompactionToolMiddleware(AgentMiddleware):
    """Middleware providing a ``compact_conversation`` tool.

    Composes with a ``CompactionMiddleware`` instance and reuses its
    engine.  Manual compaction is gated by an eligibility check (~50%
    of auto-trigger threshold).

    Usage::

        mw = CompactionMiddleware(model=model, backend=backend)
        tool_mw = CompactionToolMiddleware(mw)
        agent = create_deep_agent(middleware=[mw, tool_mw])
    """

    state_schema = CompactionState

    def __init__(self, compaction: CompactionMiddleware) -> None:
        self._compaction = compaction
        self.tools: list[BaseTool] = [self._create_compact_tool()]

    def _create_compact_tool(self) -> BaseTool:
        from langchain_core.tools import StructuredTool

        mw = self

        def sync_compact(runtime: ToolRuntime) -> Command:
            return mw._run_compact(runtime)

        async def async_compact(runtime: ToolRuntime) -> Command:
            return await mw._arun_compact(runtime)

        return StructuredTool.from_function(
            name="compact_conversation",
            description=(
                "Compact the conversation by summarizing older messages "
                "into a concise summary. Use this proactively when the "
                "conversation is getting long to free up context window "
                "space. This tool takes no arguments."
            ),
            func=sync_compact,
            coroutine=async_compact,
        )

    def _is_eligible_for_compaction(
        self, messages: list[AnyMessage]
    ) -> bool:
        config = self._compaction.config
        trigger = config.trigger
        if trigger is None:
            return False

        conditions = trigger if isinstance(trigger, list) else [trigger]
        total_tokens = token_count_with_estimation(messages)
        max_input = self._compaction._max_input_tokens

        for kind, value in conditions:
            if kind == "tokens":
                if total_tokens >= int(value * 0.5):
                    return True
            elif kind == "fraction" and max_input is not None:
                if total_tokens >= int(max_input * value * 0.5):
                    return True
            elif kind == "messages":
                if len(messages) >= int(value * 0.5):
                    return True
        return False

    @staticmethod
    def _tool_message(tool_call_id: str, text: str) -> Command:
        return Command(
            update={
                "messages": [
                    ToolMessage(content=text, tool_call_id=tool_call_id)
                ],
            }
        )

    def _run_compact(self, runtime: ToolRuntime) -> Command:
        c = self._compaction
        tool_call_id = runtime.tool_call_id or ""
        thread_id = c._get_thread_id()
        ts = c._get_thread_state(thread_id)

        messages = runtime.state.get("messages", [])
        effective = c._apply_event_to_messages(messages, ts.event)

        if not self._is_eligible_for_compaction(effective):
            return self._tool_message(
                tool_call_id,
                "Nothing to compact yet — conversation is within the token budget.",
            )

        cutoff = c._determine_cutoff_index(effective)
        if cutoff == 0:
            return self._tool_message(
                tool_call_id,
                "Nothing to compact yet — conversation is within the token budget.",
            )

        try:
            to_summarize = effective[:cutoff]

            new_messages, _, _ = compact_conversation(
                effective, c._model, c._config, cutoff_index=cutoff
            )

            # Backend offload
            backend_obj = c._backend
            if callable(backend_obj):
                backend_obj = backend_obj(runtime)
            file_path = c._offload_to_backend(
                backend_obj, to_summarize, thread_id
            )

            # Restoration
            restoration = build_restoration_context(
                backend_obj,
                to_summarize,
                dict(runtime.state),
                c._config.restoration,
            )
            new_messages = c._enrich_summary(
                new_messages, restoration, file_path
            )
        except Exception as exc:
            logger.exception("compact_conversation tool failed")
            return self._tool_message(
                tool_call_id,
                f"Compaction failed: {type(exc).__name__}: {exc}",
            )

        # Save on instance state
        state_cutoff = c._compute_state_cutoff(ts.event, cutoff)
        c._save_compaction_event(
            thread_id,
            CompactionEvent(
                cutoff_index=state_cutoff,
                summary_message=new_messages[0],
                file_path=file_path,
                strategy="full",
                tokens_before=estimate_tokens(effective),
                tokens_after=estimate_tokens(new_messages),
            ),
        )

        return self._tool_message(
            tool_call_id,
            f"Conversation compacted. Summarized {len(to_summarize)} messages.",
        )

    async def _arun_compact(self, runtime: ToolRuntime) -> Command:
        c = self._compaction
        tool_call_id = runtime.tool_call_id or ""
        thread_id = c._get_thread_id()
        ts = c._get_thread_state(thread_id)

        messages = runtime.state.get("messages", [])
        effective = c._apply_event_to_messages(messages, ts.event)

        if not self._is_eligible_for_compaction(effective):
            return self._tool_message(
                tool_call_id,
                "Nothing to compact yet — conversation is within the token budget.",
            )

        cutoff = c._determine_cutoff_index(effective)
        if cutoff == 0:
            return self._tool_message(
                tool_call_id,
                "Nothing to compact yet — conversation is within the token budget.",
            )

        try:
            to_summarize = effective[:cutoff]

            backend_obj = c._backend
            if callable(backend_obj):
                backend_obj = backend_obj(runtime)

            file_path, (new_messages, _, _) = await asyncio.gather(
                c._aoffload_to_backend(
                    backend_obj, to_summarize, thread_id
                ),
                acompact_conversation(
                    effective, c._model, c._config, cutoff_index=cutoff
                ),
            )

            restoration = await abuild_restoration_context(
                backend_obj,
                to_summarize,
                dict(runtime.state),
                c._config.restoration,
            )
            new_messages = c._enrich_summary(
                new_messages, restoration, file_path
            )
        except Exception as exc:
            logger.exception("compact_conversation tool failed")
            return self._tool_message(
                tool_call_id,
                f"Compaction failed: {type(exc).__name__}: {exc}",
            )

        state_cutoff = c._compute_state_cutoff(ts.event, cutoff)
        c._save_compaction_event(
            thread_id,
            CompactionEvent(
                cutoff_index=state_cutoff,
                summary_message=new_messages[0],
                file_path=file_path,
                strategy="full",
                tokens_before=estimate_tokens(effective),
                tokens_after=estimate_tokens(new_messages),
            ),
        )

        return self._tool_message(
            tool_call_id,
            f"Conversation compacted. Summarized {len(to_summarize)} messages.",
        )

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject compact-tool usage nudge into system prompt."""
        from deepagents.middleware._utils import append_to_system_message

        new_sys = append_to_system_message(
            request.system_message, COMPACTION_SYSTEM_PROMPT
        )
        return handler(request.override(system_message=new_sys))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        from deepagents.middleware._utils import append_to_system_message

        new_sys = append_to_system_message(
            request.system_message, COMPACTION_SYSTEM_PROMPT
        )
        return await handler(request.override(system_message=new_sys))


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_compaction_middleware(
    model: BaseChatModel,
    backend: BACKEND_TYPES,
    config: CompactionConfig | None = None,
) -> CompactionMiddleware:
    """Create a ``CompactionMiddleware`` with model-aware defaults."""
    return CompactionMiddleware(model=model, backend=backend, config=config)


def create_compaction_tool_middleware(
    model: str | BaseChatModel,
    backend: BACKEND_TYPES,
    config: CompactionConfig | None = None,
) -> CompactionToolMiddleware:
    """Create both compaction middlewares (auto + tool) in one call.

    Returns the ``CompactionToolMiddleware``.  Access the auto-middleware
    via ``tool_mw._compaction``.

    Example::

        tool_mw = create_compaction_tool_middleware("anthropic:claude-sonnet-4-6", backend)
        agent = create_deep_agent(middleware=[tool_mw._compaction, tool_mw])
    """
    if isinstance(model, str):
        from deepagents._models import resolve_model

        model = resolve_model(model)

    compaction = create_compaction_middleware(model, backend, config)
    return CompactionToolMiddleware(compaction)
