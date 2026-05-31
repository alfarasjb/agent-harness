"""ContextBuilder -- the heart of the demo.

It assembles everything sent to the model each turn and decides where the
prompt-cache breakpoints go. The whole point is that the *layout* is explicit
and inspectable:

    [ tools ]                      <- stable  (cache breakpoint #1)
    [ system: persona ]           \
    [ system: manual snapshot ]    > stable  (cache breakpoint #2)
    --------- cache prefix ends here ---------
    [ conversation history ]       <- volatile (grows every turn)
    [ newest user message ]        <- volatile

Anthropic caches the request *prefix* up to and including each
``cache_control`` marker. A hit requires that prefix to be byte-identical to a
previous request -- so stable content goes first, volatile content last. Put a
single rotating character in the prefix and the cache for everything after it
is gone.

This module is provider-aware only in the small way it must be: cache markers
ride on content blocks as ``{"cache_control": {"type": "ephemeral"}}``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from . import corpus


_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, round(len(text) / _CHARS_PER_TOKEN))


SYSTEM_PERSONA = (
    "You are the operations assistant for the deep-space survey vessel "
    "ACV Helios-IX. Answer the watch crew's questions about the ship using "
    "ONLY the operations manual provided and the tools available to you.\n\n"
    "How to work:\n"
    "- Prefer calling tools to ground every answer in specific manual blocks.\n"
    "- Use search_manual to locate, then get_block to read the exact block, "
    "including blocks cross-referenced (\"See also\") by ones you've read.\n"
    "- Chain tool calls when a question depends on several blocks (e.g. a "
    "fault that points to a procedure that points to a subsystem).\n"
    "- Cite the block IDs you used in your final answer.\n"
    "- If the manual does not cover something, say so plainly rather than "
    "guessing.\n"
    "Be concise and precise -- you are talking to trained crew."
)


@dataclass
class Layer:
    """One inspectable slice of the assembled context."""

    name: str
    role: str
    cacheable: bool
    cache_breakpoint: bool
    tokens: int
    preview: str


@dataclass
class BuiltContext:
    """Everything the agent needs to make a request, plus a UI view of it."""

    system: list[dict[str, Any]]
    tools: list[dict[str, Any]]
    layers: list[Layer]

    @property
    def stable_tokens(self) -> int:
        return sum(l.tokens for l in self.layers if l.cacheable)

    @property
    def volatile_tokens(self) -> int:
        return sum(l.tokens for l in self.layers if not l.cacheable)


class ContextBuilder:
    """Assembles system + tools and decides cache placement.

    Note on scope: the builder owns the *stable prefix* (tools + system +
    manual snapshot). The volatile conversation history and newest user message
    live in the ``messages`` array managed by the agent loop -- they are
    described here only so the inspector can show the full picture.
    """

    def __init__(self, tool_schemas: list[dict[str, Any]], cache_enabled: bool):
        self._tool_schemas = tool_schemas
        self.cache_enabled = cache_enabled

    def build(
        self, history_messages: list[dict[str, Any]], user_text: str
    ) -> BuiltContext:
        manual = corpus.render_full_manual()

        tools = [dict(t) for t in self._tool_schemas]
        if self.cache_enabled and tools:
            tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}

        persona_block: dict[str, Any] = {"type": "text", "text": SYSTEM_PERSONA}
        manual_block: dict[str, Any] = {"type": "text", "text": manual}
        if self.cache_enabled:
            manual_block["cache_control"] = {"type": "ephemeral"}
        system = [persona_block, manual_block]

        tools_json = _tools_preview(tools)
        history_tokens, history_preview = _history_view(history_messages)
        layers = [
            Layer(
                name="Tool definitions",
                role="tools",
                cacheable=self.cache_enabled,
                cache_breakpoint=self.cache_enabled,
                tokens=estimate_tokens(tools_json),
                preview=tools_json,
            ),
            Layer(
                name="System: persona + instructions",
                role="system",
                cacheable=self.cache_enabled,
                cache_breakpoint=False,
                tokens=estimate_tokens(SYSTEM_PERSONA),
                preview=SYSTEM_PERSONA,
            ),
            Layer(
                name="System: full manual snapshot",
                role="system",
                cacheable=self.cache_enabled,
                cache_breakpoint=self.cache_enabled,
                tokens=estimate_tokens(manual),
                preview=manual[:1200] + ("\n..." if len(manual) > 1200 else ""),
            ),
            Layer(
                name="Conversation history (prior turns + tool results)",
                role="history",
                cacheable=False,
                cache_breakpoint=False,
                tokens=history_tokens,
                preview=history_preview or "(empty -- first turn)",
            ),
            Layer(
                name="Newest user message",
                role="user",
                cacheable=False,
                cache_breakpoint=False,
                tokens=estimate_tokens(user_text),
                preview=user_text,
            ),
        ]
        return BuiltContext(system=system, tools=tools, layers=layers)


def _tools_preview(tools: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {t['name']}({', '.join(t['input_schema'].get('properties', {}))})"
        for t in tools
    )


def _history_view(messages: list[dict[str, Any]]) -> tuple[int, str]:
    """Crude token count + preview of the message array for the inspector."""
    total = 0
    lines: list[str] = []
    for m in messages:
        role = m.get("role", "?")
        content = m.get("content", "")
        text = content if isinstance(content, str) else _flatten_blocks(content)
        total += estimate_tokens(text)
        lines.append(f"[{role}] {text[:120]}")
    return total, "\n".join(lines)


def _flatten_blocks(content: Any) -> str:
    parts: list[str] = []
    for block in content if isinstance(content, list) else []:
        if isinstance(block, dict):
            btype = block.get("type")
            if btype == "text":
                parts.append(block.get("text", ""))
            elif btype == "tool_use":
                parts.append(f"<tool_use {block.get('name')}>")
            elif btype == "tool_result":
                inner = block.get("content", "")
                if isinstance(inner, list):
                    inner = " ".join(
                        b.get("text", "") for b in inner if isinstance(b, dict)
                    )
                parts.append(f"<tool_result {str(inner)[:80]}>")
            elif btype == "thinking":
                parts.append("<thinking ...>")
        else:
            parts.append(f"<{getattr(block, 'type', 'block')}>")
    return " ".join(parts)
