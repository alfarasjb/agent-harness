"""The agent loop: real Claude API, extended thinking, multi-step tool use.

``run_turn`` is a generator. It yields trace events as they happen so the UI can
render the loop step-by-step:

    context  -> what was assembled (for the inspector)
    step     -> a new model call begins
    thinking -> the model's reasoning for this step (extended thinking)
    text     -> visible assistant prose
    tool_call / tool_result -> one round of tool use
    usage    -> real token + cache counts for that API call
    done     -> final answer + updated message history + totals

The conversation ``messages`` array (history + tool rounds) is the volatile
part. The stable prefix (tools + system + manual) is rebuilt each turn by the
ContextBuilder; when caching is on it is byte-identical turn to turn, so turn 2+
reads it from cache instead of re-paying for it.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from typing import Any, Iterator

from anthropic import Anthropic

from .context import ContextBuilder
from . import tools as toolkit


PRICING: dict[str, dict[str, float]] = {
    "claude-sonnet-4-6": {"in": 3.0, "out": 15.0},
    "claude-opus-4-8": {"in": 15.0, "out": 75.0},
    "claude-haiku-4-5-20251001": {"in": 1.0, "out": 5.0},
}
MODELS = list(PRICING)

MAX_STEPS = 8


@dataclass
class Usage:
    """Normalized view of one API call's token usage."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @classmethod
    def from_response(cls, resp_usage: Any) -> "Usage":
        def g(name: str) -> int:
            return int(getattr(resp_usage, name, 0) or 0)

        return cls(
            input_tokens=g("input_tokens"),
            output_tokens=g("output_tokens"),
            cache_creation_input_tokens=g("cache_creation_input_tokens"),
            cache_read_input_tokens=g("cache_read_input_tokens"),
        )

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cache_creation_input_tokens += other.cache_creation_input_tokens
        self.cache_read_input_tokens += other.cache_read_input_tokens

    def costs(self, model: str) -> dict[str, float]:
        """Actual cost vs the hypothetical cost if nothing had been cached."""
        rate = PRICING.get(model, {"in": 3.0, "out": 15.0})
        in_rate, out_rate = rate["in"] / 1e6, rate["out"] / 1e6
        actual = (
            self.input_tokens * in_rate
            + self.cache_creation_input_tokens * in_rate * 1.25
            + self.cache_read_input_tokens * in_rate * 0.1
            + self.output_tokens * out_rate
        )
        uncached_input = (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )
        uncached = uncached_input * in_rate + self.output_tokens * out_rate
        return {
            "actual": actual,
            "uncached": uncached,
            "saved": uncached - actual,
        }

    @property
    def cache_hit_rate(self) -> float:
        cached = self.cache_read_input_tokens
        total = (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )
        return cached / total if total else 0.0


@dataclass
class AgentConfig:
    model: str = "claude-sonnet-4-6"
    cache_enabled: bool = True
    thinking_enabled: bool = True
    thinking_budget: int = 1500
    max_tokens: int = 4000


@dataclass
class TurnResult:
    messages: list[dict[str, Any]]
    answer: str
    usage: Usage = field(default_factory=Usage)
    steps: int = 0


class Agent:
    def __init__(self, client: Anthropic, config: AgentConfig):
        self.client = client
        self.config = config
        self.builder = ContextBuilder(toolkit.TOOL_SCHEMAS, config.cache_enabled)

    def run_turn(
        self, history: list[dict[str, Any]], user_text: str
    ) -> Iterator[dict[str, Any]]:
        """Run one user turn to completion, yielding trace events."""
        cfg = self.config

        built = self.builder.build(history, user_text)
        yield {"type": "context", "built": built}

        messages: list[dict[str, Any]] = list(history)
        messages.append({"role": "user", "content": user_text})

        turn_usage = Usage()
        final_answer = ""
        step = 0

        while step < MAX_STEPS:
            step += 1
            yield {"type": "step", "n": step}

            max_tokens = cfg.max_tokens
            if cfg.thinking_enabled:
                max_tokens = max(max_tokens, cfg.thinking_budget + 1024)

            params: dict[str, Any] = {
                "model": cfg.model,
                "max_tokens": max_tokens,
                "system": built.system,
                "tools": built.tools,
                "messages": messages,
            }
            if cfg.thinking_enabled:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": cfg.thinking_budget,
                }

            response = self.client.messages.create(**params)

            usage = Usage.from_response(response.usage)
            turn_usage.add(usage)
            yield {"type": "usage", "step": step, "usage": usage}

            # Raw API response, exactly as Anthropic returned it. Pydantic's
            # model_dump gives us the full shape -- thinking signatures,
            # tool_use ids, stop_reason, the lot. This is the provider-native
            # data before any conversion to a domain type.
            raw = response.model_dump(mode="json")
            print(
                f"\n===== RAW RESPONSE (step {step}, stop_reason="
                f"{response.stop_reason}) =====",
                file=sys.stderr,
            )
            print(json.dumps(raw, indent=2), file=sys.stderr)
            yield {
                "type": "raw",
                "step": step,
                "stop_reason": response.stop_reason,
                "data": raw,
            }

            step_text_parts: list[str] = []
            for block in response.content:
                btype = getattr(block, "type", None)
                if btype == "thinking":
                    yield {
                        "type": "thinking",
                        "step": step,
                        "text": block.thinking,
                        "signature": getattr(block, "signature", None),
                    }
                elif btype == "redacted_thinking":
                    yield {
                        "type": "thinking",
                        "step": step,
                        "text": "[redacted by Anthropic safety system]",
                        "signature": "(redacted)",
                    }
                elif btype == "text":
                    step_text_parts.append(block.text)
                    yield {"type": "text", "step": step, "text": block.text}

            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                final_answer = "\n".join(p for p in step_text_parts if p).strip()
                break

            tool_result_blocks: list[dict[str, Any]] = []
            for block in response.content:
                if getattr(block, "type", None) != "tool_use":
                    continue
                yield {
                    "type": "tool_call",
                    "step": step,
                    "id": block.id,
                    "name": block.name,
                    "input": dict(block.input),
                }
                result = toolkit.dispatch(block.name, dict(block.input))
                yield {
                    "type": "tool_result",
                    "step": step,
                    "id": block.id,
                    "name": block.name,
                    "text": result.text,
                    "meta": result.meta,
                    "is_error": result.is_error,
                }
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result.text,
                        "is_error": result.is_error,
                    }
                )
            messages.append({"role": "user", "content": tool_result_blocks})
        else:
            final_answer = (
                "(Stopped: reached the maximum tool-use steps for this demo.)"
            )

        yield {
            "type": "done",
            "result": TurnResult(
                messages=messages,
                answer=final_answer,
                usage=turn_usage,
                steps=step,
            ),
        }
