"""LLM backends and message types.

`ScriptedLLM` makes the whole chatbot deterministic in tests — including the
tool-calling loop — without a network call or an API key. The real adapters
import their SDKs lazily.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass
class Message:
    role: str                      # "system" | "user" | "assistant" | "tool"
    content: str
    name: str | None = None        # tool name, when role == "tool"

    def to_dict(self) -> dict[str, Any]:
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        return d


@dataclass
class LLMResponse:
    """A model turn: either prose, or a request to call tools, or both."""

    text: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tokens: int = 0

    @property
    def wants_tools(self) -> bool:
        return bool(self.tool_calls)


@runtime_checkable
class LLM(Protocol):
    name: str

    def complete(
        self, messages: list[Message], tools: list[dict[str, Any]] | None = None
    ) -> LLMResponse: ...


TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def parse_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Extract `<tool_call>{...}</tool_call>` blocks from a completion.

    Used by backends without native tool-calling. A malformed block is left in
    the prose untouched rather than raising or being silently deleted: if we
    can't parse it we don't know it was a tool call, and dropping text the model
    produced loses information the user may need to see.
    """
    calls: list[dict[str, Any]] = []
    spans: list[tuple[int, int]] = []

    for match in TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue  # unparseable: leave it in the prose
        if not (isinstance(payload, dict) and "name" in payload):
            continue
        calls.append({"name": payload["name"], "arguments": payload.get("arguments", {})})
        spans.append(match.span())

    cleaned = text
    for start, end in reversed(spans):  # reverse so earlier offsets stay valid
        cleaned = cleaned[:start] + cleaned[end:]
    return cleaned.strip(), calls


class ScriptedLLM:
    """Replays queued responses; used to make tests deterministic.

    Accepts plain strings (parsed for `<tool_call>` blocks) or LLMResponse
    objects. When the script runs out it falls back to `default`, so a test
    that under-specifies turns fails loudly on content rather than IndexError.
    """

    def __init__(
        self,
        responses: list[str | LLMResponse] | None = None,
        *,
        default: str = "I don't have enough information to answer that.",
        name: str = "scripted",
    ):
        self.responses = list(responses or [])
        self.default = default
        self.name = name
        self.calls: list[list[Message]] = []   # recorded for assertions

    def complete(self, messages, tools=None) -> LLMResponse:
        self.calls.append(list(messages))
        if not self.responses:
            return LLMResponse(text=self.default)
        nxt = self.responses.pop(0)
        if isinstance(nxt, LLMResponse):
            return nxt
        text, calls = parse_tool_calls(nxt)
        return LLMResponse(text=text, tool_calls=calls)


class EchoContextLLM:
    """Answers by quoting retrieved context. Offline demo backend.

    Deliberately extractive: it proves the retrieval and citation path end to
    end without pretending to be a language model.
    """

    name = "echo-context"
    MAX_CHARS = 1200
    TOP_BLOCKS = 2

    def complete(self, messages, tools=None) -> LLMResponse:
        context = next(
            (m.content for m in reversed(messages) if m.role == "system" and "CONTEXT" in m.content),
            "",
        )
        body = context.split("CONTEXT:", 1)[-1].strip()
        if not body or body.startswith("(none)"):
            return LLMResponse(text="I don't have enough information to answer that.")

        # Read the top TOP_BLOCKS retrieved passages, not just the first: the
        # answering sentence often sits in the second-ranked chunk, and a purely
        # extractive backend has no way to know that in advance.
        blocks = re.split(r"\n\n(?=\[\d+\])", body)[: self.TOP_BLOCKS]
        kept = []
        for block in blocks:
            lines = [
                ln for ln in block.splitlines()
                if not re.sub(r"^\s*\[\d+\]\s*", "", ln).startswith("#")
            ]
            prose = " ".join(ln.strip() for ln in lines if ln.strip())
            if prose:
                kept.append(prose if prose.startswith("[") else f"[{len(kept) + 1}] {prose}")
        if not kept:
            return LLMResponse(text="I don't have enough information to answer that.")
        return LLMResponse(text=" ".join(kept)[: self.MAX_CHARS])


class OpenAILLM:
    """Requires `pip install openai`."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        from openai import OpenAI

        self.client = OpenAI()
        self.model = model
        self.temperature = temperature
        self.name = f"openai:{model}"

    def complete(self, messages, tools=None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "messages": [m.to_dict() for m in messages],
        }
        if tools:
            kwargs["tools"] = [{"type": "function", "function": t} for t in tools]
        resp = self.client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        calls = [
            {"name": tc.function.name, "arguments": json.loads(tc.function.arguments or "{}")}
            for tc in (msg.tool_calls or [])
        ]
        return LLMResponse(
            text=msg.content or "",
            tool_calls=calls,
            tokens=getattr(resp.usage, "total_tokens", 0) if resp.usage else 0,
        )


class AnthropicLLM:
    """Requires `pip install anthropic`."""

    def __init__(self, model: str = "claude-sonnet-4-5", max_tokens: int = 1024):
        import anthropic

        self.client = anthropic.Anthropic()
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"anthropic:{model}"

    def complete(self, messages, tools=None) -> LLMResponse:
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        convo = [
            {"role": "assistant" if m.role == "assistant" else "user", "content": m.content}
            for m in messages
            if m.role in ("user", "assistant", "tool")
        ]
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": convo or [{"role": "user", "content": ""}],
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = [
                {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "input_schema": t.get("parameters", {"type": "object", "properties": {}}),
                }
                for t in tools
            ]
        resp = self.client.messages.create(**kwargs)
        text_parts, calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                calls.append({"name": block.name, "arguments": dict(block.input or {})})
        usage = getattr(resp, "usage", None)
        return LLMResponse(
            text="".join(text_parts),
            tool_calls=calls,
            tokens=(usage.input_tokens + usage.output_tokens) if usage else 0,
        )


class CallableLLM:
    """Wraps any `fn(messages) -> str`."""

    def __init__(self, fn: Callable[[list[Message]], str], name: str = "callable"):
        self.fn = fn
        self.name = name

    def complete(self, messages, tools=None) -> LLMResponse:
        text, calls = parse_tool_calls(self.fn(messages))
        return LLMResponse(text=text, tool_calls=calls)
