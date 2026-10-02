"""Bridge: expose RagChatbot as an agenteval Agent, plus RAG-specific graders.

This is the point of building both projects. agenteval grades any agent behind a
one-method interface; this adapter makes the chatbot one of those agents, so
every prompt change or model swap runs through the same regression gate as the
rest of the system.

Requires agenteval on the path (`pip install -e ../agent-eval-harness`).
"""
from __future__ import annotations

import re
from typing import Any

from .chat import RagChatbot

try:  # agenteval is an optional peer, not a hard dependency
    from agenteval.graders import Grader
    from agenteval.types import AgentOutput, GradeResult, TestCase, ToolCall

    AGENTEVAL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without agenteval
    AGENTEVAL_AVAILABLE = False
    Grader = object  # type: ignore[assignment,misc]


class RagbotAgent:
    """Wraps a RagChatbot in the agenteval Agent protocol.

    Resets conversation memory per case so results don't depend on evaluation
    order — otherwise a run is not reproducible and the baseline is meaningless.
    """

    def __init__(self, bot: RagChatbot, *, reset_each_case: bool = True):
        if not AGENTEVAL_AVAILABLE:
            raise ImportError(
                "agenteval is required for evaluation: pip install -e ../agent-eval-harness"
            )
        self.bot = bot
        self.reset_each_case = reset_each_case
        self.name = f"ragbot:{bot.llm.name}"
        self.last_answer = None

    def run(self, prompt: str) -> "AgentOutput":
        if self.reset_each_case:
            self.bot.reset()
        answer = self.bot.chat(prompt)
        self.last_answer = answer
        return AgentOutput(
            text=answer.text,
            tool_calls=[
                ToolCall(name=c["name"], arguments=c.get("arguments", {}))
                for c in answer.tool_calls
            ],
            latency_ms=answer.latency_ms,
            tokens=answer.tokens,
        )


if AGENTEVAL_AVAILABLE:

    class HasCitation(Grader):
        """Fails answers that assert something without citing a source.

        The characteristic RAG failure is a fluent, confident, uncited answer.
        Abstentions are exempt — refusing to answer is correct behavior, and
        punishing it would train the system toward confident guessing.
        """

        name = "has_citation"
        ABSTAIN = re.compile(
            r"don't have enough information|do not have enough information|cannot answer|"
            r"no information|not covered",
            re.IGNORECASE,
        )

        def grade(self, case: "TestCase", output: "AgentOutput") -> "GradeResult":
            if not case.metadata.get("expect_citation", True):
                return self._result(1.0, "citation not required; skipped")
            if self.ABSTAIN.search(output.text):
                return self._result(1.0, "abstained; citation not required")
            ok = bool(re.search(r"\[\d+\]", output.text))
            return self._result(1.0 if ok else 0.0, "" if ok else "answer has no citation marker")

    class Abstains(Grader):
        """Asserts the bot declines when the corpus can't support an answer.

        Out-of-scope cases are the ones that catch hallucination, and they are
        the first thing to break when a prompt is loosened.
        """

        name = "abstains"

        def grade(self, case: "TestCase", output: "AgentOutput") -> "GradeResult":
            if not case.metadata.get("expect_abstain"):
                return self._result(1.0, "not an abstain case; skipped")
            ok = bool(HasCitation.ABSTAIN.search(output.text))
            return self._result(
                1.0 if ok else 0.0,
                "" if ok else f"expected an abstention, got: {output.text[:80]!r}",
            )

    class NoToolLoop(Grader):
        """Flags a turn that hit the tool-iteration cap."""

        name = "no_tool_loop"

        def grade(self, case: "TestCase", output: "AgentOutput") -> "GradeResult":
            looped = any(
                "iteration limit reached" in str(getattr(t, "arguments", "")) for t in output.tool_calls
            ) or "iteration limit reached" in output.text
            return self._result(0.0 if looped else 1.0, "tool loop hit the cap" if looped else "")

    def default_graders() -> list[Any]:
        """Graders appropriate for a grounded RAG chatbot."""
        from agenteval.graders import Contains, NoError, ToolTrajectory

        return [NoError(), Contains(), ToolTrajectory(), HasCitation(), Abstains(), NoToolLoop()]
