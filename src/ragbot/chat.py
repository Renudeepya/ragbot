"""The RAG chatbot: retrieval, grounded generation, tool loop, memory."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .chunking import Chunk, chunk_document
from .embeddings import Embedder, HashingEmbedder
from .llm import LLM, EchoContextLLM, Message
from .retriever import Hit, HybridRetriever
from .tools import ToolRegistry

SYSTEM_PROMPT = (
    "You are a helpful assistant that answers strictly from the provided context.\n"
    "Rules:\n"
    "1. Use only the CONTEXT below. Do not rely on outside knowledge.\n"
    "2. Cite the sources you used with bracketed markers like [1] or [2].\n"
    "3. If the context does not answer the question, say you don't have enough "
    "information. Do not guess.\n"
    "4. To call a tool, emit <tool_call>{\"name\": ..., \"arguments\": {...}}</tool_call>."
)


@dataclass
class Citation:
    marker: str
    chunk_id: str
    doc_id: str
    source: str
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "source": self.source,
            "score": round(self.score, 4),
        }


@dataclass
class Answer:
    text: str
    citations: list[Citation] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    hits: list[Hit] = field(default_factory=list)
    latency_ms: float = 0.0
    tokens: int = 0
    grounded: bool = True

    @property
    def cited_markers(self) -> list[str]:
        import re
        return sorted(set(re.findall(r"\[(\d+)\]", self.text)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": [c.to_dict() for c in self.citations],
            "tool_calls": self.tool_calls,
            "latency_ms": round(self.latency_ms, 2),
            "tokens": self.tokens,
            "grounded": self.grounded,
        }


class RagChatbot:
    """Retrieval-augmented chatbot with tool calling and conversation memory.

    `max_tool_iterations` bounds the tool loop. Without a cap, a model that
    keeps requesting the same failing tool loops forever and burns the budget —
    the most common way an agent hangs in production.
    """

    def __init__(
        self,
        llm: LLM | None = None,
        retriever: HybridRetriever | None = None,
        tools: ToolRegistry | None = None,
        *,
        embedder: Embedder | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        max_history: int = 8,
        max_tool_iterations: int = 3,
        name: str = "ragbot-v1",
    ):
        self.embedder = embedder or HashingEmbedder()
        self.llm = llm or EchoContextLLM()
        self.retriever = retriever or HybridRetriever(self.embedder)
        self.tools = tools or ToolRegistry()
        self.system_prompt = system_prompt
        self.max_history = max_history
        self.max_tool_iterations = max_tool_iterations
        self.name = name
        self.history: list[Message] = []

    # ------------------------------------------------------------- ingestion
    # Smaller chunks than the 700-char default: whole-document chunks made every
    # query retrieve *something* plausible, which blunted the abstention signal.
    DEFAULT_CHUNK_SIZE = 350
    DEFAULT_OVERLAP = 60

    def ingest_text(self, doc_id: str, text: str, **meta: Any) -> int:
        chunks = chunk_document(
            doc_id,
            text,
            chunk_size=meta.pop("chunk_size", self.DEFAULT_CHUNK_SIZE),
            overlap=meta.pop("overlap", self.DEFAULT_OVERLAP),
            metadata={"title": meta.pop("title", doc_id), **meta},
        )
        return self.retriever.index(chunks)

    def ingest_path(self, path: str | Path, pattern: str = "*.md") -> int:
        p = Path(path)
        files = [p] if p.is_file() else sorted(p.glob(pattern))
        if not files:
            raise FileNotFoundError(f"no files matching {pattern!r} under {p}")
        return sum(
            self.ingest_text(f.stem, f.read_text(encoding="utf-8"), title=f.stem)
            for f in files
        )

    # --------------------------------------------------------------- context
    @staticmethod
    def _format_context(hits: list[Hit]) -> tuple[str, list[Citation]]:
        if not hits:
            return "CONTEXT:\n(none)", []
        blocks, citations = [], []
        for i, hit in enumerate(hits, start=1):
            marker = f"[{i}]"
            blocks.append(f"{marker} {hit.chunk.text}")
            citations.append(
                Citation(
                    marker=marker,
                    chunk_id=hit.chunk.id,
                    doc_id=hit.chunk.doc_id,
                    source=hit.chunk.citation,
                    score=hit.score,
                )
            )
        return "CONTEXT:\n" + "\n\n".join(blocks), citations

    def _build_messages(self, question: str, context: str) -> list[Message]:
        msgs = [Message("system", self.system_prompt), Message("system", context)]
        msgs += self.history[-self.max_history :]
        msgs.append(Message("user", question))
        return msgs

    # ------------------------------------------------------------------ chat
    def chat(self, question: str, *, k: int | None = None) -> Answer:
        started = time.perf_counter()
        hits = self.retriever.retrieve(question, k=k)
        context, citations = self._format_context(hits)
        messages = self._build_messages(question, context)

        executed: list[dict[str, Any]] = []
        tokens = 0
        response = self.llm.complete(messages, tools=self.tools.schemas() or None)
        tokens += response.tokens

        iterations = 0
        while response.wants_tools and iterations < self.max_tool_iterations:
            iterations += 1
            for call in response.tool_calls:
                result = self.tools.invoke(call["name"], call.get("arguments"))
                executed.append({**call, "result": result})
                messages.append(Message("tool", result, name=call["name"]))
            response = self.llm.complete(messages, tools=self.tools.schemas() or None)
            tokens += response.tokens

        if response.wants_tools:      # hit the cap with tools still pending
            for call in response.tool_calls:
                executed.append({**call, "result": "error: tool iteration limit reached"})

        text = response.text or "I don't have enough information to answer that."
        self.history.append(Message("user", question))
        self.history.append(Message("assistant", text))

        answer = Answer(
            text=text,
            citations=citations,
            tool_calls=executed,
            hits=hits,
            latency_ms=(time.perf_counter() - started) * 1000,
            tokens=tokens,
            grounded=bool(hits),
        )
        return answer

    def reset(self) -> None:
        self.history.clear()
