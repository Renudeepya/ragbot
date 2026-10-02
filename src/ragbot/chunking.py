"""Split documents into overlapping, retrievable chunks.

Chunking is where most RAG quality is won or lost: too large and retrieval
returns mostly irrelevant text, too small and answers lose the context that
made them correct. This splits on paragraph boundaries first and only falls
back to hard character splits when a single paragraph exceeds the budget.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Chunk:
    """A retrievable span of a source document."""

    id: str
    doc_id: str
    text: str
    ordinal: int
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def citation(self) -> str:
        title = self.metadata.get("title", self.doc_id)
        return f"{title}#{self.ordinal}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "doc_id": self.doc_id,
            "text": self.text,
            "ordinal": self.ordinal,
            "metadata": self.metadata,
        }


def _split_paragraphs(text: str) -> list[str]:
    parts = re.split(r"\n\s*\n", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Character-window split, used only when one paragraph is oversized."""
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    out = []
    for start in range(0, len(text), step):
        piece = text[start : start + size].strip()
        if piece:
            out.append(piece)
        if start + size >= len(text):
            break
    return out


def chunk_document(
    doc_id: str,
    text: str,
    *,
    chunk_size: int = 700,
    overlap: int = 120,
    metadata: dict[str, Any] | None = None,
) -> list[Chunk]:
    """Pack paragraphs into chunks of at most `chunk_size` characters.

    Adjacent paragraphs are merged while they fit, which keeps semantically
    related prose together instead of slicing mid-argument.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    metadata = dict(metadata or {})
    paragraphs = _split_paragraphs(text)
    if not paragraphs:
        return []

    packed: list[str] = []
    buf = ""
    for para in paragraphs:
        if len(para) > chunk_size:
            if buf:
                packed.append(buf)
                buf = ""
            packed.extend(_hard_split(para, chunk_size, overlap))
            continue
        candidate = f"{buf}\n\n{para}" if buf else para
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            packed.append(buf)
            buf = para
    if buf:
        packed.append(buf)

    return [
        Chunk(
            id=f"{doc_id}::{i}",
            doc_id=doc_id,
            text=t,
            ordinal=i,
            metadata=metadata,
        )
        for i, t in enumerate(packed)
    ]
