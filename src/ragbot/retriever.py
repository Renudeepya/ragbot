"""Vector store and hybrid retriever.

Dense retrieval alone misses exact identifiers — error codes, SKUs, policy
names — because embedders smooth them away. Lexical overlap catches those but
misses paraphrase. `HybridRetriever` blends both, which in practice beats either
one on a mixed question set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk
from .embeddings import Embedder, cosine, tokenize


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    dense_score: float = 0.0
    lexical_score: float = 0.0


class VectorStore:
    """In-memory vector store with cosine search and JSON persistence."""

    def __init__(self, dim: int):
        self.dim = dim
        self._chunks: list[Chunk] = []
        self._vectors: list[list[float]] = []

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def chunks(self) -> list[Chunk]:
        return list(self._chunks)

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must be the same length")
        for v in vectors:
            if len(v) != self.dim:
                raise ValueError(f"expected dim {self.dim}, got {len(v)}")
        existing = {c.id for c in self._chunks}
        for c, v in zip(chunks, vectors):
            if c.id in existing:      # re-ingesting a doc replaces its chunks
                idx = next(i for i, e in enumerate(self._chunks) if e.id == c.id)
                self._chunks[idx], self._vectors[idx] = c, v
            else:
                self._chunks.append(c)
                self._vectors.append(v)

    def search(self, query_vec: list[float], k: int = 4) -> list[tuple[Chunk, float]]:
        scored = [
            (c, cosine(query_vec, v)) for c, v in zip(self._chunks, self._vectors)
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:k]

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "dim": self.dim,
                    "records": [
                        {"chunk": c.to_dict(), "vector": v}
                        for c, v in zip(self._chunks, self._vectors)
                    ],
                }
            ),
            encoding="utf-8",
        )
        return p

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        store = cls(dim=data["dim"])
        chunks = [Chunk(**r["chunk"]) for r in data["records"]]
        vectors = [r["vector"] for r in data["records"]]
        store.add(chunks, vectors)
        return store


# Stopwords are excluded from lexical scoring. Without this, "What is the
# capital of France?" scores highly against every chunk in the corpus purely on
# what/is/the/of — which made out-of-scope questions outrank real ones and
# defeated the abstention threshold entirely.
STOPWORDS = frozenset("""
a an the and or but if then than that this these those of in on at to for from by
with without about into over under again further is are was were be been being am
do does did doing have has had having i me my we our you your he she it its they
them their what which who whom when where why how all any both each few more most
other some such no nor not only own same so too very can will just should now
""".split())


def content_tokens(text: str) -> set[str]:
    """Tokens that carry topical signal — stopwords and bare digits removed."""
    return {t for t in tokenize(text) if t not in STOPWORDS and not t.isdigit()}


def lexical_overlap(query: str, text: str) -> float:
    """Fraction of distinct content-bearing query tokens present in the text."""
    q = content_tokens(query)
    if not q:
        return 0.0
    return len(q & content_tokens(text)) / len(q)


class HybridRetriever:
    """Blends dense cosine similarity with lexical overlap.

    `alpha` is the dense weight: 1.0 is pure vector search, 0.0 pure keyword.
    `min_score` lets the caller refuse to answer rather than ground a response
    in irrelevant text — abstaining is usually better than confidently wrong.
    The 0.15 default was chosen empirically against the golden set: it separates
    genuinely out-of-corpus questions from real ones without dropping any true
    positive. Retune it when the corpus changes, and let the eval gate tell you
    whether the new value held.
    """

    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore | None = None,
        *,
        k: int = 4,
        alpha: float = 0.7,
        min_score: float = 0.15,
    ):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")
        self.embedder = embedder
        self.store = store if store is not None else VectorStore(embedder.dim)
        self.k = k
        self.alpha = alpha
        self.min_score = min_score

    def index(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        vectors = self.embedder.embed([c.text for c in chunks])
        self.store.add(chunks, vectors)
        return len(chunks)

    def retrieve(self, query: str, k: int | None = None) -> list[Hit]:
        k = k or self.k
        if len(self.store) == 0:
            return []
        qv = self.embedder.embed([query])[0]
        # Over-fetch before re-ranking so lexical signal can promote a chunk
        # that dense search ranked just outside the top-k.
        candidates = self.store.search(qv, k=min(len(self.store), max(k * 4, k)))

        hits = []
        for chunk, dense in candidates:
            lex = lexical_overlap(query, chunk.text)
            blended = self.alpha * dense + (1 - self.alpha) * lex
            hits.append(Hit(chunk=chunk, score=blended, dense_score=dense, lexical_score=lex))

        hits.sort(key=lambda h: h.score, reverse=True)
        return [h for h in hits if h.score >= self.min_score][:k]
