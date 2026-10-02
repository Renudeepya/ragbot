"""Embedding backends.

`HashingEmbedder` is a dependency-free, deterministic embedder: it hashes token
n-grams into a fixed-width vector with sublinear term weighting. It is not a
neural embedder and does not capture synonymy — but it is reproducible, needs no
network, and makes the retrieval layer testable in CI. `OpenAIEmbedder` is the
production path; both satisfy the same protocol so swapping one for the other
changes no downstream code.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol, runtime_checkable

_TOKEN = re.compile(r"[a-z0-9]+")


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    num = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return num / (na * nb)


class HashingEmbedder:
    """Deterministic bag-of-ngrams embedder using the hashing trick."""

    def __init__(self, dim: int = 512, ngram: int = 2):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if ngram < 1:
            raise ValueError("ngram must be >= 1")
        self.dim = dim
        self.ngram = ngram

    def _features(self, text: str) -> list[str]:
        toks = tokenize(text)
        feats = list(toks)
        for n in range(2, self.ngram + 1):
            feats += [" ".join(toks[i : i + n]) for i in range(len(toks) - n + 1)]
        return feats

    @staticmethod
    def _bucket(feature: str, dim: int) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        h = int.from_bytes(digest, "big")
        # Sign hashing keeps collisions from systematically inflating similarity.
        return h % dim, 1.0 if (h >> 63) & 1 else -1.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            counts: dict[str, int] = {}
            for f in self._features(text):
                counts[f] = counts.get(f, 0) + 1
            for feat, tf in counts.items():
                idx, sign = self._bucket(feat, self.dim)
                vec[idx] += sign * (1.0 + math.log(tf))  # sublinear tf
            norm = math.sqrt(sum(v * v for v in vec))
            out.append([v / norm for v in vec] if norm else vec)
        return out


class OpenAIEmbedder:
    """Production embedder. Requires `pip install openai`."""

    DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}

    def __init__(self, model: str = "text-embedding-3-small", batch_size: int = 128):
        from openai import OpenAI  # lazy: keeps the package installable without it

        self.client = OpenAI()
        self.model = model
        self.dim = self.DIMS.get(model, 1536)
        self.batch_size = batch_size

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            resp = self.client.embeddings.create(model=self.model, input=batch)
            out.extend(d.embedding for d in resp.data)
        return out
