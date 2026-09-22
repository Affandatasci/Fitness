"""
cache.py — Semantic cache for the Forge Physique RAG demo.

Stores past (question, answer) pairs in memory. A new question is
compared against every cached question by cosine similarity; if a
close enough match is found (config.CACHE_SIMILARITY_THRESHOLD), the
cached answer is returned instantly with zero LLM calls. This is what
keeps repeated or near-duplicate demo questions fast and off your Groq
free-tier rate limit, on top of the query-time speed fix already built
into agent.py (single LLM call by default, deep reasoning off).

IMPORTANT — Streamlit usage: this class must be instantiated via
@st.cache_resource in app.py so the SAME instance persists across
reruns within a session. If a fresh SemanticCache() is created on
every rerun instead, every question will always be a cache miss and
this file does nothing. See app.py's load_resources() for the correct
pattern — and note that because @st.cache_resource shares one instance
across ALL visitors to a deployed app (not one per browser tab), the
cache is effectively shared site-wide. That's fine for this public,
non-personalized FAQ content — one visitor's question warms the cache
for the next — but would NOT be appropriate if answers ever contained
anything client-specific or private.
"""

from typing import Optional

import numpy as np

import config


class SemanticCache:
    def __init__(self, embed_fn, similarity_threshold: Optional[float] = None):
        """
        embed_fn: a callable(text: str) -> vector, where the vector is
        a NORMALIZED embedding (normalize_embeddings=True) — so that a
        plain dot product between two entries equals their cosine
        similarity, with no separate normalization step needed here.
        """
        self.embed_fn = embed_fn
        self.similarity_threshold = (
            similarity_threshold if similarity_threshold is not None else config.CACHE_SIMILARITY_THRESHOLD
        )
        self.entries = []  # list of {"query": str, "embedding": np.ndarray, "answer": str}

    def get(self, query: str) -> Optional[str]:
        """Returns the cached answer if a similar-enough question was asked before, else None."""
        if not self.entries:
            return None

        query_embedding = np.asarray(self.embed_fn(query))

        best_score = -1.0
        best_answer = None
        for entry in self.entries:
            score = float(np.dot(query_embedding, entry["embedding"]))
            if score > best_score:
                best_score = score
                best_answer = entry["answer"]

        if best_score >= self.similarity_threshold:
            return best_answer
        return None

    def add(self, query: str, answer: str) -> None:
        """Stores a new (question, answer) pair for future similarity matching."""
        query_embedding = np.asarray(self.embed_fn(query))
        self.entries.append({"query": query, "embedding": query_embedding, "answer": answer})

    def __len__(self) -> int:
        return len(self.entries)
