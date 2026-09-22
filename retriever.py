"""
retriever.py — Hybrid retrieval (dense + keyword) with Reciprocal Rank
Fusion and cross-encoder reranking for the Forge Physique RAG demo.

Same conceptual pattern as the hybrid_retriever.py you've already built
on other projects: two ranked lists (dense vector search + BM25
keyword search) fused with RRF, then reranked with a cross-encoder
before the top chunks are handed to the LLM.

Run ingest.py before using this — it queries a Qdrant collection that
must already have vectors in it.
"""

import torch
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

import config


class HybridRetriever:
    def __init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Streamlit Community Cloud has no GPU, so this will land on
        # "cpu" automatically at deploy time — no code change needed
        # between local testing and deployment.
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)

        # max_length is set explicitly rather than left at the model's
        # default. Our chunks are already ~512 tokens on their own —
        # pairing a full chunk with even a short 20-30 token query
        # silently pushes past a default 512-token reranker limit and
        # truncates from the end with no warning. We instead truncate
        # the CHUNK side ourselves (see _truncate_for_reranker) so the
        # truncation is visible and intentional, and leaves headroom
        # for the query within the 512-token budget.
        self.reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512)

        self.qdrant_client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)

        self._load_corpus_for_bm25()

    def _load_corpus_for_bm25(self):
        """
        Scrolls the full Qdrant collection once at startup to build an
        in-memory BM25 index. Fine for a demo-sized corpus (tens of
        chunks) — for a much larger corpus this would need a proper
        sparse index instead of loading everything into memory.
        """
        all_points = []
        next_offset = None
        while True:
            points, next_offset = self.qdrant_client.scroll(
                collection_name=config.QDRANT_COLLECTION_NAME,
                limit=100,
                offset=next_offset,
                with_payload=True,
                with_vectors=False,
            )
            all_points.extend(points)
            if next_offset is None:
                break

        if not all_points:
            raise ValueError(
                f"Qdrant collection '{config.QDRANT_COLLECTION_NAME}' is empty. "
                f"Run ingest.py first."
            )

        # Positional list — must stay in the same order as tokenized_corpus
        # below, since BM25Okapi.get_scores() returns a plain array aligned
        # to that order with no IDs attached.
        self.corpus_ids = [p.id for p in all_points]
        self.id_to_text = {p.id: p.payload["text"] for p in all_points}
        self.id_to_source = {p.id: p.payload["source"] for p in all_points}

        tokenized_corpus = [self.id_to_text[doc_id].lower().split() for doc_id in self.corpus_ids]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def _dense_search(self, query: str, top_k: int) -> list:
        query_vector = self.embedding_model.encode(query, normalize_embeddings=True).tolist()
        results = self.qdrant_client.search(
            collection_name=config.QDRANT_COLLECTION_NAME,
            query_vector=query_vector,
            limit=top_k,
            with_payload=False,
        )
        return [point.id for point in results]

    def _sparse_search(self, query: str, top_k: int) -> list:
        """Returns a ranked list of point IDs from BM25 keyword search."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [self.corpus_ids[i] for i in ranked_indices]

    @staticmethod
    def _reciprocal_rank_fusion(ranked_lists: list) -> list:
        """Fuses multiple ranked ID lists into one, by RRF score, highest first."""
        k = config.RRF_K
        scores = {}
        for ranked_list in ranked_lists:
            for rank, doc_id in enumerate(ranked_list):
                scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
        return sorted(scores.keys(), key=lambda doc_id: scores[doc_id], reverse=True)

    @staticmethod
    def _truncate_for_reranker(text: str, max_chars: int = 1400) -> str:
        """
        Truncate chunk text before pairing it with the query for
        reranking. ~1400 chars leaves headroom for a ~20-30 token query
        within the reranker's 512-token limit, since our chunks are
        already sized at ~512 tokens (roughly 2000 chars) on their own.
        """
        return text[:max_chars]

    def retrieve(self, query: str) -> list:
        """
        Full hybrid retrieval pipeline: dense + sparse search -> RRF
        fusion -> cross-encoder rerank -> top config.TOP_K_RERANK chunks.
        Returns a list of {"text": ..., "source": ...} dicts.
        """
        dense_ids = self._dense_search(query, config.TOP_K_RETRIEVE)
        sparse_ids = self._sparse_search(query, config.TOP_K_RETRIEVE)

        fused_ids = self._reciprocal_rank_fusion([dense_ids, sparse_ids])
        candidate_ids = fused_ids[: config.TOP_K_RETRIEVE]

        pairs = [
            (query, self._truncate_for_reranker(self.id_to_text[doc_id]))
            for doc_id in candidate_ids
        ]
        rerank_scores = self.reranker.predict(pairs)

        scored = list(zip(candidate_ids, rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)
        top_ids = [doc_id for doc_id, _ in scored[: config.TOP_K_RERANK]]

        return [
            {"text": self.id_to_text[doc_id], "source": self.id_to_source[doc_id]}
            for doc_id in top_ids
        ]
