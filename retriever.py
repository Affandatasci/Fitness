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

import math
import requests
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

import config


class HybridRetriever:
    def __init__(self):
        # Forced to "cpu", not auto-detected via torch.cuda.is_available().
        #
        # This class is instantiated once at app.py module load time —
        # BEFORE any @spaces.GPU-decorated function has run. On HF
        # ZeroGPU, torch.cuda.is_available() reports True even there
        # (ZeroGPU patches it so model-loading code doesn't need special
        # casing), so this used to load onto a CUDA device that hadn't
        # actually been granted yet. Running inference through that path
        # produced NaN in the output embeddings (confirmed: encode()'s
        # output was failing Python's own json.dumps with "Out of range
        # float values are not JSON compliant", i.e. NaN/Inf in the
        # vector, not a downstream API issue).
        #
        # A single query embedding (this is per-request, not batched
        # ingestion) is well under a second on CPU for this ~335M-param
        # model, so there's no throughput reason to fight ZeroGPU's
        # virtual-CUDA behaviour for this path.
        self.embedding_model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu")

        # Same reasoning applies to the reranker: same process, same
        # instantiation point, same risk of NaN scores if left on an
        # auto-detected CUDA device. NaN reranker scores wouldn't even
        # crash — sorted() just silently mis-ranks, which is worse than
        # a crash. Forced to CPU for the same reason.
        #
        # max_length is set explicitly rather than left at the model's
        # default. Our chunks are already ~512 tokens on their own —
        # pairing a full chunk with even a short 20-30 token query
        # silently pushes past a default 512-token reranker limit and
        # truncates from the end with no warning. We instead truncate
        # the CHUNK side ourselves (see _truncate_for_reranker) so the
        # truncation is visible and intentional, and leaves headroom
        # for the query within the 512-token budget.
        self.reranker = CrossEncoder(config.RERANKER_MODEL, max_length=512, device="cpu")

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
        """Returns a ranked list of point IDs from vector similarity search.

        Calls Qdrant's /points/search REST endpoint directly with `requests`,
        instead of qdrant-client's search()/query_points() wrapper methods.

        Why: search() was removed from qdrant-client in v1.16.0. Its
        replacement, query_points(), sends the query as a nested
        {"query": {"nearest": [...]}} payload — confirmed from the
        qdrant-client 1.19.1 source — and that nested shape is what this
        Qdrant Cloud cluster's 400 "Expected some form of vector, id, or a
        type of query" error was rejecting (the error column lands inside
        the vector array, i.e. mid-parse of that nested structure).

        /points/search is the older, stable endpoint: a flat
        {"vector": [...]} body with no enum-variant ambiguity for the
        server to reject. It's still live on Qdrant Cloud for backward
        compatibility — only the Python *wrapper* method was removed from
        qdrant-client, not the server-side REST route. This also sidesteps
        needing to match qdrant-client's Python API to whatever version HF
        Spaces installs going forward.

        self.qdrant_client (the SDK) is left in place for .scroll() in
        _load_corpus_for_bm25, which already works correctly against this
        cluster — proof QDRANT_URL/QDRANT_API_KEY are valid.
        """
        query_vector = self.embedding_model.encode(query, normalize_embeddings=True).tolist()

        # Safety net: fail loudly and immediately if the embedding model
        # ever produces NaN/Inf again (e.g. a future device regression),
        # instead of letting it surface three frames deep as a generic
        # requests.exceptions.InvalidJSONError with no mention of where
        # the bad value came from.
        if not all(math.isfinite(v) for v in query_vector):
            raise RuntimeError(
                f"Embedding model produced non-finite values (NaN/Inf) for "
                f"query {query!r}. embedding_model.device={self.embedding_model.device}"
            )

        url = f"{config.QDRANT_URL.rstrip('/')}/collections/{config.QDRANT_COLLECTION_NAME}/points/search"
        headers = {"api-key": config.QDRANT_API_KEY, "Content-Type": "application/json"}
        body = {
            "vector": query_vector,
            "limit": top_k,
            "with_payload": False,
            "with_vectors": False,
        }

        response = requests.post(url, json=body, headers=headers, timeout=30)
        if response.status_code != 200:
            # Surface Qdrant's actual error text instead of a bare stack
            # trace, so any further issue is diagnosable from the log
            # in one look rather than another guess-and-redeploy round.
            raise RuntimeError(
                f"Qdrant /points/search failed ({response.status_code}): {response.text[:500]}"
            )

        return [point["id"] for point in response.json()["result"]]

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