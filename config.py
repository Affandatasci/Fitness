"""
config.py — Central settings for the Forge Physique Coaching RAG demo.

Every other file (ingest.py now; retriever.py, cache.py, agent.py,
app.py in the next batch) imports its settings from here. Nothing
below is a placeholder — these are the actual model names and values
the pipeline runs on.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# --- Secrets (from .env — never hardcode real keys here) ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is missing. Add it to .env")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise ValueError("QDRANT_URL / QDRANT_API_KEY is missing. Copy .env.example to .env and fill it in.")

# --- Groq LLM models ---
# Groq retires model names without warning (it happened to
# llama-3.1-8b-instant and llama-3.3-70b-versatile on Aug 16, 2026).
# If it's been more than a few weeks since you last checked, verify
# these are still live at console.groq.com/docs/models before a demo.
MAIN_MODEL = "gemini-2.5-flash"
LIGHT_MODEL = "gemini-2.5-flash"

# --- Embedding model (local, free, no API cost) ---
EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
EMBEDDING_DIM = 1024  # must match the vector size the Qdrant collection is created with — don't change without full re-ingestion

# --- Reranker (local, CPU-viable — roughly 20-40ms per pair) ---
# RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# --- Qdrant collection ---
# A NEW, dedicated collection name — this keeps the demo's vectors
# separate from your other projects sharing the same Qdrant Cloud cluster.
QDRANT_COLLECTION_NAME = "forge_physique_demo"

# --- Chunking ---
# Character-based chunking (roughly 4 characters per English token),
# approximating the ~512-token chunks used elsewhere in your stack
# without pulling in a separate tokenizer dependency just for this.
CHUNK_SIZE_CHARS = 2000       # ~512 tokens
CHUNK_OVERLAP_CHARS = 300     # ~75 tokens of overlap between consecutive chunks
DOCUMENTS_FOLDER = "data"  # ingest.py reads every .md/.txt file from this folder

# --- Retrieval ---
TOP_K_RETRIEVE = 5   # chunks pulled by hybrid search before reranking
TOP_K_RERANK = 4      # chunks that survive reranking and get sent to the LLM
RRF_K = 60             # standard Reciprocal Rank Fusion constant

# --- Self-correction loop toggle ---
# Off by default so demo responses stay in the 2-5s range (this is the
# fix for the multi-minute query times on the old BiomedicalRAG-Agent
# project — that project ran the full retrieve-grade-rewrite loop on
# local Ollama models for every single query). Flip to True only when
# you want to show a technical prospect the deeper reasoning path.
DEEP_REASONING_MODE_DEFAULT = False

# --- Semantic cache ---
CACHE_SIMILARITY_THRESHOLD = 0.95  # cosine similarity above which a repeated/similar question is served from cache instead of re-querying the LLM
