"""
ingest.py — One-time ingestion pipeline for the Forge Physique Coaching RAG demo.

Run this once (and again any time the source documents change) to:
1. Load every .md/.txt file from the documents/ folder
2. Chunk them into ~512-token pieces with overlap
3. Embed each chunk with mxbai-embed-large-v1 (uses GPU automatically
   if one is available — e.g. on Colab — otherwise falls back to CPU)
4. Upload the vectors + text + metadata into your Qdrant Cloud collection

Usage:
    python ingest.py

On Colab, run this from a notebook cell with:
    !python ingest.py
after uploading this file, config.py, .env, and the documents/ folder.
"""

import os
import glob
import uuid

import torch
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

import config


def load_documents(folder_path: str) -> list[dict]:
    """Read every .md/.txt file in folder_path. Returns [{"filename":..., "text":...}, ...]."""
    paths = sorted(
        glob.glob(os.path.join(folder_path, "*.md")) + glob.glob(os.path.join(folder_path, "*.txt"))
    )
    if not paths:
        raise FileNotFoundError(
            f"No .md or .txt files found in '{folder_path}'. "
            f"Put your source documents there before running ingest.py."
        )

    documents = []
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        documents.append({"filename": os.path.basename(path), "text": text})
    return documents


def chunk_text(text: str, chunk_size_chars: int, overlap_chars: int) -> list[str]:
    """
    Simple character-based chunking with overlap. chunk_size_chars=2000
    approximates ~512 tokens for English text (roughly 4 chars/token),
    which avoids adding a separate tokenizer dependency just for chunking.
    """
    if overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be smaller than chunk_size_chars")

    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = start + chunk_size_chars
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= text_length:
            break
        start = end - overlap_chars
    return chunks


def build_chunks(documents: list[dict]) -> list[dict]:
    """Chunk every document, tagging each chunk with its source filename and index."""
    all_chunks = []
    for doc in documents:
        pieces = chunk_text(doc["text"], config.CHUNK_SIZE_CHARS, config.CHUNK_OVERLAP_CHARS)
        for i, piece in enumerate(pieces):
            all_chunks.append({
                "text": piece,
                "source": doc["filename"],
                "chunk_index": i,
            })
    return all_chunks


def embed_chunks(chunks: list[dict], model: SentenceTransformer) -> list[list[float]]:
    """Embed all chunk texts in one batched call."""
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,  # required for cosine similarity search in Qdrant
    )
    return embeddings.tolist()


def upload_to_qdrant(chunks: list[dict], embeddings: list[list[float]], client: QdrantClient) -> None:
    """Create the collection if it doesn't exist yet, then upsert every chunk as a point."""
    existing = [c.name for c in client.get_collections().collections]
    if config.QDRANT_COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=config.QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=config.EMBEDDING_DIM, distance=Distance.COSINE),
        )
        print(f"Created new Qdrant collection: {config.QDRANT_COLLECTION_NAME}")
    else:
        print(f"Using existing Qdrant collection: {config.QDRANT_COLLECTION_NAME}")

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embeddings[i],
            payload={
                "text": chunks[i]["text"],
                "source": chunks[i]["source"],
                "chunk_index": chunks[i]["chunk_index"],
            },
        )
        for i in range(len(chunks))
    ]

    batch_size = 100
    for i in range(0, len(points), batch_size):
        batch = points[i : i + batch_size]
        client.upsert(collection_name=config.QDRANT_COLLECTION_NAME, points=batch)
        print(f"Uploaded {min(i + batch_size, len(points))}/{len(points)} chunks")


def main():
    print("Step 1/4 — Loading documents...")
    documents = load_documents(config.DOCUMENTS_FOLDER)
    print(f"Loaded {len(documents)} document(s): {[d['filename'] for d in documents]}")

    print("\nStep 2/4 — Chunking...")
    chunks = build_chunks(documents)
    print(f"Created {len(chunks)} chunks")

    print("\nStep 3/4 — Embedding (uses GPU automatically if available)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = SentenceTransformer(config.EMBEDDING_MODEL, device=device)
    embeddings = embed_chunks(chunks, model)

    print("\nStep 4/4 — Uploading to Qdrant Cloud...")
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    upload_to_qdrant(chunks, embeddings, client)

    print("\nDone. Ingestion complete — the collection is ready for retriever.py to query.")


if __name__ == "__main__":
    main()
