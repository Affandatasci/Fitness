from sentence_transformers import SentenceTransformer
import config

model = SentenceTransformer(config.EMBEDDING_MODEL, device="cpu", model_kwargs={"torch_dtype": "float32"})

raw = model.encode("hi", normalize_embeddings=False)
raw_has_nan = any(x != x or abs(x) == float("inf") for x in raw)
raw_norm = sum(float(x) * float(x) for x in raw) ** 0.5

print("raw output has NaN/Inf:", raw_has_nan)
print("raw output length:", len(raw))
print("raw vector norm (should NOT be near 0):", raw_norm)
print("first 5 raw values:", list(raw[:5]))