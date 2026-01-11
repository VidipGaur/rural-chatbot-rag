import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---- Paths ----
BASE_PATH = "/home/gaur2/Desktop/backend/app/rag_data"
INDEX_PATH = f"{BASE_PATH}/rag_index.faiss"
CHUNKS_PATH = f"{BASE_PATH}/rag_chunks.json"

# ---- Load once at startup ----
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

index = faiss.read_index(INDEX_PATH)

with open(CHUNKS_PATH, "r") as f:
    chunks = json.load(f)

def retrieve_context(query: str, k: int = 3) -> str:
    query_vec = embedder.encode([query]).astype("float32")
    distances, indices = index.search(query_vec, k)

    retrieved = []
    for idx in indices[0]:
        retrieved.append(chunks[idx]["chunk"])

    return "\n\n".join(retrieved)
