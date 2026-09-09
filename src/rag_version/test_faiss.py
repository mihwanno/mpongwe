import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import pdb


MODEL = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
INDEX = faiss.read_index("data/processed/dictionary.faiss")

with open("data/processed/dictionaire.json", "r", encoding="utf-8") as f:
    dictionary = json.load(f)

mapping = {}
for i, key in enumerate(dictionary):
    mapping[i] = key

def vector_search(token: str, k: int = 5):
    query_vec = MODEL.encode(
        [token],
        convert_to_numpy=True,
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = INDEX.search(query_vec, k)

    return scores[0], indices[0]


text = "manger"
_, indices = vector_search(text)
for idx in indices:
    print(mapping[idx], dictionary[mapping[idx]])
