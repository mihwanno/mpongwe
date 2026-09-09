import json
import faiss
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

# -----------------------------
# CONFIG
# -----------------------------

DICT_JSON_PATH_MP = "data/processed/dictionaire_mp.json"
DICT_JSON_PATH_FR = "data/processed/dictionaire_fr.json"

FAISS_INDEX_PATH_MP = "data/processed/index_mp.faiss"
FAISS_INDEX_PATH_FR = "data/processed/index_fr.faiss"

EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

# -----------------------------
# LOAD MODEL
# -----------------------------

print("Loading embedding model...")
model = SentenceTransformer(EMBEDDING_MODEL_NAME)

EMBEDDING_DIM = model.get_sentence_embedding_dimension()


def encode_dictionary(DICT_JSON_PATH, FAISS_INDEX_PATH):
    # -----------------------------
    # LOAD DICTIONARY
    # -----------------------------
    print("Loading dictionary...")
    with open(DICT_JSON_PATH, "r", encoding="utf-8") as f:
        dictionary = json.load(f)

    # -----------------------------
    # PREPARE TEXT FOR EMBEDDING
    # -----------------------------
    texts = []
    for key in tqdm(dictionary.keys()):
        texts.append(key)

    # --------------------------
    # COMPUTE EMBEDDINGS
    # -----------------------------
    print("Computing embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True  # 🔑 cosine similarity
    )

    embeddings = embeddings.astype("float32")

    # -----------------------------
    # BUILD FAISS INDEX
    # -----------------------------
    print("Building FAISS index...")
    index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product = cosine (since normalized)
    index.add(embeddings)

    print(f"Indexed {index.ntotal} dictionary entries")

    # -----------------------------
    # SAVE INDEX + METADATA
    # -----------------------------

    faiss.write_index(index, FAISS_INDEX_PATH)




encode_dictionary(DICT_JSON_PATH_MP, FAISS_INDEX_PATH_MP)
encode_dictionary(DICT_JSON_PATH_FR, FAISS_INDEX_PATH_FR)