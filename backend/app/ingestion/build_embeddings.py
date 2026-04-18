import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


def build_embeddings(
    chunks_path: str,
    index_path: str,
    metadata_path: str,
):
    # Load chunked data
    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [chunk["text"] for chunk in chunks]

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Generate embeddings
    embeddings = model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    # Create FAISS index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Ensure output directories exist
    Path(index_path).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)

    # Save FAISS index
    faiss.write_index(index, index_path)

    # Save metadata mapping (index → chunk info)
    metadata = {
        i: {
            "chunk_id": chunks[i]["chunk_id"],
            "source": chunks[i]["metadata"]["source"],
            "section": chunks[i]["metadata"]["section"],
        }
        for i in range(len(chunks))
    }

    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    build_embeddings(
        chunks_path="backend/data/chunks/onboarding_chunks.json",
        index_path="backend/data/embeddings/faiss.index",
        metadata_path="backend/data/metadata.json",
    )
