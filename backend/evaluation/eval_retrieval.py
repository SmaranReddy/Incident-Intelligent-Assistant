import json
from app.retrieval.search import VectorSearch
from backend.evaluation.metrics import recall_at_k


def evaluate_retrieval(k=3):
    # Load evaluation queries
    with open(
        "backend/evaluation/datasets/retrieval_queries.json",
        "r",
        encoding="utf-8",
    ) as f:
        queries = json.load(f)

    searcher = VectorSearch(
        index_path="backend/data/embeddings/faiss.index",
        metadata_path="backend/data/metadata.json",
        chunks_path="backend/data/chunks/onboarding_chunks.json",
        top_k=k,
    )

    total = 0
    correct = 0

    for item in queries:
        results = searcher.search(item["query"])
        retrieved_ids = [r["chunk_id"] for r in results]

        score = recall_at_k(
            retrieved_chunk_ids=retrieved_ids,
            relevant_chunk_id=item["relevant_chunk_id"],
            k=k,
        )

        correct += score
        total += 1

    recall = correct / total
    print(f"Recall@{k}: {recall:.2f}")


if __name__ == "__main__":
    evaluate_retrieval(k=3)
