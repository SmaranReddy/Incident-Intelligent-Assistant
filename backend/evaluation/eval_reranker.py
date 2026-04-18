import json
from app.retrieval.search import hybrid_search
from app.retrieval.rerank import rerank_results
from app.retrieval.vector_store import VectorStore
from backend.evaluation.metrics import recall_at_k, reciprocal_rank

DATASET = "backend/evaluation/datasets/retrieval_queries.json"


def evaluate():
    vector_store = VectorStore()
    vector_store.load("backend/data/embeddings/faiss.index")

    with open(DATASET, "r") as f:
        queries = json.load(f)

    hybrid_recall, rerank_recall = [], []
    hybrid_mrr, rerank_mrr = [], []

    for q in queries:
        query = q["query"]
        relevant_ids = q["relevant_chunk_ids"]

        hybrid_results = hybrid_search(query, vector_store, top_k=10)
        reranked_results = rerank_results(query, hybrid_results)

        hybrid_recall.append(recall_at_k(hybrid_results, relevant_ids, k=3))
        rerank_recall.append(recall_at_k(reranked_results, relevant_ids, k=3))

        hybrid_mrr.append(reciprocal_rank(hybrid_results, relevant_ids))
        rerank_mrr.append(reciprocal_rank(reranked_results, relevant_ids))

    print("\n=== RERANKER EVALUATION ===\n")
    print(f"Hybrid Recall@3:   {sum(hybrid_recall)/len(hybrid_recall):.2f}")
    print(f"Reranked Recall@3: {sum(rerank_recall)/len(rerank_recall):.2f}")
    print()
    print(f"Hybrid MRR:        {sum(hybrid_mrr)/len(hybrid_mrr):.2f}")
    print(f"Reranked MRR:      {sum(rerank_mrr)/len(rerank_mrr):.2f}")


if __name__ == "__main__":
    evaluate()
