import json
from app.retrieval.search import hybrid_search
from app.retrieval.vector_store import VectorStore
from backend.evaluation.metrics import recall_at_k, reciprocal_rank

DATASET = "backend/evaluation/datasets/retrieval_queries.json"


def evaluate_alpha(alpha):

    vector_store = VectorStore()
    vector_store.load("backend/data/embeddings/faiss.index")

    with open(DATASET, "r") as f:
        queries = json.load(f)

    recalls = []
    mrrs = []

    for q in queries:

        query = q["query"]
        relevant_ids = q["relevant_chunk_ids"]

        results = hybrid_search(
            query=query,
            vector_store=vector_store,
            alpha=alpha,
            top_k=10
        )

        recalls.append(recall_at_k(results, relevant_ids, k=3))
        mrrs.append(reciprocal_rank(results, relevant_ids))

    recall = sum(recalls) / len(recalls)
    mrr = sum(mrrs) / len(mrrs)

    return recall, mrr


def run_experiment():

    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]

    print("\n=== HYBRID SEARCH ALPHA SWEEP ===\n")

    for alpha in alphas:

        recall, mrr = evaluate_alpha(alpha)

        print(f"alpha={alpha:.2f}  Recall@3={recall:.2f}  MRR={mrr:.2f}")


if __name__ == "__main__":
    run_experiment()