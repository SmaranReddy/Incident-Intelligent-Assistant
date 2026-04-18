def recall_at_k(results, relevant_ids, k):
    retrieved_ids = [r["chunk_id"] for r in results[:k]]
    return 1 if any(rid in retrieved_ids for rid in relevant_ids) else 0


def reciprocal_rank(results, relevant_ids):
    for idx, r in enumerate(results):
        if r["chunk_id"] in relevant_ids:
            return 1 / (idx + 1)
    return 0
