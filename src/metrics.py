"""nDCG@5 — the competition's evaluation metric."""
import numpy as np
import pandas as pd


def dcg(relevances, k=5):
    """Discounted cumulative gain for an ordered relevance list."""
    values = np.asarray(list(relevances)[:k], dtype=float)
    if len(values) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(values) + 2))
    return float(np.sum(values / discounts))


def evaluate_ndcg_at_5(predictions: dict, qrels_frame: pd.DataFrame):
    """predictions: {query_id: [ranked document_ids]}."""
    relevance_lookup = {
        query_id: dict(zip(group["document_id"], group["relevance"]))
        for query_id, group in qrels_frame.groupby("query_id")
    }
    per_query = {}
    for query_id, judged_docs in relevance_lookup.items():
        ranked_docs = predictions.get(query_id, [])[:5]
        predicted_relevance = [judged_docs.get(doc_id, 0) for doc_id in ranked_docs]
        ideal_relevance = sorted(judged_docs.values(), reverse=True)[:5]
        ideal_dcg = dcg(ideal_relevance, k=5)
        per_query[query_id] = dcg(predicted_relevance, k=5) / ideal_dcg if ideal_dcg else 0.0
    return float(np.mean(list(per_query.values()))), per_query


def _sanity_check():
    toy_qrels = pd.DataFrame({
        "query_id": ["q1", "q1", "q1"],
        "document_id": ["d1", "d2", "d3"],
        "relevance": [3, 2, 1],
    })
    score, _ = evaluate_ndcg_at_5({"q1": ["d1", "d2", "d3"]}, toy_qrels)
    assert np.isclose(score, 1.0)


if __name__ == "__main__":
    _sanity_check()
    print("Metric sanity check passed.")
