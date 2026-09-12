"""Hybrid retrieval: BM25 + dense ensemble combined via Reciprocal Rank Fusion."""


def retrieve_hybrid(query, bm25, documents_df, ensemble, document_ids, k=5, alpha=60):
    from src.bm25_retrieval import retrieve_bm25
    from src.dense_retrieval import retrieve_dense_ensemble

    bm25_ids = retrieve_bm25(query, bm25, documents_df, k=20)
    dense_ids = retrieve_dense_ensemble(query, ensemble, document_ids, k=20)

    rrf_scores = {}
    for rank, doc_id in enumerate(bm25_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (alpha + rank + 1))
    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (alpha + rank + 1))

    sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return sorted_docs[:k]
