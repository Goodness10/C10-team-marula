"""Final submission: BM25 + dense ensemble, combined via RRF."""
from src.data_loading import load_data, validate_schema, build_document_text
from src.metrics import evaluate_ndcg_at_5
from src.bm25_retrieval import build_bm25_index
from src.dense_retrieval import build_ensemble
from src.hybrid_retrieval import retrieve_hybrid
from src.submission import predictions_to_submission

documents, train_queries, qrels, test_queries, sample_submission = load_data()
validate_schema(documents, train_queries, qrels, test_queries)

document_text = build_document_text(documents)
document_ids = documents["document_id"].to_numpy()

bm25 = build_bm25_index(documents, title_weight=2)
print("Loading and encoding with MiniLM, BGE, E5 (downloads models on first run)...")
ensemble = build_ensemble(document_text)

train_predictions = {
    row.query_id: retrieve_hybrid(row.query, bm25, documents, ensemble, document_ids, k=5, alpha=60)
    for row in train_queries.itertuples()
}
hybrid_score, per_query = evaluate_ndcg_at_5(train_predictions, qrels)
print(f"Hybrid (BM25+Dense, RRF) training-query nDCG@5: {hybrid_score:.4f}")

test_predictions = {
    row.query_id: retrieve_hybrid(row.query, bm25, documents, ensemble, document_ids, k=5, alpha=60)
    for row in test_queries.itertuples()
}
submission = predictions_to_submission(test_predictions, test_queries, documents["document_id"], k=5)
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv (final) —", submission.shape)
