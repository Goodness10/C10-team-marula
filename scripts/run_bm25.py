from src.data_loading import load_data, validate_schema
from src.metrics import evaluate_ndcg_at_5
from src.bm25_retrieval import build_bm25_index, retrieve_bm25
from src.submission import predictions_to_submission

documents, train_queries, qrels, test_queries, sample_submission = load_data()
validate_schema(documents, train_queries, qrels, test_queries)

bm25 = build_bm25_index(documents, title_weight=2)

train_predictions = {
    row.query_id: retrieve_bm25(row.query, bm25, documents, k=5)
    for row in train_queries.itertuples()
}
bm25_score, _ = evaluate_ndcg_at_5(train_predictions, qrels)
print(f"BM25 training-query nDCG@5: {bm25_score:.4f}")

test_predictions = {
    row.query_id: retrieve_bm25(row.query, bm25, documents, k=5)
    for row in test_queries.itertuples()
}
submission = predictions_to_submission(test_predictions, test_queries, documents["document_id"], k=5)
submission.to_csv("submission_bm25.csv", index=False)
print("Saved submission_bm25.csv —", submission.shape)
