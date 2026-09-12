from src.data_loading import load_data, validate_schema, build_document_text
from src.metrics import evaluate_ndcg_at_5
from src.dense_retrieval import build_ensemble, retrieve_dense_ensemble
from src.submission import predictions_to_submission

documents, train_queries, qrels, test_queries, sample_submission = load_data()
validate_schema(documents, train_queries, qrels, test_queries)

document_text = build_document_text(documents)
document_ids = documents["document_id"].to_numpy()

print("Loading and encoding with MiniLM, BGE, E5 (downloads models on first run)...")
ensemble = build_ensemble(document_text)

train_predictions = {
    row.query_id: retrieve_dense_ensemble(row.query, ensemble, document_ids, k=5)
    for row in train_queries.itertuples()
}
dense_score, _ = evaluate_ndcg_at_5(train_predictions, qrels)
print(f"Dense ensemble training-query nDCG@5: {dense_score:.4f}")

test_predictions = {
    row.query_id: retrieve_dense_ensemble(row.query, ensemble, document_ids, k=5)
    for row in test_queries.itertuples()
}
submission = predictions_to_submission(test_predictions, test_queries, documents["document_id"], k=5)
submission.to_csv("submission_dense_ensemble.csv", index=False)
print("Saved submission_dense_ensemble.csv —", submission.shape)
