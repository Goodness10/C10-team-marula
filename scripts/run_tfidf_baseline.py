from src.data_loading import load_data, validate_schema
from src.metrics import evaluate_ndcg_at_5
from src.tfidf_retrieval import build_tfidf_index, retrieve_tfidf
from src.submission import predictions_to_submission

documents, train_queries, qrels, test_queries, sample_submission = load_data()
validate_schema(documents, train_queries, qrels, test_queries)

vectorizer, document_matrix, document_ids = build_tfidf_index(documents)

train_predictions = retrieve_tfidf(train_queries, vectorizer, document_matrix, document_ids, k=5)
tfidf_score, query_scores = evaluate_ndcg_at_5(train_predictions, qrels)
print(f"TF-IDF training-query nDCG@5: {tfidf_score:.4f}")

test_predictions = retrieve_tfidf(test_queries, vectorizer, document_matrix, document_ids, k=5)
submission = predictions_to_submission(test_predictions, test_queries, documents["document_id"], k=5)
submission.to_csv("submission_tfidf.csv", index=False)
print("Saved submission_tfidf.csv —", submission.shape)
