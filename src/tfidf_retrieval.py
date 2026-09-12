"""TF-IDF baseline retrieval."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from src.data_loading import build_document_text


def build_tfidf_index(documents):
    document_text = build_document_text(documents)
    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
    )
    document_matrix = normalize(vectorizer.fit_transform(document_text))
    document_ids = documents["document_id"].to_numpy()
    return vectorizer, document_matrix, document_ids


def retrieve_tfidf(query_frame, vectorizer, document_matrix, document_ids, k=5):
    query_matrix = normalize(vectorizer.transform(query_frame["query"].fillna("").astype(str)))
    similarities = query_matrix @ document_matrix.T
    results = {}
    for row_number, query_id in enumerate(query_frame["query_id"]):
        scores = similarities.getrow(row_number).toarray().ravel()
        candidate_positions = np.argpartition(-scores, min(k, len(scores)) - 1)[:k]
        ranked_positions = candidate_positions[np.argsort(-scores[candidate_positions])]
        results[query_id] = document_ids[ranked_positions].tolist()
    return results
