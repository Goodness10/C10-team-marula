"""Convert ranked predictions into the exact Kaggle submission format."""
import pandas as pd


def predictions_to_submission(predictions: dict, query_frame, valid_document_ids, k=5):
    rows = []
    for query_id in query_frame["query_id"]:
        ranked_docs = predictions.get(query_id, [])
        assert len(ranked_docs) == k, f"{query_id} has {len(ranked_docs)} predictions, expected {k}."
        for document_id in ranked_docs:
            rows.append({"QueryId": query_id, "DocumentId": document_id})

    submission = pd.DataFrame(rows, columns=["QueryId", "DocumentId"])
    expected_query_order = query_frame["query_id"].repeat(k).tolist()
    assert list(submission.columns) == ["QueryId", "DocumentId"]
    assert len(submission) == len(query_frame) * k
    assert submission["QueryId"].tolist() == expected_query_order, "Query/rank order changed."
    assert submission.groupby("QueryId", sort=False).size().eq(k).all()
    assert not submission.duplicated(["QueryId", "DocumentId"]).any(), "A query contains duplicate documents."
    assert set(submission["DocumentId"]).issubset(set(valid_document_ids)), "Unknown document ID found."
    return submission
