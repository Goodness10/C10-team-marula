"""Load and validate the competition dataset."""
import pandas as pd
from src.config import DATA_DIR


def load_data():
    documents = pd.read_csv(DATA_DIR / "documents.csv")
    train_queries = pd.read_csv(DATA_DIR / "train_queries.csv")
    qrels = pd.read_csv(DATA_DIR / "qrels_train.csv")
    test_queries = pd.read_csv(DATA_DIR / "test_queries.csv")
    sample_submission = pd.read_csv(DATA_DIR / "sample_submission.csv")
    return documents, train_queries, qrels, test_queries, sample_submission


def validate_schema(documents, train_queries, qrels, test_queries):
    required_columns = {
        "documents": {"document_id", "title", "text"},
        "train_queries": {"query_id", "query"},
        "qrels": {"query_id", "document_id", "relevance"},
        "test_queries": {"query_id", "query"},
    }
    frames = {
        "documents": documents,
        "train_queries": train_queries,
        "qrels": qrels,
        "test_queries": test_queries,
    }
    for name, expected in required_columns.items():
        missing = expected - set(frames[name].columns)
        assert not missing, f"{name} is missing columns: {sorted(missing)}"

    assert documents["document_id"].is_unique, "Each document_id must be unique."
    assert train_queries["query_id"].is_unique, "Each training query_id must be unique."
    assert test_queries["query_id"].is_unique, "Each test query_id must be unique."
    assert qrels["relevance"].between(0, 3).all(), "Relevance values must be between 0 and 3."


def build_document_text(documents: pd.DataFrame) -> pd.Series:
    """Concatenate title + body — titles carry compact topic signal."""
    return (
        documents["title"].fillna("").astype(str)
        + ". "
        + documents["text"].fillna("").astype(str)
    )
