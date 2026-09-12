"""Central configuration: dataset location and model names.

On Kaggle, DATA_DIR is auto-detected. Locally, download the competition
data (documents.csv, train_queries.csv, qrels_train.csv, test_queries.csv,
sample_submission.csv) and point DATA_DIR at that folder, or set the
env var AGRI_RAG_DATA_DIR.
"""
import os
from pathlib import Path

_KAGGLE_DEFAULT = Path(
    "/kaggle/input/competitions/agricultural-extension-rag-smart-retrieval-for-farmers"
)


def resolve_data_dir() -> Path:
    env_override = os.environ.get("AGRI_RAG_DATA_DIR")
    if env_override:
        return Path(env_override)
    if (_KAGGLE_DEFAULT / "documents.csv").exists():
        return _KAGGLE_DEFAULT
    # Fallback: search /kaggle/input for the file if the slug changed.
    for candidate in Path("/kaggle/input").glob("**/documents.csv"):
        return candidate.parent
    # Local fallback: ./data
    local = Path("data")
    if (local / "documents.csv").exists():
        return local
    raise FileNotFoundError(
        "Could not find documents.csv. Set AGRI_RAG_DATA_DIR to the folder "
        "containing the competition CSVs."
    )


DATA_DIR = resolve_data_dir()

# HuggingFace model names used for dense retrieval (downloaded automatically;
# no Kaggle-attached model dependency, so this runs anywhere).
DENSE_MODELS = {
    "minilm": {"name": "sentence-transformers/all-MiniLM-L6-v2", "query_prefix": "", "doc_prefix": ""},
    "bge": {"name": "BAAI/bge-base-en-v1.5", "query_prefix": "", "doc_prefix": ""},
    "e5": {"name": "intfloat/e5-base-v2", "query_prefix": "query: ", "doc_prefix": "passage: "},
}
