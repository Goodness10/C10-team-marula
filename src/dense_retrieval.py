"""Dense retrieval: single encoder + 3-model ensemble (MiniLM/BGE/E5)."""
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoTokenizer
from src.config import DENSE_MODELS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class DenseEncoder:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(DEVICE)
        self.model.eval()

    def encode(self, texts, prefix="", batch_size=32, max_length=256):
        texts = [f"{prefix}{t}" for t in pd.Series(texts).fillna("").astype(str)]
        batches = []
        with torch.no_grad():
            for start in range(0, len(texts), batch_size):
                tokens = self.tokenizer(
                    texts[start:start + batch_size],
                    padding=True, truncation=True, max_length=max_length,
                    return_tensors="pt",
                ).to(DEVICE)
                output = self.model(**tokens)
                embeddings = mean_pool(output.last_hidden_state, tokens["attention_mask"])
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                batches.append(embeddings.cpu().numpy())
        return np.vstack(batches)


def retrieve_dense(query_frame, encoder, document_vectors, document_ids, k=5, query_prefix=""):
    query_vectors = encoder.encode(query_frame["query"], prefix=query_prefix, batch_size=32)
    similarities = query_vectors @ document_vectors.T
    results = {}
    for row_number, query_id in enumerate(query_frame["query_id"]):
        scores = similarities[row_number]
        candidates = np.argpartition(-scores, min(k, len(scores)) - 1)[:k]
        ranked = candidates[np.argsort(-scores[candidates])]
        results[query_id] = document_ids[ranked].tolist()
    return results


def build_ensemble(document_text):
    """Loads all 3 models and encodes the corpus once. Returns a dict of
    {name: (encoder, doc_vectors, doc_prefix, query_prefix)}."""
    ensemble = {}
    for name, cfg in DENSE_MODELS.items():
        encoder = DenseEncoder(cfg["name"])
        doc_vectors = encoder.encode(document_text, prefix=cfg["doc_prefix"])
        ensemble[name] = {
            "encoder": encoder,
            "doc_vectors": doc_vectors,
            "query_prefix": cfg["query_prefix"],
        }
    return ensemble


def retrieve_dense_ensemble(query, ensemble, document_ids, k=5):
    """Average cosine similarity across MiniLM, BGE, and E5."""
    scores_per_model = []
    for name, m in ensemble.items():
        q_vec = m["encoder"].encode([query], prefix=m["query_prefix"])
        scores_per_model.append((q_vec @ m["doc_vectors"].T)[0])
    ensemble_scores = np.mean(scores_per_model, axis=0)

    k = min(k, len(ensemble_scores))
    candidates = np.argpartition(-ensemble_scores, k - 1)[:k]
    ranked = candidates[np.argsort(-ensemble_scores[candidates])]
    return document_ids[ranked].tolist()
