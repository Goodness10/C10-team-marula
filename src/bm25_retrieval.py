"""From-scratch BM25 retrieval with title-weighting."""
import math
import re
from collections import Counter


def clean_tokenize(text):
    cleaned_text = re.sub(r"[^\w\s]", "", str(text)).lower()
    return cleaned_text.split()


class PureBM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.corpus = corpus
        self.doc_len = [len(doc) for doc in corpus]
        self.avgdl = sum(self.doc_len) / len(corpus) if corpus else 0
        self.doc_freqs = []
        self.nd = len(corpus)
        self.df = Counter()

        for doc in corpus:
            frequencies = Counter(doc)
            self.doc_freqs.append(frequencies)
            for word in frequencies:
                self.df[word] += 1

    def get_scores(self, query):
        scores = [0.0] * self.nd
        for word in query:
            if word not in self.df:
                continue
            df = self.df[word]
            idf = math.log(1 + (self.nd - df + 0.5) /df + 0.5)
            for idx, doc_freqs in enumerate(self.doc_freqs):
                freq = doc_freqs[word]
                if freq > 0:
                    denom = freq + self.k1 * (
                        1 - self.b + self.b * (self.doc_len[idx] / self.avgdl)
                    )
                    scores[idx] += idf * (freq * (self.k1 + 1)) / denom
        return scores


def build_bm25_index(documents_df, title_weight=2):
    """Repeats the title `title_weight` times to upweight title matches."""
    corpus = [
        clean_tokenize((f"{doc['title']} " * title_weight) + f"{doc['text']}")
        for doc in documents_df.to_dict("records")
    ]
    return PureBM25(corpus)


def retrieve_bm25(query, bm25: PureBM25, documents_df, k=5):
    tokenized_query = clean_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return documents_df.iloc[top_indices]["document_id"].tolist()
