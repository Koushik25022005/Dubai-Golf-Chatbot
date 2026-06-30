import os
import re
import json
import pickle
from itertools import zip_longest

import onnxruntime as ort

# onnxruntime on macOS auto-selects CoreMLExecutionProvider by default.
# Chroma's bundled MiniLM ONNX model has dynamic-shape inputs that CoreML's
# backend doesn't reliably handle, causing a "Status Message: Error
# executing model" failure at inference time. Force CPU-only execution so
# this never gets attempted.
_original_session_init = ort.InferenceSession.__init__


def _cpu_only_session_init(self, *args, **kwargs):
    kwargs.pop("providers", None)
    kwargs.pop("provider_options", None)
    kwargs["providers"] = ["CPUExecutionProvider"]
    return _original_session_init(self, *args, **kwargs)


ort.InferenceSession.__init__ = _cpu_only_session_init
ort.get_available_providers = lambda: ["CPUExecutionProvider"]

import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_ingestion", "cleaned_data.jsonl")
BM25_STATE_PATH = os.path.join(os.path.dirname(__file__), "bm25_state.pkl")


def chunk_text(text, chunk_size=500, overlap=50):
    """Simple character-based chunker with overlap. Chunking is pure text
    splitting -- it never needed torch or faiss, so this has no native
    dependencies at all, on any CPU."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if end >= len(text):
            break
    return chunks


class HybridSearchKnowledgeBase:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=DB_DIR)

        # ChromaDB's DefaultEmbeddingFunction runs all-MiniLM-L6-v2 through
        # ONNX Runtime, not PyTorch. Same weights/quality as
        # SentenceTransformerEmbeddingFunction, but no libtorch.dylib /
        # libiomp5.dylib involved -- which is what was colliding with
        # FAISS's bundled libomp.dylib and causing the segfault.
        self.embedding_func = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.chroma_client.get_or_create_collection(
            name="dubaigolf_knowledge",
            embedding_function=self.embedding_func,
        )

        # BM25 replaces TF-IDF + FAISS for sparse search. rank_bm25 is pure
        # Python -- no compiled extension, no OpenMP, nothing to collide.
        self.bm25 = None
        self.bm25_corpus_ids = []
        self.document_map = {}
        self._load_bm25_state()

    def _load_bm25_state(self):
        if os.path.exists(BM25_STATE_PATH):
            with open(BM25_STATE_PATH, "rb") as f:
                state = pickle.load(f)
            self.bm25 = state["bm25"]
            self.bm25_corpus_ids = state["ids"]
            self.document_map = state["document_map"]

    def build_index(self):
        if not os.path.exists(DATA_FILE):
            print(f"Error: {DATA_FILE} not found. Please run data ingestion first.")
            return

        documents, metadatas, ids = [], [], []
        print("Reading cleaned data...")
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                chunks = chunk_text(data["text"])
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadatas.append({"url": data["url"]})
                    ids.append(f"{data['chunk_id']}_{i}")

        if not documents:
            print("No documents to index.")
            return

        print(f"Adding {len(documents)} documents to ChromaDB (ONNX embeddings)...")
        batch_size = 5000
        for i in range(0, len(documents), batch_size):
            self.collection.add(
                documents=documents[i:i + batch_size],
                metadatas=metadatas[i:i + batch_size],
                ids=ids[i:i + batch_size],
            )

        print("Building BM25 sparse index...")
        tokenized_corpus = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.bm25_corpus_ids = ids
        self.document_map = {
            doc_id: {"text": documents[idx], "url": metadatas[idx]["url"]}
            for idx, doc_id in enumerate(ids)
        }

        with open(BM25_STATE_PATH, "wb") as f:
            pickle.dump(
                {
                    "bm25": self.bm25,
                    "ids": self.bm25_corpus_ids,
                    "document_map": self.document_map,
                },
                f,
            )

        print("Knowledge base index built successfully.")

    def search(self, query, top_k=3):
        # Dense search (ChromaDB + ONNX embeddings)
        dense_results = self.collection.query(query_texts=[query], n_results=top_k)
        dense_docs = dense_results["documents"][0] if dense_results["documents"] else []

        # Sparse search (BM25)
        sparse_docs = []
        if self.bm25 is not None:
            tokenized_query = query.lower().split()
            scores = self.bm25.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
            MIN_BM25_SCORE = 0.5  # filters out near-zero / no-overlap matches
            for idx in top_indices:
                if scores[idx] > MIN_BM25_SCORE:
                    doc_id = self.bm25_corpus_ids[idx]
                    sparse_docs.append(self.document_map[doc_id]["text"])

        # Hybrid fusion (interleaved, so sparse matches aren't crowded out
        # by dense results when top_k is small)
        combined_results = []
        seen = set()
        for dense_doc, sparse_doc in zip_longest(dense_docs, sparse_docs):
            for doc in (dense_doc, sparse_doc):
                if doc is not None and doc not in seen:
                    seen.add(doc)
                    combined_results.append(doc)

        return combined_results[:top_k]


if __name__ == "__main__":
    kb = HybridSearchKnowledgeBase()
    kb.build_index()

