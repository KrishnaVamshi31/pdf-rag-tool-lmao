import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

_model = None

def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_documents(documents):
    model = get_embedding_model()
    texts = [doc.page_content for doc in documents]
    vectors = model.encode(texts, show_progress_bar=True)
    return np.array(vectors).astype("float32")  # FAISS requires float32


class VectorStore:
    """
    Wraps a FAISS index + keeps the original Document objects alongside it,
    so that when FAISS returns "vector #7 is closest", we can map that
    back to the actual chunk text + metadata.
    """
    def __init__(self, documents, vectors):
        self.documents = documents
        # Normalize vectors so that FAISS's inner-product search
        # behaves like cosine similarity (this is the standard trick --
        # cosine similarity == dot product when vectors are unit-length)
        faiss.normalize_L2(vectors)
        self.index = faiss.IndexFlatIP(vectors.shape[1])  # IP = inner product
        self.index.add(vectors)

    def similarity_search(self, query: str, k: int = 4):
        model = get_embedding_model()
        query_vector = model.encode([query]).astype("float32")
        faiss.normalize_L2(query_vector)

        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            doc = self.documents[idx]
            results.append((doc, float(score)))
        return results


def build_vector_store(documents):
    vectors = embed_documents(documents)
    return VectorStore(documents, vectors)


if __name__ == "__main__":
    from ingestion import load_pdf, split_documents

    docs = load_pdf("sample.pdf")
    chunks = split_documents(docs)
    store = build_vector_store(chunks)

    query = "how does mutual induction work in a transformer"
    results = store.similarity_search(query, k=3)

    print(f"Query: {query}\n")
    for doc, score in results:
        print(f"Score: {score:.4f} | Page: {doc.metadata['page']} | Chunk: {doc.metadata['chunk_index']}")
        print(doc.page_content[:200])
        print("---")