import os

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from agents.rag.ingest import CHROMA_DIR, COLLECTION, EMBED_MODEL

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        if not os.path.exists(CHROMA_DIR):
            raise RuntimeError(
                "Chroma DB not found. Run `python -m agents.rag.ingest` first."
            )
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = _client.get_collection(
            name=COLLECTION,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL),
        )
    return _collection


def retrieve(query: str, k: int = 5) -> list[dict]:
    """
    Query the grid incident corpus.

    Returns a list of dicts with keys: text, doc_name, start_page, end_page, distance.
    Ordered by relevance (closest first).
    """
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=k)

    passages = []
    for text, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        passages.append({
            "text": text,
            "doc_name": meta["doc_name"],
            "start_page": meta["start_page"],
            "end_page": meta["end_page"],
            "distance": dist,
        })
    return passages


def format_passages(passages: list[dict]) -> str:
    """Format retrieved passages as a block for inclusion in a prompt."""
    lines = []
    for i, p in enumerate(passages, start=1):
        pages = (
            f"p.{p['start_page']}"
            if p["start_page"] == p["end_page"]
            else f"pp.{p['start_page']}–{p['end_page']}"
        )
        lines.append(f"[{i}] {p['doc_name']} ({pages}):")
        lines.append(p["text"])
        lines.append("")
    return "\n".join(lines).strip()
