"""
Download RAG corpus PDFs, chunk them, embed with sentence-transformers, store in Chroma.

Run once before using the LLM reporting agent:
    python -m agents.rag.ingest

Re-run with --force to re-embed (e.g. after adding new documents).
"""

import argparse
import os
import time

import chromadb
import requests
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pypdf import PdfReader

CORPUS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "corpus")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "chroma")
COLLECTION = "grid_incidents"
EMBED_MODEL = "all-MiniLM-L6-v2"

CHUNK_SIZE = 400    # words per chunk
CHUNK_OVERLAP = 80  # words of overlap between consecutive chunks

DOCUMENTS = [
    {
        "id": "ferc_nerc_2021",
        "name": "FERC/NERC Feb 2021 Texas Cold Weather Outages Report",
        "url": "https://www.ferc.gov/media/february-2021-cold-weather-outages-texas-and-south-central-united-states-ferc-nerc-and",
        "filename": "ferc_nerc_2021_texas_outages.pdf",
    },
    {
        "id": "ut_austin_2021",
        "name": "UT Austin Feb 2021 Texas Blackout Timeline",
        "url": "https://energy.utexas.edu/sites/default/files/UTAustin%20(2021)%20EventsFebruary2021TexasBlackout%2020210714.pdf",
        "filename": "ut_austin_2021_blackout_timeline.pdf",
    },
    {
        "id": "nerc_wra_2021",
        "name": "NERC 2021–2022 Winter Reliability Assessment",
        "url": "https://www.nerc.com/globalassets/programs/rapa/ra/nerc_wra_2021.pdf",
        "filename": "nerc_wra_2021.pdf",
    },
]


def _download(doc: dict) -> str | None:
    """Download a PDF. Returns path on success, None if URL serves non-PDF content."""
    path = os.path.join(CORPUS_DIR, doc["filename"])
    if os.path.exists(path):
        print(f"  already downloaded: {doc['filename']}")
        return path
    print(f"  downloading {doc['filename']} ...")
    r = requests.get(doc["url"], timeout=120, stream=True)
    r.raise_for_status()

    # Read the first 8 bytes to confirm it's a PDF before writing
    header = b""
    chunks = r.iter_content(chunk_size=8192)
    first = next(chunks, b"")
    header = first[:8]
    if not header.lstrip().startswith(b"%PDF"):
        print(f"  WARNING: URL did not return a PDF (got {header[:20]!r})")
        print(f"  Download manually and save to: {path}")
        return None

    with open(path, "wb") as f:
        f.write(first)
        for chunk in chunks:
            f.write(chunk)
    print(f"  saved ({os.path.getsize(path) / 1e6:.1f} MB)")
    return path


def _extract_text(path: str) -> list[tuple[int, str]]:
    """Returns list of (page_number, page_text) tuples."""
    reader = PdfReader(path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = " ".join(text.split())  # collapse whitespace
        if text:
            pages.append((i, text))
    return pages


def _chunk(pages: list[tuple[int, str]]) -> list[dict]:
    """Sliding window over words across all pages, preserving page metadata."""
    words_with_page = []
    for page_num, text in pages:
        for word in text.split():
            words_with_page.append((page_num, word))

    chunks = []
    i = 0
    while i < len(words_with_page):
        window = words_with_page[i: i + CHUNK_SIZE]
        text = " ".join(w for _, w in window)
        start_page = window[0][0]
        end_page = window[-1][0]
        chunks.append({"text": text, "start_page": start_page, "end_page": end_page})
        i += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def ingest(force: bool = False) -> None:
    os.makedirs(CORPUS_DIR, exist_ok=True)
    os.makedirs(CHROMA_DIR, exist_ok=True)

    embed_fn = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    if force and COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)
        print("Deleted existing collection.")

    collection = client.get_or_create_collection(
        name=COLLECTION,
        embedding_function=embed_fn,
        metadata={"hnsw:space": "cosine"},
    )

    existing = set(collection.get(include=[])["ids"])
    print(f"Existing chunks in collection: {len(existing)}")

    for doc in DOCUMENTS:
        print(f"\n[{doc['id']}] {doc['name']}")
        path = _download(doc)
        if path is None:
            print("  skipping (not a PDF — add manually to docs/corpus/)")
            continue

        print("  extracting text ...")
        pages = _extract_text(path)
        print(f"  {len(pages)} pages with text")

        chunks = _chunk(pages)
        print(f"  {len(chunks)} chunks")

        new_ids, new_texts, new_metas = [], [], []
        for j, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk_{j:04d}"
            if chunk_id in existing:
                continue
            new_ids.append(chunk_id)
            new_texts.append(chunk["text"])
            new_metas.append({
                "doc_id": doc["id"],
                "doc_name": doc["name"],
                "start_page": chunk["start_page"],
                "end_page": chunk["end_page"],
            })

        if not new_ids:
            print("  all chunks already ingested, skipping")
            continue

        # Chroma recommends batches of ≤5000
        batch = 500
        for start in range(0, len(new_ids), batch):
            collection.add(
                ids=new_ids[start: start + batch],
                documents=new_texts[start: start + batch],
                metadatas=new_metas[start: start + batch],
            )
            time.sleep(0.1)

        print(f"  ingested {len(new_ids)} new chunks")

    total = collection.count()
    print(f"\nDone. Collection '{COLLECTION}' has {total} chunks total.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--force", action="store_true", help="Re-embed from scratch")
    args = p.parse_args()
    ingest(force=args.force)
