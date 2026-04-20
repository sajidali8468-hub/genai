"""
ingest.py — Scrape GitHub repo, chunk, embed, and persist to ChromaDB + BM25.

Run once (or whenever the source repo changes):
    python ingest.py
"""

from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import chromadb

load_dotenv()

# ── Configuration ─────────────────────────────────────────────────────────────
GITHUB_REPO   = "sajid-llm"
GITHUB_TOKEN  = os.getenv("GITHUB_TOKEN", "")
CHROMA_PATH   = "./chroma_db"
BM25_PATH     = "./bm25_index.pkl"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50

# File types worth indexing from the repo
ALLOWED_EXTENSIONS = {".py", ".md", ".txt", ".rst", ".ipynb", ".json", ".yaml", ".yml"}


# ── GitHub helpers ─────────────────────────────────────────────────────────────

def _github_headers() -> dict:
    return {"Authorization": f"Bearer {GITHUB_TOKEN}", "Accept": "application/vnd.github+json"} \
        if GITHUB_TOKEN else {"Accept": "application/vnd.github+json"}


def fetch_repo_files(repo: str) -> list[dict]:
    """Return a list of {path, content} dicts for all text files in the repo."""
    tree_url = f"https://api.github.com/repos/{repo}/git/trees/HEAD?recursive=1"
    resp = requests.get(tree_url, headers=_github_headers(), timeout=30)

    if resp.status_code == 404:
        # Fallback: try 'main' branch explicitly
        tree_url = f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
        resp = requests.get(tree_url, headers=_github_headers(), timeout=30)

    resp.raise_for_status()
    tree = resp.json().get("tree", [])

    files: list[dict] = []
    for item in tree:
        if item["type"] != "blob":
            continue
        if Path(item["path"]).suffix.lower() not in ALLOWED_EXTENSIONS:
            continue

        raw_url = f"https://raw.githubusercontent.com/{repo}/HEAD/{item['path']}"
        try:
            file_resp = requests.get(raw_url, headers=_github_headers(), timeout=15)
            if file_resp.status_code == 200:
                text = file_resp.text.strip()
                if text:
                    files.append({"path": item["path"], "content": text})
                    print(f"  ✓ {item['path']}  ({len(text):,} chars)")
            time.sleep(0.05)          # gentle rate-limit
        except Exception as exc:
            print(f"  ✗ {item['path']}: {exc}")

    # Also grab the README rendered HTML as bonus context
    readme_url = f"https://api.github.com/repos/{repo}/readme"
    try:
        r = requests.get(readme_url, headers=_github_headers(), timeout=15)
        if r.status_code == 200:
            import base64, json as _json
            data = r.json()
            readme_text = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
            files.append({"path": "README.md (raw)", "content": readme_text})
            print("  ✓ README.md (raw)")
    except Exception:
        pass

    return files


# ── Main ingestion pipeline ────────────────────────────────────────────────────

def ingest() -> None:
    print(f"\n{'─'*60}")
    print(f"  Ingesting: https://github.com/{GITHUB_REPO}")
    print(f"{'─'*60}\n")

    # 1. Fetch raw files
    print("[1/5] Fetching files from GitHub …")
    raw_files = fetch_repo_files(GITHUB_REPO)
    if not raw_files:
        raise RuntimeError("No files fetched. Check GITHUB_REPO or GITHUB_TOKEN.")
    print(f"      → {len(raw_files)} files fetched\n")

    # 2. Build LangChain Documents
    print("[2/5] Building documents …")
    documents: list[Document] = []
    for f in raw_files:
        documents.append(Document(
            page_content=f["content"],
            metadata={
                "source": f"https://github.com/{GITHUB_REPO}/blob/HEAD/{f['path']}",
                "path":   f["path"],
            },
        ))

    # 3. Chunk
    print("[3/5] Splitting into chunks …")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks: list[Document] = splitter.split_documents(documents)
    print(f"      → {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})\n")

    texts    = [c.page_content for c in chunks]
    metas    = [c.metadata     for c in chunks]

    # 4. Embed + store in ChromaDB
    print("[4/5] Embedding with sentence-transformers …")
    embed_model = SentenceTransformer(EMBED_MODEL)
    embeddings  = embed_model.encode(
        texts, show_progress_bar=True, batch_size=64, normalize_embeddings=True
    )

    print("      Persisting to ChromaDB …")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection("rag_collection")
    except Exception:
        pass
    collection = client.create_collection(
        "rag_collection",
        metadata={"hnsw:space": "cosine"},
    )

    batch = 100
    for i in range(0, len(chunks), batch):
        sl = slice(i, i + batch)
        collection.add(
            ids        = [f"chunk_{j}" for j in range(i, i + len(texts[sl]))],
            documents  = texts[sl],
            embeddings = embeddings[sl].tolist(),
            metadatas  = metas[sl],
        )
    print(f"      → {collection.count()} vectors stored in {CHROMA_PATH}\n")

    # 5. Build BM25 index
    print("[5/5] Building BM25 index …")
    tokenized = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenized)

    with open(BM25_PATH, "wb") as fh:
        pickle.dump({"bm25": bm25, "texts": texts, "metadatas": metas}, fh)
    print(f"      → BM25 index saved to {BM25_PATH}\n")

    print("=" * 60)
    print(f"  Ingestion complete — {len(chunks)} chunks ready.")
    print("=" * 60)


if __name__ == "__main__":
    ingest()
