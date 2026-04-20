"""
streamlit_app.py — Self-contained RAG chatbot for Streamlit Cloud.

No separate FastAPI server needed. All RAG logic is embedded here.
Ingestion runs automatically on first startup.

Set API keys in Streamlit Cloud → App settings → Secrets:
  GROQ_API_KEY  = "gsk_..."       # free at console.groq.com (recommended)
  GITHUB_TOKEN  = "ghp_..."       # optional, avoids rate-limits
"""

from __future__ import annotations

import json
import os
import pickle
import re
import time
from pathlib import Path

import requests
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

# ── Secrets: support both .env (local) and st.secrets (Streamlit Cloud) ───────
def _secret(key: str, default: str = "") -> str:
    try:
        return st.secrets.get(key, os.getenv(key, default))
    except Exception:
        return os.getenv(key, default)

GROQ_API_KEY      = _secret("GROQ_API_KEY")
ANTHROPIC_API_KEY = _secret("ANTHROPIC_API_KEY")
OPENAI_API_KEY    = _secret("OPENAI_API_KEY")
GITHUB_TOKEN      = _secret("GITHUB_TOKEN", "")

GITHUB_REPO   = "priyanka-963/llm"
CHROMA_PATH   = "./chroma_db"
BM25_PATH     = "./bm25_index.pkl"
EMBED_MODEL   = "all-MiniLM-L6-v2"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_SIZE    = 512
CHUNK_OVERLAP = 50
HYBRID_TOP_K  = 10
RERANK_TOP_N  = 3
RRF_K         = 60
ALLOWED_EXT   = {".py", ".md", ".txt", ".rst", ".ipynb", ".json", ".yaml", ".yml"}


# ── LLM provider ─────────────────────────────────────────────────────────────
if GROQ_API_KEY:
    from openai import OpenAI as _OAI
    _llm         = _OAI(api_key=GROQ_API_KEY,
                        base_url="https://api.groq.com/openai/v1")
    LLM_PROVIDER = "openai"          # Groq is OpenAI-compatible
    LLM_MODEL    = "llama-3.3-70b-versatile"   # best free model on Groq
    FAST_MODEL   = "llama-3.1-8b-instant"      # fast cheap routing model
elif ANTHROPIC_API_KEY:
    import anthropic as _ant
    _llm         = _ant.Anthropic(api_key=ANTHROPIC_API_KEY)
    LLM_PROVIDER = "anthropic"
    LLM_MODEL    = "claude-sonnet-4-6"
    FAST_MODEL   = "claude-haiku-4-5-20251001"
elif OPENAI_API_KEY:
    from openai import OpenAI as _OAI
    _llm         = _OAI(api_key=OPENAI_API_KEY)
    LLM_PROVIDER = "openai"
    LLM_MODEL    = "gpt-4o"
    FAST_MODEL   = "gpt-4o-mini"
else:
    st.error(
        "No API key found. Add **GROQ_API_KEY** in Streamlit Cloud → Settings → Secrets.  \n"
        "Get a free key at https://console.groq.com"
    )
    st.stop()


# ── GitHub scraper ─────────────────────────────────────────────────────────────
def _gh_headers() -> dict:
    return {"Authorization": f"Bearer {GITHUB_TOKEN}"} if GITHUB_TOKEN else {}


def fetch_repo_files(repo: str) -> list[dict]:
    for branch in ("HEAD", "main", "master"):
        url  = f"https://api.github.com/repos/{repo}/git/trees/{branch}?recursive=1"
        resp = requests.get(url, headers=_gh_headers(), timeout=30)
        if resp.status_code == 200:
            break
    else:
        resp.raise_for_status()

    files = []
    for item in resp.json().get("tree", []):
        if item["type"] != "blob":
            continue
        if Path(item["path"]).suffix.lower() not in ALLOWED_EXT:
            continue
        raw = f"https://raw.githubusercontent.com/{repo}/HEAD/{item['path']}"
        try:
            r = requests.get(raw, headers=_gh_headers(), timeout=15)
            if r.status_code == 200 and r.text.strip():
                files.append({"path": item["path"], "content": r.text})
            time.sleep(0.04)
        except Exception:
            pass
    return files


# ── Ingestion ─────────────────────────────────────────────────────────────────
def run_ingestion(status_container) -> None:
    from langchain.schema import Document
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from rank_bm25 import BM25Okapi
    from sentence_transformers import SentenceTransformer
    import chromadb

    status_container.info("📥 Fetching files from GitHub …")
    raw = fetch_repo_files(GITHUB_REPO)
    if not raw:
        status_container.error("No files fetched — check GITHUB_TOKEN or repo name.")
        st.stop()

    docs = [
        Document(
            page_content=f["content"],
            metadata={"path": f["path"],
                      "source": f"https://github.com/{GITHUB_REPO}/blob/HEAD/{f['path']}"},
        )
        for f in raw
    ]

    status_container.info(f"✂️ Splitting {len(docs)} files into chunks …")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    texts  = [c.page_content for c in chunks]
    metas  = [c.metadata     for c in chunks]

    status_container.info(f"🔢 Embedding {len(chunks)} chunks …")
    model      = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, show_progress_bar=False,
                              batch_size=64, normalize_embeddings=True)

    status_container.info("💾 Storing in ChromaDB …")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection("rag_collection")
    except Exception:
        pass
    col = client.create_collection("rag_collection", metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), 100):
        sl = slice(i, i + 100)
        col.add(
            ids        = [f"chunk_{j}" for j in range(i, i + len(texts[sl]))],
            documents  = texts[sl],
            embeddings = embeddings[sl].tolist(),
            metadatas  = metas[sl],
        )

    status_container.info("🔑 Building BM25 index …")
    bm25 = BM25Okapi([t.lower().split() for t in texts])
    with open(BM25_PATH, "wb") as fh:
        pickle.dump({"bm25": bm25, "texts": texts, "metadatas": metas}, fh)

    status_container.success(f"✅ Indexed {len(chunks)} chunks — ready!")
    time.sleep(1)
    status_container.empty()


# ── Cached resource loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model …")
def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Loading reranker …")
def load_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder(RERANK_MODEL)


@st.cache_resource(show_spinner="Loading vector index …")
def load_collection():
    import chromadb
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection("rag_collection")


@st.cache_resource(show_spinner="Loading BM25 index …")
def load_bm25():
    with open(BM25_PATH, "rb") as fh:
        return pickle.load(fh)


# ── Retrieval ─────────────────────────────────────────────────────────────────
def hybrid_search(query: str) -> list[dict]:
    col   = load_collection()
    qvec  = load_embed_model().encode([query], normalize_embeddings=True)[0].tolist()
    n     = min(HYBRID_TOP_K, col.count())
    vres  = col.query(query_embeddings=[qvec], n_results=n,
                      include=["documents", "metadatas"])

    v_docs = [{"text": vres["documents"][0][i], "metadata": vres["metadatas"][0][i],
                "vector_rank": i + 1}
               for i in range(len(vres["documents"][0]))]

    bdata  = load_bm25()
    scores = bdata["bm25"].get_scores(query.lower().split())
    top_i  = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:HYBRID_TOP_K]
    b_docs = [{"text": bdata["texts"][idx], "metadata": bdata["metadatas"][idx],
                "bm25_rank": rank + 1}
               for rank, idx in enumerate(top_i)]

    rrf: dict[str, dict] = {}
    for d in v_docs:
        k = d["text"]
        rrf.setdefault(k, {"score": 0.0, "text": k, "metadata": d["metadata"]})
        rrf[k]["score"] += 1.0 / (RRF_K + d["vector_rank"])
    for d in b_docs:
        k = d["text"]
        rrf.setdefault(k, {"score": 0.0, "text": k, "metadata": d["metadata"]})
        rrf[k]["score"] += 1.0 / (RRF_K + d["bm25_rank"])

    return sorted(rrf.values(), key=lambda x: x["score"], reverse=True)[:HYBRID_TOP_K]


def rerank(query: str, candidates: list[dict]) -> list[dict]:
    if not candidates:
        return []
    pairs  = [[query, c["text"]] for c in candidates]
    scores = load_reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:RERANK_TOP_N]]


def agentic_router(query: str, context: str) -> bool:
    system = (
        "You are a relevance judge. Does the provided context contain enough "
        "specific information to answer the query accurately without hallucinating? "
        'Reply with JSON only: {"sufficient": true|false, "confidence": 0.0-1.0}'
    )
    user = f"Query: {query}\n\nContext:\n{context}"
    if LLM_PROVIDER == "anthropic":
        r = _llm.messages.create(model=FAST_MODEL, max_tokens=64,
                                  system=system,
                                  messages=[{"role": "user", "content": user}])
        raw = r.content[0].text
    else:
        r = _llm.chat.completions.create(
            model=FAST_MODEL, max_tokens=64,
            messages=[{"role": "system", "content": system},
                      {"role": "user",   "content": user}])
        raw = r.choices[0].message.content
    try:
        m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if m:
            data = json.loads(m.group())
            return not (not data.get("sufficient", True) and data.get("confidence", 0) >= 0.75)
    except Exception:
        pass
    return True   # default: allow answering


# ── Streaming generator for st.write_stream ───────────────────────────────────
SYSTEM_PROMPT = (
    "You are an expert AI/LLM assistant. Answer questions using ONLY the provided "
    "repository context. Be precise and technical. Never invent facts not in the context."
)

def answer_stream(query: str, context: str):
    user_msg = (
        f"Context (from repository):\n{context}\n\n"
        f"Question: {query}\n\nAnswer based strictly on the context above:"
    )
    if LLM_PROVIDER == "anthropic":
        with _llm.messages.stream(
            model=LLM_MODEL, max_tokens=1024, system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            yield from stream.text_stream
    else:
        stream = _llm.chat.completions.create(
            model=LLM_MODEL, max_tokens=1024, stream=True,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user",   "content": user_msg}],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


# ── Page layout ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Sajid Mock Project", page_icon="🤖", layout="wide")

with st.sidebar:
    st.title("⚙️ Settings")
    st.markdown(
        f"**Repo:** Sajid Mock Project  \n"
        f"**LLM:** `{LLM_MODEL}`  \n"
        f"**Embeddings:** `{EMBED_MODEL}`"
    )
    st.divider()
    st.markdown(
        "**Pipeline:**\n"
        "1. 🔍 Hybrid Search (BM25 + Vector)\n"
        "2. 🏆 CrossEncoder Rerank (top-3)\n"
        "3. 🤖 Agentic Router\n"
        "4. ✨ LLM Stream"
    )
    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if st.button("🔄 Re-ingest repo"):
        for key in ("rag_collection", "bm25_data", "embed_model", "reranker"):
            st.cache_resource.clear()
        if Path(BM25_PATH).exists():
            Path(BM25_PATH).unlink()
        st.rerun()

# ── Auto-ingest on first run ───────────────────────────────────────────────────
index_ready = Path(BM25_PATH).exists() and Path(CHROMA_PATH).exists()
if not index_ready:
    st.title("🤖 Sajid Mock Project")
    status = st.empty()
    status.info("🚀 First run — ingesting the repository. This takes ~1 min …")
    run_ingestion(status)
    st.rerun()

# ── Chat UI ───────────────────────────────────────────────────────────────────
st.title("🤖 Sajid Mock Project")
st.caption(f"Powered by Hybrid Search + CrossEncoder + Agentic Router | "
           f"Source: [github.com/{GITHUB_REPO}](https://github.com/{GITHUB_REPO})")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask about the LLM repository …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        candidates = hybrid_search(prompt)
        if not candidates:
            full = "I don't have enough specific info to answer that question."
            st.markdown(full)
        else:
            top_docs = rerank(prompt, candidates)
            context  = "\n\n---\n\n".join(
                f"[{d['metadata'].get('path','?')}]\n{d['text']}" for d in top_docs
            )
            if not agentic_router(prompt, context):
                full = (
                    "I don't have enough specific info in the indexed repository "
                    "to answer accurately. Try rephrasing or check the source directly."
                )
                st.markdown(full)
            else:
                full = st.write_stream(answer_stream(prompt, context))

    st.session_state.messages.append({"role": "assistant", "content": full})
