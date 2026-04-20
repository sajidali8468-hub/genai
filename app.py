"""
app.py — FastAPI RAG backend with hybrid search, CrossEncoder reranking,
         agentic routing, and SSE streaming.

Start with:
    uvicorn app:app --reload --port 8000
"""

from __future__ import annotations

import json
import os
import pickle
import re
from typing import AsyncGenerator

import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import CrossEncoder, SentenceTransformer

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
CHROMA_PATH    = "./chroma_db"
BM25_PATH      = "./bm25_index.pkl"
EMBED_MODEL    = "all-MiniLM-L6-v2"
RERANK_MODEL   = "cross-encoder/ms-marco-MiniLM-L-6-v2"
HYBRID_TOP_K   = 10    # candidates fed to reranker
RERANK_TOP_N   = 3     # docs sent to LLM
RRF_K          = 60    # RRF constant


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lazy-loaded singletons (avoids reloading on every request) ─────────────────
_embed_model:  SentenceTransformer | None = None
_reranker:     CrossEncoder        | None = None
_collection                               = None
_bm25_data:    dict                | None = None


def get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def get_reranker() -> CrossEncoder:
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker


def get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection("rag_collection")
    return _collection


def get_bm25_data() -> dict:
    global _bm25_data
    if _bm25_data is None:
        with open(BM25_PATH, "rb") as fh:
            _bm25_data = pickle.load(fh)
    return _bm25_data


# ── LLM provider selection ────────────────────────────────────────────────────
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

if ANTHROPIC_API_KEY:
    import anthropic as _ant
    _llm = _ant.Anthropic(api_key=ANTHROPIC_API_KEY)
    LLM_PROVIDER   = "anthropic"
    LLM_MODEL      = "claude-sonnet-4-6"
    ROUTER_MODEL   = "claude-haiku-4-5-20251001"     # cheap fast model for routing
elif OPENAI_API_KEY:
    from openai import OpenAI as _OAI
    _llm = _OAI(api_key=OPENAI_API_KEY)
    LLM_PROVIDER   = "openai"
    LLM_MODEL      = "gpt-4o"
    ROUTER_MODEL   = "gpt-4o-mini"
else:
    raise RuntimeError(
        "No LLM API key found. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in your .env file."
    )


# ── Pydantic models ───────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    query:  str
    top_k:  int = RERANK_TOP_N


# ── Retrieval helpers ─────────────────────────────────────────────────────────

def vector_search(query: str, k: int) -> list[dict]:
    collection = get_collection()
    n = min(k, collection.count())
    if n == 0:
        return []

    qvec = get_embed_model().encode([query], normalize_embeddings=True)[0].tolist()
    res  = collection.query(
        query_embeddings=[qvec],
        n_results=n,
        include=["documents", "metadatas"],
    )
    return [
        {"text": res["documents"][0][i], "metadata": res["metadatas"][0][i], "vector_rank": i + 1}
        for i in range(len(res["documents"][0]))
    ]


def bm25_search(query: str, k: int) -> list[dict]:
    data   = get_bm25_data()
    scores = data["bm25"].get_scores(query.lower().split())
    top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [
        {
            "text":      data["texts"][idx],
            "metadata":  data["metadatas"][idx],
            "bm25_rank": rank + 1,
            "bm25_score": float(scores[idx]),
        }
        for rank, idx in enumerate(top_idx)
    ]


def hybrid_search(query: str, top_k: int = HYBRID_TOP_K) -> list[dict]:
    """Reciprocal Rank Fusion of BM25 and vector results."""
    v_docs = vector_search(query, top_k)
    b_docs = bm25_search(query, top_k)

    rrf: dict[str, dict] = {}

    for d in v_docs:
        key = d["text"]
        rrf.setdefault(key, {"score": 0.0, "text": key, "metadata": d["metadata"]})
        rrf[key]["score"] += 1.0 / (RRF_K + d["vector_rank"])

    for d in b_docs:
        key = d["text"]
        rrf.setdefault(key, {"score": 0.0, "text": key, "metadata": d["metadata"]})
        rrf[key]["score"] += 1.0 / (RRF_K + d["bm25_rank"])

    return sorted(rrf.values(), key=lambda x: x["score"], reverse=True)[:top_k]


def rerank(query: str, candidates: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
    """CrossEncoder reranks top-K candidates, returns top-N."""
    if not candidates:
        return []
    pairs  = [[query, c["text"]] for c in candidates]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:top_n]]


# ── Agentic router ─────────────────────────────────────────────────────────────

def _call_llm_sync(system: str, user: str, model: str, max_tokens: int = 256) -> str:
    if LLM_PROVIDER == "anthropic":
        resp = _llm.messages.create(
            model=model, max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
    else:
        resp = _llm.chat.completions.create(
            model=model, max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        return resp.choices[0].message.content


def agentic_router(query: str, context: str) -> dict:
    """
    Lightweight routing step: decide whether the retrieved context
    is sufficient to answer the query faithfully.
    Returns {"has_sufficient_context": bool, "confidence": float}.
    """
    system = (
        "You are a strict relevance judge. "
        "Given a user query and retrieved context, decide whether the context "
        "contains enough specific information to answer the query accurately "
        "WITHOUT hallucinating. "
        "Reply with valid JSON only: "
        '{"has_sufficient_context": true|false, "confidence": 0.0-1.0}'
    )
    user = f"Query: {query}\n\nContext:\n{context}\n\nIs the context sufficient?"

    raw = _call_llm_sync(system, user, model=ROUTER_MODEL, max_tokens=64)
    try:
        m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
        if m:
            return json.loads(m.group())
    except Exception:
        pass
    return {"has_sufficient_context": True, "confidence": 0.5}


# ── LLM streaming generator ───────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert AI/LLM assistant. Answer questions using ONLY the provided \
repository context. Be precise and technical. If the context partially covers \
the topic, answer what you can and clearly state the limits of the available \
information. Never invent facts not present in the context.\
"""


async def stream_answer(query: str, context: str) -> AsyncGenerator[str, None]:
    user_msg = (
        f"Context (from repository):\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer based strictly on the context above:"
    )

    if LLM_PROVIDER == "anthropic":
        with _llm.messages.stream(
            model=LLM_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        ) as stream:
            for token in stream.text_stream:
                yield f"data: {json.dumps({'text': token, 'type': 'token'})}\n\n"

    else:  # openai
        stream = _llm.chat.completions.create(
            model=LLM_MODEL,
            max_tokens=1024,
            stream=True,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'text': delta, 'type': 'token'})}\n\n"

    yield f"data: {json.dumps({'type': 'done'})}\n\n"


async def _static_stream(message: str) -> AsyncGenerator[str, None]:
    yield f"data: {json.dumps({'text': message, 'type': 'token'})}\n\n"
    yield f"data: {json.dumps({'type': 'done'})}\n\n"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/ask")
async def ask(request: QueryRequest):
    """
    Full RAG pipeline:
      1. Hybrid BM25 + vector search (top-10)
      2. CrossEncoder rerank (top-3)
      3. Agentic router — refuse gracefully if context is insufficient
      4. Stream LLM answer via SSE
    """
    try:
        # ① Hybrid retrieval
        candidates = hybrid_search(request.query, top_k=HYBRID_TOP_K)
        if not candidates:
            return StreamingResponse(
                _static_stream("I don't have enough specific info to answer that question."),
                media_type="text/event-stream",
            )

        # ② Rerank
        top_docs = rerank(request.query, candidates, top_n=request.top_k)

        # ③ Build context string
        context = "\n\n---\n\n".join(
            f"[Source: {d['metadata'].get('path', 'unknown')}]\n{d['text']}"
            for d in top_docs
        )

        # ④ Agentic routing — refuse if confidence is high that context is insufficient
        routing = agentic_router(request.query, context)
        insufficient = (
            not routing.get("has_sufficient_context", True)
            and routing.get("confidence", 0) >= 0.75
        )
        if insufficient:
            msg = (
                "I don't have enough specific info in the indexed repository "
                "to answer this accurately. Consider rephrasing or checking "
                "the source directly."
            )
            return StreamingResponse(
                _static_stream(msg),
                media_type="text/event-stream",
            )

        # ⑤ Stream LLM answer
        return StreamingResponse(
            stream_answer(request.query, context),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Index not found. Run `python ingest.py` first.",
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    try:
        count = get_collection().count()
        bm25_texts = len(get_bm25_data()["texts"])
        return {
            "status":        "ok",
            "vector_chunks": count,
            "bm25_chunks":   bm25_texts,
            "llm_provider":  LLM_PROVIDER,
            "llm_model":     LLM_MODEL,
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
