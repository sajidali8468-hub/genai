"""
ui.py — Streamlit chat UI that streams responses from the FastAPI backend.

Run with:
    streamlit run ui.py
"""

import json

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="LLM Repo Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    top_k = st.slider("Top-K docs to retrieve", min_value=1, max_value=5, value=3)
    st.divider()
    st.caption("Pipeline:")
    st.markdown(
        "1. 🔍 **Hybrid Search** — BM25 + Vector (top-10)\n"
        "2. 🏆 **CrossEncoder Rerank** — keeps top-K\n"
        "3. 🤖 **Agentic Router** — checks sufficiency\n"
        "4. ✨ **LLM Stream** — SSE token-by-token"
    )
    st.divider()

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

    # Health check
    try:
        h = requests.get(f"{API_BASE}/health", timeout=3).json()
        if h.get("status") == "ok":
            st.success(f"API online — {h.get('vector_chunks', 0)} chunks indexed")
        else:
            st.error(f"API error: {h.get('detail', 'unknown')}")
    except Exception:
        st.warning("API offline — start `uvicorn app:app --port 8000`")

# ── Chat state ────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 LLM Repository RAG Chatbot")
st.caption("Ask anything about the [priyanka-963/llm](https://github.com/priyanka-963/llm) repo.")

# ── Render history ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ─────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about the LLM repository …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder  = st.empty()
        full_response = ""
        error_msg     = None

        try:
            with requests.post(
                f"{API_BASE}/ask",
                json={"query": prompt, "top_k": top_k},
                stream=True,
                timeout=120,
                headers={"Accept": "text/event-stream"},
            ) as resp:
                resp.raise_for_status()

                for raw_line in resp.iter_lines():
                    if not raw_line:
                        continue
                    line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                    if not line.startswith("data: "):
                        continue
                    payload = json.loads(line[6:])
                    if payload.get("type") == "token":
                        full_response += payload.get("text", "")
                        placeholder.markdown(full_response + "▌")
                    elif payload.get("type") == "done":
                        break

            placeholder.markdown(full_response)

        except requests.exceptions.ConnectionError:
            error_msg = (
                "❌ Cannot connect to API server.  \n"
                "Run: `uvicorn app:app --reload --port 8000`"
            )
        except requests.exceptions.Timeout:
            error_msg = "❌ Request timed out. The LLM is taking too long — try a shorter question."
        except Exception as exc:
            error_msg = f"❌ Unexpected error: {exc}"

        if error_msg:
            placeholder.error(error_msg)
            full_response = error_msg

    st.session_state.messages.append({"role": "assistant", "content": full_response})
