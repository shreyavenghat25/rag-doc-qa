"""
Streamlit frontend for RAG Document Q&A.

Local (with the FastAPI service running):   streamlit run frontend/app.py
Self-contained demo (no API server needed): RAG_MODE=embedded streamlit run frontend/app.py
"""
import html
import os
import sys
from pathlib import Path

import streamlit as st

# Put the repo root first on the path so `import app...` resolves to the backend
# package, not to this file (which Streamlit runs from the frontend/ folder).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from frontend.backends import ApiBackend, BackendError, EmbeddedBackend  # noqa: E402

MODE = os.getenv("RAG_MODE", "api").lower()          # "api" | "embedded"
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")
SAMPLE_DOC = ROOT / "README.md"

st.set_page_config(page_title="RAG Document Q&A", page_icon="🔍", layout="wide")


# ─── Backend selection ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model (first visit only)…")
def warm_up_models():
    """Load the shared models once per process, not once per visitor."""
    from app.core.embedder import get_embedder
    return get_embedder()


def get_backend():
    if MODE != "embedded":
        return ApiBackend(API_BASE)
    if "rag_session" not in st.session_state:
        warm_up_models()
        from app.core.session import RAGSession
        st.session_state.rag_session = RAGSession()  # private index for this visitor
    return EmbeddedBackend(st.session_state.rag_session)


backend = get_backend()

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .citation-box {
    background: rgba(79, 110, 247, 0.08);
    border-left: 3px solid #4f6ef7;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
    color: inherit;
  }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────
def run_index(spinner: str, fn, *args):
    with st.spinner(spinner):
        try:
            result = fn(*args)
        except BackendError as e:
            st.error(str(e))
            return
    if result.get("status") == "duplicate":
        st.info("This document is already indexed.")
    else:
        st.success(f"✅ Indexed {result['chunks']} chunks")


def render_citations(citations: list[dict]):
    if not citations:
        return
    with st.expander(f"📎 {len(citations)} source(s) cited"):
        for c in citations:
            # Escape document text: it comes from user-uploaded files and is rendered as HTML.
            filename = html.escape(str(c.get("filename", "Unknown")))
            preview = html.escape(str(c.get("text_preview", "")))
            st.markdown(
                f'<div class="citation-box">'
                f'<strong>[Source {c["source_n"]}]</strong> {filename} · page {c.get("page", "?")}'
                f'<br><small>{preview}</small>'
                f'</div>',
                unsafe_allow_html=True,
            )


def render_meta(meta: dict):
    if not meta:
        return
    cols = st.columns(3)
    cols[0].metric("Latency", f"{meta.get('latency_ms') or 0:.0f} ms")
    cols[1].metric("Chunks retrieved", meta.get("chunks_retrieved") or "–")
    cols[2].metric("Tokens used", meta.get("tokens_used") or "–")


# ─── Sidebar — documents & settings ──────────────────────────────────────────
with st.sidebar:
    st.title("📄 Documents")

    if MODE == "embedded":
        st.caption("Your documents are private to this browser tab and are cleared when you close it.")
        if SAMPLE_DOC.exists() and st.button("Try a sample: this project's README"):
            run_index("Indexing sample…", backend.index_text, SAMPLE_DOC.read_text(), "README.md")

    tab_upload, tab_url, tab_text = st.tabs(["PDF", "URL", "Text"])

    with tab_upload:
        use_semantic = st.checkbox("Semantic chunking", value=True, key="sem_pdf")
        uploaded_file = st.file_uploader("Upload PDF (indexed automatically)", type=["pdf"])
        # Index as soon as a file is chosen; remember it so reruns don't re-index.
        indexed_uploads = st.session_state.setdefault("indexed_uploads", set())
        if uploaded_file is not None and uploaded_file.file_id not in indexed_uploads:
            indexed_uploads.add(uploaded_file.file_id)
            run_index(f"Indexing {uploaded_file.name}…", backend.index_pdf,
                      uploaded_file.getvalue(), uploaded_file.name, use_semantic)

    with tab_url:
        url_input = st.text_input("URL", placeholder="https://en.wikipedia.org/wiki/...")
        use_semantic_url = st.checkbox("Semantic chunking", value=True, key="sem_url")
        if st.button("Index URL", disabled=not url_input):
            run_index("Fetching & indexing…", backend.index_url, url_input, use_semantic_url)

    with tab_text:
        text_input = st.text_area("Paste text", height=150)
        text_name = st.text_input("Label", value="pasted_text")
        if st.button("Index Text", disabled=not text_input):
            run_index("Indexing…", backend.index_text, text_input, text_name)

    st.divider()
    st.subheader("📚 Indexed Documents")
    docs = []
    try:
        docs = backend.list_documents()
        if docs:
            for doc in docs:
                st.markdown(f"- **{html.escape(doc['filename'])}** ({doc['total_chunks']} chunks)")
        else:
            st.caption("No documents indexed yet.")
    except BackendError as e:
        st.caption(str(e))

    st.divider()
    st.subheader("⚙️ Retrieval Settings")
    top_k_retrieve = st.slider("Candidates to retrieve", 5, 50, 20)
    top_k_rerank = st.slider("Final chunks after rerank", 1, 10, 5)
    use_reranker = st.checkbox("Use cross-encoder reranker", value=True)
    use_streaming = st.checkbox("Stream response", value=True)


# ─── Main — chat ─────────────────────────────────────────────────────────────
st.title("🔍 RAG Document Q&A")
st.caption("Hybrid BM25 + dense retrieval · cross-encoder reranking · inline citations")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_citations(msg.get("citations", []))
        render_meta(msg.get("meta", {}))

if not docs:
    st.info("👈 Add a document first: upload a PDF, paste text, or click **Try a sample** in the sidebar.")

if query := st.chat_input(
    "Ask a question about your documents..." if docs else "Add a document to start asking questions",
    disabled=not docs,
):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        full_text, citations, meta = "", [], {}

        if use_streaming:
            placeholder = st.empty()
            try:
                for event in backend.query_stream(query, top_k_retrieve, top_k_rerank):
                    if event["type"] == "token":
                        full_text += event["text"]
                        placeholder.markdown(full_text + "▌")
                    elif event["type"] == "citations":
                        citations = event.get("citations", [])
                        meta = {"latency_ms": event.get("latency_ms")}
                    elif event["type"] == "error":
                        st.error(f"Generation failed: {event.get('text')}")
            except BackendError as e:
                st.error(str(e))
            full_text = full_text or "Error during streaming."
            placeholder.markdown(full_text)
        else:
            with st.spinner("Retrieving and generating…"):
                try:
                    result = backend.query(query, top_k_retrieve, top_k_rerank, use_reranker)
                except BackendError as e:
                    result = None
                    st.error(str(e))
            if result:
                full_text = result["answer"]
                citations = result.get("citations", [])
                meta = {k: result.get(k) for k in ("latency_ms", "chunks_retrieved", "tokens_used")}
                st.markdown(full_text)
            else:
                full_text = "Query failed."

        render_citations(citations)
        render_meta(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_text, "citations": citations, "meta": meta}
    )
