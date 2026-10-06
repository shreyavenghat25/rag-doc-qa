"""
Streamlit frontend for RAG Document Q&A.
Run: streamlit run frontend/app.py
"""
import html
import json
import os

import requests
import streamlit as st

# Local: defaults to the API on :8000. Docker / deployed: set API_BASE in the environment.
API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1").rstrip("/")
REQUEST_TIMEOUT = 120  # first query loads the reranker model, which can take a while

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🔍",
    layout="wide"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
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
  .metric-row { display: flex; gap: 1rem; margin-bottom: 1rem; }
  .stAlert { border-radius: 8px; }
  .answer-block {
    background: #f9f9f9;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    border: 1px solid #e0e0e0;
    line-height: 1.7;
  }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────
def error_detail(resp: requests.Response, fallback: str) -> str:
    """Pull FastAPI's error detail out of a response, even if the body isn't JSON."""
    try:
        detail = resp.json().get("detail", fallback)
    except ValueError:
        detail = resp.text or fallback
    return f"{fallback} ({resp.status_code}): {detail}"


def index_request(method_path: str, spinner: str, **kwargs):
    """POST to an indexing endpoint and show success / failure in the sidebar."""
    with st.spinner(spinner):
        try:
            r = requests.post(f"{API_BASE}/{method_path}", timeout=REQUEST_TIMEOUT, **kwargs)
        except requests.RequestException as e:
            st.error(f"API not reachable at {API_BASE}: {e}")
            return
    if r.ok:
        st.success(f"✅ Indexed {r.json()['chunks']} chunks")
    else:
        st.error(error_detail(r, "Indexing failed"))


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


# ─── Sidebar — Document Upload ─────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Documents")

    tab_upload, tab_url, tab_text = st.tabs(["PDF", "URL", "Text"])

    with tab_upload:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        use_semantic = st.checkbox("Semantic chunking", value=True, key="sem_pdf")
        if st.button("Index PDF", disabled=uploaded_file is None):
            index_request(
                "upload", "Indexing...",
                files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                params={"use_semantic_chunking": use_semantic},
            )

    with tab_url:
        url_input = st.text_input("URL", placeholder="https://en.wikipedia.org/wiki/...")
        use_semantic_url = st.checkbox("Semantic chunking", value=True, key="sem_url")
        if st.button("Index URL", disabled=not url_input):
            index_request(
                "index/url", "Fetching & indexing...",
                json={"url": url_input, "use_semantic_chunking": use_semantic_url},
            )

    with tab_text:
        text_input = st.text_area("Paste text", height=150)
        text_name = st.text_input("Label", value="pasted_text")
        if st.button("Index Text", disabled=not text_input):
            index_request(
                "index/text", "Indexing...",
                json={"text": text_input, "filename": text_name},
            )

    st.divider()
    st.subheader("📚 Indexed Documents")
    try:
        docs_resp = requests.get(f"{API_BASE}/documents", timeout=3)
        if docs_resp.ok:
            docs = docs_resp.json()["documents"]
            if docs:
                for doc in docs:
                    st.markdown(f"- **{doc['filename']}** ({doc['total_chunks']} chunks)")
            else:
                st.caption("No documents indexed yet.")
    except requests.RequestException:
        st.caption(f"API not reachable at {API_BASE}")

    st.divider()
    st.subheader("⚙️ Retrieval Settings")
    top_k_retrieve = st.slider("Candidates to retrieve", 5, 50, 20)
    top_k_rerank = st.slider("Final chunks after rerank", 1, 10, 5)
    use_reranker = st.checkbox("Use cross-encoder reranker", value=True)
    use_streaming = st.checkbox("Stream response", value=False)


# ─── Main — Chat Interface ─────────────────────────────────────────────────
st.title("🔍 RAG Document Q&A")
st.caption("Hybrid BM25 + dense retrieval · cross-encoder reranking · inline citations")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_citations(msg.get("citations", []))
        render_meta(msg.get("meta", {}))

# Query input
if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        params = {
            "query": query,
            "top_k_retrieve": top_k_retrieve,
            "top_k_rerank": top_k_rerank,
        }
        full_text, citations, meta = "", [], {}

        if use_streaming:
            placeholder = st.empty()
            try:
                with requests.get(
                    f"{API_BASE}/query/stream", params=params, stream=True, timeout=REQUEST_TIMEOUT
                ) as resp:
                    resp.raise_for_status()
                    for line in resp.iter_lines(decode_unicode=True):
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue
                        if event["type"] == "token":
                            full_text += event["text"]
                            placeholder.markdown(full_text + "▌")
                        elif event["type"] == "citations":
                            citations = event.get("citations", [])
                            meta = {"latency_ms": event.get("latency_ms")}
                        elif event["type"] == "error":
                            st.error(f"Generation failed: {event.get('text')}")
                placeholder.markdown(full_text or "_No answer generated._")
            except requests.RequestException as e:
                st.error(f"Streaming request failed: {e}")
                full_text = full_text or "Error during streaming."

        else:
            with st.spinner("Retrieving and generating..."):
                try:
                    r = requests.post(
                        f"{API_BASE}/query",
                        json={**params, "use_reranker": use_reranker},
                        timeout=REQUEST_TIMEOUT,
                    )
                except requests.RequestException as e:
                    r = None
                    st.error(f"API not reachable at {API_BASE}: {e}")

            if r is not None and r.ok:
                result = r.json()
                full_text = result["answer"]
                citations = result.get("citations", [])
                meta = {k: result.get(k) for k in ("latency_ms", "chunks_retrieved", "tokens_used")}
                st.markdown(full_text)
            else:
                if r is not None:
                    st.error(error_detail(r, "Query failed"))
                full_text = "Query failed."

        render_citations(citations)
        render_meta(meta)

    st.session_state.messages.append(
        {"role": "assistant", "content": full_text, "citations": citations, "meta": meta}
    )
