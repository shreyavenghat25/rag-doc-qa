"""
Streamlit frontend for RAG Document Q&A.
Run: streamlit run frontend/app.py
"""
import json
import time
import requests
import streamlit as st

API_BASE = "http://localhost:8001/api/v1"

st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="🔍",
    layout="wide"
)

# ─── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  .citation-box {
    background: #f0f4ff;
    border-left: 3px solid #4f6ef7;
    border-radius: 4px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 13px;
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


# ─── Sidebar — Document Upload ─────────────────────────────────────────────
with st.sidebar:
    st.title("📄 Documents")

    tab_upload, tab_url, tab_text = st.tabs(["PDF", "URL", "Text"])

    with tab_upload:
        uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
        use_semantic = st.checkbox("Semantic chunking", value=True, key="sem_pdf")
        if st.button("Index PDF", disabled=uploaded_file is None):
            with st.spinner("Indexing..."):
                r = requests.post(
                    f"{API_BASE}/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                    params={"use_semantic_chunking": use_semantic}
                )
                if r.ok:
                    d = r.json()
                    st.success(f"✅ Indexed {d['chunks']} chunks")
                else:
                    st.error(r.json().get("detail", "Upload failed"))

    with tab_url:
        url_input = st.text_input("URL", placeholder="https://en.wikipedia.org/wiki/...")
        use_semantic_url = st.checkbox("Semantic chunking", value=True, key="sem_url")
        if st.button("Index URL", disabled=not url_input):
            with st.spinner("Fetching & indexing..."):
                r = requests.post(f"{API_BASE}/index/url", json={
                    "url": url_input,
                    "use_semantic_chunking": use_semantic_url
                })
                if r.ok:
                    d = r.json()
                    st.success(f"✅ Indexed {d['chunks']} chunks")
                else:
                    st.error(r.json().get("detail", "Indexing failed"))

    with tab_text:
        text_input = st.text_area("Paste text", height=150)
        text_name = st.text_input("Label", value="pasted_text")
        if st.button("Index Text", disabled=not text_input):
            with st.spinner("Indexing..."):
                r = requests.post(f"{API_BASE}/index/text", json={
                    "text": text_input,
                    "filename": text_name
                })
                if r.ok:
                    d = r.json()
                    st.success(f"✅ Indexed {d['chunks']} chunks")
                else:
                    st.error(r.json().get("detail", "Indexing failed"))

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
    except Exception:
        st.caption("API not reachable.")

    st.divider()
    st.subheader("⚙️ Retrieval Settings")
    top_k_retrieve = st.slider("Candidates to retrieve", 5, 50, 20)
    top_k_rerank = st.slider("Final chunks after rerank", 1, 10, 5)
    use_reranker = st.checkbox("Use cross-encoder reranker", value=True)
    use_streaming = st.checkbox("Stream response", value=False)


# ─── Main — Chat Interface ─────────────────────────────────────────────────
st.title("🔍 RAG Document Q&A")
st.caption("FAANG-grade RAG: hybrid BM25 + dense retrieval · cross-encoder reranking · citations")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"📎 {len(msg['citations'])} source(s) cited"):
                for c in msg["citations"]:
                    st.markdown(
                        f'<div class="citation-box">'
                        f'<strong>[Source {c["source_n"]}]</strong> '
                        f'{c.get("filename", "Unknown")} · page {c.get("page", "?")} '
                        f'<br><small>{c.get("text_preview", "")}</small>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
        if msg.get("meta"):
            m = msg["meta"]
            cols = st.columns(3)
            cols[0].metric("Latency", f"{m.get('latency_ms', 0):.0f} ms")
            cols[1].metric("Chunks retrieved", m.get("chunks_retrieved", 0))
            cols[2].metric("Tokens used", m.get("tokens_used", 0))

# Query input
if query := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        if use_streaming:
            # SSE streaming
            placeholder = st.empty()
            full_text = ""
            citations = []

            try:
                with requests.get(
                    f"{API_BASE}/query/stream",
                    params={"query": query, "top_k_retrieve": top_k_retrieve, "top_k_rerank": top_k_rerank},
                    stream=True,
                    timeout=60
                ) as resp:
                    for line in resp.iter_lines():
                        if not line:
                            continue
                        line = line.decode("utf-8")
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            event = json.loads(data_str)
                            if event["type"] == "token":
                                full_text += event["text"]
                                placeholder.markdown(full_text + "▌")
                            elif event["type"] == "citations":
                                citations = event["citations"]
                        except json.JSONDecodeError:
                            pass

                placeholder.markdown(full_text)

            except Exception as e:
                st.error(f"Streaming error: {e}")
                full_text = "Error during streaming."

            msg = {"role": "assistant", "content": full_text, "citations": citations, "meta": {}}

        else:
            # Non-streaming
            with st.spinner("Retrieving and generating..."):
                r = requests.post(f"{API_BASE}/query", json={
                    "query": query,
                    "top_k_retrieve": top_k_retrieve,
                    "top_k_rerank": top_k_rerank,
                    "use_reranker": use_reranker,
                })

            if r.ok:
                result = r.json()
                st.markdown(result["answer"])
                citations = result.get("citations", [])
                meta = {
                    "latency_ms": result.get("latency_ms", 0),
                    "chunks_retrieved": result.get("chunks_retrieved", 0),
                    "tokens_used": result.get("tokens_used", 0),
                }

                if citations:
                    with st.expander(f"📎 {len(citations)} source(s) cited"):
                        for c in citations:
                            st.markdown(
                                f'<div class="citation-box">'
                                f'<strong>[Source {c["source_n"]}]</strong> '
                                f'{c.get("filename", "Unknown")} · page {c.get("page", "?")} '
                                f'<br><small>{c.get("text_preview", "")}</small>'
                                f'</div>',
                                unsafe_allow_html=True
                            )

                cols = st.columns(3)
                cols[0].metric("Latency", f"{meta['latency_ms']:.0f} ms")
                cols[1].metric("Chunks", meta["chunks_retrieved"])
                cols[2].metric("Tokens", meta["tokens_used"])

                msg = {"role": "assistant", "content": result["answer"], "citations": citations, "meta": meta}
            else:
                st.error("Query failed.")
                msg = {"role": "assistant", "content": "Query failed.", "citations": [], "meta": {}}

    st.session_state.messages.append(msg)
