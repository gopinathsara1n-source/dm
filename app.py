"""
DocuMind Streamlit application.

app.py contains UI/session-state only. RAG logic lives in rag_pipeline.py.
"""

import os
import streamlit as st

from about import render_about
from rag_pipeline import RAGPipeline


st.set_page_config(
    page_title="DocuMind — PDF AI Assistant",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1200px; }
.hero {
    padding: 1.5rem 1.7rem;
    border-radius: 22px;
    background: linear-gradient(135deg, #eef2ff 0%, #ecfeff 50%, #fdf2f8 100%);
    border: 1px solid rgba(99,102,241,.15);
    margin-bottom: 1rem;
}
.hero h1 { margin: 0; font-size: 2.35rem; }
.hero p { margin: .45rem 0 0; color: #475569; }
.metric-card {
    padding: 1rem;
    border-radius: 16px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}
.source-card {
    padding: .9rem 1rem;
    border-radius: 14px;
    background: #f8fafc;
    border-left: 4px solid #6366f1;
    margin: .55rem 0;
}
.small-muted { color: #64748b; font-size: .9rem; }
</style>
""",
    unsafe_allow_html=True,
)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing_error" not in st.session_state:
    st.session_state.processing_error = None


def get_secret(name: str):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets[name]
    except Exception:
        return None


def progress_callback(stage, message, value=None):
    if "progress_bar" in st.session_state:
        if value is not None:
            st.session_state.progress_bar.progress(min(max(value, 0.0), 1.0))
        st.session_state.progress_text = message


st.markdown(
    """
<div class="hero">
    <h1>📘 DocuMind</h1>
    <p>Ask questions about your PDF using an intelligent document assistant.</p>
</div>
""",
    unsafe_allow_html=True,
)

tab_main, tab_about = st.tabs(["💬 Ask Your Document", "ℹ️ About & Help"])

with tab_main:
    st.markdown("### Upload a document")
    st.caption("Upload a PDF and DocuMind will prepare it for questions.")

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="PDF files only.",
    )

    if uploaded_file is not None:
        current_name = st.session_state.get("document_name")
        if current_name != uploaded_file.name:
            st.session_state.pipeline = None
            st.session_state.messages = []
            st.session_state.processing_error = None
            st.session_state.document_name = uploaded_file.name

            jina_key = get_secret("JINA_API_KEY")
            gemini_key = get_secret("GEMINI_API_KEY")

            if not jina_key or not gemini_key:
                st.error(
                    "API keys are missing. Add JINA_API_KEY and GEMINI_API_KEY "
                    "to Streamlit secrets or environment variables."
                )
            else:
                st.session_state.progress_bar = st.progress(0)
                st.session_state.progress_text = "Starting..."
                status_box = st.empty()

                try:
                    pipeline = RAGPipeline(
                        jina_api_key=jina_key,
                        gemini_api_key=gemini_key,
                        progress_callback=progress_callback,
                    )
                    with st.spinner("Processing your PDF. This may take a while..."):
                        pipeline.ingest_pdf(
                            uploaded_file.getvalue(),
                            uploaded_file.name,
                        )
                    st.session_state.pipeline = pipeline
                    st.session_state.messages = []
                    st.session_state.processing_error = None
                    status_box.success("✅ Document processed and ready.")
                    st.session_state.progress_bar.empty()
                except Exception as exc:
                    st.session_state.processing_error = str(exc)
                    st.session_state.pipeline = None
                    st.session_state.progress_bar.empty()
                    status_box.error(
                        "Processing failed. Check the error details below."
                    )

    pipeline = st.session_state.pipeline

    if st.session_state.processing_error:
        with st.expander("Technical error details", expanded=False):
            st.code(st.session_state.processing_error)

    if pipeline is not None:
        info = pipeline.get_document_info()
        cols = st.columns(4)
        cards = [
            ("📄 Pages", info.get("pages", 0)),
            ("🧩 Chunks", info.get("chunks", 0)),
            ("📐 Vectors", info.get("vectors", 0)),
            ("🔢 Dimension", info.get("embedding_dimension", 0)),
        ]
        for col, (label, value) in zip(cols, cards):
            with col:
                st.markdown(
                    f'<div class="metric-card"><b>{label}</b><br>'
                    f'<span style="font-size:1.35rem">{value}</span></div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        c1, c2 = st.columns([5, 1])
        with c1:
            st.success(f"Ready: **{info.get('name', 'document')}**")
        with c2:
            if st.button("🗑️ Clear", use_container_width=True):
                pipeline.clear_document()
                st.session_state.pipeline = None
                st.session_state.messages = []
                st.session_state.processing_error = None
                st.session_state.document_name = None
                st.rerun()

        st.markdown("### 💬 Ask a question")

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📚 Supporting sources"):
                        for item in message["sources"]:
                            chunk = item["chunk"]
                            st.markdown(
                                f'<div class="source-card">'
                                f'<b>Source {item["rank"]}</b> · '
                                f'Page {chunk.get("page_start")}–{chunk.get("page_end")} · '
                                f'Score {item["score"]:.4f}<br>'
                                f'<span class="small-muted">'
                                f'{chunk.get("section") or "No section"}'
                                f'</span></div>',
                                unsafe_allow_html=True,
                            )
                            st.caption(chunk.get("content", "")[:1200])

        question = st.chat_input("Ask something about your PDF…")
        if question:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching the document and generating an answer..."):
                    try:
                        result = pipeline.ask(question)
                        st.markdown(result["answer"])
                        with st.expander("📚 Supporting sources"):
                            for item in result["results"]:
                                chunk = item["chunk"]
                                st.markdown(
                                    f'<div class="source-card">'
                                    f'<b>Source {item["rank"]}</b> · '
                                    f'Page {chunk.get("page_start")}–{chunk.get("page_end")} · '
                                    f'Score {item["score"]:.4f}<br>'
                                    f'<span class="small-muted">'
                                    f'{chunk.get("section") or "No section"}'
                                    f'</span></div>',
                                    unsafe_allow_html=True,
                                )
                                st.caption(chunk.get("content", "")[:1200])
                            st.caption(f"Answer model: {result['model_used']}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["results"],
                        })
                    except Exception as exc:
                        error_text = f"Unable to answer this question: {exc}"
                        st.error(error_text)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_text,
                        })
    else:
        st.info(
            "👆 Upload a PDF above to start. Your document is processed only "
            "after upload, and questions become available when processing finishes."
        )

with tab_about:
    render_about()
