"""
Streamlit RAG app — chat with your PDF.

Pipeline: Docling ingestion -> Gemini vision descriptions for charts/images
-> structure-aware chunking -> Gemini Embedding -> FAISS -> Gemini answer
generation. Ported from a Colab notebook that previously used a local
BGE-M3 sentence-transformer for embeddings (which crashed on Streamlit
Cloud's memory limits); this version calls the Gemini Embedding API
instead, so no local embedding model has to be loaded into memory.

Run with:
    streamlit run app.py
"""

import os
import tempfile
import time
from pathlib import Path

import streamlit as st
from google import genai

import rag_pipeline as rp

# --------------------------------------------------------------------------
# PAGE CONFIG + STYLE
# --------------------------------------------------------------------------

st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main .block-container { padding-top: 2rem; max-width: 950px; }

    .app-hero {
        background: linear-gradient(120deg, #4f46e5 0%, #7c3aed 55%, #db2777 100%);
        padding: 28px 32px;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.4rem;
    }
    .app-hero h1 { margin: 0; font-size: 1.7rem; }
    .app-hero p { margin: 6px 0 0 0; opacity: 0.92; font-size: 0.95rem; }

    .stat-card {
        background: var(--secondary-background-color);
        border: 1px solid rgba(128,128,128,0.18);
        border-radius: 12px;
        padding: 10px 14px;
        text-align: center;
    }
    .stat-card .num { font-size: 1.35rem; font-weight: 700; }
    .stat-card .label { font-size: 0.72rem; opacity: 0.7; text-transform: uppercase; letter-spacing: .04em; }

    [data-testid="stChatMessage"] { border-radius: 14px; }

    .source-pill {
        display: inline-block;
        font-size: 0.72rem;
        padding: 2px 9px;
        border-radius: 999px;
        background: rgba(124,58,237,0.12);
        color: #7c3aed;
        margin-right: 6px;
        margin-bottom: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# SESSION STATE
# --------------------------------------------------------------------------

defaults = {
    "doc_index": None,
    "messages": [],           # chat history for display
    "chat_history": [],       # compact history passed to the LLM
    "work_dir": None,
    "processed_file_id": None,
    "client": None,
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


def get_client(api_key: str):
    if not api_key:
        return None
    if st.session_state.client is None or st.session_state.get("_api_key_used") != api_key:
        st.session_state.client = genai.Client(api_key=api_key)
        st.session_state["_api_key_used"] = api_key
    return st.session_state.client


@st.cache_resource(show_spinner=False)
def get_converter(do_ocr: bool):
    return rp.build_converter(do_ocr=do_ocr)


# --------------------------------------------------------------------------
# SIDEBAR
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### ⚙️ Setup")

    default_key = os.environ.get("GEMINI_API_KEY", "") or st.secrets.get("GEMINI_API_KEY", "") if hasattr(st, "secrets") else os.environ.get("GEMINI_API_KEY", "")
    try:
        default_key = st.secrets["GEMINI_API_KEY"]
    except Exception:
        default_key = os.environ.get("GEMINI_API_KEY", "")

    api_key = st.text_input(
        "Gemini API key",
        value=default_key,
        type="password",
        help="Get a free key at aistudio.google.com/apikey. Stored only in this session.",
    )

    with st.expander("Advanced settings"):
        embedding_model = st.text_input("Embedding model", value=rp.EMBEDDING_MODEL_NAME)
        embedding_dim = st.number_input("Embedding dimensions", value=rp.EMBEDDING_OUTPUT_DIMENSIONALITY, step=64)
        top_k = st.slider("Chunks retrieved per question", 3, 10, 5)
        describe_visuals = st.checkbox("Describe charts/images with Gemini Vision", value=True)
        enable_ocr = st.checkbox(
            "Enable OCR for scanned pages",
            value=False,
            help="Only needed for scanned/image-only PDFs. Requires libgl1 etc. "
                 "(see packages.txt) — leave off for normal text PDFs, it's faster too.",
        )

    st.divider()
    st.markdown("### 📄 Document")

    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

    process_clicked = st.button(
        "🚀 Process document",
        use_container_width=True,
        type="primary",
        disabled=not (uploaded_file and api_key),
    )

    if st.session_state.doc_index is not None:
        st.success(f"Loaded: **{st.session_state.doc_index.document_name}**")
        if st.button("🗑️ Clear document & chat", use_container_width=True):
            st.session_state.doc_index = None
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.session_state.processed_file_id = None
            st.rerun()

    st.divider()
    st.caption(
        "Pipeline: Docling ingestion → Gemini Vision for charts/images → "
        "structure-aware chunking → Gemini Embedding → FAISS → Gemini answer generation."
    )

# --------------------------------------------------------------------------
# HERO
# --------------------------------------------------------------------------

st.markdown(
    """
    <div class="app-hero">
        <h1>📄 Chat with your PDF</h1>
        <p>Upload a document — including ones full of charts and tables — and ask it questions.
        Powered by Docling + Gemini Vision + Gemini Embedding + FAISS.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# PROCESSING
# --------------------------------------------------------------------------

if process_clicked and uploaded_file and api_key:
    client = get_client(api_key)

    work_dir = Path(tempfile.mkdtemp(prefix="rag_"))
    pdf_path = work_dir / uploaded_file.name
    pdf_path.write_bytes(uploaded_file.getvalue())

    progress_bar = st.progress(0.0)
    status_text = st.empty()

    def progress_cb(frac, msg):
        progress_bar.progress(frac)
        status_text.markdown(f"**{msg}**")

    try:
        converter = get_converter(enable_ocr)
        vision_models = rp.VISION_MODELS if describe_visuals else []

        doc_index = rp.process_pdf(
            client=client,
            pdf_path=pdf_path,
            work_dir=work_dir,
            converter=converter,
            embedding_model=embedding_model,
            embedding_dim=int(embedding_dim),
            vision_models=vision_models if describe_visuals else [],
            progress_cb=progress_cb,
        )

        st.session_state.doc_index = doc_index
        st.session_state.work_dir = str(work_dir)
        st.session_state.processed_file_id = uploaded_file.name
        st.session_state.messages = []
        st.session_state.chat_history = []

        progress_bar.progress(1.0)
        status_text.markdown("✅ **Ready — ask away below!**")
        time.sleep(0.6)
        st.rerun()

    except Exception as e:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Processing failed: {e}")

# --------------------------------------------------------------------------
# MAIN AREA
# --------------------------------------------------------------------------

doc_index = st.session_state.doc_index

if doc_index is None:
    st.info("👈 Add your Gemini API key and upload a PDF in the sidebar, then click **Process document** to get started.")
    st.stop()

stats = doc_index.stats
cols = st.columns(5)
stat_items = [
    ("Pages", stats.get("pages", "—")),
    ("Chunks", stats.get("chunks", "—")),
    ("Tables", stats.get("table_elements", "—")),
    ("Visuals described", stats.get("visuals_described", "—")),
    ("Avg chunk size", f"{stats.get('avg_chunk_chars', '—')} ch"),
]
for col, (label, value) in zip(cols, stat_items):
    col.markdown(
        f'<div class="stat-card"><div class="num">{value}</div><div class="label">{label}</div></div>',
        unsafe_allow_html=True,
    )

with st.expander("🔍 Debug: search the raw chunk text (no AI involved)"):
    st.caption(
        "If the assistant keeps saying 'not available', check here first — this does a plain "
        "keyword search across every chunk, so you can confirm whether the document actually "
        "contains what you're looking for before assuming retrieval is broken."
    )
    debug_query = st.text_input("Keyword to search for", key="debug_keyword")
    if debug_query:
        hits = [
            c for c in doc_index.chunks
            if debug_query.lower() in c["content"].lower()
        ]
        st.write(f"**{len(hits)} chunk(s) contain \"{debug_query}\"**")
        for c in hits[:15]:
            st.markdown(
                f"— page {c['page_start']}–{c['page_end']}"
                + (f" · _{c.get('section')}_" if c.get("section") else "")
            )
            st.caption(c["content"][:500] + ("…" if len(c["content"]) > 500 else ""))
            st.markdown("---")
        if not hits:
            st.info("No chunk contains that exact text — the document likely doesn't cover it, "
                     "or it uses different wording. Try a shorter/simpler keyword.")

st.write("")

# ---- render chat history ----
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="📄" if msg["role"] == "assistant" else None):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(f"📚 {len(msg['sources'])} source chunk(s) used"):
                for item in msg["sources"]:
                    chunk = item["chunk"]
                    tags = ""
                    if chunk.get("has_table"):
                        tags += '<span class="source-pill">table</span>'
                    if chunk.get("has_visual"):
                        tags += '<span class="source-pill">chart/image</span>'
                    st.markdown(
                        f"**Source {item['rank']}** · score {item['score']:.3f} · "
                        f"page {chunk['page_start']}–{chunk['page_end']}"
                        + (f" · _{chunk.get('section')}_" if chunk.get("section") else "")
                        + f"<br>{tags}",
                        unsafe_allow_html=True,
                    )
                    st.caption(chunk["content"][:600] + ("…" if len(chunk["content"]) > 600 else ""))
                    st.markdown("---")

# ---- chat input ----
question = st.chat_input("Ask a question about the document...")

if question:
    client = get_client(api_key)
    if client is None:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar="📄"):
        with st.spinner("Searching the document and thinking..."):
            try:
                results = rp.retrieve_chunks(client, doc_index, question, top_k=top_k)
                answer, model_used = rp.generate_answer(
                    client, question, results,
                    chat_history=st.session_state.chat_history,
                )
            except Exception as e:
                answer, results, model_used = f"Sorry, something went wrong: {e}", [], None

        st.markdown(answer)
        if results:
            with st.expander(f"📚 {len(results)} source chunk(s) used"):
                for item in results:
                    chunk = item["chunk"]
                    tags = ""
                    if chunk.get("has_table"):
                        tags += '<span class="source-pill">table</span>'
                    if chunk.get("has_visual"):
                        tags += '<span class="source-pill">chart/image</span>'
                    st.markdown(
                        f"**Source {item['rank']}** · score {item['score']:.3f} · "
                        f"page {chunk['page_start']}–{chunk['page_end']}"
                        + (f" · _{chunk.get('section')}_" if chunk.get("section") else "")
                        + f"<br>{tags}",
                        unsafe_allow_html=True,
                    )
                    st.caption(chunk["content"][:600] + ("…" if len(chunk["content"]) > 600 else ""))
                    st.markdown("---")

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": results})
    st.session_state.chat_history.append({"question": question, "answer": answer})
