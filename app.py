import streamlit as st

from about import render_about
from rag_pipeline import (
    clear_document,
    get_document_info,
    process_pdf,
    query_document,
)

st.set_page_config(
    page_title="DocuMind — Intelligent PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
    .documind-brand {display:flex; align-items:center; gap:.8rem; margin-bottom:.2rem;}
    .documind-logo {
        width:44px; height:44px; border-radius:14px; display:flex;
        align-items:center; justify-content:center; font-size:24px;
        background:linear-gradient(135deg,#6366f1,#06b6d4); color:white;
        box-shadow:0 8px 24px rgba(79,70,229,.22);
    }
    .documind-title {font-size:2rem; font-weight:800; line-height:1.1;}
    .documind-subtitle {color:#64748b; margin-top:.25rem; font-size:1rem;}
    .hero {
        padding:1.5rem 1.6rem; border:1px solid rgba(100,116,139,.18);
        border-radius:20px; background:linear-gradient(135deg,rgba(99,102,241,.08),rgba(6,182,212,.06));
        margin:1rem 0 1.2rem;
    }
    .status-card {
        border:1px solid rgba(100,116,139,.18); border-radius:16px;
        padding:1rem 1.1rem; background:rgba(248,250,252,.72); margin:.7rem 0 1rem;
    }
    .status-ok {color:#15803d; font-weight:700;}
    .metric-label {font-size:.78rem; color:#64748b;}
    .metric-value {font-size:1.15rem; font-weight:700;}
    .source-pill {
        display:inline-block; padding:.28rem .55rem; margin:.15rem .25rem .15rem 0;
        border-radius:999px; background:#eef2ff; color:#3730a3; font-size:.8rem;
    }
    .small-muted {color:#64748b; font-size:.85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

def init_state():
    defaults = {
        "document": None,
        "document_info": None,
        "messages": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_state()

def reset_document():
    clear_document(st.session_state.get("document"))
    st.session_state.document = None
    st.session_state.document_info = None
    st.session_state.messages = []

st.markdown(
    """
    <div class="documind-brand">
      <div class="documind-logo">📄</div>
      <div>
        <div class="documind-title">DocuMind</div>
        <div class="documind-subtitle">Intelligent PDF Assistant</div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_ask, tab_about = st.tabs(["💬 Ask your PDF", "✨ About & Help"])

with tab_ask:
    st.markdown(
        """
        <div class="hero">
          <h2 style="margin:0;">Ask your document anything</h2>
          <p style="margin:.45rem 0 0;color:#64748b;">
          Upload a PDF, let DocuMind build its searchable representation,
          and ask natural-language questions about the document.
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### 📄 Current Document")
        if st.session_state.document_info:
            info = st.session_state.document_info
            st.write(f"**File:** {info.get('document_name', '—')}")
            st.write(f"**Pages:** {info.get('pages', '—')}")
            st.write(f"**Chunks:** {info.get('chunks', '—')}")
            st.markdown('<span class="status-ok">● Ready</span>', unsafe_allow_html=True)
        else:
            st.caption("No document has been processed yet.")

        st.divider()
        st.markdown("### ⚙️ Document Controls")
        if st.button("🔄 New Document", use_container_width=True):
            reset_document()
            st.rerun()

        st.divider()
        st.markdown("### ℹ️ Quick Tips")
        st.caption("Ask specific questions for more focused retrieval.")
        st.caption("Use follow-up questions while the same PDF is loaded.")
        st.caption("Check supporting pages for important answers.")

    uploaded = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload one PDF document to create a searchable RAG index.",
    )

    if uploaded is not None:
        previous_name = st.session_state.get("uploaded_name")
        previous_size = st.session_state.get("uploaded_size")
        changed = (
            uploaded.name != previous_name
            or uploaded.size != previous_size
        )

        if changed:
            st.session_state.uploaded_name = uploaded.name
            st.session_state.uploaded_size = uploaded.size
            st.session_state.document = None
            st.session_state.document_info = None
            st.session_state.messages = []

        if st.session_state.document is None:
            if st.button("🚀 Process Document", type="primary", use_container_width=True):
                try:
                    with st.status("Processing your document…", expanded=True) as status:
                        st.write("Reading and parsing the PDF…")
                        document = process_pdf(
                            uploaded.getvalue(),
                            uploaded.name,
                        )
                        info = get_document_info(document)
                        st.session_state.document = document
                        st.session_state.document_info = info
                        status.update(
                            label="✅ Document ready!",
                            state="complete",
                            expanded=False,
                        )
                    st.rerun()
                except Exception as exc:
                    import logging
                    logging.getLogger("documind.app").exception(
                        "Document processing failed",
                        exc_info=exc,
                    )
                    st.error(
                        "❌ Unable to process this document."
                    )
                    st.caption(
                        "The PDF could not be processed. Please try another PDF "
                        "or check the application logs for the underlying error."
                    )

    if st.session_state.document_info:
        info = st.session_state.document_info
        st.markdown(
            f"""
            <div class="status-card">
              <div style="font-size:1.05rem;font-weight:750;">📄 {info.get('document_name','Document')}</div>
              <div class="small-muted" style="margin-top:.35rem;">
                <span class="status-ok">✓ Document processed successfully</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f'<div class="metric-label">Pages</div><div class="metric-value">{info.get("pages","—")}</div>', unsafe_allow_html=True)
        with c2:
            st.markdown(f'<div class="metric-label">Searchable chunks</div><div class="metric-value">{info.get("chunks","—")}</div>', unsafe_allow_html=True)
        with c3:
            st.markdown(f'<div class="metric-label">Indexed elements</div><div class="metric-value">{info.get("elements","—")}</div>', unsafe_allow_html=True)

    if st.session_state.document is None:
        st.info("Upload a PDF and click **Process Document** to start asking questions.")
    else:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if message.get("sources"):
                    with st.expander("📚 View supporting pages"):
                        for source in message["sources"]:
                            page = source.get("page")
                            score = source.get("score")
                            label = f"Page {page}" if page is not None else "Page unavailable"
                            if score is not None:
                                label += f" · similarity {score:.3f}"
                            st.markdown(f"- **{label}**")

        question = st.chat_input("Ask a question about your PDF…")
        if question:
            st.session_state.messages.append(
                {"role": "user", "content": question}
            )
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching the document and generating an answer…"):
                    try:
                        result = query_document(
                            st.session_state.document,
                            question,
                        )
                        answer = result["answer"]
                        sources = result.get("sources", [])
                        st.markdown(answer)
                        if sources:
                            with st.expander("📚 View supporting pages"):
                                for source in sources:
                                    page = source.get("page")
                                    score = source.get("score")
                                    label = f"Page {page}" if page is not None else "Page unavailable"
                                    if score is not None:
                                        label += f" · similarity {score:.3f}"
                                    st.markdown(f"- **{label}**")
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                                "sources": sources,
                            }
                        )
                    except Exception:
                        st.error(
                            "❌ Sorry, I couldn't generate an answer for this question."
                        )
                        st.caption(
                            "Please check that JINA_API_KEY and GEMINI_API_KEY "
                            "are configured correctly, then try again."
                        )

with tab_about:
    render_about()
