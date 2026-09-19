"""
DocuMind Streamlit Application

UI/session-state layer only.
The RAG architecture remains inside rag_pipeline.py.
"""

import os
import time

import streamlit as st

from about import render_about
from rag_pipeline import RAGPipeline


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="DocuMind",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION STATE
# ============================================================

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "messages" not in st.session_state:
    st.session_state.messages = []

if "processing_error" not in st.session_state:
    st.session_state.processing_error = None

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "processing_started" not in st.session_state:
    st.session_state.processing_started = None


# ============================================================
# API KEY
# ============================================================

def get_secret(name: str):
    """Read API key from environment variables or Streamlit secrets."""

    value = os.getenv(name)

    if value:
        return value

    try:
        return st.secrets[name]
    except Exception:
        return None


# ============================================================
# RESET DOCUMENT
# ============================================================

def reset_document():
    """Clear current document and chat state."""

    pipeline = st.session_state.pipeline

    if pipeline is not None:
        try:
            pipeline.clear_document()
        except Exception:
            pass

    st.session_state.pipeline = None
    st.session_state.messages = []
    st.session_state.processing_error = None
    st.session_state.document_name = None
    st.session_state.processing_started = None


# ============================================================
# PROCESSING STATE
# ============================================================

PROCESSING_STAGES = [
    "📄 Reading PDF",
    "🔍 Extracting document content",
    "👁️ Understanding visuals",
    "🧩 Building document structure",
    "✂️ Creating intelligent chunks",
    "🧠 Generating embeddings",
    "⚡ Building search index",
    "✅ Finalizing document",
]


def get_processing_stage(message: str):
    """
    Convert existing RAG pipeline messages into
    user-friendly processing information.

    RAG architecture is not changed.
    """

    text = message.lower()

    if "reading pdf" in text:
        return (
            0,
            "📄 Reading PDF",
            "Opening the PDF and preparing it for document analysis.",
        )

    if "docling complete" in text:
        return (
            2,
            "👁️ Understanding visuals",
            message,
        )

    if "analyzing visual" in text:
        return (
            2,
            "👁️ Understanding visuals",
            message,
        )

    if "embedding chunks" in text:
        return (
            5,
            "🧠 Generating embeddings",
            "Converting document chunks into semantic vectors for fast question retrieval.",
        )

    if "document is ready" in text:
        return (
            7,
            "✅ Finalizing document",
            "The document is ready for questions.",
        )

    return (
        1,
        "🔍 Processing document",
        message,
    )


# ============================================================
# MAIN PROCESSING UI
# ============================================================

def process_document(uploaded_file, jina_key, gemini_key):
    """
    Run the existing RAG pipeline with a native Streamlit
    progress/status interface.
    """

    st.session_state.processing_started = time.time()

    # --------------------------------------------------------
    # Native Streamlit status container
    # --------------------------------------------------------

    with st.status(
        "Preparing your document...",
        expanded=True,
    ) as status:

        progress = st.progress(
            0,
            text="Starting document processing...",
        )

        current_stage = st.empty()

        details = st.empty()

        try:

            # ------------------------------------------------
            # Progress callback
            # ------------------------------------------------

            def progress_callback(stage, message, value=None):

                stage_index, title, description = (
                    get_processing_stage(message)
                )

                # --------------------------------------------
                # Overall progress
                # --------------------------------------------

                if value is not None:

                    value = float(value)

                    # The pipeline reports embedding progress
                    # separately. Map it to the later part of
                    # the overall progress bar.

                    if "embedding chunks" in message.lower():

                        overall = 0.45 + (value * 0.48)

                    elif stage == "complete":

                        overall = 1.0

                    else:

                        overall = value

                    overall = max(
                        0.0,
                        min(1.0, overall),
                    )

                    progress.progress(
                        overall,
                        text=f"{int(overall * 100)}% — {title}",
                    )

                # --------------------------------------------
                # Current stage
                # --------------------------------------------

                current_stage.markdown(
                    f"### {title}"
                )

                details.caption(
                    description
                )

                # --------------------------------------------
                # Processing timeline
                # --------------------------------------------

                completed = stage_index

                for index, stage_name in enumerate(
                    PROCESSING_STAGES
                ):

                    if index < completed:

                        st.markdown(
                            f"✓ {stage_name}"
                        )

                    elif index == stage_index:

                        st.markdown(
                            f"⏳ **{stage_name}**"
                        )

                    else:

                        st.caption(
                            f"○ {stage_name}"
                        )

            # ------------------------------------------------
            # Create pipeline
            # ------------------------------------------------

            pipeline = RAGPipeline(
                jina_api_key=jina_key,
                gemini_api_key=gemini_key,
                progress_callback=progress_callback,
            )

            # ------------------------------------------------
            # IMPORTANT:
            # Existing RAG architecture is called unchanged.
            # ------------------------------------------------

            pipeline.ingest_pdf(
                uploaded_file.getvalue(),
                uploaded_file.name,
            )

            # ------------------------------------------------
            # Completed
            # ------------------------------------------------

            progress.progress(
                1.0,
                text="100% — Document ready",
            )

            current_stage.markdown(
                "### ✅ Document ready"
            )

            elapsed = (
                time.time()
                - st.session_state.processing_started
            )

            details.success(
                f"Processing completed in {elapsed:.1f} seconds. "
                "You can now ask questions about the document."
            )

            status.update(
                label="Document processed successfully",
                state="complete",
                expanded=False,
            )

            st.session_state.pipeline = pipeline
            st.session_state.messages = []
            st.session_state.processing_error = None

            return pipeline

        except Exception as exc:

            status.update(
                label="Document processing failed",
                state="error",
                expanded=True,
            )

            raise exc


# ============================================================
# HEADER
# ============================================================

st.title("📘 DocuMind")

st.caption(
    "Ask questions, explore information, and understand your PDF "
    "with an AI-powered document assistant."
)


# ============================================================
# TABS
# ============================================================

tab_main, tab_about = st.tabs(
    [
        "💬 Ask Your Document",
        "ℹ️ About & Help",
    ]
)


# ============================================================
# MAIN TAB
# ============================================================

with tab_main:

    st.header("Upload your document")

    st.caption(
        "Upload a PDF to extract its content, understand its visuals, "
        "build a searchable knowledge base, and ask questions."
    )

    # --------------------------------------------------------
    # Upload
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        help="Upload one PDF document.",
    )

    # --------------------------------------------------------
    # New file detected
    # --------------------------------------------------------

    if uploaded_file is not None:

        current_name = st.session_state.document_name

        if current_name != uploaded_file.name:

            # Clear previous state
            reset_document()

            st.session_state.document_name = (
                uploaded_file.name
            )

            jina_key = get_secret(
                "JINA_API_KEY"
            )

            gemini_key = get_secret(
                "GEMINI_API_KEY"
            )

            # ------------------------------------------------
            # API keys
            # ------------------------------------------------

            if not jina_key or not gemini_key:

                st.error(
                    "API keys are missing. Please configure "
                    "`JINA_API_KEY` and `GEMINI_API_KEY` "
                    "in Streamlit Secrets."
                )

            else:

                try:

                    process_document(
                        uploaded_file,
                        jina_key,
                        gemini_key,
                    )

                    st.rerun()

                except Exception as exc:

                    st.session_state.pipeline = None

                    st.session_state.processing_error = (
                        str(exc)
                    )

                    st.error(
                        "The document could not be processed."
                    )


    # ========================================================
    # ERROR
    # ========================================================

    if st.session_state.processing_error:

        with st.expander(
            "🔧 Technical error details",
            expanded=False,
        ):

            st.exception(
                Exception(
                    st.session_state.processing_error
                )
            )


    # ========================================================
    # CURRENT PIPELINE
    # ========================================================

    pipeline = st.session_state.pipeline


    # ========================================================
    # READY DOCUMENT
    # ========================================================

    if pipeline is not None:

        info = pipeline.get_document_info()

        # ----------------------------------------------------
        # Document information
        # ----------------------------------------------------

        st.divider()

        document_col, action_col = st.columns(
            [5, 1]
        )

        with document_col:

            st.subheader(
                f"📄 {info.get('name', 'Document')}"
            )

            st.success(
                "Document ready for questions"
            )

        with action_col:

            if st.button(
                "Clear",
                use_container_width=True,
            ):

                reset_document()

                st.rerun()


        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        metric_cols = st.columns(4)

        metric_cols[0].metric(
            "Pages",
            info.get("pages", 0),
        )

        metric_cols[1].metric(
            "Chunks",
            info.get("chunks", 0),
        )

        metric_cols[2].metric(
            "Vectors",
            info.get("vectors", 0),
        )

        metric_cols[3].metric(
            "Embedding Dimension",
            info.get("embedding_dimension", 0),
        )


        st.divider()


        # ----------------------------------------------------
        # Chat header
        # ----------------------------------------------------

        st.subheader(
            "💬 Ask your document"
        )

        st.caption(
            "Ask about facts, tables, charts, figures, comparisons, "
            "summaries, or specific pages."
        )


        # ----------------------------------------------------
        # Conversation
        # ----------------------------------------------------

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

                # --------------------------------------------
                # Supporting sources
                # --------------------------------------------

                if message.get("sources"):

                    with st.expander(
                        "📚 Supporting sources"
                    ):

                        for item in message["sources"]:

                            chunk = item["chunk"]

                            page_start = chunk.get(
                                "page_start"
                            )

                            page_end = chunk.get(
                                "page_end"
                            )

                            section = (
                                chunk.get("section")
                                or "Document section"
                            )

                            if page_start == page_end:

                                page_text = (
                                    f"Page {page_start}"
                                )

                            else:

                                page_text = (
                                    f"Pages "
                                    f"{page_start}–{page_end}"
                                )

                            st.markdown(
                                f"**Source {item['rank']}**  \n"
                                f"{page_text} · "
                                f"{section}"
                            )

                            st.caption(
                                chunk.get(
                                    "content",
                                    "",
                                )[:1200]
                            )

                            st.divider()


        # ----------------------------------------------------
        # Chat input
        # ----------------------------------------------------

        question = st.chat_input(
            "Ask something about your PDF..."
        )

        if question:

            # -----------------------------------------------
            # User
            # -----------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):

                st.markdown(
                    question
                )


            # -----------------------------------------------
            # Assistant
            # -----------------------------------------------

            with st.chat_message("assistant"):

                with st.status(
                    "Finding the answer...",
                    expanded=True,
                ) as answer_status:

                    st.write(
                        "🔎 Searching the document..."
                    )

                    st.write(
                        "🧠 Selecting the most relevant information..."
                    )

                    try:

                        result = pipeline.ask(
                            question
                        )

                        answer_status.update(
                            label="Answer generated",
                            state="complete",
                            expanded=False,
                        )

                        st.markdown(
                            result["answer"]
                        )

                        # ------------------------------------
                        # Sources
                        # ------------------------------------

                        with st.expander(
                            "📚 Supporting sources"
                        ):

                            for item in result["results"]:

                                chunk = item["chunk"]

                                page_start = chunk.get(
                                    "page_start"
                                )

                                page_end = chunk.get(
                                    "page_end"
                                )

                                section = (
                                    chunk.get("section")
                                    or "Document section"
                                )

                                if page_start == page_end:

                                    page_text = (
                                        f"Page {page_start}"
                                    )

                                else:

                                    page_text = (
                                        f"Pages "
                                        f"{page_start}–{page_end}"
                                    )

                                st.markdown(
                                    f"**Source {item['rank']}**"
                                )

                                st.caption(
                                    f"{page_text} · "
                                    f"{section} · "
                                    f"similarity "
                                    f"{item['score']:.4f}"
                                )

                                st.write(
                                    chunk.get(
                                        "content",
                                        "",
                                    )[:1200]
                                )

                                st.divider()

                            st.caption(
                                f"Answer model: "
                                f"{result['model_used']}"
                            )


                        # ------------------------------------
                        # Save assistant message
                        # ------------------------------------

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": result["answer"],
                                "sources": result["results"],
                            }
                        )

                    except Exception as exc:

                        answer_status.update(
                            label="Unable to generate answer",
                            state="error",
                            expanded=True,
                        )

                        st.error(
                            "I couldn't generate an answer "
                            "for that question."
                        )

                        with st.expander(
                            "Technical details"
                        ):

                            st.exception(
                                exc
                            )

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": (
                                    "I couldn't generate an answer "
                                    "for that question."
                                ),
                            }
                        )


    # ========================================================
    # EMPTY STATE
    # ========================================================

    else:

        st.divider()

        st.info(
            "📄 Upload a PDF above to get started."
        )

        example_cols = st.columns(3)

        with example_cols[0]:

            st.markdown(
                "**📑 Understand**"
            )

            st.caption(
                "Summarize sections and identify key findings."
            )

        with example_cols[1]:

            st.markdown(
                "**📊 Explore**"
            )

            st.caption(
                "Ask about tables, charts, figures, and numbers."
            )

        with example_cols[2]:

            st.markdown(
                "**🔎 Find**"
            )

            st.caption(
                "Locate information and relevant pages quickly."
            )


# ============================================================
# ABOUT TAB
# ============================================================

with tab_about:

    render_about()

