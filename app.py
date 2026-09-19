
"""
DocuMind Streamlit Application

UI/session-state layer only.
RAG architecture remains inside rag_pipeline.py.
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
# CUSTOM UI
# ============================================================

st.markdown(
    """
<style>

/* ---------- GLOBAL ---------- */

.block-container {
    max-width: 1120px;
    padding-top: 1.8rem;
    padding-bottom: 4rem;
}

[data-testid="stHeader"] {
    background: transparent;
}

footer {
    visibility: hidden;
}

/* ---------- HERO ---------- */

.hero {
    padding: 1.8rem 2rem;
    border-radius: 20px;
    background:
        linear-gradient(
            135deg,
            rgba(99,102,241,0.10),
            rgba(14,165,233,0.08),
            rgba(168,85,247,0.08)
        );
    border: 1px solid rgba(99,102,241,0.14);
    margin-bottom: 1.3rem;
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 750;
    letter-spacing: -0.7px;
    margin-bottom: 0.25rem;
}

.hero-subtitle {
    color: #64748b;
    font-size: 1rem;
    margin: 0;
}

/* ---------- SECTION TITLE ---------- */

.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    margin-top: 0.5rem;
    margin-bottom: 0.2rem;
}

.section-subtitle {
    color: #64748b;
    font-size: 0.92rem;
    margin-bottom: 1rem;
}

/* ---------- UPLOAD ---------- */

.upload-card {
    padding: 1.4rem;
    border-radius: 18px;
    border: 1px solid #e2e8f0;
    background: #ffffff;
    box-shadow: 0 4px 18px rgba(15,23,42,0.04);
    margin-bottom: 1rem;
}

/* ---------- DOCUMENT CARD ---------- */

.document-card {
    padding: 1.1rem 1.25rem;
    border-radius: 17px;
    border: 1px solid #e2e8f0;
    background: #ffffff;
    margin: 0.8rem 0 1rem 0;
}

.document-name {
    font-weight: 700;
    font-size: 1.02rem;
}

.document-status {
    color: #16a34a;
    font-size: 0.86rem;
    font-weight: 600;
}

/* ---------- METRICS ---------- */

.metric-card {
    padding: 0.9rem 1rem;
    border-radius: 15px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
}

.metric-label {
    color: #64748b;
    font-size: 0.78rem;
    margin-bottom: 0.25rem;
}

.metric-value {
    font-size: 1.35rem;
    font-weight: 700;
}

/* ---------- PROCESSING ---------- */

.processing-card {
    padding: 1.35rem 1.45rem;
    border-radius: 18px;
    background: #ffffff;
    border: 1px solid #e2e8f0;
    box-shadow: 0 4px 18px rgba(15,23,42,0.04);
    margin: 0.8rem 0 1.2rem 0;
}

.processing-title {
    font-size: 1.05rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.processing-detail {
    color: #64748b;
    font-size: 0.88rem;
    margin-bottom: 0.8rem;
}

.stage-active {
    font-weight: 700;
    margin-bottom: 0.35rem;
}

.stage-complete {
    color: #16a34a;
    font-size: 0.88rem;
    margin: 0.22rem 0;
}

.stage-pending {
    color: #94a3b8;
    font-size: 0.88rem;
    margin: 0.22rem 0;
}

/* ---------- SOURCES ---------- */

.source-card {
    padding: 0.8rem 0.95rem;
    border-radius: 13px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    margin: 0.45rem 0;
}

.source-title {
    font-weight: 650;
    font-size: 0.88rem;
}

.source-meta {
    color: #64748b;
    font-size: 0.78rem;
    margin-top: 0.15rem;
}

/* ---------- EMPTY STATE ---------- */

.empty-state {
    text-align: center;
    padding: 3.2rem 1rem;
    color: #64748b;
}

.empty-icon {
    font-size: 3rem;
    margin-bottom: 0.5rem;
}

.empty-title {
    color: #334155;
    font-size: 1.2rem;
    font-weight: 700;
}

.empty-description {
    max-width: 580px;
    margin: auto;
    font-size: 0.92rem;
}

/* ---------- CHAT ---------- */

[data-testid="stChatMessage"] {
    border-radius: 15px;
}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "pipeline": None,
    "messages": [],
    "processing_error": None,
    "document_name": None,
    "processing_started": None,
    "processing_finished": None,
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPERS
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


def reset_document_state():
    """Reset the current document/session."""

    pipeline = st.session_state.get("pipeline")

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
    st.session_state.processing_finished = None


# ============================================================
# PROCESSING UI
# ============================================================

def create_processing_ui():
    """
    Create all UI elements required while the PDF is being processed.

    Returns:
        dictionary containing Streamlit placeholders.
    """

    container = st.container()

    with container:
        st.markdown(
            """
            <div class="processing-card">
                <div class="processing-title">
                    ⚙️ Preparing your document
                </div>
                <div class="processing-detail">
                    DocuMind is analyzing the PDF and building its searchable
                    knowledge base. You can follow the progress below.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        progress = st.progress(0)

        percentage = st.empty()

        current_stage = st.empty()

        detail = st.empty()

        stage_list = st.empty()

    return {
        "progress": progress,
        "percentage": percentage,
        "current_stage": current_stage,
        "detail": detail,
        "stage_list": stage_list,
    }


PROCESSING_STAGES = [
    "📄 Read PDF",
    "🔍 Extract document content",
    "👁️ Understand visuals",
    "🧩 Build document structure",
    "✂️ Create intelligent chunks",
    "🧠 Generate embeddings",
    "⚡ Build search index",
    "✅ Finalize document",
]


def render_stage_list(container, completed_count, active_index):
    """Render a clean processing stage list."""

    html = ""

    for index, stage in enumerate(PROCESSING_STAGES):

        if index < completed_count:
            html += (
                f'<div class="stage-complete">'
                f'✓ {stage}'
                f'</div>'
            )

        elif index == active_index:
            html += (
                f'<div class="stage-active">'
                f'⏳ {stage}'
                f'</div>'
            )

        else:
            html += (
                f'<div class="stage-pending">'
                f'○ {stage}'
                f'</div>'
            )

    container.markdown(
        html,
        unsafe_allow_html=True,
    )


def make_progress_callback(ui):
    """
    Creates a callback compatible with RAGPipeline.

    No RAG logic is changed here.
    This only translates the pipeline's existing progress messages
    into a user-friendly Streamlit progress display.
    """

    def callback(stage, message, value=None):

        message_lower = message.lower()

        # ----------------------------------------------
        # Determine current stage
        # ----------------------------------------------

        if "docling complete" in message_lower:

            active_index = 2
            completed_count = 2

            title = "👁️ Understanding document visuals"
            explanation = (
                "Text and document elements have been extracted. "
                "Now DocuMind is analyzing charts, figures and other "
                "visual content."
            )

        elif "analyzing visual" in message_lower:

            active_index = 2
            completed_count = 2

            title = "👁️ Understanding visuals"
            explanation = message

        elif "embedding chunks" in message_lower:

            active_index = 5
            completed_count = 5

            title = "🧠 Creating semantic embeddings"
            explanation = (
                "Each document chunk is being converted into a numerical "
                "representation so relevant information can be found when "
                "you ask a question."
            )

        elif stage == "complete":

            active_index = 7
            completed_count = 7

            title = "✅ Finalizing document"
            explanation = (
                "The searchable document index has been created. "
                "Your PDF is now ready for questions."
            )

        elif stage == "processing":

            active_index = 0
            completed_count = 0

            title = "📄 Processing PDF"
            explanation = message

        else:

            active_index = 0
            completed_count = 0

            title = "⚙️ Processing document"
            explanation = message

        # ----------------------------------------------
        # Progress bar
        # ----------------------------------------------

        if value is not None:

            # The pipeline reports embedding progress from 0 again.
            # Keep the overall UI progress visually continuous.
            if "embedding chunks" in message_lower:

                overall_value = 0.45 + (float(value) * 0.50)

            elif stage == "complete":

                overall_value = 1.0

            else:

                overall_value = float(value)

            overall_value = max(
                0.0,
                min(1.0, overall_value),
            )

            ui["progress"].progress(overall_value)

            ui["percentage"].markdown(
                f"**{int(overall_value * 100)}% complete**"
            )

        # ----------------------------------------------
        # Current stage
        # ----------------------------------------------

        ui["current_stage"].markdown(
            f"### {title}"
        )

        ui["detail"].markdown(
            f'<div class="processing-detail">{explanation}</div>',
            unsafe_allow_html=True,
        )

        # ----------------------------------------------
        # Stage list
        # ----------------------------------------------

        render_stage_list(
            ui["stage_list"],
            completed_count,
            active_index,
        )

    return callback


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
<div class="hero">
    <div class="hero-title">📘 DocuMind</div>
    <p class="hero-subtitle">
        Intelligent question answering for your PDF documents
    </p>
</div>
""",
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================

tab_main, tab_about = st.tabs(
    [
        "💬  Ask Your Document",
        "ℹ️  About & Help",
    ]
)


# ============================================================
# MAIN TAB
# ============================================================

with tab_main:

    # --------------------------------------------------------
    # Upload section
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Upload your document</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-subtitle">'
        'Upload a PDF and DocuMind will prepare it for intelligent '
        'question answering.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="PDF files only.",
    )

    # --------------------------------------------------------
    # New document uploaded
    # --------------------------------------------------------

    if uploaded_file is not None:

        current_name = st.session_state.get("document_name")

        if current_name != uploaded_file.name:

            # Reset previous document
            reset_document_state()

            st.session_state.document_name = uploaded_file.name
            st.session_state.processing_started = time.time()

            jina_key = get_secret("JINA_API_KEY")
            gemini_key = get_secret("GEMINI_API_KEY")

            # ------------------------------------------------
            # API KEY CHECK
            # ------------------------------------------------

            if not jina_key or not gemini_key:

                st.error(
                    "API keys are missing. Please configure "
                    "`JINA_API_KEY` and `GEMINI_API_KEY` "
                    "in Streamlit Secrets."
                )

            else:

                # ------------------------------------------------
                # PROCESSING UI
                # ------------------------------------------------

                ui = create_processing_ui()

                callback = make_progress_callback(ui)

                try:

                    pipeline = RAGPipeline(
                        jina_api_key=jina_key,
                        gemini_api_key=gemini_key,
                        progress_callback=callback,
                    )

                    # ------------------------------------------------
                    # RUN EXISTING RAG PIPELINE
                    # ------------------------------------------------

                    pipeline.ingest_pdf(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                    )

                    # ------------------------------------------------
                    # SUCCESS
                    # ------------------------------------------------

                    st.session_state.pipeline = pipeline

                    st.session_state.messages = []

                    st.session_state.processing_error = None

                    st.session_state.processing_finished = time.time()

                    elapsed = (
                        st.session_state.processing_finished
                        - st.session_state.processing_started
                    )

                    # Final progress display
                    ui["progress"].progress(1.0)

                    ui["percentage"].markdown(
                        "**100% complete**"
                    )

                    ui["current_stage"].markdown(
                        "### ✅ Your document is ready"
                    )

                    ui["detail"].markdown(
                        f'<div class="processing-detail">'
                        f'Processing completed in {elapsed:.1f} seconds. '
                        f'You can now ask questions about your document.'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                    render_stage_list(
                        ui["stage_list"],
                        len(PROCESSING_STAGES),
                        -1,
                    )

                    time.sleep(0.8)

                    # Remove processing UI after completion
                    ui["progress"].empty()
                    ui["percentage"].empty()
                    ui["current_stage"].empty()
                    ui["detail"].empty()
                    ui["stage_list"].empty()

                    st.success(
                        "Document processed successfully. "
                        "You can now ask questions."
                    )

                    st.rerun()

                except Exception as exc:

                    st.session_state.processing_error = str(exc)

                    st.session_state.pipeline = None

                    # Show failure without hiding useful information
                    ui["current_stage"].markdown(
                        "### ❌ Processing stopped"
                    )

                    ui["detail"].markdown(
                        '<div class="processing-detail">'
                        'Something went wrong while preparing the document. '
                        'See the technical details below if you need to '
                        'troubleshoot the deployment.'
                        '</div>',
                        unsafe_allow_html=True,
                    )

                    st.error(
                        "Document processing failed."
                    )


    # ========================================================
    # CURRENT PIPELINE
    # ========================================================

    pipeline = st.session_state.pipeline


    # ========================================================
    # ERROR DETAILS
    # ========================================================

    if st.session_state.processing_error:

        with st.expander(
            "🔧 Technical error details",
            expanded=False,
        ):
            st.code(
                st.session_state.processing_error
            )


    # ========================================================
    # READY DOCUMENT
    # ========================================================

    if pipeline is not None:

        info = pipeline.get_document_info()

        # ----------------------------------------------------
        # Document header
        # ----------------------------------------------------

        st.markdown(
            f"""
            <div class="document-card">
                <div class="document-name">
                    📄 {info.get("name", "Document")}
                </div>
                <div class="document-status">
                    ● Ready for questions
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        cols = st.columns(4)

        metrics = [
            (
                "📄",
                "Pages",
                info.get("pages", 0),
            ),
            (
                "🧩",
                "Chunks",
                info.get("chunks", 0),
            ),
            (
                "⚡",
                "Vectors",
                info.get("vectors", 0),
            ),
            (
                "🔢",
                "Dimensions",
                info.get("embedding_dimension", 0),
            ),
        ]

        for col, (icon, label, value) in zip(
            cols,
            metrics,
        ):

            with col:

                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">
                            {icon} {label}
                        </div>
                        <div class="metric-value">
                            {value:,}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ----------------------------------------------------
        # Clear document
        # ----------------------------------------------------

        st.write("")

        clear_col, _ = st.columns([1, 5])

        with clear_col:

            if st.button(
                "🗑️ Clear document",
                use_container_width=True,
            ):

                reset_document_state()

                st.rerun()

        # ----------------------------------------------------
        # Chat
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Ask your document</div>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<div class="section-subtitle">'
            'Ask questions, summarize sections, compare values, '
            'or explore information contained in the PDF.'
            '</div>',
            unsafe_allow_html=True,
        )

        # ----------------------------------------------------
        # Previous messages
        # ----------------------------------------------------

        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

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
                                    f"Pages {page_start}–{page_end}"
                                )

                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <div class="source-title">
                                        Source {item["rank"]} · {page_text}
                                    </div>
                                    <div class="source-meta">
                                        {section}
                                        &nbsp; · &nbsp;
                                        similarity {item["score"]:.4f}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.caption(
                                chunk.get(
                                    "content",
                                    "",
                                )[:1200]
                            )


        # ----------------------------------------------------
        # Chat input
        # ----------------------------------------------------

        question = st.chat_input(
            "Ask a question about this PDF..."
        )

        if question:

            # ------------------------------------------------
            # User message
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "user",
                    "content": question,
                }
            )

            with st.chat_message("user"):

                st.markdown(question)

            # ------------------------------------------------
            # Assistant
            # ------------------------------------------------

            with st.chat_message("assistant"):

                status = st.empty()

                status.markdown(
                    "🔎 **Searching the document...**"
                )

                try:

                    result = pipeline.ask(
                        question
                    )

                    status.empty()

                    # ----------------------------------------
                    # Answer
                    # ----------------------------------------

                    st.markdown(
                        result["answer"]
                    )

                    # ----------------------------------------
                    # Sources
                    # ----------------------------------------

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

                            if page_start == page_end:

                                page_text = (
                                    f"Page {page_start}"
                                )

                            else:

                                page_text = (
                                    f"Pages {page_start}–{page_end}"
                                )

                            section = (
                                chunk.get("section")
                                or "Document section"
                            )

                            st.markdown(
                                f"""
                                <div class="source-card">
                                    <div class="source-title">
                                        Source {item["rank"]} · {page_text}
                                    </div>
                                    <div class="source-meta">
                                        {section}
                                        &nbsp; · &nbsp;
                                        similarity {item["score"]:.4f}
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            st.caption(
                                chunk.get(
                                    "content",
                                    "",
                                )[:1200]
                            )

                        st.caption(
                            f"Answer generated using "
                            f"{result['model_used']}"
                        )

                    # ----------------------------------------
                    # Save conversation
                    # ----------------------------------------

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": result["answer"],
                            "sources": result["results"],
                        }
                    )

                except Exception as exc:

                    status.empty()

                    error_text = (
                        "I couldn't generate an answer for that "
                        "question. Please try again."
                    )

                    st.error(
                        error_text
                    )

                    with st.expander(
                        "Technical details"
                    ):

                        st.code(
                            str(exc)
                        )

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": error_text,
                        }
                    )


    # ========================================================
    # EMPTY STATE
    # ========================================================

    else:

        st.markdown(
            """
            <div class="empty-state">

                <div class="empty-icon">
                    📄
                </div>

                <div class="empty-title">
                    Your document workspace is ready
                </div>

                <p class="empty-description">
                    Upload a PDF above to extract its content,
                    understand its visuals, build a searchable
                    knowledge base, and start asking questions.
                </p>

            </div>
            """,
            unsafe_allow_html=True,
        )


# ============================================================
# ABOUT TAB
# ============================================================

with tab_about:

    render_about()

