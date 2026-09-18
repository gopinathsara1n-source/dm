import streamlit as st

def render_about():
    st.markdown(
        """
        # ✨ About & Help

        **DocuMind** is an AI-powered PDF question-answering assistant
        based on Retrieval-Augmented Generation (RAG). Upload a PDF,
        let the application build a searchable representation of its
        text, tables, and supported visual information, then ask questions
        in natural language.
        """
    )

    st.markdown("## 🔎 How It Works")
    st.markdown(
        """
        The application follows the same architecture as the supplied
        working RAG notebook:

        **PDF Upload → Docling document processing → Text/Table/Image extraction
        → Gemini visual descriptions → Structure-aware chunking → Jina embeddings
        → FAISS similarity search → Retrieved context → Gemini answer generation**

        Text is processed as section-aware chunks. Tables and visual descriptions
        are kept as intact retrieval units. Jina's retrieval-specific passage/query
        embedding tasks are used, and FAISS performs inner-product similarity search.
        """
    )

    st.markdown("## 🚀 How to Use")
    st.markdown(
        """
        1. Upload a PDF on **Ask your PDF**.
        2. Click **Process Document** and wait for processing to finish.
        3. Ask a question in the chat box.
        4. Review the answer and expand **Supporting pages** when available.
        5. Continue with follow-up questions about the same document.
        6. Use **New Document** before working with another PDF.
        """
    )

    st.markdown("## 💡 Example Questions")
    examples = [
        "What is the main objective of this document?",
        "Summarize the key findings.",
        "What are the important financial figures?",
        "What does the document say about this topic?",
        "Compare the values mentioned for different years.",
        "Which page contains information about this topic?",
    ]
    for item in examples:
        st.markdown(f"- {item}")

    st.markdown("## 🧩 Features")
    st.markdown(
        """
        - PDF question answering through retrieval-augmented generation
        - Structure-aware text, table, and supported visual retrieval
        - Jina Embeddings v4 for document/query embeddings
        - FAISS vector similarity search
        - Gemini-based visual description and answer generation
        - Supporting page information when available
        - Persistent chat history for the active document
        - Reset/new-document workflow
        - User-friendly error handling
        """
    )

    st.markdown("## ⚠️ Limitations")
    st.markdown(
        """
        - Answers depend on the information successfully extracted and retrieved
          from the uploaded document.
        - If the retrieved context does not contain enough information, the
          assistant is instructed to say that the information is unavailable.
        - Large or visually complex PDFs can take longer to process.
        - Visual understanding depends on what is readable in the extracted images.
        - AI-generated answers should be verified when accuracy is critical.
        """
    )

    st.markdown("## 🛠️ Tips for Better Questions")
    st.markdown(
        """
        - Ask one clear question at a time.
        - Include a year, metric, category, or section name when relevant.
        - For comparisons, name the values or periods you want compared.
        - Ask for the page when you need to locate supporting information.
        - If an answer seems incomplete, ask a more specific follow-up question.
        """
    )

    st.markdown("## 🆘 Support / Troubleshooting")
    st.markdown(
        """
        **PDF processing fails:** confirm the file is a valid, readable PDF and
        try uploading it again.

        **The answer cannot be found:** rephrase the question using terminology
        that appears in the document, or ask for a specific page/topic.

        **API/model error:** verify the required API secrets are configured and
        that the services are available.

        **Unexpected application error:** check the Streamlit server logs for
        the underlying exception; the UI intentionally avoids exposing raw
        tracebacks to normal users.
        """
    )
