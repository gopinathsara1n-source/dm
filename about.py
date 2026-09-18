"""About & Help page for DocuMind."""

import streamlit as st


def render_about():
    st.markdown("## 📘 About DocuMind")
    st.markdown(
        "DocuMind is an AI-powered PDF question-answering assistant based on "
        "Retrieval-Augmented Generation (RAG). Upload a PDF, wait for processing, "
        "and ask natural-language questions about the document."
    )

    st.markdown("### ⚙️ How it works")
    st.markdown(
        """
**PDF Upload**
→ **Docling Processing**
→ **Text + Table + Picture Extraction**
→ **Gemini Visual Description**
→ **Canonical Document**
→ **Structure-aware Chunking**
→ **Jina Embeddings v4**
→ **FAISS Vector Search**
→ **Relevant Context**
→ **Gemini Answer**
        """
    )

    st.info(
        "The answer-generation step is instructed to use only the retrieved "
        "document context and to say when the required information is not available."
    )

    st.markdown("### 🚀 How to use")
    steps = [
        "Upload a PDF in the **Ask Your Document** tab.",
        "Wait until processing finishes.",
        "Ask a question in the chat box.",
        "Review the answer and expand **Supporting sources** to see retrieved pages.",
        "Ask follow-up questions about the same PDF.",
        "Use **Clear document** before starting with another PDF.",
    ]
    for i, step in enumerate(steps, 1):
        st.markdown(f"**{i}.** {step}")

    st.markdown("### 💡 Example questions")
    examples = [
        "What is the main objective of this document?",
        "Summarize the key findings.",
        "What are the important financial figures?",
        "What does the document say about a specific topic?",
        "Compare the values mentioned for different years.",
        "Which page contains information about a specific topic?",
        "What does the chart on page X show?",
    ]
    for example in examples:
        st.markdown(f"• {example}")

    st.markdown("### ✨ Features")
    features = [
        "PDF question answering",
        "Structure-aware retrieval",
        "Text, table and visual-description support",
        "Page-aware supporting sources",
        "Chat-style follow-up questions",
        "Jina embedding API with batched ingestion",
        "FAISS similarity search",
        "Gemini answer generation with fallback models",
        "Clear document / start a new document",
    ]
    cols = st.columns(2)
    for i, feature in enumerate(features):
        cols[i % 2].markdown(f"✅ {feature}")

    st.markdown("### ⚠️ Limitations")
    limitations = [
        "Answers depend on information successfully extracted from the uploaded PDF.",
        "If the required information is absent from the retrieved context, the assistant should say it is unavailable.",
        "Large or visually complex PDFs can take longer to process.",
        "Jina and Gemini API availability, quotas and rate limits can affect processing.",
        "AI-generated answers should be verified when accuracy is critical.",
    ]
    for item in limitations:
        st.markdown(f"• {item}")

    st.markdown("### 🆘 Support / Help")
    st.markdown(
        """
**If PDF processing fails**
- Check that the file is a valid PDF.
- Try a smaller or simpler PDF.
- Check the API keys and service quotas.
- Review the Streamlit deployment logs.

**If an answer is not useful**
- Ask a more specific question.
- Include a year, topic, table name, metric or page reference when known.
- Ask one multi-part question only when the document clearly contains all parts.

**For better questions**
- Ask directly about information contained in the PDF.
- Use exact terminology from the document where possible.
- For tables, mention the row/column or metric you need.
- For charts, mention the title, year or category if visible.
        """
    )

    st.caption("DocuMind • Intelligent PDF Question Answering")
