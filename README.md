# DocuMind — Streamlit PDF RAG Assistant

## 1. Overview

DocuMind is a two-tab Streamlit application for uploading a PDF and asking questions about it.

The supplied working notebook is the source of truth for the RAG architecture. The application modularizes that workflow without changing its core sequence or notebook parameters.

## 2. Features

- PDF upload
- Docling extraction of text, tables and pictures
- Gemini visual descriptions
- Structure-aware chunking
- Jina Embeddings v4
- Batched embedding requests
- FAISS `IndexFlatIP`
- Top-5 semantic retrieval
- Gemini answer generation with fallback models
- Page-aware supporting sources
- Chat-style follow-up questions
- Clear/start-new-document control
- About, usage and support tab

## 3. Architecture

```text
PDF
 ↓
Docling
 ├── Text
 ├── Tables
 └── Pictures
       ↓
    Gemini visual descriptions
 ↓
Canonical document
 ↓
Structure-aware chunking
 ↓
Jina Embeddings v4
 ↓
Normalized vectors
 ↓
FAISS IndexFlatIP
 ↓
User question
 ↓
Jina retrieval.query embedding
 ↓
Top 5 chunks
 ↓
Context
 ↓
Gemini
 ↓
Answer + supporting pages
```

## 4. Project structure

```text
documind_rag_streamlit/
├── app.py
├── rag_pipeline.py
├── about.py
├── requirements.txt
└── README.md
```

### `rag_pipeline.py`
Contains the complete RAG architecture and exposes:
- `ingest_pdf(...)`
- `retrieve_chunks(...)`
- `generate_answer(...)`
- `ask(...)`
- `get_document_info(...)`
- `clear_document(...)`

### `about.py`
Contains only the About & Help tab content.

### `app.py`
Contains Streamlit configuration, styling, tabs, session state and user interaction. It calls the RAG module and the About module.

## 5. Installation

Recommended Python: **3.12**.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install:

```bash
pip install -r requirements.txt
```

## 6. API keys

Create `.streamlit/secrets.toml`:

```toml
JINA_API_KEY = "your-jina-key"
GEMINI_API_KEY = "your-gemini-key"
```

For deployment, add the same secrets through Streamlit's Secrets settings.

Environment variables with the same names are also supported.

## 7. Run locally

```bash
streamlit run app.py
```

## 8. Deploy to Streamlit

1. Push the project to GitHub.
2. Create a Streamlit app from the repository.
3. Set the main file to `app.py`.
4. Add `JINA_API_KEY` and `GEMINI_API_KEY` under app secrets.
5. Deploy.

## 9. Important architecture note

The notebook-specific Google Colab `userdata` and Google Drive mounting were replaced with Streamlit-compatible API-key and temporary local storage handling.

This is an interface/deployment adaptation only. The RAG sequence, models, retrieval method, prompt rules, chunking parameters, embedding dimensions, batch size, FAISS metric and top-k retrieval remain based on the supplied working notebook.

## 10. Known limitations

- Processing large PDFs can take time.
- Complex PDFs can require substantial CPU/RAM for Docling.
- Jina and Gemini API quotas/rate limits can affect processing.
- Visual analysis requires Gemini calls.
- Uploaded-document information must be successfully extracted/retrieved for reliable answers.
- AI answers should be verified for high-stakes use.
