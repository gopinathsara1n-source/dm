# DocuMind — Intelligent PDF Assistant

DocuMind is a Streamlit-based PDF Retrieval-Augmented Generation (RAG) assistant.

It converts the supplied working Jupyter Notebook architecture into a modular application without replacing its core RAG components.

## Architecture

The application preserves this sequence:

```text
PDF
 ↓
Docling PDF processing
 ↓
Text + Tables + Pictures
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
Jina retrieval.query embedding
 ↓
Top-5 similarity retrieval
 ↓
Retrieved context
 ↓
Gemini answer generation
```

### Important implementation note

The notebook uses Google Drive as its persistent Colab workspace. In Streamlit, an uploaded PDF is instead placed in an isolated temporary workspace for that session. This changes storage mechanics only; the extraction, visual-understanding, chunking, embedding, FAISS retrieval, prompt, and LLM logic are retained.

## Project Structure

```text
documind/
├── app.py
├── rag_pipeline.py
├── about.py
├── requirements.txt
└── README.md
```

### `app.py`

Contains Streamlit configuration, UI, tabs, session state, chat rendering, upload controls, and calls into the RAG module.

### `rag_pipeline.py`

Contains the RAG implementation converted from the notebook:

- Docling extraction
- table extraction
- picture extraction/filtering
- Gemini visual understanding
- canonical document construction
- structure-aware chunking
- Jina Embeddings v4
- FAISS IndexFlatIP
- Jina query embedding
- top-k retrieval
- Gemini answer generation
- source/page metadata

### `about.py`

Contains the About & Help tab only.

## Requirements

The deployment uses `opencv-python-headless` rather than the GUI-enabled
`opencv-python`. This is important for Streamlit Cloud/Linux server
environments because GUI OpenCV can require the system library
`libGL.so.1`, which is not available in the hosted runtime.

The notebook did not specify exact package versions, so `requirements.txt`
uses compatible minimum versions where practical rather than inventing
notebook-specific pins.

Use a Python version supported by the current releases of Docling and Streamlit. Python 3.11 or 3.12 is a practical deployment target.

## API Secrets

DocuMind requires:

- `JINA_API_KEY`
- `GEMINI_API_KEY`

### Streamlit secrets

Create:

```text
.streamlit/secrets.toml
```

with:

```toml
JINA_API_KEY = "your-jina-key"
GEMINI_API_KEY = "your-gemini-key"
```

Do not commit this file to GitHub.

For deployment, add the same secrets in the Streamlit app's Secrets configuration.

Environment variables with the same names are also supported.

## Local Installation

```bash
python -m venv .venv
```

Activate the environment.

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

## Using the Application

1. Open DocuMind.
2. Upload a PDF.
3. Click **Process Document**.
4. Wait for the document to be converted, visual information to be described where applicable, chunks to be created, embeddings to be generated, and the FAISS index to be built.
5. Ask questions in the chat box.
6. Expand supporting pages when available.
7. Click **New Document** before switching to another PDF.

## RAG Behavior

The answer-generation prompt is preserved from the supplied notebook. The assistant is instructed to:

- use only retrieved document context;
- avoid outside knowledge;
- avoid inventing information;
- compare retrieved sources;
- preserve dates, categories, units, metrics and numerical values;
- calculate only from numbers present in the retrieved context;
- state when the retrieved context is insufficient;
- answer all parts of multi-part questions;
- mention page numbers when useful.

The chat history is maintained by Streamlit for the UI. The underlying retrieval remains question-based top-k FAISS retrieval, as in the notebook; chat history is not injected into the RAG prompt as a new conversational retrieval architecture.

## Visual Processing

The notebook enables Docling picture extraction and separately sends sufficiently large extracted pictures to Gemini for searchable visual descriptions. Those descriptions become `visual` canonical elements and intact chunks.

The notebook's visual-size filtering is preserved:

- minimum width: 200
- minimum height: 80

## Chunking

The notebook's structure-aware chunking parameters are preserved:

- target characters: 1800
- maximum characters: 3200
- minimum standalone characters: 150

Section headers update the active section. Tables and visual descriptions are kept as intact chunks. Page headers and footers are removed from the chunking stream.

## Embeddings and Retrieval

The notebook uses:

- model: `jina-embeddings-v4`
- output dimensionality: `2048`
- document task: `retrieval.passage`
- query task: `retrieval.query`
- batch size: `100`
- FAISS: `IndexFlatIP`
- normalized embeddings
- top-k: `5`

## Error Handling

The UI displays user-friendly errors. Detailed underlying failures are logged through Python logging.

API/model calls retain the retry and fallback approach from the notebook.

## Limitations

- Processing can be slow for very large PDFs, especially PDFs containing many images.
- Visual descriptions consume Gemini API requests.
- Embedding the document consumes Jina API quota.
- Answer generation consumes Gemini API quota.
- PDF extraction quality depends on the source document.
- A retrieved context may still be insufficient for a question even when related content exists elsewhere in the document.
- AI-generated answers should be verified when accuracy is important.

## Deployment

A typical Streamlit deployment flow is:

1. Push the project files to a GitHub repository.
2. Create a Streamlit app pointing to `app.py`.
3. Add `JINA_API_KEY` and `GEMINI_API_KEY` in the deployment's Secrets configuration.
4. Deploy.
5. Test with a representative PDF.

Do not place API keys in source code or commit secret files.

## Security Considerations

Each processed upload receives an isolated temporary workspace. When the user starts a new document, the previous workspace is removed where possible.

The application does not display API keys, raw prompts, embedding vectors, FAISS internals, or raw retrieved chunks in the normal user interface.


## Streamlit Cloud / Linux OpenCV Note

Docling can load OCR/layout components through Transformers and OpenCV.
On hosted Linux environments, `opencv-python` may fail with:

```text
ImportError: libGL.so.1: cannot open shared object file
```

Use `opencv-python-headless` in `requirements.txt`. Do not install both
`opencv-python` and `opencv-python-headless`.

The included `.streamlit/config.toml` also disables Streamlit's source
file watcher. This prevents Streamlit from repeatedly introspecting large
third-party packages such as Transformers; it is not a substitute for the
headless OpenCV fix.
