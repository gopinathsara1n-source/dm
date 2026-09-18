"""
DocuMind RAG pipeline.

This module is a Streamlit-compatible modular conversion of the supplied
working notebook. The core RAG sequence is preserved:

Docling -> Gemini visual descriptions -> canonical elements
-> structure-aware chunks -> Jina Embeddings v4 -> FAISS IndexFlatIP
-> Jina query embedding -> top-k retrieval -> Gemini answer generation.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import requests
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from google import genai
from google.genai import types
from pypdf import PdfReader

LOGGER = logging.getLogger("documind.rag")

JINA_EMBEDDING_MODEL_NAME = "jina-embeddings-v4"
JINA_EMBEDDING_OUTPUT_DIMENSIONALITY = 2048
JINA_EMBEDDING_ENDPOINT = "https://api.jina.ai/v1/embeddings"

JINA_EMBEDDING_BATCH_SIZE = 100
JINA_EMBEDDING_INTER_BATCH_DELAY = 1.0
JINA_EMBEDDING_MIN_REQUEST_INTERVAL = 0.20
JINA_EMBEDDING_MAX_RETRIES = 6

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
]
ANSWER_GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

TARGET_CHARS = 1800
MAX_CHARS = 3200
MIN_STANDALONE_CHARS = 150
VISUAL_MIN_WIDTH = 200
VISUAL_MIN_HEIGHT = 80

MAX_VISUAL_RETRIES = 5
VISUAL_DELAY_BETWEEN_REQUESTS = 3
VISUAL_INITIAL_RETRY_DELAY = 2
VISUAL_MAX_RETRY_DELAY = 30
ANSWER_MAX_RETRIES = 4
ANSWER_BASE_DELAY = 3

_last_request_time = 0.0


def _get_secret(name: str) -> str:
    value = os.getenv(name)
    if value:
        return value

    try:
        import streamlit as st
        value = st.secrets.get(name)
        if value:
            return str(value)
    except Exception:
        pass

    return ""


def _require_keys() -> tuple[str, str]:
    jina_key = _get_secret("JINA_API_KEY")
    gemini_key = _get_secret("GEMINI_API_KEY")
    missing = []
    if not jina_key:
        missing.append("JINA_API_KEY")
    if not gemini_key:
        missing.append("GEMINI_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing required API secret(s): " + ", ".join(missing)
        )
    return jina_key, gemini_key


def _clean_text(text: Any) -> str:
    if text is None:
        return ""
    text = str(text).replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _get_attr_or_key(obj: Any, name: str, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _object_to_dict(obj: Any) -> dict:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(mode="python")
        except Exception:
            pass
    if hasattr(obj, "dict"):
        try:
            return obj.dict()
        except Exception:
            pass
    return {}


def _get_item_text(item: Any) -> str:
    text = _get_attr_or_key(item, "text")
    if isinstance(text, str) and text.strip():
        return _clean_text(text)

    orig = _get_attr_or_key(item, "orig")
    if isinstance(orig, str) and orig.strip():
        return _clean_text(orig)

    if isinstance(item, dict):
        for key in ("text", "orig", "content"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return _clean_text(value)
    return ""


def _get_page_number(item: Any):
    prov = _get_attr_or_key(item, "prov", [])
    if not prov:
        return None
    try:
        return _get_attr_or_key(prov[0], "page_no", None)
    except Exception:
        return None


def _extract_table_cells(table: Any) -> list[dict]:
    cells = []
    try:
        dataframe = table.export_to_dataframe()
        if dataframe is not None:
            for row_idx, row in dataframe.iterrows():
                for col_idx, value in enumerate(row):
                    if pd.isna(value):
                        value = ""
                    cells.append({
                        "row": int(row_idx),
                        "col": int(col_idx),
                        "text": _clean_text(value),
                    })
            if cells:
                return cells
    except Exception:
        pass

    table_dict = _object_to_dict(table)
    data = table_dict.get("data", {})
    grid = data.get("grid", []) if isinstance(data, dict) else []
    if isinstance(grid, list):
        for row_idx, row in enumerate(grid):
            if not isinstance(row, list):
                continue
            for col_idx, cell in enumerate(row):
                value = (
                    cell.get("text") or cell.get("value") or ""
                    if isinstance(cell, dict)
                    else cell
                )
                cells.append({
                    "row": row_idx,
                    "col": col_idx,
                    "text": _clean_text(value),
                })
    return cells


def _table_to_text(table: Any) -> str:
    try:
        dataframe = table.export_to_dataframe()
        if dataframe is not None:
            dataframe = dataframe.fillna("")
            lines = []
            columns = [_clean_text(x) for x in dataframe.columns]
            if any(columns):
                lines.append(" | ".join(columns))
            for _, row in dataframe.iterrows():
                values = [_clean_text(v) for v in row.tolist()]
                if any(values):
                    lines.append(" | ".join(v if v else "-" for v in values))
            result = "\n".join(lines).strip()
            if result:
                return result
    except Exception:
        pass

    cells = _extract_table_cells(table)
    if not cells:
        return _get_item_text(table)

    max_row = max(x["row"] for x in cells)
    max_col = max(x["col"] for x in cells)
    matrix = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for cell in cells:
        matrix[cell["row"]][cell["col"]] = cell["text"]

    lines = []
    for row in matrix:
        if any(cell.strip() for cell in row):
            lines.append(" | ".join(cell.strip() or "-" for cell in row))
    return "\n".join(lines)


def _classify_gemini_error(error: Exception) -> str:
    text = str(error).upper()
    if any(x in text for x in ["401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED", "INVALID API KEY"]):
        return "authentication"
    if any(x in text for x in ["429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "QUOTA", "TOO MANY REQUESTS"]):
        return "quota"
    if any(x in text for x in ["503", "UNAVAILABLE", "SERVICE UNAVAILABLE", "SERVER BUSY", "INTERNAL", "500", "502", "504"]):
        return "temporary"
    return "permanent"


def _build_visual_prompt(page_number):
    return f"""
You are analyzing a visual extracted from page {page_number}
of a complex PDF document for a Retrieval-Augmented Generation
(RAG) system.

Create a highly accurate, self-contained description of the visual.

Include, when visible:
1. Chart or figure title.
2. Type of visual.
3. Important categories, labels and legends.
4. Important numerical values.
5. Dates, years or periods.
6. Units.
7. Trends, comparisons and relationships.
8. Forecast values, if present.
9. Important annotations.
10. Source, if visible.
11. Meaning or interpretation directly supported by the visual.

Do NOT invent information that is not visible.
Do NOT provide opinions or recommendations.
Preserve exact terminology, names, labels and numerical values
whenever they are readable.

Return ONLY the description.
""".strip()


def _call_gemini_visual(client, image_path, page_number, model_name):
    prompt = _build_visual_prompt(page_number)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    last_error = None
    for attempt in range(MAX_VISUAL_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Part.from_bytes(
                        data=image_bytes,
                        mime_type="image/png",
                    ),
                    prompt,
                ],
            )
            if not response.text:
                raise ValueError("Gemini returned an empty response.")
            return response.text.strip(), "success"
        except Exception as exc:
            last_error = exc
            error_type = _classify_gemini_error(exc)
            LOGGER.warning(
                "%s | attempt %s/%s | %s | %s",
                model_name,
                attempt + 1,
                MAX_VISUAL_RETRIES,
                error_type,
                str(exc)[:300],
            )
            if error_type in {"authentication", "permanent"}:
                return None, error_type
            if attempt < MAX_VISUAL_RETRIES - 1:
                delay = min(
                    VISUAL_INITIAL_RETRY_DELAY * (2 ** attempt)
                    + random.uniform(0, 1),
                    VISUAL_MAX_RETRY_DELAY,
                )
                time.sleep(delay)
    return None, _classify_gemini_error(last_error or RuntimeError("Unknown error"))


def _build_canonical_document(docling_data, gemini_visuals):
    texts_by_ref = {
        item.get("self_ref"): item
        for item in docling_data.get("texts", [])
        if item.get("self_ref")
    }
    tables_by_ref = {
        item.get("self_ref"): item
        for item in docling_data.get("tables", [])
        if item.get("self_ref")
    }
    pictures_by_ref = {
        item.get("self_ref"): item
        for item in docling_data.get("pictures", [])
        if item.get("self_ref")
    }
    groups_by_ref = {
        item.get("self_ref"): item
        for item in docling_data.get("groups", [])
        if item.get("self_ref")
    }
    gemini_by_picture = {
        item["picture_id"]: item
        for item in gemini_visuals
        if item.get("status") == "success"
    }

    picture_ref_to_id = {}
    for idx, picture in enumerate(docling_data.get("pictures", []), start=1):
        ref = picture.get("self_ref")
        if ref:
            picture_ref_to_id[ref] = f"picture_{idx:03d}"

    def resolve_ref(ref):
        if isinstance(ref, dict):
            ref = ref.get("$ref")
        if not isinstance(ref, str):
            return None
        if ref in texts_by_ref:
            return "text", texts_by_ref[ref]
        if ref in tables_by_ref:
            return "table", tables_by_ref[ref]
        if ref in pictures_by_ref:
            return "picture", pictures_by_ref[ref]
        if ref in groups_by_ref:
            return "group", groups_by_ref[ref]
        return None

    canonical_elements = []
    body_children = docling_data.get("body", {}).get("children", [])

    for position, child in enumerate(body_children):
        resolved = resolve_ref(child)
        if resolved is None:
            continue
        item_type, item = resolved
        page_no = _get_page_number(item)

        if item_type == "text":
            text = _get_item_text(item)
            if text:
                canonical_elements.append({
                    "element_id": f"element_{len(canonical_elements)+1:06d}",
                    "type": "text",
                    "position": position,
                    "page": page_no,
                    "text": text,
                    "label": item.get("label", "text"),
                    "source": {"docling_ref": item.get("self_ref")},
                })

        elif item_type == "table":
            table_text = _table_to_text(item)
            if table_text:
                canonical_elements.append({
                    "element_id": f"element_{len(canonical_elements)+1:06d}",
                    "type": "table",
                    "position": position,
                    "page": page_no,
                    "text": table_text,
                    "content": table_text,
                    "source": {"docling_ref": item.get("self_ref")},
                })

        elif item_type == "picture":
            picture_ref = item.get("self_ref")
            picture_id = picture_ref_to_id.get(picture_ref)
            result = gemini_by_picture.get(picture_id)
            if not result:
                continue
            description = (result.get("description") or "").strip()
            if not description:
                continue
            canonical_elements.append({
                "element_id": f"element_{len(canonical_elements)+1:06d}",
                "type": "visual",
                "position": position,
                "page": page_no,
                "picture_id": picture_id,
                "image_path": result.get("image_path"),
                "visual_description": description,
                "text": description,
                "content": description,
                "source": {
                    "docling_ref": picture_ref,
                    "gemini_model": result.get("model"),
                },
            })

    return {
        "schema_version": "1.0",
        "document_name": docling_data.get("name"),
        "source": {"type": "pdf", "origin": docling_data.get("origin")},
        "statistics": {
            "total_elements": len(canonical_elements),
            "text_elements": sum(x["type"] == "text" for x in canonical_elements),
            "table_elements": sum(x["type"] == "table" for x in canonical_elements),
            "visual_elements": sum(x["type"] == "visual" for x in canonical_elements),
        },
        "elements": canonical_elements,
    }


def _structure_aware_chunk(canonical_elements):
    remove_labels = {"page_header", "page_footer"}
    chunking_elements = [
        e for e in canonical_elements
        if not (e["type"] == "text" and e.get("label", "text") in remove_labels)
    ]

    def normalize_spaces(text):
        return _clean_text(text)

    def split_sentences(text):
        text = normalize_spaces(text)
        if not text:
            return []
        paragraphs = re.split(r"\n\s*\n", text)
        sentences = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            parts = re.split(
                r"(?<=[.!?])\s+(?=[A-Z₹0-9(\[])",
                paragraph,
            )
            sentences.extend(part.strip() for part in parts if part.strip())
        return sentences

    def is_probable_page_number(text):
        text = normalize_spaces(text)
        return bool(
            text
            and (
                re.fullmatch(r"\d{1,4}", text)
                or re.fullmatch(r"[ivxlcdmIVXLCDM]{1,8}", text)
            )
        )

    def is_low_value_fragment(text):
        text = normalize_spaces(text)
        return not text or is_probable_page_number(text) or len(text) < MIN_STANDALONE_CHARS

    chunks = []
    current_parts = []
    current_element_ids = []
    current_element_types = []
    current_pages = []
    current_section = ""

    def current_text():
        return "\n\n".join(x for x in current_parts if x.strip()).strip()

    def flush_chunk():
        nonlocal current_parts, current_element_ids, current_element_types, current_pages
        text = current_text()
        if not text:
            current_parts = []
            current_element_ids = []
            current_element_types = []
            current_pages = []
            return
        chunks.append({
            "chunk_id": f"chunk_{len(chunks):06d}",
            "page_start": min(current_pages) if current_pages else None,
            "page_end": max(current_pages) if current_pages else None,
            "section": current_section,
            "content": text,
            "element_ids": current_element_ids.copy(),
            "element_types": current_element_types.copy(),
            "has_table": "table" in current_element_types,
            "has_visual": "visual" in current_element_types,
        })
        current_parts = []
        current_element_ids = []
        current_element_types = []
        current_pages = []

    def add_element(element, content):
        content = normalize_spaces(content)
        if not content:
            return
        current_parts.append(content)
        current_element_ids.append(element["element_id"])
        current_element_types.append(element["type"])
        if element.get("page") is not None:
            current_pages.append(element["page"])

    def add_text_sentence(element, sentence):
        candidate = current_text()
        if candidate:
            candidate = candidate + "\n\n" + sentence
        else:
            candidate = sentence

        if len(candidate) <= TARGET_CHARS:
            add_element(element, sentence)
            return

        if current_text():
            flush_chunk()
            if len(sentence) <= MAX_CHARS:
                add_element(element, sentence)
                return

        words = sentence.split()
        buffer = []
        for word in words:
            test = " ".join(buffer + [word])
            if len(test) <= MAX_CHARS:
                buffer.append(word)
            else:
                if buffer:
                    add_element(element, " ".join(buffer))
                    flush_chunk()
                buffer = [word]
        if buffer:
            add_element(element, " ".join(buffer))

    for element in chunking_elements:
        element_type = element["type"]
        label = element.get("label", "")
        page = element.get("page")

        if element_type == "text" and label == "section_header":
            header = normalize_spaces(element.get("text", ""))
            if not header:
                continue
            flush_chunk()
            current_section = header
            add_element(element, header)
            continue

        if element_type == "table":
            flush_chunk()
            table_text = normalize_spaces(element.get("content") or element.get("text") or "")
            if table_text:
                add_element(element, table_text)
                flush_chunk()
            continue

        if element_type == "visual":
            flush_chunk()
            description = normalize_spaces(
                element.get("visual_description")
                or element.get("content")
                or element.get("text")
                or ""
            )
            if not description:
                continue
            visual_text = (
                "[VISUAL]\n"
                f"Picture ID: {element.get('picture_id')}\n"
                f"Page: {page}\n\n"
                f"{description}"
            )
            add_element(element, visual_text)
            flush_chunk()
            continue

        if element_type == "text":
            text = normalize_spaces(element.get("text") or element.get("content") or "")
            if not text or is_low_value_fragment(text):
                continue
            for sentence in split_sentences(text):
                add_text_sentence(element, normalize_spaces(sentence))

    flush_chunk()

    clean_chunks = []
    for chunk in chunks:
        content = normalize_spaces(chunk["content"])
        if len(content) >= MIN_STANDALONE_CHARS:
            chunk["content"] = content
            clean_chunks.append(chunk)
            continue

        if clean_chunks:
            previous = clean_chunks[-1]
            merged = previous["content"] + "\n\n" + content
            if len(merged) <= MAX_CHARS:
                previous["content"] = merged
                previous["page_end"] = max(
                    previous["page_end"] or 0,
                    chunk["page_end"] or 0,
                )
                previous["element_ids"].extend(chunk["element_ids"])
                previous["element_types"].extend(chunk["element_types"])
                previous["has_table"] = previous["has_table"] or chunk["has_table"]
                previous["has_visual"] = previous["has_visual"] or chunk["has_visual"]
                continue

        chunk["content"] = content
        clean_chunks.append(chunk)

    for i, chunk in enumerate(clean_chunks):
        chunk["chunk_id"] = f"chunk_{i:06d}"
    return clean_chunks


def _embedding_error_is_retryable(exc):
    text = str(exc).upper()
    return any(
        x in text
        for x in [
            "429", "RATE LIMIT", "TOO MANY REQUESTS",
            "500", "502", "503", "504",
            "UNAVAILABLE", "INTERNAL", "TIMEOUT", "DEADLINE",
        ]
    )


def _wait_for_request_slot():
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < JINA_EMBEDDING_MIN_REQUEST_INTERVAL:
        time.sleep(JINA_EMBEDDING_MIN_REQUEST_INTERVAL - elapsed)
    _last_request_time = time.monotonic()


def _embed_batch(jina_key, batch_texts, batch_start_index):
    headers = {
        "Authorization": f"Bearer {jina_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": JINA_EMBEDDING_MODEL_NAME,
        "input": batch_texts,
        "task": "retrieval.passage",
        "embedding_type": "float",
        "dimensions": JINA_EMBEDDING_OUTPUT_DIMENSIONALITY,
        "truncate": False,
    }

    for attempt in range(1, JINA_EMBEDDING_MAX_RETRIES + 1):
        try:
            _wait_for_request_slot()
            response = requests.post(
                JINA_EMBEDDING_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=180,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Jina HTTP {response.status_code}: {response.text[:500]}"
                )
            body = response.json()
            data = body.get("data", [])
            if len(data) != len(batch_texts):
                raise ValueError(
                    f"Jina returned {len(data)} embeddings for {len(batch_texts)} inputs."
                )
            data = sorted(data, key=lambda item: int(item["index"]))
            vectors = []
            for local_index, item in enumerate(data):
                values = np.asarray(item["embedding"], dtype="float32")
                if values.shape[0] != JINA_EMBEDDING_OUTPUT_DIMENSIONALITY:
                    raise ValueError(
                        f"Unexpected embedding dimension {values.shape[0]} "
                        f"for chunk {batch_start_index + local_index + 1}."
                    )
                if not np.isfinite(values).all():
                    raise ValueError("NaN/Inf detected in Jina embedding.")
                vectors.append(values)
            return vectors
        except Exception as exc:
            LOGGER.warning(
                "Embedding batch %s attempt %s/%s failed: %s",
                batch_start_index + 1,
                attempt,
                JINA_EMBEDDING_MAX_RETRIES,
                str(exc)[:300],
            )
            if not _embedding_error_is_retryable(exc) or attempt == JINA_EMBEDDING_MAX_RETRIES:
                raise
            time.sleep(min(2 ** (attempt - 1) + random.uniform(0, 1), 30))
    raise RuntimeError("Embedding request failed.")


def _embed_documents(jina_key, chunks):
    texts = []
    for chunk in chunks:
        title = chunk.get("section") or "none"
        content = str(chunk.get("content", "")).strip()
        if not content:
            raise ValueError(f"Empty chunk content: {chunk.get('chunk_id')}")
        texts.append(f"title: {title} | text: {content}")

    all_vectors = []
    for start in range(0, len(texts), JINA_EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + JINA_EMBEDDING_BATCH_SIZE]
        vectors = _embed_batch(jina_key, batch, start)
        all_vectors.extend(vectors)
        if start + JINA_EMBEDDING_BATCH_SIZE < len(texts):
            time.sleep(JINA_EMBEDDING_INTER_BATCH_DELAY)

    embeddings = np.vstack(all_vectors).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)
    return embeddings


def _build_faiss(chunks, embeddings):
    metadata = [
        {
            "faiss_index": i,
            "chunk_id": chunk["chunk_id"],
            "page_start": chunk["page_start"],
            "page_end": chunk["page_end"],
            "section": chunk["section"],
            "element_ids": chunk["element_ids"],
            "element_types": chunk["element_types"],
            "has_table": chunk["has_table"],
            "has_visual": chunk["has_visual"],
        }
        for i, chunk in enumerate(chunks)
    ]
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, metadata


def _embed_query(jina_key, question):
    global _last_request_time
    headers = {
        "Authorization": f"Bearer {jina_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    payload = {
        "model": JINA_EMBEDDING_MODEL_NAME,
        "input": [question],
        "task": "retrieval.query",
        "embedding_type": "float",
        "dimensions": JINA_EMBEDDING_OUTPUT_DIMENSIONALITY,
        "truncate": False,
    }

    for attempt in range(1, JINA_EMBEDDING_MAX_RETRIES + 1):
        try:
            _wait_for_request_slot()
            response = requests.post(
                JINA_EMBEDDING_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=60,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Jina HTTP {response.status_code}: {response.text[:500]}"
                )
            data = response.json().get("data", [])
            if not data:
                raise ValueError("Jina returned no query embedding.")
            vector = np.asarray(data[0]["embedding"], dtype="float32")
            if vector.shape[0] != JINA_EMBEDDING_OUTPUT_DIMENSIONALITY:
                raise ValueError(
                    f"Unexpected Jina query embedding dimension {vector.shape[0]}."
                )
            if not np.isfinite(vector).all():
                raise ValueError("NaN/Inf detected in Jina query embedding.")
            vector = vector / max(np.linalg.norm(vector), 1e-12)
            return vector.reshape(1, -1)
        except Exception as exc:
            LOGGER.warning(
                "Query embedding attempt %s/%s failed: %s",
                attempt,
                JINA_EMBEDDING_MAX_RETRIES,
                str(exc)[:300],
            )
            if not _embedding_error_is_retryable(exc) or attempt == JINA_EMBEDDING_MAX_RETRIES:
                raise
            time.sleep(min(2 ** (attempt - 1) + random.uniform(0, 1), 30))
    raise RuntimeError("Query embedding failed.")


def _retrieve(index, chunks, metadata, jina_key, question, top_k=5):
    query_embedding = _embed_query(jina_key, question)
    scores, indices = index.search(query_embedding, top_k)
    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        results.append({
            "rank": rank,
            "score": float(score),
            "chunk": chunks[idx],
            "metadata": metadata[idx],
        })
    return results


def _build_context(results):
    context_parts = []
    for item in results:
        chunk = item["chunk"]
        context_parts.append(
            f"""
SOURCE {item["rank"]}
Similarity score: {item["score"]:.4f}
Chunk ID: {chunk["chunk_id"]}
Page: {chunk["page_start"]} - {chunk["page_end"]}
Section: {chunk.get("section")}

CONTENT:
{chunk["content"]}
""".strip()
        )
    return "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(context_parts)


def _is_retryable_gemini_answer_error(exc):
    error_text = str(exc).upper()
    return any(
        code in error_text
        for code in [
            "429", "500", "502", "503", "504",
            "UNAVAILABLE", "RESOURCE_EXHAUSTED",
            "INTERNAL", "DEADLINE",
        ]
    )


def _call_gemini_answer(client, prompt):
    last_error = None
    for model_name in ANSWER_GEMINI_MODELS:
        for attempt in range(1, ANSWER_MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                if not response.text:
                    raise ValueError("Gemini returned an empty response.")
                return response.text.strip(), model_name
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "Answer model %s attempt %s/%s failed: %s",
                    model_name,
                    attempt,
                    ANSWER_MAX_RETRIES,
                    str(exc)[:300],
                )
                if not _is_retryable_gemini_answer_error(exc):
                    break
                if attempt < ANSWER_MAX_RETRIES:
                    time.sleep(
                        ANSWER_BASE_DELAY * (2 ** (attempt - 1))
                        + random.uniform(0, 2)
                    )
    raise RuntimeError("All Gemini answer-generation attempts failed.") from last_error


def _generate_answer(client, question, results):
    context = _build_context(results)
    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the retrieved
document context below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. Carefully compare all retrieved sources before answering.
4. Prefer the source that directly answers the exact question.
5. Do not select a source merely because it has the highest similarity score.
6. Pay close attention to dates, categories, units, metrics,
   geographic scope, and time periods.
7. Preserve numerical values from the document.
8. If calculation is required, calculate it using the numbers provided in the context.
9. If the retrieved context does not contain enough information,
   clearly say that the information is not available.
10. For multi-part questions, answer every part.
11. Mention the relevant page number when useful.
12. Do not mention FAISS, embeddings, chunks, or retrieval unless specifically asked.

FORMAT RULES:
- Return clean plain text or simple Markdown.
- Never use LaTeX.
- Use % for percentages.
- Use normal Markdown headings and bullet points when helpful.

RETRIEVED DOCUMENT CONTEXT
==========================

{context}

==========================

USER QUESTION
=============

{question}

==========================

FINAL ANSWER
============

Give a concise, accurate answer based strictly on the
retrieved document context.
"""
    return _call_gemini_answer(client, prompt)


def process_pdf(pdf_bytes: bytes, filename: str) -> dict:
    """
    Process one uploaded PDF using the notebook's existing pipeline.

    The notebook's Drive persistence is intentionally replaced only by an
    isolated temporary workspace because Streamlit uploads are runtime inputs.
    The extraction/chunking/embedding/retrieval/model logic is retained.
    """
    jina_key, gemini_key = _require_keys()

    if not pdf_bytes or not pdf_bytes.startswith(b"%PDF"):
        raise ValueError("The uploaded file does not appear to be a valid PDF.")

    workdir = Path(tempfile.mkdtemp(prefix="documind_"))
    pdf_path = workdir / filename
    pdf_path.write_bytes(pdf_bytes)

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_description = False

    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options
            )
        }
    )

    result = converter.convert(str(pdf_path))
    doc = result.document
    docling_data = result.document.export_to_dict()

    image_dir = workdir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    # Notebook Stage 2: extract all Docling pictures and filter visuals.
    page_text = {}
    for item, _level in doc.iterate_items():
        text = _get_item_text(item)
        if not text:
            continue
        page_no = _get_page_number(item)
        if page_no is not None:
            page_text.setdefault(page_no, []).append(text)

    picture_manifest = []
    for idx, picture in enumerate(doc.pictures, start=1):
        picture_id = f"picture_{idx:03d}"
        image_path = image_dir / f"{picture_id}.png"
        try:
            image = picture.get_image(doc)
            if image is None:
                continue
            image.save(image_path)

            page_no = None
            bbox = None
            try:
                prov = _get_attr_or_key(picture, "prov", [])
                if prov:
                    first = prov[0]
                    page_no = _get_attr_or_key(first, "page_no", None)
                    b = _get_attr_or_key(first, "bbox", None)
                    if b is not None:
                        bbox = {
                            "l": float(_get_attr_or_key(b, "l", 0)),
                            "t": float(_get_attr_or_key(b, "t", 0)),
                            "r": float(_get_attr_or_key(b, "r", 0)),
                            "b": float(_get_attr_or_key(b, "b", 0)),
                        }
            except Exception:
                pass

            if bbox:
                bbox_width = abs(bbox["r"] - bbox["l"])
                bbox_height = abs(bbox["b"] - bbox["t"])
            else:
                bbox_width = image.width
                bbox_height = image.height

            nearby_text = []
            if page_no is not None:
                for p in (page_no - 1, page_no, page_no + 1):
                    nearby_text.extend(page_text.get(p, []))
            seen = set()
            cleaned_context = []
            for text in nearby_text:
                text = text.strip()
                if text and text not in seen:
                    seen.add(text)
                    cleaned_context.append(text)

            picture_manifest.append({
                "picture_id": picture_id,
                "index": idx - 1,
                "page": page_no,
                "image_path": str(image_path),
                "bbox": bbox,
                "width": image.width,
                "height": image.height,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "nearby_text": cleaned_context,
            })
        except Exception as exc:
            LOGGER.warning("Error extracting %s: %s", picture_id, exc)

    visual_manifest = []
    for record in picture_manifest:
        width = record.get("bbox_width", record.get("width", 0))
        height = record.get("bbox_height", record.get("height", 0))
        if width < VISUAL_MIN_WIDTH or height < VISUAL_MIN_HEIGHT:
            continue
        image_path = Path(record["image_path"])
        if not image_path.exists():
            continue
        visual_manifest.append({
            "picture_id": record["picture_id"],
            "page": record["page"],
            "image_path": str(image_path),
            "width": round(width, 2),
            "height": round(height, 2),
            "area": round(width * height, 2),
            "nearby_text": record.get("nearby_text", []),
        })
    visual_manifest.sort(
        key=lambda x: (
            x["page"] if x["page"] is not None else 0,
            x["picture_id"],
        )
    )

    # Notebook Stage 3: Gemini visual understanding with fallback/retry.
    gemini_client = genai.Client(api_key=gemini_key)
    gemini_results = []
    for item in visual_manifest:
        final = None
        for model_name in GEMINI_MODELS:
            description, status = _call_gemini_visual(
                gemini_client,
                Path(item["image_path"]),
                item["page"],
                model_name,
            )
            if description:
                final = {
                    "picture_id": item["picture_id"],
                    "page": item["page"],
                    "image_path": item["image_path"],
                    "model": model_name,
                    "description": description,
                    "status": "success",
                }
                break
            if status == "authentication":
                final = {
                    **item,
                    "description": None,
                    "model": model_name,
                    "status": "authentication_error",
                }
                break
        if final is None:
            final = {
                **item,
                "description": None,
                "model": None,
                "status": "failed",
            }
        gemini_results.append(final)

    # Notebook Stage 4: canonical document.
    canonical_document = _build_canonical_document(
        docling_data,
        gemini_results,
    )

    # Notebook Stage 5: structure-aware chunking.
    chunks = _structure_aware_chunk(canonical_document["elements"])
    if not chunks:
        raise ValueError("No searchable content could be created from the PDF.")

    # Notebook Stage 6: Jina Embeddings v4 + normalized FAISS inner product.
    embeddings = _embed_documents(jina_key, chunks)
    if embeddings.shape != (
        len(chunks),
        JINA_EMBEDDING_OUTPUT_DIMENSIONALITY,
    ):
        raise ValueError("Unexpected embedding matrix shape.")

    index, metadata = _build_faiss(chunks, embeddings)

    return {
        "document_name": filename,
        "pages": total_pages,
        "chunks": chunks,
        "metadata": metadata,
        "index": index,
        "canonical_document": canonical_document,
        "gemini_visuals": gemini_results,
        "workspace": str(workdir),
        "embeddings_shape": list(embeddings.shape),
    }


def query_document(document: dict, question: str, top_k: int = 5) -> dict:
    if not document:
        raise ValueError("No document is currently loaded.")
    question = (question or "").strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    jina_key, gemini_key = _require_keys()
    results = _retrieve(
        document["index"],
        document["chunks"],
        document["metadata"],
        jina_key,
        question,
        top_k=top_k,
    )
    if not results:
        return {
            "answer": "I could not find relevant information in the uploaded document.",
            "sources": [],
            "model": None,
            "retrieved": [],
        }

    client = genai.Client(api_key=gemini_key)
    answer, model_used = _generate_answer(client, question, results)

    sources = []
    seen_pages = set()
    for item in results:
        chunk = item["chunk"]
        start = chunk.get("page_start")
        end = chunk.get("page_end")
        if start is None:
            page = None
        elif end is None or end == start:
            page = str(start)
        else:
            page = f"{start}–{end}"
        key = (page, round(item["score"], 6))
        if key in seen_pages:
            continue
        seen_pages.add(key)
        sources.append({
            "page": page,
            "score": item["score"],
            "rank": item["rank"],
        })

    return {
        "answer": answer,
        "sources": sources,
        "model": model_used,
        "retrieved": results,
    }


def get_document_info(document: dict) -> dict:
    canonical = document["canonical_document"]
    stats = canonical.get("statistics", {})
    return {
        "document_name": document.get("document_name"),
        "pages": document.get("pages"),
        "chunks": len(document.get("chunks", [])),
        "elements": stats.get("total_elements", 0),
        "text_elements": stats.get("text_elements", 0),
        "table_elements": stats.get("table_elements", 0),
        "visual_elements": stats.get("visual_elements", 0),
    }


def clear_document(document: dict | None) -> None:
    if not document:
        return
    workspace = document.get("workspace")
    if not workspace:
        return
    try:
        import shutil
        shutil.rmtree(workspace, ignore_errors=True)
    except Exception:
        LOGGER.exception("Unable to clean document workspace.")
