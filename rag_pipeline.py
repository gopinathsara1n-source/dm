"""
rag_pipeline.py
================
Core Retrieval-Augmented-Generation pipeline used by the Streamlit app.

This is a Streamlit-friendly refactor of the original Colab notebook
("final_RAG_gemini_embedding.ipynb"). The logic (Docling ingestion ->
Gemini vision descriptions -> canonical document -> structure-aware
chunking -> Gemini Embedding -> FAISS -> Gemini answer generation) is
preserved, but:

  * Google Colab / Google Drive mounting is removed.
  * All intermediate artifacts live in a per-session temp folder
    instead of Google Drive.
  * Every long-running stage accepts an optional ``progress_cb``
    callback ``(fraction: float, message: str) -> None`` so the UI
    can show live progress instead of printing to stdout.
  * The embedding step no longer assumes a multi-day quota budget --
    it embeds everything in one run with safe rate limiting, since a
    Streamlit session is short-lived.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from pypdf import PdfReader

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions

from google import genai
from google.genai import types

import faiss

ProgressCB = Optional[Callable[[float, str], None]]


def _tick(cb: ProgressCB, frac: float, msg: str) -> None:
    if cb is not None:
        try:
            cb(min(max(frac, 0.0), 1.0), msg)
        except Exception:
            pass


# ------------------------------------------------------------------
# CONFIG DEFAULTS (mirrors Stage 1 of the notebook)
# ------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "gemini-embedding-2"
EMBEDDING_OUTPUT_DIMENSIONALITY = 768

VISION_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
]

ANSWER_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
]

GEMINI_EMBEDDING_MIN_REQUEST_INTERVAL = 0.75
GEMINI_EMBEDDING_BATCH_SIZE = 20
GEMINI_EMBEDDING_INTER_BATCH_DELAY = 3
GEMINI_EMBEDDING_MAX_RETRIES = 6

VISUAL_MIN_WIDTH = 200
VISUAL_MIN_HEIGHT = 80

TARGET_CHARS = 1800
MAX_CHARS = 3200
MIN_STANDALONE_CHARS = 150

VISION_MAX_RETRIES = 5
VISION_INITIAL_RETRY_DELAY = 2
VISION_MAX_RETRY_DELAY = 30
VISION_DELAY_BETWEEN_REQUESTS = 1.0


@dataclass
class DocumentIndex:
    """Everything needed to answer questions about one processed PDF."""

    document_name: str
    total_pages: int
    chunks: list = field(default_factory=list)
    metadata: list = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    index: Optional[faiss.Index] = None
    stats: dict = field(default_factory=dict)
    image_dir: Optional[str] = None


# ------------------------------------------------------------------
# DOCLING CONVERTER (build once, reuse)
# ------------------------------------------------------------------

def build_converter(do_ocr: bool = False) -> DocumentConverter:
    """
    do_ocr=False by default: Docling's OCR engines pull in opencv-python,
    which needs the system library libGL.so.1. Streamlit Community Cloud
    doesn't have it installed unless it's listed in packages.txt, so this
    stays off unless the caller explicitly turns it on (and packages.txt
    is present). Most PDFs made from Word/PowerPoint/Docs etc. already
    have a text layer and don't need OCR at all.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_table_structure = True
    pipeline_options.generate_picture_images = True
    pipeline_options.do_picture_description = False
    pipeline_options.do_ocr = do_ocr

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )


# ------------------------------------------------------------------
# HELPERS (verbatim logic from Stage 2 of the notebook)
# ------------------------------------------------------------------

def get_attr_or_key(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def clean_text(text) -> str:
    if text is None:
        return ""
    text = str(text)
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_item_text(item) -> str:
    text = get_attr_or_key(item, "text", None)
    if isinstance(text, str) and text.strip():
        return clean_text(text)

    orig = get_attr_or_key(item, "orig", None)
    if isinstance(orig, str) and orig.strip():
        return clean_text(orig)

    if isinstance(item, dict):
        for key in ["text", "orig", "content"]:
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return clean_text(value)
    return ""


def get_page_number(item):
    prov = get_attr_or_key(item, "prov", [])
    if not prov:
        return None
    try:
        first_prov = prov[0]
        page_no = get_attr_or_key(first_prov, "page_no", None)
        if page_no is not None:
            return page_no
    except Exception:
        pass
    return None


def extract_table_cells(table):
    cells = []
    try:
        dataframe = table.export_to_dataframe()
        if dataframe is not None:
            for row_idx, row in dataframe.iterrows():
                for col_idx, value in enumerate(row):
                    if pd.isna(value):
                        value = ""
                    cells.append({"row": int(row_idx), "col": int(col_idx), "text": clean_text(value)})
            if cells:
                return cells
    except Exception:
        pass

    table_dict = table if isinstance(table, dict) else {}
    data = table_dict.get("data", {})
    if isinstance(data, dict):
        grid = data.get("grid", [])
        if isinstance(grid, list):
            for row_idx, row in enumerate(grid):
                if not isinstance(row, list):
                    continue
                for col_idx, cell in enumerate(row):
                    if isinstance(cell, dict):
                        value = cell.get("text") or cell.get("value") or ""
                    else:
                        value = cell
                    cells.append({"row": row_idx, "col": col_idx, "text": clean_text(value)})
    return cells


def table_to_text(table) -> str:
    try:
        dataframe = table.export_to_dataframe()
        if dataframe is not None:
            dataframe = dataframe.fillna("")
            lines = []
            columns = [clean_text(x) for x in dataframe.columns]
            if any(columns):
                lines.append(" | ".join(columns))
            for _, row in dataframe.iterrows():
                values = [clean_text(value) for value in row.tolist()]
                if any(values):
                    lines.append(" | ".join(v if v else "-" for v in values))
            result_text = "\n".join(lines).strip()
            if result_text:
                return result_text
    except Exception:
        pass

    cells = extract_table_cells(table)
    if not cells:
        return get_item_text(table)

    max_row = max(x["row"] for x in cells)
    max_col = max(x["col"] for x in cells)
    matrix = [["" for _ in range(max_col + 1)] for _ in range(max_row + 1)]
    for cell in cells:
        matrix[cell["row"]][cell["col"]] = cell["text"]

    lines = []
    for row in matrix:
        if not any(cell.strip() for cell in row):
            continue
        lines.append(" | ".join(cell.strip() if cell.strip() else "-" for cell in row))
    return "\n".join(lines)


# ------------------------------------------------------------------
# STAGE 2 — DOCLING INGESTION (text + tables + pictures)
# ------------------------------------------------------------------

def ingest_pdf(pdf_path: Path, image_dir: Path, converter: DocumentConverter, progress_cb: ProgressCB = None):
    image_dir.mkdir(parents=True, exist_ok=True)

    _tick(progress_cb, 0.02, "Reading PDF with Docling (text, tables, images)...")
    result = converter.convert(str(pdf_path))
    doc = result.document
    docling_data = result.document.export_to_dict()

    # ---- extract every picture to PNG on disk ----
    picture_manifest = []
    total_pics = max(len(doc.pictures), 1)
    for idx, picture in enumerate(doc.pictures, start=1):
        picture_id = f"picture_{idx:03d}"
        image_path = image_dir / f"{picture_id}.png"
        try:
            image = picture.get_image(doc)
            if image is None:
                continue
            image.save(image_path)

            page_no, bbox = None, None
            prov = get_attr_or_key(picture, "prov", [])
            if prov:
                first_prov = prov[0]
                page_no = get_attr_or_key(first_prov, "page_no", None)
                b = get_attr_or_key(first_prov, "bbox", None)
                if b is not None:
                    bbox = {
                        "l": float(get_attr_or_key(b, "l", 0)),
                        "t": float(get_attr_or_key(b, "t", 0)),
                        "r": float(get_attr_or_key(b, "r", 0)),
                        "b": float(get_attr_or_key(b, "b", 0)),
                    }

            if bbox:
                bbox_width = abs(bbox["r"] - bbox["l"])
                bbox_height = abs(bbox["b"] - bbox["t"])
            else:
                bbox_width, bbox_height = image.width, image.height

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
            })
        except Exception:
            pass

        _tick(progress_cb, 0.05 + 0.10 * (idx / total_pics), f"Extracting images ({idx}/{total_pics})...")

    # ---- page text lookup, used to give each picture some nearby context ----
    page_text = defaultdict(list)
    for item, _level in doc.iterate_items():
        text = get_item_text(item)
        if not text:
            continue
        page_no = get_page_number(item)
        if page_no is not None:
            page_text[page_no].append(text)

    picture_context_manifest = []
    for picture_info in picture_manifest:
        page_no = picture_info["page"]
        nearby_text = []
        if page_no is not None:
            nearby_text.extend(page_text.get(page_no - 1, []))
            nearby_text.extend(page_text.get(page_no, []))
            nearby_text.extend(page_text.get(page_no + 1, []))

        seen, cleaned_context = set(), []
        for text in nearby_text:
            text = text.strip()
            if text and text not in seen:
                seen.add(text)
                cleaned_context.append(text)

        picture_context_manifest.append({**picture_info, "nearby_text": cleaned_context})

    # ---- filter to visuals big enough to matter ----
    visual_manifest = []
    for record in picture_context_manifest:
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
            "nearby_text": record.get("nearby_text", []),
        })

    visual_manifest.sort(key=lambda x: (x["page"] if x["page"] is not None else 0, x["picture_id"]))

    _tick(progress_cb, 0.18, "Document ingestion complete.")
    return docling_data, picture_manifest, visual_manifest


# ------------------------------------------------------------------
# STAGE 3 — GEMINI VISUAL UNDERSTANDING
# ------------------------------------------------------------------

def build_visual_prompt(page_number) -> str:
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


def _classify_gemini_error(error) -> str:
    text = str(error).upper()
    if any(x in text for x in ["401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED", "INVALID API KEY"]):
        return "authentication"
    if any(x in text for x in ["429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "QUOTA", "TOO MANY REQUESTS"]):
        return "quota"
    if any(x in text for x in ["503", "UNAVAILABLE", "SERVICE UNAVAILABLE", "SERVER BUSY", "INTERNAL", "500", "502", "504"]):
        return "temporary"
    return "permanent"


def _call_vision_with_retry(client, image_path, page_number, model_name, vision_models):
    prompt = build_visual_prompt(page_number)
    with open(image_path, "rb") as f:
        image_bytes = f.read()

    for attempt in range(VISION_MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[types.Part.from_bytes(data=image_bytes, mime_type="image/png"), prompt],
            )
            if not response.text:
                raise ValueError("Gemini returned an empty response.")
            return response.text.strip(), "success"
        except Exception as e:
            error_type = _classify_gemini_error(e)
            if error_type == "authentication":
                return None, "authentication"
            if error_type == "permanent":
                return None, "permanent"
            if attempt < VISION_MAX_RETRIES - 1:
                delay = min(VISION_INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1), VISION_MAX_RETRY_DELAY)
                time.sleep(delay)
    return None, "failed"


def describe_visuals(client, visual_manifest, vision_models=None, progress_cb: ProgressCB = None):
    vision_models = vision_models or VISION_MODELS
    results = []
    total = max(len(visual_manifest), 1)

    for n, item in enumerate(visual_manifest, start=1):
        picture_id = item["picture_id"]
        page_number = item["page"]
        image_path = Path(item["image_path"])

        _tick(progress_cb, 0.18 + 0.32 * (n / total), f"Describing visual {n}/{len(visual_manifest)} (Gemini vision)...")

        if not image_path.exists():
            results.append({"picture_id": picture_id, "page": page_number, "description": None, "status": "image_missing"})
            continue

        record = None
        for model_name in vision_models:
            description, status = _call_vision_with_retry(client, image_path, page_number, model_name, vision_models)
            if description:
                record = {
                    "picture_id": picture_id, "page": page_number, "image_path": str(image_path),
                    "model": model_name, "description": description, "status": "success",
                }
                break
            if status == "authentication":
                record = {"picture_id": picture_id, "page": page_number, "description": None, "status": "authentication_error"}
                break

        if record is None:
            record = {"picture_id": picture_id, "page": page_number, "description": None, "status": "failed"}

        results.append(record)

        if record["status"] == "authentication_error":
            raise RuntimeError("Gemini API key was rejected while describing visuals. Check your API key.")

        if n < len(visual_manifest):
            time.sleep(VISION_DELAY_BETWEEN_REQUESTS)

    return results


# ------------------------------------------------------------------
# STAGE 4 — CANONICAL DOCUMENT (text + table + visual descriptions)
# ------------------------------------------------------------------

def build_canonical_elements(docling_data, gemini_visuals, progress_cb: ProgressCB = None):
    texts_by_ref = {item.get("self_ref"): item for item in docling_data.get("texts", []) if item.get("self_ref")}
    tables_by_ref = {item.get("self_ref"): item for item in docling_data.get("tables", []) if item.get("self_ref")}
    pictures_by_ref = {item.get("self_ref"): item for item in docling_data.get("pictures", []) if item.get("self_ref")}
    groups_by_ref = {item.get("self_ref"): item for item in docling_data.get("groups", []) if item.get("self_ref")}

    gemini_by_picture = {item["picture_id"]: item for item in gemini_visuals if item.get("status") == "success"}

    picture_ref_to_id = {}
    for idx, picture in enumerate(docling_data.get("pictures", []), start=1):
        self_ref = picture.get("self_ref")
        if self_ref:
            picture_ref_to_id[self_ref] = f"picture_{idx:03d}"

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

    def get_page_info(item):
        prov = item.get("prov", [])
        if prov and isinstance(prov[0], dict):
            return prov[0].get("page_no")
        return None

    canonical_elements = []
    body_children = docling_data.get("body", {}).get("children", [])
    total = max(len(body_children), 1)

    for position, child in enumerate(body_children):
        if position % 40 == 0:
            _tick(progress_cb, 0.50 + 0.05 * (position / total), "Building canonical document...")

        resolved = resolve_ref(child)
        if resolved is None:
            continue
        item_type, item = resolved
        page_no = get_page_info(item)

        if item_type == "text":
            text = get_item_text(item)
            if not text:
                continue
            canonical_elements.append({
                "element_id": f"element_{len(canonical_elements)+1:06d}", "type": "text", "position": position,
                "page": page_no, "text": text, "label": item.get("label", "text"),
            })

        elif item_type == "table":
            table_text = table_to_text(item)
            if not table_text:
                continue
            canonical_elements.append({
                "element_id": f"element_{len(canonical_elements)+1:06d}", "type": "table", "position": position,
                "page": page_no, "text": table_text, "content": table_text,
            })

        elif item_type == "picture":
            picture_ref = item.get("self_ref")
            picture_id = picture_ref_to_id.get(picture_ref)
            gemini_result = gemini_by_picture.get(picture_id)
            if not gemini_result:
                continue
            description = (gemini_result.get("description") or "").strip()
            if not description:
                continue
            canonical_elements.append({
                "element_id": f"element_{len(canonical_elements)+1:06d}", "type": "visual", "position": position,
                "page": page_no, "picture_id": picture_id, "image_path": gemini_result.get("image_path"),
                "visual_description": description, "text": description, "content": description,
            })

    REMOVE_LABELS = {"page_header", "page_footer"}
    chunking_elements = [
        e for e in canonical_elements
        if not (e["type"] == "text" and e.get("label", "text") in REMOVE_LABELS)
    ]

    stats = dict(Counter(x["type"] for x in chunking_elements))
    _tick(progress_cb, 0.55, "Canonical document ready.")
    return chunking_elements, stats


# ------------------------------------------------------------------
# STAGE 5 — STRUCTURE-AWARE CHUNKING
# ------------------------------------------------------------------

def normalize_spaces(text) -> str:
    text = text or ""
    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_sentences(text) -> list:
    text = normalize_spaces(text)
    if not text:
        return []
    paragraphs = re.split(r"\n\s*\n", text)
    sentences = []
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z₹0-9(\[])", paragraph)
        for part in parts:
            part = part.strip()
            if part:
                sentences.append(part)
    return sentences


def is_probable_page_number(text) -> bool:
    text = normalize_spaces(text)
    if not text:
        return False
    if re.fullmatch(r"\d{1,4}", text):
        return True
    if re.fullmatch(r"[ivxlcdmIVXLCDM]{1,8}", text):
        return True
    return False


def is_low_value_fragment(text) -> bool:
    text = normalize_spaces(text)
    if not text:
        return True
    if is_probable_page_number(text):
        return True
    return len(text) < MIN_STANDALONE_CHARS


def chunk_elements(chunking_elements, progress_cb: ProgressCB = None):
    chunks = []
    current_parts, current_element_ids, current_element_types, current_pages = [], [], [], []
    current_section = ""

    def current_text():
        return "\n\n".join(x for x in current_parts if x.strip()).strip()

    def flush_chunk():
        nonlocal current_parts, current_element_ids, current_element_types, current_pages
        text = current_text()
        if not text:
            current_parts, current_element_ids, current_element_types, current_pages = [], [], [], []
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
        current_parts, current_element_ids, current_element_types, current_pages = [], [], [], []

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
        candidate = candidate + "\n\n" + sentence if candidate else sentence

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

    total = max(len(chunking_elements), 1)
    for i, element in enumerate(chunking_elements):
        if i % 40 == 0:
            _tick(progress_cb, 0.55 + 0.05 * (i / total), "Chunking document...")

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
            description = normalize_spaces(element.get("visual_description") or element.get("content") or element.get("text") or "")
            if not description:
                continue
            visual_text = f"[VISUAL]\nPicture ID: {element.get('picture_id')}\nPage: {page}\n\n{description}"
            add_element(element, visual_text)
            flush_chunk()
            continue

        if element_type == "text":
            text = normalize_spaces(element.get("text") or element.get("content") or "")
            if not text or is_low_value_fragment(text):
                continue
            sentences = split_sentences(text)
            for sentence in sentences:
                add_text_sentence(element, normalize_spaces(sentence))

    flush_chunk()

    # merge very small chunks into the previous one
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
                previous["page_end"] = max(previous["page_end"] or 0, chunk["page_end"] or 0)
                previous["element_ids"].extend(chunk["element_ids"])
                previous["element_types"].extend(chunk["element_types"])
                previous["has_table"] = previous["has_table"] or chunk["has_table"]
                previous["has_visual"] = previous["has_visual"] or chunk["has_visual"]
                continue
        chunk["content"] = content
        clean_chunks.append(chunk)

    chunks = clean_chunks
    for i, chunk in enumerate(chunks):
        chunk["chunk_id"] = f"chunk_{i:06d}"

    _tick(progress_cb, 0.60, f"Chunking complete — {len(chunks)} chunks.")
    return chunks


# ------------------------------------------------------------------
# STAGE 6 — GEMINI EMBEDDING + FAISS
# ------------------------------------------------------------------

def _prepare_document(chunk) -> str:
    title = chunk.get("section") or "none"
    content = str(chunk.get("content", "")).strip()
    return f"title: {title} | text: {content}"


def _embedding_error_is_retryable(exc) -> bool:
    text = str(exc).upper()
    return any(x in text for x in [
        "429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "TOO MANY REQUESTS",
        "500", "502", "503", "504", "UNAVAILABLE", "INTERNAL", "DEADLINE",
    ])


def embed_chunks(client, chunks, embedding_model=EMBEDDING_MODEL_NAME,
                  output_dim=EMBEDDING_OUTPUT_DIMENSIONALITY, progress_cb: ProgressCB = None):
    texts = [_prepare_document(chunk) for chunk in chunks]
    embeddings = [None] * len(texts)
    last_request_time = [0.0]

    def wait_for_slot():
        elapsed = time.monotonic() - last_request_time[0]
        if elapsed < GEMINI_EMBEDDING_MIN_REQUEST_INTERVAL:
            time.sleep(GEMINI_EMBEDDING_MIN_REQUEST_INTERVAL - elapsed)
        last_request_time[0] = time.monotonic()

    def embed_one(text, chunk_number):
        for attempt in range(1, GEMINI_EMBEDDING_MAX_RETRIES + 1):
            try:
                wait_for_slot()
                response = client.models.embed_content(
                    model=embedding_model,
                    contents=text,
                    config=types.EmbedContentConfig(output_dimensionality=output_dim),
                )
                if not response.embeddings:
                    raise ValueError("Gemini returned no embedding.")
                values = np.asarray(response.embeddings[0].values, dtype="float32")
                if values.shape[0] != output_dim or not np.isfinite(values).all():
                    raise ValueError(f"Bad embedding for chunk {chunk_number}.")
                return values
            except Exception as exc:
                if not _embedding_error_is_retryable(exc) or attempt == GEMINI_EMBEDDING_MAX_RETRIES:
                    raise
                delay = min(2 ** (attempt - 1) + random.uniform(0, 1), 30)
                time.sleep(delay)

    total = max(len(texts), 1)
    for batch_start in range(0, len(texts), GEMINI_EMBEDDING_BATCH_SIZE):
        batch_indices = list(range(batch_start, min(batch_start + GEMINI_EMBEDDING_BATCH_SIZE, len(texts))))
        for idx in batch_indices:
            embeddings[idx] = embed_one(texts[idx], idx + 1)
            _tick(progress_cb, 0.60 + 0.35 * ((idx + 1) / total), f"Embedding chunk {idx + 1}/{total} (Gemini)...")
        if batch_start + GEMINI_EMBEDDING_BATCH_SIZE < len(texts):
            time.sleep(GEMINI_EMBEDDING_INTER_BATCH_DELAY)

    embeddings = np.vstack(embeddings).astype("float32")
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.clip(norms, 1e-12, None)

    metadata = []
    for i, chunk in enumerate(chunks):
        metadata.append({
            "faiss_index": i, "chunk_id": chunk["chunk_id"], "page_start": chunk["page_start"],
            "page_end": chunk["page_end"], "section": chunk["section"],
            "has_table": chunk["has_table"], "has_visual": chunk["has_visual"],
        })

    _tick(progress_cb, 0.96, "Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    _tick(progress_cb, 1.0, "Document ready for chat.")
    return embeddings, metadata, index


# ------------------------------------------------------------------
# STAGE 7 — RETRIEVAL + ANSWER GENERATION
# ------------------------------------------------------------------

def _prepare_query(question: str) -> str:
    return f"task: search result | query: {question}"


def embed_query(client, question, embedding_model=EMBEDDING_MODEL_NAME, output_dim=EMBEDDING_OUTPUT_DIMENSIONALITY):
    response = client.models.embed_content(
        model=embedding_model,
        contents=_prepare_query(question),
        config=types.EmbedContentConfig(output_dimensionality=output_dim),
    )
    if not response.embeddings:
        raise ValueError("Gemini returned no query embedding.")
    vector = np.asarray(response.embeddings[0].values, dtype="float32")
    vector = vector / max(np.linalg.norm(vector), 1e-12)
    return vector.reshape(1, -1)


def retrieve_chunks(client, doc_index: DocumentIndex, question, top_k=5):
    query_embedding = embed_query(client, question)
    scores, indices = doc_index.index.search(query_embedding, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        results.append({
            "rank": rank, "score": float(score),
            "chunk": doc_index.chunks[idx], "metadata": doc_index.metadata[idx],
        })
    return results


def build_context(results) -> str:
    context_parts = []
    for item in results:
        chunk = item["chunk"]
        context_parts.append(
            f"""SOURCE {item["rank"]}
Similarity score: {item["score"]:.4f}
Chunk ID: {chunk["chunk_id"]}
Page: {chunk["page_start"]} - {chunk["page_end"]}
Section: {chunk.get("section")}

CONTENT:
{chunk["content"]}""".strip()
        )
    return "\n\n" + ("\n\n" + "=" * 60 + "\n\n").join(context_parts)


def _is_retryable_error(exc) -> bool:
    error_text = str(exc).upper()
    retryable_codes = ["429", "500", "502", "503", "504", "UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL", "DEADLINE"]
    return any(code in error_text for code in retryable_codes)


def _call_gemini_with_fallback(client, prompt, models, max_retries=4, base_delay=3):
    last_error = None
    for model_name in models:
        for attempt in range(1, max_retries + 1):
            try:
                response = client.models.generate_content(model=model_name, contents=prompt)
                if not response.text:
                    raise ValueError("Gemini returned an empty response.")
                return response.text.strip(), model_name
            except Exception as exc:
                last_error = exc
                if not _is_retryable_error(exc):
                    break
                if attempt < max_retries:
                    time.sleep(base_delay * (2 ** (attempt - 1)) + random.uniform(0, 2))
    raise RuntimeError("All Gemini models failed.") from last_error


def generate_answer(client, question, results, answer_models=None, chat_history=None):
    answer_models = answer_models or ANSWER_MODELS
    context = build_context(results)

    history_block = ""
    if chat_history:
        turns = []
        for turn in chat_history[-4:]:
            turns.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
        history_block = (
            "\nRECENT CONVERSATION (for follow-up context only — the "
            "document context above is still the single source of truth)\n"
            "===========================================================\n\n"
            + "\n\n".join(turns) + "\n"
        )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the retrieved
document context below.

IMPORTANT RULES:

1. Do not use outside knowledge.
2. Do not invent information.
3. Carefully compare all retrieved sources before answering.
4. Prefer the source that directly answers the exact question.
5. Do not select a source merely because it has the highest
   similarity score.
6. Pay close attention to:
   - dates
   - categories
   - units
   - metrics
   - geographic scope
   - time periods
7. Preserve numerical values from the document.
8. If calculation is required, calculate it using the
   numbers provided in the context.
9. If the retrieved context does not contain enough information,
   clearly say that the information is not available.
10. For multi-part questions, answer every part.
11. Mention the relevant page number when useful.
12. Do not mention FAISS, embeddings, chunks, or retrieval
    unless specifically asked.

FORMAT RULES:
- Return clean plain text or simple Markdown.
- Never use LaTeX.
- Use % for percentages.
- Use normal Markdown headings and bullet points when helpful.
{history_block}
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
    return _call_gemini_with_fallback(client, prompt, answer_models)


# ------------------------------------------------------------------
# FULL PIPELINE ORCHESTRATOR
# ------------------------------------------------------------------

def process_pdf(
    client,
    pdf_path: Path,
    work_dir: Path,
    converter: DocumentConverter,
    embedding_model: str = EMBEDDING_MODEL_NAME,
    embedding_dim: int = EMBEDDING_OUTPUT_DIMENSIONALITY,
    vision_models=None,
    progress_cb: ProgressCB = None,
) -> DocumentIndex:
    image_dir = work_dir / "images"
    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)

    docling_data, picture_manifest, visual_manifest = ingest_pdf(pdf_path, image_dir, converter, progress_cb)

    gemini_visuals = describe_visuals(client, visual_manifest, vision_models, progress_cb)

    chunking_elements, element_stats = build_canonical_elements(docling_data, gemini_visuals, progress_cb)

    chunks = chunk_elements(chunking_elements, progress_cb)
    if not chunks:
        raise ValueError("No text could be extracted from this PDF.")

    embeddings, metadata, index = embed_chunks(client, chunks, embedding_model, embedding_dim, progress_cb)

    sizes = [len(c["content"]) for c in chunks]
    stats = {
        "pages": total_pages,
        "pictures_found": len(picture_manifest),
        "visuals_described": sum(1 for g in gemini_visuals if g.get("status") == "success"),
        "chunks": len(chunks),
        "chunks_with_tables": sum(c["has_table"] for c in chunks),
        "chunks_with_visuals": sum(c["has_visual"] for c in chunks),
        "avg_chunk_chars": round(sum(sizes) / len(sizes)) if sizes else 0,
        "text_elements": element_stats.get("text", 0),
        "table_elements": element_stats.get("table", 0),
        "visual_elements": element_stats.get("visual", 0),
    }

    return DocumentIndex(
        document_name=pdf_path.name,
        total_pages=total_pages,
        chunks=chunks,
        metadata=metadata,
        embeddings=embeddings,
        index=index,
        stats=stats,
        image_dir=str(image_dir),
    )
