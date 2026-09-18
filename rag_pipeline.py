"""
rag_pipeline.py

The working RAG architecture is preserved from the supplied notebook:

PDF
 -> Docling extraction (text + tables + pictures)
 -> Gemini visual descriptions
 -> canonical document
 -> structure-aware chunking
 -> Jina Embeddings v4 (retrieval.passage)
 -> normalized FAISS IndexFlatIP
 -> Jina query embedding (retrieval.query)
 -> top-k retrieval
 -> Gemini answer generation using retrieved context only

The only intentional adaptation is replacing Google Colab/Google Drive I/O
with local temporary/session storage suitable for Streamlit.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import requests
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
from google import genai
from google.genai import types
from pypdf import PdfReader


# ============================================================
# NOTEBOOK PARAMETERS — KEPT UNCHANGED
# ============================================================

JINA_EMBEDDING_MODEL_NAME = "jina-embeddings-v4"
JINA_EMBEDDING_OUTPUT_DIMENSIONALITY = 2048
JINA_EMBEDDING_ENDPOINT = "https://api.jina.ai/v1/embeddings"

JINA_EMBEDDING_RPM_LIMIT = 500
JINA_EMBEDDING_TPM_LIMIT = 2_000_000
JINA_EMBEDDING_SAFE_RPM = 400
JINA_EMBEDDING_BATCH_SIZE = 100
JINA_EMBEDDING_INTER_BATCH_DELAY = 1.0
JINA_EMBEDDING_MIN_REQUEST_INTERVAL = 0.20
JINA_EMBEDDING_MAX_RETRIES = 6
JINA_EMBEDDING_CHECKPOINT_EVERY = 100

TARGET_CHARS = 1800
MAX_CHARS = 3200
MIN_STANDALONE_CHARS = 150
TOP_K = 5

GEMINI_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-3.5-flash",
    "gemini-3.7-flash",
]

GEMINI_MAX_RETRIES = 5
GEMINI_DELAY_BETWEEN_REQUESTS = 3
GEMINI_INITIAL_RETRY_DELAY = 2
GEMINI_MAX_RETRY_DELAY = 30


class RAGPipeline:
    """One isolated RAG session for one uploaded PDF."""

    def __init__(
        self,
        jina_api_key: str,
        gemini_api_key: str,
        work_dir: str | Path | None = None,
        progress_callback=None,
    ):
        self.jina_api_key = jina_api_key
        self.gemini_api_key = gemini_api_key
        self.progress_callback = progress_callback

        self.root = Path(work_dir) if work_dir else Path(
            tempfile.mkdtemp(prefix="documind_rag_")
        )
        self.source_dir = self.root / "source"
        self.docling_dir = self.root / "docling"
        self.image_dir = self.root / "images"
        self.gemini_dir = self.root / "gemini"
        self.canonical_dir = self.root / "canonical"
        self.chunk_dir = self.root / "chunks"
        self.vector_dir = self.root / "vectors"

        for directory in (
            self.source_dir,
            self.docling_dir,
            self.image_dir,
            self.gemini_dir,
            self.canonical_dir,
            self.chunk_dir,
            self.vector_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        self.jina_headers = {
            "Authorization": f"Bearer {self.jina_api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        self.gemini_client = genai.Client(api_key=self.gemini_api_key)

        # Notebook state
        self.last_request_time = 0.0
        self.chunks: list[dict[str, Any]] = []
        self.metadata: list[dict[str, Any]] = []
        self.index = None
        self.total_pages = 0
        self.document_name = ""
        self.statistics: dict[str, Any] = {}
        self.last_results: list[dict[str, Any]] = []
        self.model_used = None

    def _progress(self, stage: str, message: str, value: float | None = None):
        if self.progress_callback:
            self.progress_callback(stage, message, value)

    # ========================================================
    # STAGE 2 — DOCLING INGESTION
    # ========================================================

    @staticmethod
    def get_attr_or_key(obj, name, default=None):
        if isinstance(obj, dict):
            return obj.get(name, default)
        try:
            return getattr(obj, name, default)
        except Exception:
            return default

    @staticmethod
    def object_to_dict(obj):
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

    @staticmethod
    def clean_text(text):
        if text is None:
            return ""
        text = str(text).replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    def get_item_text(self, item):
        text = self.get_attr_or_key(item, "text", None)
        if isinstance(text, str) and text.strip():
            return self.clean_text(text)

        orig = self.get_attr_or_key(item, "orig", None)
        if isinstance(orig, str) and orig.strip():
            return self.clean_text(orig)

        if isinstance(item, dict):
            for key in ("text", "orig", "content"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    return self.clean_text(value)
        return ""

    def get_page_number(self, item):
        prov = self.get_attr_or_key(item, "prov", [])
        if not prov:
            return None
        try:
            return self.get_attr_or_key(prov[0], "page_no", None)
        except Exception:
            return None

    def extract_table_cells(self, table):
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
                            "text": self.clean_text(value),
                        })
                if cells:
                    return cells
        except Exception:
            pass

        table_dict = self.object_to_dict(table)
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
                        cells.append({
                            "row": row_idx,
                            "col": col_idx,
                            "text": self.clean_text(value),
                        })
        return cells

    def table_to_text(self, table):
        try:
            dataframe = table.export_to_dataframe()
            if dataframe is not None:
                dataframe = dataframe.fillna("")
                lines = [
                    " | ".join(self.clean_text(x) for x in dataframe.columns)
                ]
                if not any(lines[0]):
                    lines = []
                for _, row in dataframe.iterrows():
                    values = [self.clean_text(v) for v in row.tolist()]
                    if any(values):
                        lines.append(" | ".join(v if v else "-" for v in values))
                result = "\n".join(lines).strip()
                if result:
                    return result
        except Exception:
            pass

        cells = self.extract_table_cells(table)
        if not cells:
            return self.get_item_text(table)

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

    def ingest_pdf(self, pdf_bytes: bytes, filename: str):
        self.document_name = filename
        pdf_path = self.source_dir / Path(filename).name
        pdf_path.write_bytes(pdf_bytes)
        self.pdf_path = pdf_path

        reader = PdfReader(str(pdf_path))
        self.total_pages = len(reader.pages)

        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_table_structure = True
        pipeline_options.generate_picture_images = True
        pipeline_options.do_picture_description = False
        pipeline_options.do_ocr = False

        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        self._progress("processing", "Reading PDF with Docling...", 0.10)
        result = converter.convert(str(pdf_path))
        doc = result.document

        docling_path = self.docling_dir / "docling_document.json"
        docling_data = result.document.export_to_dict()
        docling_path.write_text(
            json.dumps(docling_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        item_count = text_count = table_count = picture_count = 0
        for item, _level in doc.iterate_items():
            item_count += 1
            item_type = type(item).__name__.lower()
            if "text" in item_type:
                text_count += 1
            elif "table" in item_type:
                table_count += 1
            elif "picture" in item_type:
                picture_count += 1

        picture_manifest = []
        for idx, picture in enumerate(doc.pictures, start=1):
            picture_id = f"picture_{idx:03d}"
            image_path = self.image_dir / f"{picture_id}.png"
            try:
                image = picture.get_image(doc)
                if image is None:
                    continue
                image.save(image_path)
                page_no = None
                bbox = None
                try:
                    prov = self.get_attr_or_key(picture, "prov", [])
                    if prov:
                        first = prov[0]
                        page_no = self.get_attr_or_key(first, "page_no", None)
                        b = self.get_attr_or_key(first, "bbox", None)
                        if b is not None:
                            bbox = {
                                "l": float(self.get_attr_or_key(b, "l", 0)),
                                "t": float(self.get_attr_or_key(b, "t", 0)),
                                "r": float(self.get_attr_or_key(b, "r", 0)),
                                "b": float(self.get_attr_or_key(b, "b", 0)),
                            }
                except Exception:
                    pass

                picture_manifest.append({
                    "picture_id": picture_id,
                    "page": page_no,
                    "image_path": str(image_path),
                    "bbox": bbox,
                })
            except Exception:
                continue

        (self.image_dir / "picture_manifest.json").write_text(
            json.dumps(picture_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        self._progress(
            "processing",
            f"Docling complete — {self.total_pages} pages, {len(picture_manifest)} pictures.",
            0.30,
        )

        self._process_visuals(picture_manifest)
        self._build_canonical(docling_data)
        self._build_chunks()
        self._build_embeddings_and_faiss()

        self.statistics = {
            "pages": self.total_pages,
            "docling_items": item_count,
            "text_items": text_count,
            "tables": table_count,
            "pictures": picture_count,
            "chunks": len(self.chunks),
            "vectors": int(self.index.ntotal),
            "embedding_dimension": int(self.index.d),
        }
        self._progress("complete", "Document is ready for questions.", 1.0)
        return self.statistics

    # ========================================================
    # STAGE 3 — GEMINI VISUAL UNDERSTANDING
    # ========================================================

    def build_visual_prompt(self, page_number):
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

    @staticmethod
    def classify_gemini_error(error):
        text = str(error).upper()
        if any(x in text for x in ["401", "403", "UNAUTHENTICATED", "PERMISSION_DENIED", "INVALID API KEY"]):
            return "authentication"
        if any(x in text for x in ["429", "RESOURCE_EXHAUSTED", "RATE LIMIT", "QUOTA", "TOO MANY REQUESTS"]):
            return "quota"
        if any(x in text for x in ["503", "UNAVAILABLE", "SERVICE UNAVAILABLE", "SERVER BUSY", "INTERNAL", "500", "502", "504"]):
            return "temporary"
        return "permanent"

    def call_gemini_visual(self, image_path, page_number, model_name):
        prompt = self.build_visual_prompt(page_number)
        image_bytes = Path(image_path).read_bytes()

        last_status = "permanent"
        for attempt in range(GEMINI_MAX_RETRIES):
            try:
                response = self.gemini_client.models.generate_content(
                    model=model_name,
                    contents=[
                        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                        prompt,
                    ],
                )
                if not response.text:
                    raise ValueError("Gemini returned an empty response.")
                return response.text.strip(), "success"
            except Exception as exc:
                last_status = self.classify_gemini_error(exc)
                if last_status == "authentication":
                    return None, last_status
                if last_status == "permanent":
                    return None, last_status
                if attempt < GEMINI_MAX_RETRIES - 1:
                    delay = min(
                        GEMINI_INITIAL_RETRY_DELAY * (2 ** attempt) + random.uniform(0, 1),
                        GEMINI_MAX_RETRY_DELAY,
                    )
                    time.sleep(delay)
        return None, last_status

    def _process_visuals(self, visual_manifest):
        results_path = self.gemini_dir / "visual_descriptions.json"
        if results_path.exists():
            existing_results = json.loads(results_path.read_text(encoding="utf-8"))
        else:
            existing_results = []

        results_by_picture = {
            item["picture_id"]: item for item in existing_results
        }

        for n, item in enumerate(visual_manifest, start=1):
            picture_id = item["picture_id"]
            existing = results_by_picture.get(picture_id)
            if existing and existing.get("description"):
                continue

            self._progress(
                "processing",
                f"Analyzing visual {n}/{len(visual_manifest)}...",
                0.30 + 0.10 * (n / max(len(visual_manifest), 1)),
            )

            image_path = Path(item["image_path"])
            result = {
                "picture_id": picture_id,
                "page": item["page"],
                "image_path": str(image_path),
                "description": None,
                "model": None,
                "status": "failed",
            }

            if not image_path.exists():
                result["status"] = "image_missing"
            else:
                for model_name in GEMINI_MODELS:
                    description, status = self.call_gemini_visual(
                        image_path, item["page"], model_name
                    )
                    if description:
                        result.update({
                            "model": model_name,
                            "description": description,
                            "status": "success",
                        })
                        break
                    if status == "authentication":
                        result["status"] = "authentication_error"
                        break

            results_by_picture[picture_id] = result
            results = sorted(
                results_by_picture.values(),
                key=lambda x: (x.get("page", 0) or 0, x.get("picture_id", "")),
            )
            results_path.write_text(
                json.dumps(results, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            if n < len(visual_manifest):
                time.sleep(GEMINI_DELAY_BETWEEN_REQUESTS)

    # ========================================================
    # STAGE 4 — CANONICAL DOCUMENT
    # ========================================================

    def _build_canonical(self, docling_data):
        gemini_path = self.gemini_dir / "visual_descriptions.json"
        gemini_visuals = json.loads(gemini_path.read_text(encoding="utf-8")) if gemini_path.exists() else []

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

        def get_page_info(item):
            prov = item.get("prov", [])
            if prov and isinstance(prov[0], dict):
                return prov[0].get("page_no")
            return None

        canonical_elements = []
        for position, child in enumerate(docling_data.get("body", {}).get("children", [])):
            resolved = resolve_ref(child)
            if resolved is None:
                continue

            item_type, item = resolved
            page_no = get_page_info(item)

            if item_type == "text":
                text = self.get_item_text(item)
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
                table_text = self.table_to_text(item)
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
                gemini_result = gemini_by_picture.get(picture_id)
                if not gemini_result:
                    continue
                description = (gemini_result.get("description") or "").strip()
                if not description:
                    continue
                canonical_elements.append({
                    "element_id": f"element_{len(canonical_elements)+1:06d}",
                    "type": "visual",
                    "position": position,
                    "page": page_no,
                    "picture_id": picture_id,
                    "image_path": gemini_result.get("image_path"),
                    "visual_description": description,
                    "text": description,
                    "content": description,
                    "source": {
                        "docling_ref": picture_ref,
                        "gemini_model": gemini_result.get("model"),
                    },
                })

        canonical_document = {
            "schema_version": "1.0",
            "document_name": docling_data.get("name") or self.document_name,
            "source": {"type": "pdf", "origin": docling_data.get("origin")},
            "statistics": {
                "total_elements": len(canonical_elements),
                "text_elements": sum(x["type"] == "text" for x in canonical_elements),
                "table_elements": sum(x["type"] == "table" for x in canonical_elements),
                "visual_elements": sum(x["type"] == "visual" for x in canonical_elements),
            },
            "elements": canonical_elements,
        }

        path = self.canonical_dir / "canonical_document.json"
        path.write_text(
            json.dumps(canonical_document, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        remove_labels = {"page_header", "page_footer"}
        self.chunking_elements = [
            element for element in canonical_elements
            if not (
                element["type"] == "text"
                and element.get("label", "text") in remove_labels
            )
        ]

    # ========================================================
    # STAGE 5 — STRUCTURE-AWARE CHUNKING
    # ========================================================

    @staticmethod
    def normalize_spaces(text):
        text = text or ""
        text = text.replace("\xa0", " ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def split_sentences(cls, text):
        text = cls.normalize_spaces(text)
        if not text:
            return []
        paragraphs = re.split(r"\n\s*\n", text)
        sentences = []
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            parts = re.split(r"(?<=[.!?])\s+(?=[A-Z₹0-9(\[])", paragraph)
            sentences.extend(part.strip() for part in parts if part.strip())
        return sentences

    @classmethod
    def is_probable_page_number(cls, text):
        text = cls.normalize_spaces(text)
        return bool(
            re.fullmatch(r"\d{1,4}", text)
            or re.fullmatch(r"[ivxlcdmIVXLCDM]{1,8}", text)
        )

    @classmethod
    def is_low_value_fragment(cls, text):
        text = cls.normalize_spaces(text)
        return not text or cls.is_probable_page_number(text) or len(text) < MIN_STANDALONE_CHARS

    def _build_chunks(self):
        chunks = []
        current_parts = []
        current_element_ids = []
        current_element_types = []
        current_pages = []
        current_section = ""

        def current_text():
            return "\n\n".join(x for x in current_parts if x.strip()).strip()

        def flush_chunk():
            nonlocal current_parts, current_element_ids, current_element_types
            nonlocal current_pages

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
            content = self.normalize_spaces(content)
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

        for element in self.chunking_elements:
            element_type = element["type"]
            label = element.get("label", "")

            if element_type == "text" and label == "section_header":
                header = self.normalize_spaces(element.get("text", ""))
                if not header:
                    continue
                flush_chunk()
                current_section = header
                add_element(element, header)
                continue

            if element_type == "table":
                flush_chunk()
                table_text = self.normalize_spaces(
                    element.get("content") or element.get("text") or ""
                )
                if table_text:
                    add_element(element, table_text)
                    flush_chunk()
                continue

            if element_type == "visual":
                flush_chunk()
                description = self.normalize_spaces(
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
                    f"Page: {element.get('page')}\n\n"
                    f"{description}"
                )
                add_element(element, visual_text)
                flush_chunk()
                continue

            if element_type == "text":
                text = self.normalize_spaces(
                    element.get("text") or element.get("content") or ""
                )
                if not text or self.is_low_value_fragment(text):
                    continue
                for sentence in self.split_sentences(text):
                    add_text_sentence(element, self.normalize_spaces(sentence))

        flush_chunk()

        clean_chunks = []
        for chunk in chunks:
            content = self.normalize_spaces(chunk["content"])
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
                    previous["has_table"] |= chunk["has_table"]
                    previous["has_visual"] |= chunk["has_visual"]
                    continue

            chunk["content"] = content
            clean_chunks.append(chunk)

        for i, chunk in enumerate(clean_chunks):
            chunk["chunk_id"] = f"chunk_{i:06d}"

        self.chunks = clean_chunks
        (self.chunk_dir / "chunks_final_v2.json").write_text(
            json.dumps(self.chunks, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # ========================================================
    # STAGE 6 — JINA EMBEDDING API + FAISS
    # ========================================================

    def wait_for_request_slot(self):
        elapsed = time.monotonic() - self.last_request_time
        if elapsed < JINA_EMBEDDING_MIN_REQUEST_INTERVAL:
            time.sleep(JINA_EMBEDDING_MIN_REQUEST_INTERVAL - elapsed)
        self.last_request_time = time.monotonic()

    @staticmethod
    def embedding_error_is_retryable(exc):
        text = str(exc).upper()
        return any(
            x in text for x in [
                "429", "RATE LIMIT", "TOO MANY REQUESTS",
                "500", "502", "503", "504",
                "UNAVAILABLE", "INTERNAL", "TIMEOUT", "DEADLINE",
            ]
        )

    def prepare_document(self, chunk):
        title = chunk.get("section") or "none"
        content = str(chunk.get("content", "")).strip()
        if not content:
            raise ValueError(f"Empty chunk content: {chunk.get('chunk_id')}")
        return f"title: {title} | text: {content}"

    def embed_batch(self, batch_texts, batch_start_index):
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
                self.wait_for_request_slot()
                response = requests.post(
                    JINA_EMBEDDING_ENDPOINT,
                    headers=self.jina_headers,
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
                        f"Jina returned {len(data)} embeddings for "
                        f"{len(batch_texts)} inputs in batch starting at "
                        f"chunk {batch_start_index + 1}."
                    )

                data = sorted(data, key=lambda item: int(item["index"]))
                batch_embeddings = []
                for local_index, item in enumerate(data):
                    values = np.asarray(item["embedding"], dtype="float32")
                    if values.shape[0] != JINA_EMBEDDING_OUTPUT_DIMENSIONALITY:
                        raise ValueError(
                            f"Unexpected embedding dimension {values.shape[0]} "
                            f"for chunk {batch_start_index + local_index + 1}."
                        )
                    if not np.isfinite(values).all():
                        raise ValueError(
                            f"NaN/Inf detected for chunk "
                            f"{batch_start_index + local_index + 1}."
                        )
                    batch_embeddings.append(values)
                return batch_embeddings, body.get("usage", {})
            except Exception as exc:
                if (
                    not self.embedding_error_is_retryable(exc)
                    or attempt == JINA_EMBEDDING_MAX_RETRIES
                ):
                    raise
                time.sleep(min(2 ** (attempt - 1) + random.uniform(0, 1), 30))

        raise RuntimeError("Jina embedding failed.")

    def _build_embeddings_and_faiss(self):
        texts = [self.prepare_document(chunk) for chunk in self.chunks]
        embedding_map = {}
        total = len(texts)

        for batch_start in range(0, total, JINA_EMBEDDING_BATCH_SIZE):
            batch_indices = list(
                range(batch_start, min(batch_start + JINA_EMBEDDING_BATCH_SIZE, total))
            )
            batch_texts = [texts[idx] for idx in batch_indices]

            self._progress(
                "embedding",
                f"Embedding chunks {batch_indices[0] + 1}-{batch_indices[-1] + 1} of {total}...",
                batch_start / max(total, 1),
            )
            batch_embeddings, _usage = self.embed_batch(
                batch_texts, batch_indices[0]
            )
            for idx, vector in zip(batch_indices, batch_embeddings):
                embedding_map[idx] = vector

            if batch_start + JINA_EMBEDDING_BATCH_SIZE < total:
                time.sleep(JINA_EMBEDDING_INTER_BATCH_DELAY)

        embeddings = np.vstack(
            [embedding_map[i] for i in range(len(self.chunks))]
        ).astype("float32")

        assert embeddings.shape == (
            len(self.chunks),
            JINA_EMBEDDING_OUTPUT_DIMENSIONALITY,
        )
        assert np.isfinite(embeddings).all()

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.clip(norms, 1e-12, None)

        np.save(self.vector_dir / "embeddings.npy", embeddings)

        self.metadata = []
        for i, chunk in enumerate(self.chunks):
            self.metadata.append({
                "faiss_index": i,
                "chunk_id": chunk["chunk_id"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"],
                "section": chunk["section"],
                "element_ids": chunk["element_ids"],
                "element_types": chunk["element_types"],
                "has_table": chunk["has_table"],
                "has_visual": chunk["has_visual"],
            })

        (self.vector_dir / "embedding_metadata.json").write_text(
            json.dumps(self.metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        faiss.write_index(self.index, str(self.vector_dir / "faiss.index"))

    # ========================================================
    # STAGE 7 — QUERY + RETRIEVAL + ANSWER
    # ========================================================

    def embed_query(self, question):
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
                self.wait_for_request_slot()
                response = requests.post(
                    JINA_EMBEDDING_ENDPOINT,
                    headers=self.jina_headers,
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
                if (
                    not self.embedding_error_is_retryable(exc)
                    or attempt == JINA_EMBEDDING_MAX_RETRIES
                ):
                    raise
                time.sleep(min(2 ** (attempt - 1) + random.uniform(0, 1), 30))

        raise RuntimeError("Jina query embedding failed.")

    def retrieve_chunks(self, question, top_k=TOP_K):
        query_embedding = self.embed_query(question)
        scores, indices = self.index.search(query_embedding, top_k)
        results = []

        for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
            if idx < 0:
                continue
            results.append({
                "rank": rank,
                "score": float(score),
                "chunk": self.chunks[idx],
                "metadata": self.metadata[idx],
            })

        return results

    @staticmethod
    def build_context(results):
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

    @staticmethod
    def is_retryable_gemini_error(exc):
        error_text = str(exc).upper()
        retryable_codes = [
            "429", "500", "502", "503", "504",
            "UNAVAILABLE", "RESOURCE_EXHAUSTED", "INTERNAL", "DEADLINE",
        ]
        return any(code in error_text for code in retryable_codes)

    def call_gemini_with_fallback(
        self, prompt, models=GEMINI_MODELS, max_retries=4, base_delay=3
    ):
        last_error = None
        for model_name in models:
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.gemini_client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    if not response.text:
                        raise ValueError("Gemini returned an empty response.")
                    return response.text.strip(), model_name
                except Exception as exc:
                    last_error = exc
                    if not self.is_retryable_gemini_error(exc):
                        break
                    if attempt < max_retries:
                        delay = (
                            base_delay * (2 ** (attempt - 1))
                            + random.uniform(0, 2)
                        )
                        time.sleep(delay)
        raise RuntimeError("All Gemini models failed.") from last_error

    def generate_answer(self, question, results):
        context = self.build_context(results)
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
        return self.call_gemini_with_fallback(prompt)

    def ask(self, question, top_k=TOP_K):
        question = question.strip()
        if not question:
            raise ValueError("Question cannot be empty.")
        results = self.retrieve_chunks(question, top_k=top_k)
        answer, model_used = self.generate_answer(question, results)
        self.last_results = results
        self.model_used = model_used
        return {
            "answer": answer,
            "model_used": model_used,
            "results": results,
        }

    def get_document_info(self):
        return {
            "name": self.document_name,
            **self.statistics,
        }

    def clear_document(self):
        try:
            shutil.rmtree(self.root, ignore_errors=True)
        except Exception:
            pass
        self.chunks = []
        self.metadata = []
        self.index = None
        self.last_results = []
        self.statistics = {}
        self.document_name = ""
