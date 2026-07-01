import os
import json
import sqlite3
import shutil
import csv
import re
import hashlib

import streamlit as st

JSONL_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data_ingestion", "cleaned_data.jsonl")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
BM25_STATE_PATH = os.path.join(os.path.dirname(__file__), "bm25_state.pkl")


def _invalidate_cached_kb():
    """The chat-facing page caches its HybridSearchKnowledgeBase instance
    with @st.cache_resource so it doesn't reload the embedding model on
    every query. That cached instance holds its own PersistentClient
    pointed at chroma_db -- once we delete/recreate the collection here
    (a *different* PersistentClient, in the same process), the chat page's
    cached client goes stale and its next read/write throws SQLite error
    1032 (readonly db moved). Clearing the resource cache forces the chat
    page to construct a fresh instance on its next run.

    NOTE: this clears ALL @st.cache_resource caches app-wide, not just the
    knowledge base, since we don't have a reference to that specific
    cached function from here. If that's too broad, import the chat page's
    cached getter function directly and call its own .clear() instead.
    """
    try:
        st.cache_resource.clear()
    except Exception:
        # Calling this outside of an active Streamlit script run (e.g. a
        # one-off CLI re-index) shouldn't blow up the whole operation.
        pass


def _chunk_text(text, chunk_size=500, overlap=50):
    """Shared character-based chunker used by all import functions."""
    text = re.sub(r"\s+", " ", text).strip()
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if end >= len(text):
            break
    return chunks


def _load_existing_ids():
    """Return the set of chunk_ids already present in cleaned_data.jsonl."""
    existing_ids = set()
    if os.path.exists(JSONL_FILE):
        with open(JSONL_FILE, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "chunk_id" in data:
                        existing_ids.add(data["chunk_id"])
                except json.JSONDecodeError:
                    pass
    return existing_ids


def export_to_sqlite(output_db_path):
    """
    Reads cleaned_data.jsonl and exports it to an SQLite database so admins can easily edit it.
    """
    if not os.path.exists(JSONL_FILE):
        raise FileNotFoundError(f"Source data {JSONL_FILE} not found.")

    if os.path.exists(output_db_path):
        os.remove(output_db_path)

    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE knowledge (
            chunk_id TEXT PRIMARY KEY,
            url TEXT,
            text TEXT
        )
    ''')

    with open(JSONL_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            cursor.execute(
                "INSERT INTO knowledge (chunk_id, url, text) VALUES (?, ?, ?)",
                (data.get("chunk_id"), data.get("url"), data.get("text")),
            )

    conn.commit()
    conn.close()
    return output_db_path


def import_from_sqlite(input_db_path, rebuild_index=True):
    """
    Reads the uploaded SQLite database, appends new records to cleaned_data.jsonl,
    and optionally triggers a rebuild.
    """
    if not os.path.exists(input_db_path):
        raise FileNotFoundError(f"Uploaded DB {input_db_path} not found.")

    existing_ids = _load_existing_ids()

    conn = sqlite3.connect(input_db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT url, chunk_id, text FROM knowledge")
    rows = cursor.fetchall()
    conn.close()

    new_records_added = 0
    with open(JSONL_FILE, "a", encoding="utf-8") as f:
        for row in rows:
            chunk_id = row[1]
            if chunk_id not in existing_ids:
                f.write(json.dumps({"url": row[0], "chunk_id": chunk_id, "text": row[2]}) + "\n")
                existing_ids.add(chunk_id)
                new_records_added += 1

    if new_records_added > 0 and rebuild_index:
        rebuild_knowledge_base()

    return new_records_added


def rebuild_knowledge_base():
    """Wipes the existing vector indexes and rebuilds them from cleaned_data.jsonl."""
    from knowledge_base.vector_db_onnx_bm25 import HybridSearchKnowledgeBase
    kb = HybridSearchKnowledgeBase()

    try:
        kb.chroma_client.delete_collection("dubaigolf_knowledge")
    except ValueError:
        pass

    kb.collection = kb.chroma_client.get_or_create_collection(
        name="dubaigolf_knowledge",
        embedding_function=kb.embedding_func,
    )

    if os.path.exists(BM25_STATE_PATH):
        try:
            os.remove(BM25_STATE_PATH)
        except OSError:
            pass

    kb.build_index()
    _invalidate_cached_kb()


def _clear_vector_index():
    from knowledge_base.vector_db_onnx_bm25 import HybridSearchKnowledgeBase
    kb = HybridSearchKnowledgeBase()

    try:
        kb.chroma_client.delete_collection("dubaigolf_knowledge")
    except ValueError:
        pass

    if os.path.exists(BM25_STATE_PATH):
        try:
            os.remove(BM25_STATE_PATH)
        except OSError:
            pass


def _rebuild_vector_index():
    from knowledge_base.vector_db_onnx_bm25 import HybridSearchKnowledgeBase
    kb = HybridSearchKnowledgeBase()
    kb.build_index()
    _invalidate_cached_kb()


def import_from_txt(input_txt_path, rebuild_index=True):
    """
    Parses a scraped-text file in the format produced by the Dubai Golf
    crawler:

        # https://dubaigolf.com/some/page/
        Page body text all on one line (or a few lines) ...

        # https://dubaigolf.com/another/page/
        ...

    Each `# <url>` line starts a new page section. The body text that
    follows (up to the next `#` header or EOF) is cleaned, chunked, and
    appended to cleaned_data.jsonl.

    Duplicate URL+content combinations are skipped via a content hash so
    re-running after adding more PDFs/pages doesn't bloat the jsonl.
    The same chunk_id deduplication used by the other importers is also
    applied, so it's safe to call multiple times.
    """
    if not os.path.exists(input_txt_path):
        raise FileNotFoundError(f"Uploaded TXT {input_txt_path} not found.")

    # --- Parse into (url, body_text) sections ----------------------------
    sections = []          # list of (url, raw_text)
    current_url = None
    current_lines = []

    with open(input_txt_path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.rstrip("\n")
            if stripped.startswith("# http"):
                # Save previous section if it had content
                if current_url and current_lines:
                    sections.append((current_url, " ".join(current_lines)))
                current_url = stripped[2:].strip()   # drop "# "
                current_lines = []
            else:
                body = stripped.strip()
                if body:
                    current_lines.append(body)

    # Flush the last section
    if current_url and current_lines:
        sections.append((current_url, " ".join(current_lines)))

    if not sections:
        raise ValueError("No URL-delimited sections found in the TXT file.")

    # --- Build chunks, deduplicating by content hash ---------------------
    # Two consecutive scrape passes of the same URL produce identical text.
    # We use a (url, content_hash) key so repeated blocks are skipped
    # before they even become chunk_ids, without touching the jsonl.
    seen_url_hashes = set()
    rows = []

    for url, body in sections:
        slug = re.sub(r"[^a-z0-9]+", "_", url.lower()).strip("_")[:60]
        content_hash = hashlib.md5(body.encode()).hexdigest()[:8]
        dedup_key = (url, content_hash)

        if dedup_key in seen_url_hashes:
            continue
        seen_url_hashes.add(dedup_key)

        for chunk_idx, chunk in enumerate(_chunk_text(body)):
            if not chunk.strip():
                continue
            rows.append({
                # Include content_hash in the chunk_id so re-uploading a
                # revised version of the same URL produces new IDs and
                # gets indexed, rather than being silently skipped.
                "chunk_id": f"txt_{slug}_{content_hash}_{chunk_idx}",
                "url": url,
                "text": chunk,
            })

    if not rows:
        raise ValueError("No usable text chunks could be extracted from the TXT file.")

    # --- Append only genuinely new chunks to the jsonl -------------------
    existing_ids = _load_existing_ids()
    new_records_added = 0

    os.makedirs(os.path.dirname(JSONL_FILE), exist_ok=True)
    with open(JSONL_FILE, "a", encoding="utf-8") as f:
        for row in rows:
            if row["chunk_id"] not in existing_ids:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing_ids.add(row["chunk_id"])
                new_records_added += 1

    if new_records_added > 0 and rebuild_index:
        _clear_vector_index()
        _rebuild_vector_index()

    return new_records_added


def import_from_pdf(input_pdf_path, rebuild_index=True):
    """
    Extracts text from the uploaded PDF (OCR fallback for scanned pages),
    chunks it, and APPENDS to cleaned_data.jsonl.
    """
    if not os.path.exists(input_pdf_path):
        raise FileNotFoundError(f"Uploaded PDF {input_pdf_path} not found.")

    import pdfplumber
    import pytesseract
    from pdf2image import convert_from_path

    min_chars_before_ocr_fallback = 20
    ocr_dpi = 300
    ocr_lang = "eng"

    slug = re.sub(
        r"[^a-z0-9]+", "_",
        os.path.splitext(os.path.basename(input_pdf_path))[0].lower()
    ).strip("_")

    rows = []
    ocr_page_images = None

    with pdfplumber.open(input_pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            if len(text.strip()) < min_chars_before_ocr_fallback:
                if ocr_page_images is None:
                    ocr_page_images = convert_from_path(input_pdf_path, dpi=ocr_dpi)
                if page_index < len(ocr_page_images):
                    text = pytesseract.image_to_string(ocr_page_images[page_index], lang=ocr_lang)

            if not text.strip():
                continue

            for chunk_idx, chunk in enumerate(_chunk_text(text)):
                if not chunk.strip():
                    continue
                rows.append({
                    "chunk_id": f"pdf_{slug}_p{page_index + 1}_{chunk_idx}",
                    "url": f"file://{os.path.abspath(input_pdf_path)}#page={page_index + 1}",
                    "text": chunk,
                })

    if not rows:
        raise ValueError("No text could be extracted from the uploaded PDF.")

    existing_ids = _load_existing_ids()
    new_records_added = 0
    with open(JSONL_FILE, "a", encoding="utf-8") as f:
        for row in rows:
            if row["chunk_id"] not in existing_ids:
                f.write(json.dumps(row) + "\n")
                existing_ids.add(row["chunk_id"])
                new_records_added += 1

    if new_records_added > 0 and rebuild_index:
        _clear_vector_index()
        _rebuild_vector_index()

    return new_records_added


def import_from_csv(input_csv_path, rebuild_index=True):
    """
    Reads an uploaded CSV with columns chunk_id, url, text,
    and APPENDS new rows to cleaned_data.jsonl.
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Uploaded CSV {input_csv_path} not found.")

    rows = []
    with open(input_csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = {(name or "").strip().lower(): name for name in (reader.fieldnames or [])}
        required = {"chunk_id", "url", "text"}
        missing = required - fieldnames.keys()
        if missing:
            raise ValueError(f"CSV is missing required column(s): {', '.join(sorted(missing))}")

        for row in reader:
            rows.append({
                "chunk_id": row[fieldnames["chunk_id"]],
                "url": row[fieldnames["url"]],
                "text": row[fieldnames["text"]],
            })

    if not rows:
        raise ValueError("CSV contained no rows.")

    existing_ids = _load_existing_ids()
    new_records_added = 0
    with open(JSONL_FILE, "a", encoding="utf-8") as f:
        for row in rows:
            if row["chunk_id"] not in existing_ids:
                f.write(json.dumps(row) + "\n")
                existing_ids.add(row["chunk_id"])
                new_records_added += 1

    if new_records_added > 0 and rebuild_index:
        _clear_vector_index()
        _rebuild_vector_index()

    return new_records_added
