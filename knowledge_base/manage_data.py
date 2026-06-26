import os
import json
import sqlite3
import shutil
import csv
import re

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


def export_to_sqlite(output_db_path):
    """
    Reads cleaned_data.jsonl and exports it to an SQLite database so admins can easily edit it.
    """
    if not os.path.exists(JSONL_FILE):
        raise FileNotFoundError(f"Source data {JSONL_FILE} not found.")

    # Create a new SQLite database
    if os.path.exists(output_db_path):
        os.remove(output_db_path)
        
    conn = sqlite3.connect(output_db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE knowledge (
            chunk_id TEXT PRIMARY KEY,
            url TEXT,
            text TEXT
        )
    ''')
    
    # Insert data
    with open(JSONL_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            cursor.execute(
                'INSERT INTO knowledge (chunk_id, url, text) VALUES (?, ?, ?)',
                (data.get("chunk_id"), data.get("url"), data.get("text"))
            )
            
    conn.commit()
    conn.close()
    return output_db_path

def import_from_sqlite(input_db_path, rebuild_index=True):
    """
    Reads the uploaded SQLite database, and appends new records to cleaned_data.jsonl,
    clears the vector database, and optionally triggers a rebuild.
    """
    if not os.path.exists(input_db_path):
        raise FileNotFoundError(f"Uploaded DB {input_db_path} not found.")
        
    # Load existing chunk_ids to avoid duplicates when appending
    existing_ids = set()
    if os.path.exists(JSONL_FILE):
        with open(JSONL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "chunk_id" in data:
                        existing_ids.add(data["chunk_id"])
                except json.JSONDecodeError:
                    pass
                    
    conn = sqlite3.connect(input_db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT url, chunk_id, text FROM knowledge')
    rows = cursor.fetchall()
    
    # Append to the jsonl file
    new_records_added = 0
    with open(JSONL_FILE, 'a', encoding='utf-8') as f:
        for row in rows:
            chunk_id = row[1]
            if chunk_id not in existing_ids:
                data = {
                    "url": row[0],
                    "chunk_id": chunk_id,
                    "text": row[2]
                }
                f.write(json.dumps(data) + "\n")
                existing_ids.add(chunk_id)
                new_records_added += 1
            
    conn.close()
    
    # Only rebuild if there were actually new records appended and rebuild_index is True
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
        pass # Collection might not exist
        
    # Re-create the collection to ensure it's empty
    kb.collection = kb.chroma_client.get_or_create_collection(
        name="dubaigolf_knowledge",
        embedding_function=kb.embedding_func,
    )
    
    if os.path.exists(BM25_STATE_PATH):
        try:
            os.remove(BM25_STATE_PATH)
        except OSError:
            pass
            
    # Rebuild index
    kb.build_index()

    # Force the chat page's cached HybridSearchKnowledgeBase to reload --
    # otherwise it keeps using a PersistentClient pointed at the
    # collection we just deleted/recreated above, and the next query
    # through it throws SQLite error 1032 (readonly db moved).
    _invalidate_cached_kb()


def _clear_vector_index():
    """Shared by import_from_pdf/import_from_csv: wipe the existing vector
    index and BM25 state so the next build_index() starts clean. (Same
    two checks import_from_sqlite does inline above.)"""
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

    # Same reasoning as in rebuild_knowledge_base() -- invalidate the
    # chat page's cached instance so it doesn't hold a stale client.
    _invalidate_cached_kb()


def import_from_pdf(input_pdf_path, rebuild_index=True):
    """
    Extracts text from the uploaded PDF, chunks it, and APPENDS
    it to cleaned_data.jsonl, and optionally triggers a rebuild.
    """
    if not os.path.exists(input_pdf_path):
        raise FileNotFoundError(f"Uploaded PDF {input_pdf_path} not found.")

    # Imported here rather than at module level so manage_data.py doesn't
    # require pdfplumber/pytesseract/pdf2image (and the tesseract/poppler
    # system binaries) unless a PDF is actually uploaded.
    import pdfplumber
    import pytesseract
    from pdf2image import convert_from_path

    min_chars_before_ocr_fallback = 20
    ocr_dpi = 300
    ocr_lang = "eng"
    chunk_size = 500
    chunk_overlap = 50

    def chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap):
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

    slug = re.sub(
        r"[^a-z0-9]+", "_",
        os.path.splitext(os.path.basename(input_pdf_path))[0].lower()
    ).strip("_")

    rows = []
    with pdfplumber.open(input_pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text = page.extract_text() or ""

            if len(text.strip()) < min_chars_before_ocr_fallback:
                # Likely a scanned page with no usable text layer --
                # rasterize just this page and OCR it.
                images = convert_from_path(
                    input_pdf_path,
                    dpi=ocr_dpi,
                    first_page=page_index + 1,
                    last_page=page_index + 1,
                )
                text = pytesseract.image_to_string(images[0], lang=ocr_lang) if images else ""

            if not text.strip():
                continue

            for chunk_idx, chunk in enumerate(chunk_text(text)):
                if not chunk.strip():
                    continue
                rows.append({
                    "chunk_id": f"pdf_{slug}_p{page_index + 1}_{chunk_idx}",
                    "url": f"file://{os.path.abspath(input_pdf_path)}#page={page_index + 1}",
                    "text": chunk,
                })

    if not rows:
        raise ValueError("No text could be extracted from the uploaded PDF.")

    # Load existing chunk_ids to avoid duplicates when appending
    existing_ids = set()
    if os.path.exists(JSONL_FILE):
        with open(JSONL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "chunk_id" in data:
                        existing_ids.add(data["chunk_id"])
                except json.JSONDecodeError:
                    pass

    new_records_added = 0
    with open(JSONL_FILE, 'a', encoding='utf-8') as f:
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
    Reads an uploaded CSV, APPENDS to cleaned_data.jsonl,
    and optionally triggers a rebuild.
    """
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Uploaded CSV {input_csv_path} not found.")

    rows = []
    with open(input_csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)

        # Accept header names case-insensitively (e.g. "Chunk_ID" or "URL").
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

    # Load existing chunk_ids to avoid duplicates when appending
    existing_ids = set()
    if os.path.exists(JSONL_FILE):
        with open(JSONL_FILE, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    if "chunk_id" in data:
                        existing_ids.add(data["chunk_id"])
                except json.JSONDecodeError:
                    pass

    new_records_added = 0
    with open(JSONL_FILE, 'a', encoding='utf-8') as f:
        for row in rows:
            if row["chunk_id"] not in existing_ids:
                f.write(json.dumps(row) + "\n")
                existing_ids.add(row["chunk_id"])
                new_records_added += 1

    if new_records_added > 0 and rebuild_index:
        _clear_vector_index()
        _rebuild_vector_index()

    return new_records_added
