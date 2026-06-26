import os
import re
import json
import glob

# pip install pdfplumber pytesseract pdf2image pillow
# Also requires system binaries (not pip-installable):
#   macOS:  brew install tesseract poppler
#   Ubuntu: sudo apt-get install tesseract-ocr poppler-utils
import pdfplumber
import pytesseract
from pdf2image import convert_from_path

# --- Configuration --------------------------------------------------------

# PDFs live in the same folder as this script -- the knowledge_base folder,
# right alongside vector_db_onnx_bm25.py and chroma_db/.
PDF_DIR = os.path.dirname(__file__)

# This is the exact same path vector_db_onnx_bm25.py's build_index() reads
# from. Writing here means the PDF-derived chunks get picked up by the
# existing indexing pipeline with zero changes to that script -- just
# re-run build_index() afterward and it'll embed/BM25-index everything
# (scraped web pages + PDFs) together.
DATA_FILE = os.path.join(os.path.dirname(PDF_DIR), "data_ingestion", "cleaned_data.jsonl")

# OCR is comparatively slow (rasterize page -> Tesseract), so it's only
# invoked per-page, and only when a page's embedded text layer is too thin
# to be real extracted text -- i.e. the page is a scanned image rather than
# a digitally-generated one. Pages with a normal text layer skip OCR
# entirely and just use pdfplumber's extraction, which is faster and more
# accurate than re-OCRing text that's already there.
MIN_CHARS_BEFORE_OCR_FALLBACK = 20
OCR_DPI = 300
OCR_LANG = "eng"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Same character-based chunker as vector_db_onnx_bm25.py, duplicated
    here so this script doesn't need to import that module (which pulls in
    onnxruntime/chromadb/rank_bm25 -- none of that is needed just to OCR
    and chunk a PDF)."""
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


def slugify(path):
    name = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def extract_page_text(pdf_path, page_index, plumber_page):
    """Try the embedded text layer first. Only fall back to OCR if the
    page looks like a scanned image (i.e. almost no text came out).
    Returns (text, used_ocr)."""
    text = plumber_page.extract_text() or ""

    if len(text.strip()) >= MIN_CHARS_BEFORE_OCR_FALLBACK:
        return text, False

    # Rasterize just this one page and OCR it -- cheaper than converting
    # the whole PDF to images up front when most pages won't need it.
    images = convert_from_path(
        pdf_path,
        dpi=OCR_DPI,
        first_page=page_index + 1,
        last_page=page_index + 1,
    )
    ocr_text = pytesseract.image_to_string(images[0], lang=OCR_LANG) if images else ""
    return ocr_text, True


def process_pdf(pdf_path):
    """Extract (with OCR fallback) and chunk a single PDF. Returns rows
    shaped exactly like cleaned_data.jsonl: text, url, chunk_id."""
    filename = os.path.basename(pdf_path)
    slug = slugify(pdf_path)
    rows = []

    print(f"Processing {filename}...")
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            text, used_ocr = extract_page_text(pdf_path, page_index, page)
            if used_ocr:
                print(f"  page {page_index + 1}: no text layer, ran OCR")

            if not text.strip():
                continue

            for chunk_idx, chunk in enumerate(chunk_text(text)):
                if not chunk.strip():
                    continue
                rows.append({
                    "text": chunk,
                    # Page-level reference so retrieved chunks can still be
                    # traced back to where they came from, same role the
                    # "url" field plays for scraped web pages.
                    "url": f"file://{os.path.abspath(pdf_path)}#page={page_index + 1}",
                    "chunk_id": f"pdf_{slug}_p{page_index + 1}_{chunk_idx}",
                })

    print(f"  -> {len(rows)} chunks")
    return rows


def load_existing_chunk_ids():
    """So re-running this script after adding new PDFs doesn't duplicate
    chunks already appended from a previous run."""
    existing = set()
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    existing.add(json.loads(line)["chunk_id"])
    return existing


def append_to_data_file(rows):
    os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
    with open(DATA_FILE, "a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    pdf_paths = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {PDF_DIR}.")
        return

    all_rows = []
    for pdf_path in pdf_paths:
        all_rows.extend(process_pdf(pdf_path))

    existing_ids = load_existing_chunk_ids()
    new_rows = [r for r in all_rows if r["chunk_id"] not in existing_ids]
    skipped = len(all_rows) - len(new_rows)

    if not new_rows:
        print("\nNothing new to add (all chunks already in cleaned_data.jsonl).")
        return

    append_to_data_file(new_rows)
    print(f"\nAppended {len(new_rows)} new chunks from {len(pdf_paths)} PDF(s) to {DATA_FILE}")
    if skipped:
        print(f"Skipped {skipped} chunks already present from a previous run.")
    print("Now run vector_db_onnx_bm25.py's build_index() to (re)index everything -- web pages + PDFs together.")


if __name__ == "__main__":
    main()
