# Dubai Golf Chatbot ⛳️

An AI-powered chatbot that answers questions about **Dubai Golf** using content scraped directly from the official Dubai Golf website. It combines a hybrid retrieval system (vector + keyword search) with an LLM to provide accurate, grounded answers through a simple chat interface.

## How It Works

The project is a retrieval-augmented generation (RAG) pipeline made up of three stages:

1. **Data Ingestion** — Scrapes structured content (titles, headings, body text, club sections) from the Dubai Golf website, then cleans it into a usable format.
2. **Knowledge Base** — Builds a **hybrid search index** combining:
   - **ChromaDB** for semantic (vector) search, using lightweight ONNX-based embeddings (no PyTorch/FAISS dependency, for speed and Mac/Intel compatibility)
   - **BM25** for keyword/lexical search, blended with the vector results
   - A separate pipeline for ingesting **PDF documents** via OCR (`pdfplumber` + `pytesseract`)
3. **Frontend** — A password-protected Streamlit chat app that:
   - Retrieves the most relevant context chunks for a user's question from the hybrid search index
   - Sends that context + the question to an LLM via **OpenRouter** (free-tier `gpt-oss-20b` / `gpt-oss-120b` models)
   - Displays the response and stores chat history per session (SQLite)
   - Collects 5-star feedback on each response

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Vector Search | ChromaDB |
| Keyword Search | rank-bm25 |
| Embeddings | ONNX Runtime (CPU-only) |
| LLM Access | OpenAI SDK → OpenRouter API |
| Chat Storage | SQLite |
| PDF/OCR Ingestion | pdfplumber, pytesseract |

## Project Structure

```
Dubai-Golf-Chatbot/
├── data_ingestion/
│   ├── scraper.py          # Scrapes dubaigolf.com
│   └── cleaner.py          # Cleans scraped data
├── knowledge_base/
│   ├── vector_db_onnx_bm25.py   # Builds hybrid ChromaDB + BM25 index
│   ├── pdf_ocr_ingest.py        # Ingests PDF/OCR content into the knowledge base
│   └── chroma_db/               # Generated locally — not tracked in git
├── frontend/
│   ├── app.py               # Streamlit chat application
│   └── database.py          # Chat history & feedback storage (SQLite)
├── requirements_app.txt
└── .env                      # Local-only secrets, not tracked
```

## Prerequisites

- **Python 3.11** (recommended — newer versions can break some pinned dependencies)
- **Git**
- An **OpenRouter API key** (free-tier models are available) — [openrouter.ai](https://openrouter.ai)
- **Tesseract OCR** — only required if you plan to run the PDF/OCR ingestion step

---

## Installation

### macOS

1. **Clone the repository**
   ```bash
   git clone https://github.com/Koushik25022005/Dubai-Golf-Chatbot.git
   cd Dubai-Golf-Chatbot
   ```

2. **Create and activate a virtual environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements_app.txt
   ```

4. **(Optional) Install Tesseract OCR** — needed only for PDF ingestion
   ```bash
   brew install tesseract
   ```

5. **Set up environment variables** — create a `.env` file in the project root:
   ```
   TS_CHAT_OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

### Windows

1. **Clone the repository**
   ```powershell
   git clone https://github.com/Koushik25022005/Dubai-Golf-Chatbot.git
   cd Dubai-Golf-Chatbot
   ```

2. **Create and activate a virtual environment**
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install dependencies**
   ```powershell
   pip install -r requirements_app.txt
   ```

4. **(Optional) Install Tesseract OCR** — needed only for PDF ingestion
   - Download the installer from [UB-Mannheim's Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki)
   - Add the install directory (typically `C:\Program Files\Tesseract-OCR`) to your system `PATH`

5. **Set up environment variables** — create a `.env` file in the project root:
   ```
   TS_CHAT_OPENROUTER_API_KEY=your_openrouter_api_key_here
   ```

---

## Execution

These commands are identical on macOS and Windows once your virtual environment is activated.

### 1. Scrape and clean source data

```bash
python data_ingestion/scraper.py
python data_ingestion/cleaner.py
```

### 2. Build the knowledge base

```bash
python knowledge_base/vector_db_onnx_bm25.py
python knowledge_base/pdf_ocr_ingest.py
```

> This builds the local `knowledge_base/chroma_db/` folder containing the hybrid ChromaDB + BM25 index. Only re-run this when source content changes — the chatbot reads from this index at runtime rather than rebuilding it on every launch.

### 3. Run the chatbot

```bash
streamlit run frontend/app.py
```

The app opens at `http://localhost:8501`. You'll be prompted for a password before accessing the chat (set in `app.py` — change this before any real-world deployment).

---

## Notes on Deployment

If deploying to **Streamlit Community Cloud**:

- Pin the Python version to **3.11** via "Advanced settings" at deploy time — newer defaults can break pinned dependencies.
- Add `TS_CHAT_OPENROUTER_API_KEY` under **Settings → Secrets** in TOML format rather than relying on `.env`.
- The `chroma_db/` knowledge base exceeds GitHub's file size limits and is **not committed to git** — it's distributed as a GitHub Release asset and downloaded/extracted automatically the first time the app starts.

---

## License

This project is for educational and demonstration purposes, built on publicly available content from the Dubai Golf website.
