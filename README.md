# 📄 RAG Document Assistant

A Retrieval-Augmented Generation (RAG) web app that answers questions from your
PDF and TXT documents. Built with **Streamlit**, **LangChain**, **FAISS**, and
**Google Gemini**.

🔗 **Live app:** https://rag-using-faiss.streamlit.app/

---

## ✨ Features

- **Ask questions** about your documents in natural language — grounded only in
  the uploaded content (no hallucinated answers).
- **Source-cited answers** — every response shows the source file and page, with
  expandable snippets of the retrieved chunks.
- **Admin-gated uploads** — only an authenticated admin can add new PDF/TXT
  documents to the knowledge base. Regular users can ask questions freely.
- **Persistent vector index** — documents are embedded once and stored in a
  local FAISS index that is reused on subsequent runs.

---

## 🔐 Access

| Action | Login required? | Credentials |
|---|---|---|
| Ask questions | ❌ No | — anyone can use it |
| Upload new documents | ✅ Yes (admin only) | **Username:** `rag_admin` · **Password:** `rag_password_12#` |

> To upload, open the **Admin Panel** in the left sidebar, log in with the admin
> credentials above, and the file uploader will appear.

---

## 🏗️ Architecture

```
                          ┌────────────────────────────┐
                          │        User (browser)       │
                          └─────────────┬──────────────┘
                                        │  questions / admin upload
                                        ▼
                          ┌────────────────────────────┐
                          │     Streamlit UI            │
                          │     (rag_ui_app.py)         │
                          │  • Q&A box (public)         │
                          │  • Admin Panel (login-gated)│
                          └───────┬───────────┬─────────┘
                                  │           │
              ask question        │           │  admin uploads PDF/TXT
                                  ▼           ▼
                  ┌──────────────────┐   ┌─────────────────────────┐
                  │  Retriever       │   │  Document loader         │
                  │  (FAISS top-k)   │   │  PyPDFLoader/TextLoader  │
                  └────────┬─────────┘   └────────────┬────────────┘
                           │                          │ split into chunks
                           │                          ▼
                           │             ┌─────────────────────────┐
                           │             │ CharacterTextSplitter    │
                           │             └────────────┬────────────┘
                           │                          │ embed
                           ▼                          ▼
                  ┌─────────────────────────────────────────────────┐
                  │   Google Generative AI Embeddings                │
                  │   (models/gemini-embedding-001)                  │
                  └────────────────────────┬────────────────────────┘
                                           ▼
                              ┌──────────────────────────┐
                              │   FAISS vector store      │
                              │   (faiss_index/)          │
                              └────────────┬─────────────┘
                                           │ retrieved context
                                           ▼
                              ┌──────────────────────────┐
                              │   Gemini LLM              │
                              │   (gemini-2.5-flash-lite) │
                              │   prompt → answer          │
                              └────────────┬─────────────┘
                                           ▼
                              answer + cited sources → user
```

### Pipeline flow

1. **Ingestion** — PDF/TXT files in `information_storage/` are loaded
   (`PyPDFLoader` / `TextLoader`) and tagged with their source filename.
2. **Chunking** — documents are split into overlapping chunks with
   `CharacterTextSplitter` (1000 chars, 100 overlap).
3. **Embedding** — chunks are converted to vectors via Google's
   `gemini-embedding-001`.
4. **Indexing** — vectors are stored in a **FAISS** index (`faiss_index/`),
   built once and reloaded on later runs.
5. **Retrieval** — at query time the retriever fetches the top-4 most relevant
   chunks.
6. **Generation** — the chunks + question are sent to **Gemini
   2.5 Flash Lite**, which answers strictly from the provided context and cites
   sources.
7. **Admin upload** — an authenticated admin can upload a new file; it is saved,
   embedded, added to the live FAISS index, and the cache is refreshed.

---

## 🧰 Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | Google `gemini-embedding-001` |
| LLM | Google `gemini-2.5-flash-lite` |
| Vector store | FAISS (CPU) |
| Document loaders | PyPDFLoader, TextLoader |
| Hosting | Streamlit Community Cloud |

---

## 🚀 Run locally

```bash
# 1. Clone
git clone https://github.com/debnath-sumit/rag-using-faiss.git
cd rag-using-faiss

# 2. Install dependencies (using uv)
uv sync
# ...or with pip:
pip install -r requirements.txt

# 3. Set your Google API key
echo 'GOOGLE_API_KEY=your-key-here' > .env

# 4. Run
streamlit run rag_ui_app.py
```

Get a free Google API key at https://aistudio.google.com/apikey.

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push the repo to GitHub.
2. At https://share.streamlit.io, create a new app pointing to
   `rag_ui_app.py` on the `main` branch.
3. In **Manage app → Settings → Secrets**, add:
   ```toml
   GOOGLE_API_KEY = "your-key"
   ```
4. Deploy.

> ⚠️ **Note:** Streamlit Cloud's filesystem is ephemeral, so documents uploaded
> through the Admin Panel (and changes to the FAISS index) may not persist across
> app restarts. For durable uploads, back the storage with an external service
> (e.g. S3/GCS).

---

## 📁 Project structure

```
rag-using-faiss/
├── rag_ui_app.py          # Main Streamlit app (deployed)
├── rag_txt_app.py         # CLI version (terminal Q&A loop)
├── information_storage/    # Source PDF/TXT documents
├── faiss_index/            # Generated FAISS vector index (git-ignored)
├── requirements.txt        # pip dependencies
├── pyproject.toml          # uv/project dependencies
└── README.md
```
