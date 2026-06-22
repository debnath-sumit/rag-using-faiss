# 📄 RAG Document Assistant

A Retrieval-Augmented Generation (RAG) web app that answers questions from your
PDF, TXT, and DOCX documents. Built with **Streamlit**, **LangChain**, **FAISS**,
local **HuggingFace embeddings**, and **Groq** (Llama 3.3 70B).

Embeddings run **locally** (no API, no rate limits), and the corpus is embedded
**offline** so the deployed app only loads a prebuilt index — this is what makes
it scale to many documents at zero cost.

🔗 **Live app:** https://rag-using-faiss.streamlit.app/

---

## ✨ Features

- **Ask questions** about your documents in natural language — grounded only in
  the uploaded content (no hallucinated answers).
- **Source-cited answers** — every response shows the source file and page, with
  expandable snippets of the retrieved chunks.
- **Admin-gated uploads & deletes** — only an authenticated admin can add or
  remove PDF/TXT/DOCX documents in the knowledge base. Regular users can ask
  questions freely.
- **Offline ingestion, scales cheaply** — embed the whole corpus once with
  `python ingest.py` (local model, no API cost), commit the FAISS index, and the
  deployed app just loads it. Incremental: unchanged files are skipped, changed
  files re-embedded, deleted files removed — no full re-embed.
- **Committed vector index** — `faiss_index/` is checked into the repo so it
  survives Streamlit Cloud's ephemeral-disk redeploys.

---

## 🔐 Access

| Action | Login required? | Credentials |
|---|---|---|
| Ask questions | ❌ No | — anyone can use it |
| Upload / delete documents | ✅ Yes (admin only) | **Username:** `rag_admin` · **Password:** `rag_password_12#` |

> To upload, open the **Admin Panel** in the left sidebar, log in with the admin
> credentials above, and the file uploader will appear.

---

## 🏗️ Architecture

```
   OFFLINE (you, on your machine)              ONLINE (deployed app)
   ────────────────────────────────            ─────────────────────────────

   information_storage/*.pdf/.txt/.docx          ┌────────────────────────┐
                │                                 │     User (browser)     │
                ▼                                 └───────────┬────────────┘
   ┌──────────────────────────┐                              │ question
   │  ingest.py               │                              ▼
   │  • load + chunk          │                  ┌────────────────────────┐
   │    (Recursive splitter)  │                  │   Streamlit UI         │
   │  • embed LOCALLY         │                  │   (rag_ui_app.py)      │
   │    (bge-small-en-v1.5)   │                  │  • Q&A box (public)    │
   │  • manifest: skip/update │                  │  • Admin Panel (gated) │
   └────────────┬─────────────┘                  └───────────┬────────────┘
                │ save_local                       embed query │ (bge-small)
                ▼                                               ▼
   ┌──────────────────────────┐   git commit     ┌────────────────────────┐
   │   FAISS index            │ ───────────────▶ │   Retriever (top-k)    │
   │   faiss_index/           │                  └───────────┬────────────┘
   │   + manifest.json        │                              │ context
   └──────────────────────────┘                              ▼
                                                  ┌────────────────────────┐
                                                  │   Groq LLM             │
                                                  │   llama-3.3-70b        │
                                                  │   prompt → answer      │
                                                  └───────────┬────────────┘
                                                              ▼
                                                  answer + cited sources → user
```

### Pipeline flow

**Offline (run `python ingest.py` locally, then commit `faiss_index/`):**

1. **Ingestion** — PDF/TXT/DOCX files in `information_storage/` are loaded
   (`PyPDFLoader` / `TextLoader` / `Docx2txtLoader`) and tagged with their
   source filename.
2. **Chunking** — documents are split with `RecursiveCharacterTextSplitter`
   (1000 chars, 100 overlap).
3. **Embedding** — chunks are embedded **locally** with HuggingFace
   `BAAI/bge-small-en-v1.5` — no API calls, no rate limits, no cost.
4. **Indexing** — vectors are stored in a **FAISS** index (`faiss_index/`)
   alongside a `manifest.json` that tracks each file's content hash and vector
   IDs. Re-running ingestion skips unchanged files, re-embeds changed ones, and
   drops vectors for deleted files — never a full re-embed.

**Online (the deployed app):**

5. **Load** — the app loads the prebuilt FAISS index read-only; the only thing
   it embeds at runtime is the user's question (one tiny, fast call).
6. **Retrieval** — the retriever fetches the top-4 most relevant chunks.
7. **Generation** — chunks + question go to **Groq Llama 3.3 70B**, which
   answers strictly from the provided context and cites sources.
8. **Admin upload / delete** — an authenticated admin can upload a file (embedded
   locally and added to the index under stable IDs) or delete one (its vectors
   are removed surgically by ID — no rebuild). On Streamlit Cloud these runtime
   changes are temporary; for durable changes, re-run `ingest.py` and commit.

---

## 🧰 Tech stack

| Layer | Technology |
|---|---|
| UI | Streamlit |
| Orchestration | LangChain |
| Embeddings | HuggingFace `BAAI/bge-small-en-v1.5` (local, via sentence-transformers) |
| LLM | Groq `llama-3.3-70b-versatile` |
| Vector store | FAISS (CPU) |
| Document loaders | PyPDFLoader, TextLoader, Docx2txtLoader |
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

# 3. Set your Groq API key (embeddings are local — no other key needed)
echo 'GROQ_API_KEY=your-key-here' > .env

# 4. Build the index (first run downloads the ~130MB embedding model)
python ingest.py            # incremental; use --rebuild to start fresh

# 5. Run
streamlit run rag_ui_app.py
```

Get a free Groq API key at https://console.groq.com.

> **Adding documents:** drop files into `information_storage/`, run
> `python ingest.py` (only new/changed files are embedded), then commit the
> updated `faiss_index/`.

---

## ☁️ Deployment (Streamlit Community Cloud)

1. Push the repo to GitHub.
2. At https://share.streamlit.io, create a new app pointing to
   `rag_ui_app.py` on the `main` branch.
3. In **Manage app → Settings → Secrets**, add:
   ```toml
   GROQ_API_KEY = "your-key"
   ```
4. Deploy.

> ✅ **Durable index:** because `faiss_index/` is committed to the repo, the
> deployed app loads the prebuilt index on every redeploy — no re-embedding, no
> data loss. To update the knowledge base, run `python ingest.py` locally and
> push the updated `faiss_index/`.
>
> ⚠️ **Note:** documents uploaded through the Admin Panel at runtime still live
> on Streamlit Cloud's ephemeral disk and won't persist across restarts. The
> durable path is the offline `ingest.py` + commit workflow above.

---

## 📁 Project structure

```
rag-using-faiss/
├── rag_ui_app.py          # Main Streamlit app (deployed) — loads index, serves Q&A
├── ingest.py              # Offline ingester — builds/updates faiss_index/
├── rag_core.py            # Shared building blocks (embeddings, loaders, manifest)
├── rag_txt_app.py         # CLI version (terminal Q&A loop)
├── information_storage/    # Source PDF/TXT/DOCX documents
├── faiss_index/            # Prebuilt FAISS index + manifest.json (committed)
├── requirements.txt        # pip dependencies
├── pyproject.toml          # uv/project dependencies
└── README.md
```
