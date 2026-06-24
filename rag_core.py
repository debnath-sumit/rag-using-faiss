"""Shared RAG building blocks used by both the offline ingester and the app.

Embeddings run locally (no API, no rate limits) via a small sentence-transformer
model, so ingestion can happen offline and the live app only embeds the query.
"""

import hashlib
import json
import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

FOLDER_PATH = "information_storage"
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "documents"
MANIFEST_PATH = os.path.join(CHROMA_PATH, "manifest.json")

# Local embedding model. Small (~130MB), strong retrieval quality, CPU-friendly.
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".docx")


def get_embeddings():
    """Local sentence-transformer embeddings (normalized for cosine-like ranking)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )


def get_splitter():
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
    )


def load_single_file(file_path, file_name):
    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_name.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_name.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        return []

    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = file_name
    return docs


def list_source_files(folder_path=FOLDER_PATH):
    if not os.path.isdir(folder_path):
        return []
    return sorted(
        f for f in os.listdir(folder_path) if f.endswith(SUPPORTED_EXTENSIONS)
    )


def file_hash(file_path):
    """Content hash so the ingester can skip files that have not changed."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def chunk_ids_for_file(file_name, chunks):
    """Stable, unique vector IDs per chunk so a file can be deleted surgically."""
    return [f"{file_name}::{i}" for i in range(len(chunks))]


def load_manifest():
    """Maps source_file -> {"hash": str, "ids": [chunk ids in the index]}."""
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            return json.load(f)
    return {}


def save_manifest(manifest):
    os.makedirs(CHROMA_PATH, exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def get_vectorstore(embeddings):
    """Open the persisted Chroma collection (creating it on first write)."""
    return Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME,
    )


def load_vectorstore(embeddings):
    """Load the prebuilt Chroma collection, or None if it does not exist yet."""
    if not os.path.exists(os.path.join(CHROMA_PATH, "chroma.sqlite3")):
        return None
    return get_vectorstore(embeddings)
