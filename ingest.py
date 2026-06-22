"""Offline ingestion: build/update the FAISS index from `information_storage/`.

Run this locally whenever documents change, then commit `faiss_index/` so the
deployed app picks up the prebuilt index (Streamlit Cloud disk is ephemeral).

    python ingest.py            # incremental: only new/changed/removed files
    python ingest.py --rebuild  # wipe and rebuild the whole index from scratch

Embeddings run locally, so this has no API cost and no rate limits even for a
large corpus. The live app never does this heavy work.
"""

import argparse
import shutil

from rag_core import (
    FAISS_INDEX_PATH,
    FOLDER_PATH,
    chunk_ids_for_file,
    file_hash,
    get_embeddings,
    get_splitter,
    list_source_files,
    load_manifest,
    load_single_file,
    load_vectorstore,
    save_manifest,
)

import os

from langchain_community.vectorstores import FAISS


def embed_file(vectorstore, embeddings, splitter, file_name, manifest):
    """Embed one file's chunks and add them under stable IDs. Returns chunk count."""
    full_path = os.path.join(FOLDER_PATH, file_name)
    docs = load_single_file(full_path, file_name)
    chunks = splitter.split_documents(docs)
    if not chunks:
        print(f"  ! {file_name}: no extractable text, skipping")
        return vectorstore, 0

    ids = chunk_ids_for_file(file_name, chunks)

    if vectorstore is None:
        vectorstore = FAISS.from_documents(chunks, embeddings, ids=ids)
    else:
        vectorstore.add_documents(chunks, ids=ids)

    manifest[file_name] = {"hash": file_hash(full_path), "ids": ids}
    return vectorstore, len(chunks)


def main():
    parser = argparse.ArgumentParser(description="Build/update the FAISS index.")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Wipe the index and re-embed every document from scratch.",
    )
    args = parser.parse_args()

    embeddings = get_embeddings()
    splitter = get_splitter()

    if args.rebuild and os.path.exists(FAISS_INDEX_PATH):
        shutil.rmtree(FAISS_INDEX_PATH)
        print("Rebuild: cleared existing index.")

    vectorstore = load_vectorstore(embeddings)
    manifest = {} if args.rebuild else load_manifest()

    source_files = list_source_files()
    if not source_files:
        print(f"No documents found in {FOLDER_PATH}/. Nothing to ingest.")
        return

    total_new_chunks = 0

    # Add or update new/changed files.
    for file_name in source_files:
        full_path = os.path.join(FOLDER_PATH, file_name)
        current_hash = file_hash(full_path)
        entry = manifest.get(file_name)

        if entry and entry.get("hash") == current_hash:
            print(f"  = {file_name}: unchanged, skipping")
            continue

        if entry and vectorstore is not None:
            # Changed file: drop its old vectors before re-embedding.
            vectorstore.delete(ids=entry["ids"])
            print(f"  ~ {file_name}: changed, re-embedding")
        else:
            print(f"  + {file_name}: new, embedding")

        vectorstore, n = embed_file(
            vectorstore, embeddings, splitter, file_name, manifest
        )
        total_new_chunks += n

    # Remove files that are no longer on disk.
    removed = [name for name in list(manifest) if name not in source_files]
    for file_name in removed:
        if vectorstore is not None:
            vectorstore.delete(ids=manifest[file_name]["ids"])
        del manifest[file_name]
        print(f"  - {file_name}: removed from index")

    if vectorstore is None:
        print("Index is empty after ingestion (no embeddable content).")
        return

    vectorstore.save_local(FAISS_INDEX_PATH)
    save_manifest(manifest)

    print(
        f"\nDone. Files indexed: {len(manifest)} | "
        f"new/changed chunks this run: {total_new_chunks}"
    )
    print(f"Index saved to {FAISS_INDEX_PATH}/ — commit it to deploy.")


if __name__ == "__main__":
    main()
