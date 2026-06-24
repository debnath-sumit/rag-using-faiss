import os
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from rag_core import (
    CHROMA_PATH,
    FOLDER_PATH,
    chunk_ids_for_file,
    file_hash,
    get_embeddings,
    get_splitter,
    get_vectorstore,
    list_source_files,
    load_manifest,
    load_single_file,
    load_vectorstore,
    save_manifest,
)

load_dotenv()

st.set_page_config(page_title="RAG Document Assistant", layout="wide")

# Chat answers run on Groq (free tier, fast). Embeddings are local, so no
# Google key is needed anymore.
groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    try:
        groq_api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        groq_api_key = None

if not groq_api_key:
    st.error(
        "GROQ_API_KEY is not set. Get a free key at https://console.groq.com "
        "and add it.\n\nOn Streamlit Cloud: Manage app → Settings → Secrets:\n\n"
        'GROQ_API_KEY = "your-key"'
    )
    st.stop()

os.environ["GROQ_API_KEY"] = groq_api_key

st.title("📄 RAG Document Assistant")
st.write("Ask questions from your PDF, TXT, and DOCX files.")

os.makedirs(FOLDER_PATH, exist_ok=True)

splitter = get_splitter()


@st.cache_resource
def get_embedder():
    # Loaded once per app process; used to embed the query (and admin uploads).
    return get_embeddings()


embeddings = get_embedder()


@st.cache_resource
def build_rag_pipeline():
    """Load the prebuilt Chroma index read-only and wire up the Groq chain.

    The heavy lifting (embedding the whole corpus) happens offline in
    `ingest.py`; here we only load what's already built.
    """
    vectorstore = load_vectorstore(embeddings)
    if vectorstore is None:
        raise ValueError("No Chroma index found. Run `python ingest.py` first.")

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        api_key=groq_api_key,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are a helpful QA/SDET document assistant.

Answer the user's question using only the provided context.

If the answer is not available in the context, say:
"I don't know from the provided documents."

When possible, mention the source file name.

Context:
{context}
"""),
        ("human", "{question}")
    ])

    parser = StrOutputParser()
    chain = prompt | llm | parser

    return vectorstore, retriever, chain


try:
    vectorstore, retriever, chain = build_rag_pipeline()
    rag_ready = True
except ValueError:
    # No index built yet (e.g. fresh deploy, or admin deleted the last file).
    vectorstore = retriever = chain = None
    rag_ready = False

# Counts come from the manifest — cheap, no re-reading documents.
manifest = load_manifest()
file_count = len(manifest)
chunk_count = sum(len(entry.get("ids", [])) for entry in manifest.values())

st.sidebar.header("Admin Panel")

# Hardcoded admin credentials. Only an authenticated admin can upload files.
ADMIN_USERNAME = "rag_admin"
ADMIN_PASSWORD = "rag_password_12#"

if "is_admin" not in st.session_state:
    st.session_state.is_admin = False

if not st.session_state.is_admin:
    with st.sidebar.form("admin_login"):
        st.write("Admin login required to upload documents.")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login")

    if submitted:
        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            st.session_state.is_admin = True
            st.sidebar.success("Logged in as admin.")
            st.rerun()
        else:
            st.sidebar.error("Invalid username or password.")

uploaded_file = None

if st.session_state.is_admin:
    if st.sidebar.button("Log out"):
        st.session_state.is_admin = False
        st.rerun()

    st.sidebar.caption(
        "Uploads here are indexed locally with no API cost. Note: on Streamlit "
        "Cloud these changes are temporary (ephemeral disk). For durable "
        "changes, run `python ingest.py` locally and commit `chroma_db/`."
    )

    uploaded_file = st.sidebar.file_uploader(
        "Upload new document",
        type=["pdf", "txt", "docx"]
    )

if uploaded_file is not None:
    save_path = os.path.join(FOLDER_PATH, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    new_docs = load_single_file(save_path, uploaded_file.name)
    new_chunks = splitter.split_documents(new_docs)

    if not new_chunks:
        st.sidebar.error("No extractable text in that file — nothing indexed.")
    else:
        ids = chunk_ids_for_file(uploaded_file.name, new_chunks)

        with st.spinner("Embedding and indexing..."):
            if vectorstore is None:
                vectorstore = get_vectorstore(embeddings)
            else:
                # Re-upload of an existing file: drop old vectors first.
                old = manifest.get(uploaded_file.name)
                if old:
                    vectorstore.delete(ids=old["ids"])
            vectorstore.add_documents(new_chunks, ids=ids)

            manifest[uploaded_file.name] = {
                "hash": file_hash(save_path),
                "ids": ids,
            }
            # Chroma persists on write; only the manifest needs an explicit save.
            save_manifest(manifest)

        st.cache_resource.clear()
        st.sidebar.success(f"{uploaded_file.name} uploaded and indexed.")
        st.rerun()


if st.session_state.is_admin:
    st.sidebar.subheader("Delete a document")

    existing_files = list_source_files()

    if not existing_files:
        st.sidebar.info("No documents to delete.")
    else:
        file_to_delete = st.sidebar.selectbox(
            "Select a file to delete",
            existing_files
        )

        if st.sidebar.button("Delete selected document"):
            path = os.path.join(FOLDER_PATH, file_to_delete)
            if os.path.exists(path):
                os.remove(path)

            # Surgically remove just this file's vectors — no full re-embed.
            entry = manifest.get(file_to_delete)
            if entry and vectorstore is not None:
                vectorstore.delete(ids=entry["ids"])
            if entry:
                del manifest[file_to_delete]
                save_manifest(manifest)

            st.cache_resource.clear()
            st.sidebar.success(f"{file_to_delete} deleted from the index.")
            st.rerun()


st.sidebar.header("Document Info")
st.sidebar.write(f"Files indexed: {file_count}")
st.sidebar.write(f"Chunks indexed: {chunk_count}")
st.sidebar.write(f"Folder: `{FOLDER_PATH}`")
st.sidebar.write(f"Chroma index: `{CHROMA_PATH}`")

if not rag_ready:
    st.warning(
        "No documents are indexed yet. Run `python ingest.py` locally (and "
        "commit `chroma_db/`), or log in as admin and upload a document."
    )

question = st.text_input("Ask a question:", disabled=not rag_ready)

if st.button("Ask", disabled=not rag_ready):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        docs = retriever.invoke(question)

        context_parts = []

        for doc in docs:
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "N/A")

            context_parts.append(
                f"Source file: {source}\nPage: {page}\nContent:\n{doc.page_content}"
            )

        context = "\n\n".join(context_parts)

        try:
            with st.spinner("Thinking..."):
                response = chain.invoke({
                    "context": context,
                    "question": question
                })
        except Exception as e:
            st.error(f"The model call failed: {e}")
            st.stop()

        st.subheader("Answer")
        st.write(response)

        st.subheader("Retrieved Sources")

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "N/A")

            with st.expander(f"Source {i}: {source}, Page: {page}"):
                st.write(doc.page_content)
