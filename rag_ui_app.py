import os
import shutil
import streamlit as st
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

st.set_page_config(page_title="RAG Document Assistant", layout="wide")

# Resolve the Google API key from env (.env locally) or Streamlit secrets (cloud).
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    try:
        google_api_key = st.secrets["GOOGLE_API_KEY"]
    except Exception:
        google_api_key = None

if not google_api_key:
    st.error(
        "GOOGLE_API_KEY is not set. On Streamlit Cloud, add it under "
        "Manage app → Settings → Secrets:\n\n"
        'GOOGLE_API_KEY = "your-key"'
    )
    st.stop()

os.environ["GOOGLE_API_KEY"] = google_api_key

st.title("📄 RAG Document Assistant")
st.write("Ask questions from your PDF and TXT files.")

folder_path = "information_storage"
faiss_index_path = "faiss_index"

os.makedirs(folder_path, exist_ok=True)


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


def load_all_documents():
    documents = []

    for file_name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file_name)
        docs = load_single_file(full_path, file_name)
        documents.extend(docs)

    return documents


splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100
)

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=google_api_key
)


@st.cache_resource
def build_rag_pipeline():
    if os.path.exists(faiss_index_path):
        vectorstore = FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        documents = load_all_documents()
        chunks = splitter.split_documents(documents)

    else:
        documents = load_all_documents()

        if not documents:
            raise ValueError("No PDF or TXT files found in information_storage folder.")

        chunks = splitter.split_documents(documents)

        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(faiss_index_path)

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash-lite",
        temperature=0.3,
        google_api_key=google_api_key
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

    return vectorstore, retriever, chain, len(documents), len(chunks)


try:
    vectorstore, retriever, chain, doc_count, chunk_count = build_rag_pipeline()
    rag_ready = True
except ValueError:
    # No documents indexed yet (e.g. admin deleted the last file).
    vectorstore = retriever = chain = None
    doc_count = chunk_count = 0
    rag_ready = False

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

    uploaded_file = st.sidebar.file_uploader(
        "Upload new document",
        type=["pdf", "txt", "docx"]
    )

if uploaded_file is not None:
    save_path = os.path.join(folder_path, uploaded_file.name)

    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if vectorstore is None:
        # No index yet (first document) — let the pipeline build it from scratch.
        if os.path.exists(faiss_index_path):
            shutil.rmtree(faiss_index_path)
    else:
        new_docs = load_single_file(save_path, uploaded_file.name)
        new_chunks = splitter.split_documents(new_docs)

        vectorstore.add_documents(new_chunks)
        vectorstore.save_local(faiss_index_path)

    st.cache_resource.clear()

    st.sidebar.success(f"{uploaded_file.name} uploaded and indexed.")
    st.rerun()


if st.session_state.is_admin:
    st.sidebar.subheader("Delete a document")

    existing_files = sorted(
        f for f in os.listdir(folder_path)
        if f.endswith((".pdf", ".txt", ".docx"))
    )

    if not existing_files:
        st.sidebar.info("No documents to delete.")
    else:
        file_to_delete = st.sidebar.selectbox(
            "Select a file to delete",
            existing_files
        )

        if st.sidebar.button("Delete selected document"):
            os.remove(os.path.join(folder_path, file_to_delete))

            # Removing a file from disk does not remove its vectors, so
            # rebuild the FAISS index from the remaining documents.
            if os.path.exists(faiss_index_path):
                shutil.rmtree(faiss_index_path)

            st.cache_resource.clear()

            st.sidebar.success(f"{file_to_delete} deleted. Index will rebuild.")
            st.rerun()


st.sidebar.header("Document Info")
st.sidebar.write(f"Documents/pages loaded: {doc_count}")
st.sidebar.write(f"Chunks created: {chunk_count}")
st.sidebar.write(f"Folder: `{folder_path}`")
st.sidebar.write(f"FAISS index: `{faiss_index_path}`")

if not rag_ready:
    st.warning("No documents are indexed yet. An admin needs to upload a document first.")

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

        with st.spinner("Thinking..."):
            response = chain.invoke({
                "context": context,
                "question": question
            })

        st.subheader("Answer")
        st.write(response)

        st.subheader("Retrieved Sources")

        for i, doc in enumerate(docs, start=1):
            source = doc.metadata.get("source_file", "Unknown")
            page = doc.metadata.get("page", "N/A")

            with st.expander(f"Source {i}: {source}, Page: {page}"):
                st.write(doc.page_content)