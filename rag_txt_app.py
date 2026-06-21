from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (TextLoader,PyPDFLoader)
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os

load_dotenv()

documents = []
folder_path = "information_storage"

for file_name in os.listdir(folder_path):

    full_path = os.path.join(folder_path, file_name)

    if file_name.endswith(".pdf"):
        #print(f"Loading PDF: {file_name}")
        loader = PyPDFLoader(full_path)
    elif file_name.endswith(".txt"):
        #print(f"Loading TXT: {file_name}")
        loader = TextLoader(full_path)
    else:
        #print(f"Skipping unsupported file: {file_name}")
        continue

    loaded_docs = loader.load()

    for doc in loaded_docs:
        doc.metadata["source_file"] = file_name
        documents.append(doc)

#print(f"\nLoaded {len(documents)} documents/pages")

if not documents:
    raise ValueError("No PDF or TXT files found in information_storage folder.")


# 2. Split document into smaller chunks
splitter = CharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
#print(f"Created {len(chunks)} chunks")


# 3. Convert text chunks into embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)

# 4. Store embeddings in FAISS vector database
vectorstore = FAISS.from_documents(chunks, embeddings)

# 5. Create retriever
retriever = vectorstore.as_retriever()

# 6. Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0.3
)

prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are a helpful QA/SDET assistant.
Answer only from the given context.
If the answer is not in the context, say: I don't know from the provided document.

Context:
{context}
"""),
    ("human", "{question}")
])

parser = StrOutputParser()

chain = prompt | llm | parser

print("RAG Text Assistant started. Type 'quit' to stop.\n")

while True:
    question = input("You: ")

    if question.lower() == "quit":
        break

    # 7. Search relevant document chunks
    docs = retriever.invoke(question)


    # print("\n========== RETRIEVED DOCS ==========\n")
    #
    # for doc in docs:
    #     print(doc.page_content)
    #
    # print("\n====================================\n")

    context = "\n\n".join([doc.page_content for doc in docs])

    # print("\nRetrieved Context:")
    # print(context)
    # print("\n-------------------")

    # 8. Send context + question to Gemini
    response = chain.invoke({
        "context": context,
        "question": question
    })

    print("\nAssistant:")

    print(response)
    print()