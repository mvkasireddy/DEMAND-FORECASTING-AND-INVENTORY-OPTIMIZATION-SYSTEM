import pandas as pd

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def create_vector_db():
    # Load PDF
    loader = PyPDFLoader("business_strategy.pdf")
    documents = loader.load()

    # Split text into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    # Load embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Create FAISS DB
    vector_db = FAISS.from_documents(chunks, embeddings)

    return vector_db


def retrieve_context(vector_db, query):
    docs = vector_db.similarity_search(query, k=3)
    return "\n".join([doc.page_content for doc in docs])