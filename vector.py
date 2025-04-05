"""
Langchain --> For building the chatbot logic.
chromadb --> For storing and retrieving medical knowledge.
langchain_google_genai --> For using an LLM model "gemini-2.0-flash" 
"""

import os
import sys

import chromadb
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = "AIzaSyAJtgMst73-JU9VXEnVwWVERDIhAJK5Wi8"

pdf_path = "medical_books/"

documents = []

sys.stderr = open(os.devnull,"w")

for pdf in os.listdir(pdf_path):
    if pdf.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(pdf_path,pdf))
        docs = loader.load()
        documents.extend(docs)


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
docs = text_splitter.split_documents(documents)


# store
DB_PATH = "medical_chromadb/"
if not os.path.exists(DB_PATH):      
    chroma_client = chromadb.PersistentClient(path=DB_PATH)

    vectorstore = Chroma(
        collection_name="medical_docs",
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        client=chroma_client
    )

    BATCH_SIZE = 100  
    for i in range(0, len(docs), BATCH_SIZE):
        batch = docs[i : i + BATCH_SIZE] 
        vectorstore.add_documents(batch) 

    print("Medical documents successfully embedded into ChromaDB!")

else:
    print("Loading existing ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=DB_PATH)
    vectorstore = Chroma(
        collection_name="medical_docs",
        embedding_function=GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        persist_directory=DB_PATH
    )

retriever = vectorstore.as_retriever(search_kwargs={"k": 15})