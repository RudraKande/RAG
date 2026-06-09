from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from embeddings import embedding_model

loader = PyPDFLoader("docs/handbook.pdf")

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Number of chunks:", len(chunks))

db = Chroma.from_documents(
    chunks,
    embedding_model,
    persist_directory="chroma_db"
)

print("Embeddings stored successfully")