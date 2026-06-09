from langchain_chroma import Chroma

from embeddings import embedding_model

db = Chroma(
    persist_directory="../chroma_db",
    embedding_function=embedding_model
)