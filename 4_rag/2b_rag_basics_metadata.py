import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CURRENT_DIR = os.path.dirname(__file__)
DB_DIR = os.path.join(CURRENT_DIR, "db")
PERSIST_DIRECTORY = os.path.join(DB_DIR, "chroma_db_with_metadata")

model_name = "nomic-ai/nomic-embed-text-v1"
model_kwargs = {"device": "cpu", "trust_remote_code": True}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

db = Chroma(embedding_function=embedding_model, persist_directory=PERSIST_DIRECTORY)

retriever = db.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 3, "score_threshold": 0.4},
)

query = "How did Juliet die?"

relevant_docs = retriever.invoke(query)

print("\n--- Relevant Documents ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")
    print(f"Source: {doc.metadata['source']}\n")
