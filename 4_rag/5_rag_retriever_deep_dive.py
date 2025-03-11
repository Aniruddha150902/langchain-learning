import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CURRENT_DIR = os.path.dirname(__file__)
PERSISTENT_DIRECTORY = os.path.join(CURRENT_DIR, "db", "chroma_db_with_metadata")

model_name = "nomic-ai/nomic-embed-text-v1"
model_kwargs = {"device": "cpu", "trust_remote_code": True}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


def query_vector_store(embedding_model, store_name, query, search_type, search_kwargs):
    if not os.path.exists(PERSISTENT_DIRECTORY):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            embedding_function=embedding_model, persist_directory=PERSISTENT_DIRECTORY
        )
        retriever = db.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
        relevant_docs = retriever.invoke(query)
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")


query = "How did Juliet die?"

print("\n--- Using Similarity Search ---")
query_vector_store(
    embedding_model, "chroma_db_with_metadata", query, "similarity", {"k": 3}
)

print("\n--- Using Max Marginal Relevance (MMR) ---")
query_vector_store(
    embedding_model,
    "chroma_db_with_metadata",
    query,
    "mmr",
    {"k": 3, "fetch_k": 20, "lambda_mult": 0.5},
)

print("\n--- Using Similarity Score Threshold ---")
query_vector_store(
    embedding_model,
    "chroma_db_with_metadata",
    query,
    "similarity_search_threshold",
    {"k": 3, "search_threshold": 0.4},
)

print("Querying demonstrations with different search types completed.")
