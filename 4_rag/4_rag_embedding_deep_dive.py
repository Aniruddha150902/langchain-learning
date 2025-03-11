import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import CharacterTextSplitter

CURRENT_DIR = os.path.dirname(__file__)
DB_DIR = os.path.join(CURRENT_DIR, "db")
FILE_PATH = os.path.join(CURRENT_DIR, "books", "odyssey.txt")

if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(
        f"The file {FILE_PATH} does not exist. Please check the path."
    )

loader = TextLoader(FILE_PATH, encoding="utf-8")
raw_documents = loader.load()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

print("\n--- Document Chunks Information ---")
print(f"Number of document chunks: {len(documents)}")
print(f"Sample chunk:\n{documents[0].page_content}\n")


def create_vector_store(documents, embedding_model, store_name):
    persistent_directory = os.path.join(DB_DIR, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        db = Chroma.from_documents(
            documents, embedding_model, persist_directory=persistent_directory
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(f"Vector store {store_name} already exists. No need to initialize.")


print("\n--- Using Llama Embeddings ---")
llama_embedding_model = OllamaEmbeddings(model="llama2:latest")
create_vector_store(documents, llama_embedding_model, "chroma_db_llama")

print("\n--- Using Nomic Embeddings ---")
model_name = "nomic-ai/nomic-embed-text-v1"
model_kwargs = {"device": "cpu", "trust_remote_code": True}
encode_kwargs = {"normalize_embeddings": True}
nomic_embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)
create_vector_store(documents, nomic_embedding_model, "chroma_db_nomic")

print("Embedding demonstrations for Ollama and Hugging Face completed.")


def query_vector_store(embedding_model, store_name, query):
    persistent_directory = os.path.join(DB_DIR, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            embedding_function=embedding_model, persist_directory=persistent_directory
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"k": 3, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs, 1):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")


query = "Who is Odysseus' wife?"

query_vector_store(llama_embedding_model, "chroma_db_llama", query)
query_vector_store(nomic_embedding_model, "chroma_db_nomic", query)

print("Querying demonstrations completed.")
