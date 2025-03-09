import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter

CURRENT_DIR = os.path.dirname(__file__)
FILE_PATH = os.path.join(CURRENT_DIR, "books", "odyssey.txt")
PERSIST_DIRECTORY = os.path.join(CURRENT_DIR, "db", "chroma_db")

if not os.path.exists(PERSIST_DIRECTORY):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(
            f"The file {FILE_PATH} does not exist. Please check the path."
        )

    loader = TextLoader(FILE_PATH, "utf-8")
    raw_documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(documents)}")
    print(f"Sample chunk:\n{documents[0].page_content}\n")

    print("\n--- Creating embeddings ---")
    model_name = "nomic-ai/nomic-embed-text-v1"
    model_kwargs = {"device": "cpu", "trust_remote_code": True}
    encode_kwargs = {"normalize_embeddings": True}
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )
    print("\n--- Finished creating embeddings ---")

    print("\n--- Creating vector store ---")
    db = Chroma.from_documents(
        documents, embedding_model, persist_directory=PERSIST_DIRECTORY
    )
    print("\n--- Finished creating vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
