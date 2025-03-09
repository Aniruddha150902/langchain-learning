import os

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

CURRENT_DIR = os.path.dirname(__file__)
BOOKS_DIR = os.path.join(CURRENT_DIR, "books")
DB_DIR = os.path.join(CURRENT_DIR, "db")
PERSIST_DIRECTORY = os.path.join(DB_DIR, "chroma_db_with_metadata")

if not os.path.exists(PERSIST_DIRECTORY):
    print("Persistent directory does not exist. Initializing vector store...")

    if not os.path.exists(BOOKS_DIR):
        raise FileNotFoundError(
            f"The directory {BOOKS_DIR} does not exist. Please check the path."
        )

    book_files = [fp for fp in os.listdir(BOOKS_DIR) if fp.endswith(".txt")]

    documents = []

    for book_file in book_files:
        file_path = os.path.join(BOOKS_DIR, book_file)
        loader = TextLoader(file_path, "utf-8")
        raw_documents = loader.load()

        for document in raw_documents:
            document.metadata = {"source": os.path.basename(book_file)}
            documents.append(document)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(docs)}")

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

    print("\n--- Creating and persisting vector store ---")
    db = Chroma.from_documents(
        docs, embedding_model, persist_directory=PERSIST_DIRECTORY
    )
    print("\n--- Finished creating and persisting vector store ---")

else:
    print("Vector store already exists. No need to initialize.")
