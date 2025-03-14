import os

from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import FireCrawlLoader
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

CURRENT_DIR = os.path.dirname(__file__)
DB_DIR = os.path.join(CURRENT_DIR, "db")
PERSISTENT_DIRECTORY = os.path.join(DB_DIR, "chroma_db_firecrawl")


def create_vectore_store():
    """Crawl the website, split the content, create embeddings, and persist the vector store."""
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    if not FIRECRAWL_API_KEY:
        raise ValueError("FIRECRAWL_API_KEY environment variable not set")

    print("Begin crawling the website...")
    loader = FireCrawlLoader("https://apple.com", mode="scrape")
    raw_documents = loader.load()
    print("Finished crawling the website.")

    for doc in raw_documents:
        for key, value in doc.metadata.items():
            if isinstance(value, list):
                doc.metadata[key] = ", ".join(map(str, value))

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    documents = text_splitter.split_documents(raw_documents)

    print("\n--- Document Chunks Information ---")
    print(f"Number of document chunks: {len(documents)}")
    print(f"Sample chunk:\n{documents[0].page_content}\n")

    print(f"\n--- Creating vector store in {PERSISTENT_DIRECTORY} ---")
    db = Chroma.from_documents(
        documents, embedding_model, persist_directory=PERSISTENT_DIRECTORY
    )
    print(f"--- Finished creating vector store in {PERSISTENT_DIRECTORY} ---")


model_name = "nomic-ai/nomic-embed-text-v1"
model_kwargs = {"device": "cpu", "trust_remote_code": True}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)

if not os.path.exists(PERSISTENT_DIRECTORY):
    create_vectore_store()
else:
    print("already exists")


def query_vector_store(query):
    """Query the vector store with the specified question."""

    db = Chroma(
        embedding_function=embedding_model, persist_directory=PERSISTENT_DIRECTORY
    )

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)

    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")


query = "Apple Intelligence?"

query_vector_store(query)
