import os

from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
    TextSplitter,
    TokenTextSplitter,
)

CURRENT_DIR = os.path.dirname(__file__)
DB_DIR = os.path.join(CURRENT_DIR, "db")
FILE_PATH = os.path.join(CURRENT_DIR, "books", "romeo_and_juliet.txt")

loader = TextLoader(FILE_PATH, "utf-8")
raw_docuements = loader.load()

model_name = "nomic-ai/nomic-embed-text-v1"
model_kwargs = {"device": "cpu", "trust_remote_code": True}
encode_kwargs = {"normalize_embeddings": True}
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs,
)


def create_vector_store(store_name, docs):
    persistent_directory = os.path.join(DB_DIR, store_name)
    if not os.path.exists(persistent_directory):
        print(f"\n--- Creating vector store {store_name} ---")
        db = Chroma.from_documents(
            docs, embedding_model, persist_directory=persistent_directory
        )
        print(f"--- Finished creating vector store {store_name} ---")
    else:
        print(f"Vector store {store_name} already exists. No need to initialize.")


char_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
char_docs = char_splitter.split_documents(raw_docuements)
create_vector_store("chroma_db_char", char_docs)

sentence_splitter = SentenceTransformersTokenTextSplitter(chunk_size=1000)
sentence_docs = sentence_splitter.split_documents(raw_docuements)
create_vector_store("chroma_db_sent", sentence_docs)

token_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=0)
token_docs = token_splitter.split_documents(raw_docuements)
create_vector_store("chroma_db_token", token_docs)

recur_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
recur_docs = recur_splitter.split_documents(raw_docuements)
create_vector_store("chroma_db_recur", recur_docs)


class CustomTextSplitter(TextSplitter):
    def split_text(self, text):
        return text.split("\n\n")


custom_splitter = CustomTextSplitter()
custom_docs = custom_splitter.split_documents(raw_docuements)
create_vector_store("chroma_db_custom", custom_docs)


def query_vector_store(store_name, query):
    persistent_directory = os.path.join(DB_DIR, store_name)
    if os.path.exists(persistent_directory):
        print(f"\n--- Querying the Vector Store {store_name} ---")
        db = Chroma(
            embedding_function=embedding_model, persist_directory=persistent_directory
        )
        retriever = db.as_retriever(
            search_type="similarity_score_threshold",
            search_kargs={"k": 1, "score_threshold": 0.1},
        )
        relevant_docs = retriever.invoke(query)
        print(f"\n--- Relevant Documents for {store_name} ---")
        for i, doc in enumerate(relevant_docs):
            print(f"Document {i}:\n{doc.page_content}\n")
            if doc.metadata:
                print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")
    else:
        print(f"Vector store {store_name} does not exist.")


query = "How did Juliet die?"

query_vector_store("chroma_db_char", query)
query_vector_store("chroma_db_sent", query)
query_vector_store("chroma_db_token", query)
query_vector_store("chroma_db_recur", query)
query_vector_store("chroma_db_custom", query)
