import os

from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

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

db = Chroma(embedding_function=embedding_model, persist_directory=PERSISTENT_DIRECTORY)
retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

query = "How can I learn more about LangChain?"
relevant_docs = retriever.invoke(query)

for i, docs in enumerate(relevant_docs):
    print(f"\nDOCUMENT {i} :\n{docs.page_content}\n")

combined_input = (
    "Here are some documents that might help answer the question: "
    + query
    + "\n\nRelevant Documents:\n"
    + "\n\n".join([doc.page_content for doc in relevant_docs])
    + "\n\nPlease provide an answer based only on the provided documents. If the answer is not found in the documents, respond with 'I'm not sure'."
)
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=combined_input),
]

model = ChatOllama(model="llama2:latest")
result = model.invoke(messages)
print("\n--- Generated Response ---")
print("Content only:")
print(result.content)
