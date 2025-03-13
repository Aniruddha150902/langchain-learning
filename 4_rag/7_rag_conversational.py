import os

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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
    search_kwargs={"k": 3},
)

llm = ChatOllama(model="llama2:latest")


contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(content=contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt_template
)


qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

qa_prompt_template = ChatPromptTemplate.from_messages(
    messages=[
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt_template)


rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)


def continual_chat():
    print("Start chatting with the AI! Type 'exit' to end the conversation.")
    chat_history = []

    while True:
        query = input("You : ")
        if query.lower() == "exit":
            break

        result = rag_chain.invoke({"chat_history": chat_history, "input": query})
        print(f"AI : {result["answer"]}")

        chat_history.append(HumanMessage(query))
        chat_history.append(AIMessage(result["answer"]))


if __name__ == "__main__":
    continual_chat()
