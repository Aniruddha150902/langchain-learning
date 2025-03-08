from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama2:latest")

chat_history = []

system_message = SystemMessage(content="You are a helpful AI assistant.")
chat_history.append(system_message)

while True:
    query = input("You : ")

    if query.lower() == "exit":
        break

    chat_history.append(HumanMessage(content=query))

    response = model.invoke(chat_history)
    print(f"AI : {response.content}\n")

    chat_history.append(AIMessage(content=response.content))

print("---- MESSAGE HISTORY ----")
print(chat_history)
