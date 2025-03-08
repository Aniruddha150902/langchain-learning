from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama2:latest")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?")
]

result = model.invoke(messages)
print(f"Answer from llama2 AI: {result.content}\n")


model = ChatOllama(model="deepseek-r1:1.5b")

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is 81 divided by 9?")
]

result = model.invoke(messages)
print(f"Answer from deepseek AI: {result.content}\n")
