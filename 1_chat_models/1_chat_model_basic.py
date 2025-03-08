from langchain_ollama import ChatOllama

model = ChatOllama(model='llama2:latest')
result = model.invoke("What is 81 divided by 9?")
print("Full result:")
print(result)
print("Content only:")
print(result.content)
