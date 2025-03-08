from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

model = ChatOllama(model="llama2:latest")

prompt_template = ChatPromptTemplate(messages=[
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
])

chain = prompt_template | model | StrOutputParser()

response = chain.invoke({"topic": "lawyers", "joke_count": 3})

print(response)
