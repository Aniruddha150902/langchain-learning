from langchain_ollama import ChatOllama
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

model = ChatOllama(model="llama2:latest")

print("\n---- Prompt Template with One Placeholder ----\n")
template_string = "Tell me a joke about {topic}"
prompt_template = ChatPromptTemplate.from_template(template=template_string)
prompt = prompt_template.invoke({"topic": "cat"})
response = model.invoke(prompt)
print(response.content)

print("\n---- Prompt Template with Multiple Placeholders ----\n")
template_string_multiple = "Tell me a {adjective} joke about a {animal}"
prompt_template_multiple = ChatPromptTemplate.from_template(
    template=template_string_multiple)
prompt_multiple = prompt_template_multiple.invoke(
    {"adjective": "funny", "animal": "panda"})
response = model.invoke(prompt_multiple)
print(response.content)

print("\n---- Prompt Template from Messages ----\n")
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    ("human", "Tell me {joke_count} jokes."),
]
prompt_template_messages = ChatPromptTemplate.from_messages(messages=messages)
prompt_messages = prompt_template_messages.invoke(
    {"topic": "lawyer", "joke_count": 3})
response = model.invoke(prompt_messages)
print(response.content)

print("\n---- Prompt Template from Messages Variation without tuples ----\n")
messages = [
    ("system", "You are a comedian who tells jokes about {topic}."),
    HumanMessage(content="Tell me 3 jokes.")
]
# The Below Does not Work
# messages = [
#     ("system", "You are a comedian who tells jokes about {topic}."),
#     HumanMessage(content="Tell me {joke_count} jokes.")
# ]
prompt_template_messages = ChatPromptTemplate.from_messages(messages=messages)
prompt_messages = prompt_template_messages.invoke(
    {"topic": "lawyer"})
response = model.invoke(prompt_messages)
print(response.content)
