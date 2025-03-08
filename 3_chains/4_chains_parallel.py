from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnableParallel
from langchain_ollama import ChatOllama

model = ChatOllama(model="llama2:latest")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an expert product reviewer."),
        ("human", "List the main features of the product {product_name}."),
    ]
)


def analyse_pros(features: str):
    pros_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the pros of these features.",
            ),
        ]
    )
    return pros_template.format_prompt(features=features)


def analyse_cons(features: str):
    cons_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            (
                "human",
                "Given these features: {features}, list the cons of these features.",
            ),
        ]
    )
    return cons_template.format_prompt(features=features)


def combine_pros_cons(pros: str, cons: str):
    return f"\nPros:\n{pros}\n\nCons:\n{cons}\n"


pros_branch_chain = (
    RunnableLambda(lambda x: analyse_pros(x)) | model | StrOutputParser()
)
cons_branch_chain = (
    RunnableLambda(lambda x: analyse_cons(x)) | model | StrOutputParser()
)

chain = (
    prompt_template
    | model
    | StrOutputParser()
    | RunnableParallel(pros=pros_branch_chain, cons=cons_branch_chain)
    | RunnableLambda(lambda x: print(x) or combine_pros_cons(x["pros"], x["cons"]))
)

response = chain.invoke({"product_name": "MacBook Pro"})
print(response)
