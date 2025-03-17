from datetime import datetime

from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.tools import Tool
from langchain_ollama import ChatOllama


def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    now = datetime.now()
    return now.strftime("%I:%M %p")


llm = ChatOllama(model="llama2:latest", temperature=0)

tools = [
    Tool(
        name="Time",
        func=get_current_time,
        description="Useful for when you need to know the current time",
    )
]

prompt = hub.pull("hwchase17/react")


agent = create_react_agent(llm, tools, prompt, stop_sequence=True)

agent_executer = AgentExecutor.from_agent_and_tools(
    verbose=True, agent=agent, tools=tools
)

response = agent_executer.invoke({"input": "What time is it?"})
print("Response : ", response)
