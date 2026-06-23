import os
from dotenv import load_dotenv
from typing import Annotated, TypedDict

from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

if not groq_api_key or not tavily_api_key:
    raise ValueError("Missing API keys. Check your .env file.")

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearch(max_results=3, tavily_api_key=tavily_api_key)
tools = [search_tool]

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
llm_with_tools = llm.bind_tools(tools)


def call_model(state: AgentState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

graph = StateGraph(AgentState)

graph.add_node("agent", call_model)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")

graph.add_conditional_edges(
    "agent",
    should_continue,
    {"tools": "tools", END: END}
)

graph.add_edge("tools", "agent")

app = graph.compile()

if __name__ == "__main__":
    question = "What are the latest developments in AI agents in 2026?"

    result = app.invoke({
        "messages": [{"role": "user", "content": question}]
    })

    print("=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(result["messages"][-1].content)