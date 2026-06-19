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

# ---- A small, self-contained ReAct loop (same pattern as Phase 2) ----
# This is its own tiny graph, used internally once per sub-question.

class MiniState(TypedDict):
    messages: Annotated[list, add_messages]

search_tool = TavilySearch(max_results=3, tavily_api_key=tavily_api_key)
tools = [search_tool]

llm = ChatGroq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
llm_with_tools = llm.bind_tools(tools)

def call_model(state: MiniState):
    response = llm_with_tools.invoke(state["messages"])
    return {"messages": [response]}

tool_node = ToolNode(tools)

def should_continue(state: MiniState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

mini_graph = StateGraph(MiniState)
mini_graph.add_node("agent", call_model)
mini_graph.add_node("tools", tool_node)
mini_graph.set_entry_point("agent")
mini_graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
mini_graph.add_edge("tools", "agent")

mini_app = mini_graph.compile()


# ---- The actual node used in the BIG multi-agent graph ----
SEARCH_SYSTEM_PROMPT = """You are a research assistant. Answer the following
sub-question as accurately and concisely as possible. Use the search tool if
you need current or factual information. Cite the source URLs you used in
your answer."""

def web_searcher_node(state):
    plan = state["plan"]
    search_results = []

    print(f"\n[WEB SEARCHER] Researching {len(plan)} sub-questions...")

    for i, sub_question in enumerate(plan, start=1):
        print(f"  ({i}/{len(plan)}) {sub_question}")

        result = mini_app.invoke({
            "messages": [
                {"role": "system", "content": SEARCH_SYSTEM_PROMPT},
                {"role": "user", "content": sub_question}
            ]
        })

        finding = result["messages"][-1].content
        search_results.append(f"Q: {sub_question}\nA: {finding}")

    print("[WEB SEARCHER] Done.\n")

    return {
        "search_results": search_results,
        "next": "summarizer"
    }


# Quick standalone test
if __name__ == "__main__":
    test_state = {
        "plan": [
            "What specific software engineering tasks are most likely to be automated by AI agents?",
            "What new skills are emerging as a result of AI agents in software engineering?"
        ]
    }
    result = web_searcher_node(test_state)
    print("=" * 60)
    print("SEARCH RESULTS:")
    print("=" * 60)
    for r in result["search_results"]:
        print(r)
        print("-" * 40)