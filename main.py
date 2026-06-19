from langgraph.graph import StateGraph, END

from state import ResearchState
from agents.planner import planner_node
from agents.web_searcher import web_searcher_node
from agents.summarizer import summarizer_node
from agents.writer import writer_node
from agents.supervisor import supervisor_router


def build_graph():
    graph = StateGraph(ResearchState)

    # Register each specialist as a node
    graph.add_node("planner", planner_node)
    graph.add_node("web_searcher", web_searcher_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("writer", writer_node)

    # Start at the planner
    graph.set_entry_point("planner")

    # After EACH node, route based on what that node set in state["next"].
    # We reuse the same supervisor_router function for all of them since
    # the logic ("read state['next'] and go there") is identical every time.
    routing_map = {
        "web_searcher": "web_searcher",
        "summarizer": "summarizer",
        "writer": "writer",
        "END": END,
    }

    graph.add_conditional_edges("planner", supervisor_router, routing_map)
    graph.add_conditional_edges("web_searcher", supervisor_router, routing_map)
    graph.add_conditional_edges("summarizer", supervisor_router, routing_map)
    graph.add_conditional_edges("writer", supervisor_router, routing_map)

    return graph.compile()


app = build_graph()


if __name__ == "__main__":
    topic = input("Enter a research topic: ")

    print(f"\n{'='*60}")
    print(f"Starting research on: {topic}")
    print(f"{'='*60}")

    result = app.invoke({"topic": topic})

    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print(f"{'='*60}\n")
    print(result["final_report"])

    # Save the report to a file too, since markdown reports are meant to be
    # read/shared, not just printed to terminal
    with open("report_output.md", "w", encoding="utf-8") as f:
        f.write(result["final_report"])

    print(f"\n[Saved to report_output.md]")