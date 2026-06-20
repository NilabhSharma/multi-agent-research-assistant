from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
import asyncio

from state import ResearchState
from agents.planner import planner_node
from agents.web_searcher import web_searcher_node
from agents.summarizer import summarizer_node
from agents.writer import writer_node
from agents.reviser import reviser_node
from agents.supervisor import supervisor_router


def route_from_start(state):
    """
    Decides where to ENTER the graph. If there's a follow-up instruction
    present, skip straight to the reviser (don't re-run the whole research
    pipeline). Otherwise, start fresh at the planner.
    """
    if state.get("follow_up_instruction"):
        return "reviser"
    return "planner"


def build_graph(checkpointer):
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_node)
    graph.add_node("web_searcher", web_searcher_node)
    graph.add_node("summarizer", summarizer_node)
    graph.add_node("writer", writer_node)
    graph.add_node("reviser", reviser_node)

    # Conditional entry point: planner for new topics, reviser for follow-ups
    graph.add_conditional_edges(
        START,
        route_from_start,
        {"planner": "planner", "reviser": "reviser"}
    )

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
    graph.add_edge("reviser", END)

    # checkpointer is what gives us memory across turns - every node's
    # output gets saved here, tied to a thread_id
    return graph.compile(checkpointer=checkpointer)


def save_report(report_text):
    with open("report_output.md", "w", encoding="utf-8") as f:
        f.write(report_text)
    print("\n[Saved to report_output.md]")


async def main():
    # AsyncSqliteSaver stores checkpoints in a local file - no external
    # service, completely free. The .db file is created automatically.
    # It's an async context manager, so we use "async with" to open it.
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app = build_graph(checkpointer)

        # The thread_id ties together every turn of this conversation. As
        # long as we reuse the same thread_id, the graph has access to
        # everything that happened before (topic, plan, search results,
        # summary, report).
        thread_id = "session-1"
        config = {"configurable": {"thread_id": thread_id}}

        topic = input("Enter a research topic: ")

        print(f"\n{'='*60}")
        print(f"Starting research on: {topic}")
        print(f"{'='*60}")

        result = await app.ainvoke(
            {"topic": topic, "follow_up_instruction": ""}, config=config
        )

        print(f"\n{'='*60}")
        print("FINAL REPORT")
        print(f"{'='*60}\n")
        print(result["final_report"])
        save_report(result["final_report"])

        # Follow-up loop
        while True:
            print(f"\n{'-'*60}")
            follow_up = input(
                "Enter a follow-up instruction to revise the report, "
                "or press Enter to quit: "
            )
            if not follow_up.strip():
                print("Done. Goodbye!")
                break

            result = await app.ainvoke(
                {"follow_up_instruction": follow_up},
                config=config
            )

            print(f"\n{'='*60}")
            print("REVISED REPORT")
            print(f"{'='*60}\n")
            print(result["final_report"])
            save_report(result["final_report"])


if __name__ == "__main__":
    asyncio.run(main())