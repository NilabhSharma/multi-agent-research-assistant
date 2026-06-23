import streamlit as st
import asyncio
import uuid
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from main import build_graph

st.set_page_config(page_title="Research Assistant", page_icon="🔍", layout="wide")

if "final_report" not in st.session_state:
    st.session_state.final_report = None
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())


st.title("🔍 Multi-Agent Research Assistant")
st.caption(
    "Powered by LangGraph + Groq + Tavily, with an MCP tool layer. "
    "Enter a topic and the planner, web searcher, summarizer, and writer "
    "agents will work together to produce a cited report."
)

if st.session_state.final_report:
    if st.button("🔄 Start New Topic"):
        st.session_state.final_report = None
        st.session_state.thread_id = str(uuid.uuid4())
        st.rerun()


async def run_pipeline(input_state, status_box):
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as checkpointer:
        app = build_graph(checkpointer)
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        async for update in app.astream(input_state, config=config):
            for node_name, node_output in update.items():
                if node_name == "planner":
                    plan = node_output.get("plan", [])
                    status_box.markdown(
                        "**📝 Planner** generated sub-questions:\n\n"
                        + "\n".join(f"- {q}" for q in plan)
                    )
                elif node_name == "web_searcher":
                    status_box.markdown("**🔎 Web Searcher** finished gathering sources.")
                elif node_name == "summarizer":
                    status_box.markdown("**📚 Summarizer** condensed the findings.")
                elif node_name == "writer":
                    status_box.markdown("**✍️ Writer** is finalizing the report...")
                elif node_name == "reviser":
                    status_box.markdown("**🛠️ Reviser** is applying your follow-up...")

        snapshot = await app.aget_state(config)
        return snapshot.values


def save_report(report_text):
    with open("report_output.md", "w", encoding="utf-8") as f:
        f.write(report_text)

with st.form("topic_form"):
    topic = st.text_input("Research topic", placeholder="e.g. The future of solid-state batteries")
    submitted = st.form_submit_button("Generate Report")

if submitted and topic.strip():
    status_box = st.empty()
    with st.spinner("Running multi-agent pipeline..."):
        result = asyncio.run(run_pipeline(
            {"topic": topic, "follow_up_instruction": ""},
            status_box
        ))
    st.session_state.final_report = result["final_report"]
    save_report(result["final_report"])
    status_box.success("Report complete!")
    st.rerun()

if st.session_state.final_report:
    st.divider()
    st.markdown(st.session_state.final_report)
    st.download_button(
        "Download report as .md",
        data=st.session_state.final_report,
        file_name="report_output.md",
        mime="text/markdown"
    )

    st.divider()
    st.subheader("Ask a follow-up")
    with st.form("followup_form"):
        follow_up = st.text_input(
            "Follow-up instruction",
            placeholder="e.g. Expand the section on manufacturing costs"
        )
        followup_submitted = st.form_submit_button("Apply Follow-up")

    if followup_submitted and follow_up.strip():
        status_box = st.empty()
        with st.spinner("Revising report..."):
            result = asyncio.run(run_pipeline(
                {"follow_up_instruction": follow_up},
                status_box
            ))
        st.session_state.final_report = result["final_report"]
        save_report(result["final_report"])
        status_box.success("Report updated!")
        st.rerun()