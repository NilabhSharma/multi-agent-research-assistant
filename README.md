# Multi-Agent Autonomous Research Assistant

A production-grade multi-agent AI pipeline built with LangGraph that autonomously researches any topic and produces a structured, source-cited markdown report. Specialized sub-agents (planner, web searcher, summarizer, writer) are orchestrated by a supervisor, with multi-turn follow-up support, MCP tool layer, LangSmith tracing, and a Streamlit UI.

---

## Demo

**Input:** A research topic ("The environmental impact of EV battery production")

**Output:** A structured markdown report with themed sections, inline citations, and a sources list — generated autonomously in 30–90 seconds.

**Follow-ups:** Ask the system to expand any section, add new angles, or compare with related topics — it searches for fresh information and revises the report in place.

---

## Architecture

```
User Input (Streamlit UI)
        │
        ▼
┌─────────────────────────────────────────────┐
│              LangGraph Pipeline             │
│                                             │
│  START ──► Planner                          │
│               │ (sub-questions)             │
│               ▼                             │
│           Web Searcher ◄──────────────┐     │
│           (ReAct loop, per question)  │     │
│               │ (search results)      │     │
│               ▼                       │     │
│           Summarizer                  │     │
│               │ (condensed findings)  │     │
│               ▼                       │     │
│           Writer                      │     │
│               │ (final report)        │     │
│               ▼                       │     │
│  Follow-up? ──► Reviser ──────────────┘     │
│               │                             │
│              END                            │
└─────────────────────────────────────────────┘
        │
        ▼
  MCP Server (Tavily web search tool)
  LangSmith (observability tracing)
  AsyncSqliteSaver (conversation memory)
```

**Key design decisions:**
- Supervisor is a lightweight Python router (not an LLM call) — cheaper and more reliable for a fixed pipeline
- Web searcher and reviser use `llama-3.1-8b-instant` for consistent tool-calling; other agents use `llama-3.3-70b-versatile`
- MCP server runs as a persistent HTTP process (streamable-HTTP transport), matching production deployment patterns
- Output validation guards catch malformed LLM tool-call responses before they reach the report

---

## Tech Stack

| Component | Tool |
|---|---|
| Agent orchestration | LangGraph + LangChain |
| LLM provider | Groq API (free tier) — Llama 3.1 / 3.3 |
| Web search | Tavily API (free tier) |
| Tool layer | FastMCP (streamable-HTTP MCP server) |
| Conversation memory | AsyncSqliteSaver (local SQLite) |
| Observability | LangSmith (free tier) |
| UI | Streamlit |
| Language | Python 3.10 |

**100% free to run** — no paid APIs required.

---

## Project Structure

```
research-assistant/
├── app.py                  # Streamlit UI with live agent progress streaming
├── main.py                 # Terminal entrypoint (same pipeline, no UI)
├── state.py                # Shared ResearchState TypedDict
├── agents/
│   ├── planner.py          # Breaks topic into sub-questions
│   ├── web_searcher.py     # ReAct loop per sub-question via MCP
│   ├── summarizer.py       # Condenses raw findings
│   ├── writer.py           # Produces final markdown report
│   ├── reviser.py          # Applies follow-up instructions (MCP-grounded)
│   └── supervisor.py       # Lightweight routing function
├── mcp_server/
│   └── research_server.py  # FastMCP server exposing Tavily web_search tool
├── test_groq.py            # Standalone Groq API connectivity test
├── test_tavily.py          # Standalone Tavily API connectivity test
└── test_mcp_client.py      # Standalone MCP client connectivity test
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/NilabhSharma/multi-agent-research-assistant.git
cd multi-agent-research-assistant
```

### 2. Create and activate a virtual environment

```bash
# Windows
py -3.10 -m venv venv
venv\Scripts\activate

# Mac/Linux
python3.10 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install langgraph langchain langchain-groq langchain-tavily
pip install langchain-mcp-adapters mcp fastmcp
pip install langgraph-checkpoint-sqlite aiosqlite
pip install python-dotenv groq tavily-python streamlit
pip install langsmith
```

### 4. Get free API keys

| Service | URL | Notes |
|---|---|---|
| Groq | https://console.groq.com | Free LLM API — no credit card |
| Tavily | https://tavily.com | 1,000 free searches/month |
| LangSmith | https://smith.langchain.com | Free observability tier |

### 5. Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=gsk_your_key_here
TAVILY_API_KEY=tvly_your_key_here
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_your_key_here
LANGCHAIN_PROJECT=multi-agent-research-assistant
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com
```

> **Note for users outside North America:** if you get 403 errors from LangSmith, check your regional endpoint at smith.langchain.com → Settings → API Keys → "Configure environment". Replace the endpoint value accordingly (e.g. `https://apac.api.smith.langchain.com` for Asia-Pacific).

---

## Running the App

This project requires **two terminal windows** running simultaneously.

### Terminal 1 — Start the MCP server

```bash
# Activate venv first
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux

python mcp_server/research_server.py
```

Expected output:
```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
```

Leave this running. **The MCP server must be running before you start the app.**

### Terminal 2 — Launch the Streamlit UI

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

### Alternatively: run from the terminal (no UI)

```bash
python main.py
```

---

## Usage

1. **Enter a research topic** in the text box and click "Generate Report"
2. **Watch live progress** — the planner's sub-questions, searcher status, summarizer, and writer all update in real time
3. **Read and download** the generated markdown report
4. **Ask follow-ups** — "Expand the section on X", "Add a section comparing Y", "Include more recent data on Z"
5. **Start fresh** — click "Start New Topic" to reset the conversation and research a new subject

---

## Features

- **Multi-agent orchestration** — five specialized agents with a supervisor router
- **MCP tool layer** — Tavily search exposed via a standardized MCP server, not direct SDK coupling
- **Stateful ReAct loops** — each sub-question gets its own search cycle; agents can retry on tool-call failures
- **Multi-turn memory** — SQLite-backed checkpointing; follow-ups revise the existing report without restarting the pipeline
- **Hallucination guards** — reviser agent uses live search for any new claims; output validation catches malformed LLM responses before they reach the report
- **LangSmith observability** — full trace visibility across every LLM call and tool invocation
- **Streamlit UI** — live streaming progress, inline markdown rendering, download button

---

## Acknowledgements

Built entirely on free-tier infrastructure:
- [Groq](https://groq.com) for fast, free LLM inference
- [Tavily](https://tavily.com) for web search
- [LangChain / LangGraph](https://www.langchain.com) for agent orchestration
- [FastMCP](https://github.com/jlowin/fastmcp) for the MCP server layer
- [LangSmith](https://smith.langchain.com) for tracing
