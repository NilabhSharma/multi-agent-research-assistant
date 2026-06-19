def supervisor_router(state):
    """
    Lightweight router: simply reads the 'next' field that the previous
    node already decided, and tells LangGraph where to go.

    This is NOT an LLM call - it's a plain Python function. Since our
    pipeline order is fixed (planner -> web_searcher -> summarizer ->
    writer -> END), each node already knows what should happen after it,
    so the supervisor's only job is to read that decision and act on it.
    """
    next_step = state.get("next", "END")
    print(f"[SUPERVISOR] Routing to: {next_step}")
    return next_step