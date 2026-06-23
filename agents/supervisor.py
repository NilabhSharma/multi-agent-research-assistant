def supervisor_router(state):

    next_step = state.get("next", "END")
    print(f"[SUPERVISOR] Routing to: {next_step}")
    return next_step