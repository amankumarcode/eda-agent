from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from core.state import AgentState
from graph.nodes.evaluator import evaluator_node
from graph.nodes.insight_critic import insight_critic_node
from graph.nodes.narrator import narrator_node
from graph.nodes.output_router import output_router_node
from graph.nodes.profiler import profiler_node
from graph.nodes.stat_analyst import stat_analyst_node
from graph.nodes.supervisor import supervisor_node
from graph.nodes.viz_agent import viz_agent_node


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_supervisor(state: AgentState):
    """Conditional edge function called after every supervisor execution.

    Returns a list of Send objects for parallel fan-out, or a single
    node name / END for serial routing. Unknown next_action defaults to END.
    """
    next_action = state.get("next_action", "")
    plan = state.get("plan")
    completed = state.get("completed_agents", [])
    rerun_agent = state.get("rerun_agent")
    rerun_count = state.get("rerun_count", 0)

    # Dispatch mode — fan out to agents via Send()
    if next_action == "dispatch":
        # Targeted rerun from critic
        if rerun_agent and rerun_count <= 1:
            return [Send(rerun_agent, state)]

        # Full parallel fan-out from plan
        if plan:
            agents_to_send = [a for a in plan.agents if a not in completed]
            if agents_to_send:
                return [Send(a, state) for a in agents_to_send]

    # Serial routing by next_action signal
    routing = {
        "critique": "insight_critic",
        "narrate":  "narrator",
        "evaluate": "evaluator",
        "output":   "output_router",
        "complete": END,
    }
    return routing.get(next_action, END)


def route_after_evaluator(state: AgentState) -> str:
    """Conditional edge function called after the evaluator node.

    Returns "supervisor" for a full replan, "output_router" otherwise.
    """
    next_action = state.get("next_action", "output")
    if next_action == "replan":
        return "supervisor"
    return "output_router"


def route_specialist_to_supervisor(state: AgentState) -> str:
    """Edge function used by specialist nodes — always reports to supervisor."""
    return "supervisor"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph(checkpointer):
    """Construct and compile the full EDA agent StateGraph.

    Topology: supervisor hub with parallel specialist fan-out via Send(),
    critic/narrator/evaluator as serial post-processing nodes, and an
    optional replan loop triggered by the evaluator's weak verdict.

    Args:
        checkpointer: A LangGraph checkpointer (SqliteSaver or
                      MemorySaver) for state persistence and HITL support.

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    builder = StateGraph(AgentState)

    # --- register all nodes ---
    builder.add_node("supervisor",     supervisor_node)
    builder.add_node("profiler",       profiler_node)
    builder.add_node("stat_analyst",   stat_analyst_node)
    builder.add_node("viz_agent",      viz_agent_node)
    builder.add_node("insight_critic", insight_critic_node)
    builder.add_node("narrator",       narrator_node)
    builder.add_node("evaluator",      evaluator_node)
    builder.add_node("output_router",  output_router_node)

    # --- entry point ---
    builder.add_edge(START, "supervisor")

    # --- supervisor hub: conditional fan-out or serial node routing ---
    builder.add_conditional_edges(
        "supervisor",
        route_after_supervisor,
        {
            "profiler":       "profiler",
            "stat_analyst":   "stat_analyst",
            "viz_agent":      "viz_agent",
            "insight_critic": "insight_critic",
            "narrator":       "narrator",
            "evaluator":      "evaluator",
            "output_router":  "output_router",
            "supervisor":     "supervisor",
            END:              END,
        },
    )

    # --- specialists always report back to supervisor ---
    builder.add_edge("profiler",     "supervisor")
    builder.add_edge("stat_analyst", "supervisor")
    builder.add_edge("viz_agent",    "supervisor")

    # --- critic and narrator report back to supervisor ---
    builder.add_edge("insight_critic", "supervisor")
    builder.add_edge("narrator",       "supervisor")

    # --- evaluator: conditional route to replan (supervisor) or output ---
    builder.add_conditional_edges(
        "evaluator",
        route_after_evaluator,
        {
            "supervisor":    "supervisor",
            "output_router": "output_router",
        },
    )

    # --- output_router is terminal ---
    builder.add_edge("output_router", END)

    # --- compile with checkpointer ---
    # HITL is handled by interrupt() inside supervisor_node itself;
    # interrupt_before is not needed and would fire an empty interrupt
    # before every supervisor invocation.
    return builder.compile(checkpointer=checkpointer)


def get_graph(checkpointer):
    """Public convenience function. Returns the compiled graph.

    This is the function imported by run.py and adapters.
    """
    return build_graph(checkpointer)
