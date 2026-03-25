from typing import List, Optional, Union

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import Send, interrupt

from core.config import get_agent_config, get_llm_config
from core.dataframe_registry import get as get_df
from core.schema import AnalysisPlan, EDARequest
from core.state import AgentState
from core.tool_registry import get_sample_rows, get_schema

SUPERVISOR_PLAN_PROMPT: str = """You are a senior data analyst planning an EDA investigation.

You have been given a dataset schema and sample rows. Your task is to generate a focused
analysis plan that directly targets the user's stated goal — not a generic EDA checklist.

## Agent selection rules
- Always include "profiler" — it is the foundation for all downstream analysis.
- Include "stat_analyst" if the goal involves relationships, correlations, drivers,
  predictions, or anomaly detection.
- Include "viz_agent" if the goal involves understanding distributions or patterns,
  or if a report or dashboard output format is requested.

## Plan quality rules
- steps must be human-readable, goal-specific, and ordered by logical dependency.
- parallel should group agents that can run concurrently — profiler, stat_analyst,
  and viz_agent can always run in parallel with each other.
- Do not pad the plan with generic steps like "collect data" or "clean data".
  Focus exclusively on analysis steps that serve the stated goal.
"""

SUPERVISOR_REPLAN_PROMPT: str = """You are a senior data analyst re-planning an EDA investigation.

A previous analysis attempt was evaluated as weak and must be improved. An independent
judge identified specific gaps and provided retry instructions — take them seriously.
They represent concrete deficiencies in the prior analysis.

## Your task
Generate a new, more targeted AnalysisPlan that directly addresses the retry instructions.
Be more specific than the original plan. Name the exact columns, relationships, or
outputs that must be covered.

## Agent selection rules
- Always include "profiler" — it is the foundation for all downstream analysis.
- Include "stat_analyst" if the goal involves relationships, correlations, drivers,
  predictions, or anomaly detection.
- Include "viz_agent" if the goal involves understanding distributions or patterns,
  or if a report or dashboard output format is requested.

## Plan quality rules
- steps must be human-readable, goal-specific, and ordered by logical dependency.
- parallel should group agents that can run concurrently — profiler, stat_analyst,
  and viz_agent can always run in parallel with each other.
"""

_SPECIALISTS = ["profiler", "stat_analyst", "viz_agent"]


def _inspect_dataset(df) -> dict:
    """Call get_schema and get_sample_rows directly to inspect the dataset.

    Returns {"schema": dict, "sample": list}. Never raises.
    """
    schema: dict = {}
    sample: list = []
    try:
        schema = get_schema(df)
    except Exception:
        pass
    try:
        sample = get_sample_rows(df, 3)
    except Exception:
        pass
    return {"schema": schema, "sample": sample}


def _generate_plan(
    llm,
    request: EDARequest,
    dataset_info: dict,
    retry_instructions: Optional[str] = None,
    available_agents: Optional[List[str]] = None,
) -> AnalysisPlan:
    """Make a structured LLM call to generate an AnalysisPlan.

    Uses SUPERVISOR_REPLAN_PROMPT when retry_instructions is provided,
    SUPERVISOR_PLAN_PROMPT otherwise.
    """
    schema = dataset_info.get("schema", {})
    sample = dataset_info.get("sample", [])[:3]

    parts = [
        f"## User goal",
        f"{request.goal}",
        "",
        f"## Dataset schema",
        f"- Shape: {schema.get('shape', 'unknown')}",
        f"- Columns: {schema.get('columns', [])}",
        f"- Numeric columns: {schema.get('numeric_cols', [])}",
        f"- Categorical columns: {schema.get('categorical_cols', [])}",
        f"- Dtypes: {schema.get('dtypes', {})}",
        "",
        f"## Sample rows (first {len(sample)})",
        str(sample),
    ]

    if available_agents is not None:
        parts += [
            "",
            f"## Available agents (only use these): {available_agents}",
        ]

    if retry_instructions:
        parts += [
            "",
            "## IMPROVEMENT REQUIRED",
            "The previous analysis was evaluated as weak. An independent judge identified",
            "the following specific gaps that MUST be addressed in the new plan:",
            "",
            retry_instructions,
            "",
            "Generate a more targeted plan that explicitly addresses the above gaps.",
        ]
    else:
        parts += [
            "",
            "Generate a focused AnalysisPlan that directly targets the user's goal.",
        ]

    human_message = "\n".join(parts)
    prompt = SUPERVISOR_REPLAN_PROMPT if retry_instructions else SUPERVISOR_PLAN_PROMPT
    structured_llm = llm.with_structured_output(AnalysisPlan)
    plan: AnalysisPlan = structured_llm.invoke([
        SystemMessage(content=prompt),
        HumanMessage(content=human_message),
    ])

    # Enforce session linkage
    plan = plan.model_copy(update={
        "session_id": request.session_id,
        "output_formats": request.output_formats,
    })
    return plan


def _format_plan_for_display(plan: AnalysisPlan) -> str:
    """Format an AnalysisPlan as readable text for HITL display."""
    steps_text = "\n".join(
        f"  {i + 1}. {step}" for i, step in enumerate(plan.steps)
    )
    parallel_text = ", ".join(
        f"[{', '.join(group)}]" for group in plan.parallel
    )
    formats_text = ", ".join(str(f) for f in plan.output_formats)

    return (
        f"Goal: {plan.goal}\n\n"
        f"Steps:\n{steps_text}\n\n"
        f"Agents to run: {', '.join(plan.agents)}\n"
        f"Parallel groups: {parallel_text}\n"
        f"Output formats: {formats_text}"
    )


def _all_dispatched_complete(state: AgentState) -> bool:
    """Return True if every agent in dispatched_agents appears in completed_agents."""
    dispatched: List[str] = state.get("dispatched_agents", [])
    completed: List[str] = state.get("completed_agents", [])
    if not dispatched:
        return False
    return all(a in completed for a in dispatched)


def _determine_mode(state: AgentState) -> str:
    """Determine which operational mode the supervisor should run in."""
    plan = state.get("plan")
    plan_approved = state.get("plan_approved", False)
    next_action = state.get("next_action")
    completed: List[str] = state.get("completed_agents", [])
    dispatched: List[str] = state.get("dispatched_agents", [])
    rerun_agent = state.get("rerun_agent")
    rerun_count = state.get("rerun_count", 0)

    # no plan yet — first invocation
    if plan is None or not plan_approved:
        return "plan"

    # evaluator explicitly asked for replan
    if next_action == "replan":
        return "replan"

    # explicit routing signals from other nodes — always honour these
    if next_action in ("critique", "narrate", "evaluate",
                       "output", "complete"):
        return "route"

    # critic wants a targeted rerun
    if (rerun_agent and rerun_count <= 1 and
            rerun_agent not in completed):
        return "dispatch"

    # check if all planned agents have completed
    planned_agents = plan.agents if plan else []
    all_completed = all(a in completed for a in planned_agents)

    if all_completed:
        # all agents done — route to critique if not done yet
        if "insight_critic" not in completed:
            return "route"  # _handle_route_mode will set "critique"
        if "narrator" not in completed:
            return "route"
        if "evaluator" not in completed:
            return "route"
        return "route"  # output_router

    # agents dispatched but not all done yet — check if we
    # need to dispatch remaining agents
    already_dispatched = set(dispatched)
    need_dispatch = [a for a in planned_agents
                     if a not in already_dispatched]

    if need_dispatch:
        return "dispatch"

    # all dispatched, waiting for completion — but since we are
    # being called, some agent just completed. Check again:
    if all_completed:
        return "route"

    # still waiting — agents are running, this is a spurious
    # supervisor call from a completing agent. Route to check.
    return "route"


def _handle_plan_mode(state: AgentState) -> dict:
    """MODE 1 — Generate plan and pause for HITL approval."""
    request = state["request"]
    df = get_df(request.session_id)
    if df is None:
        raise ValueError(
            f"DataFrame not found for session {request.session_id}. "
            "Registry may have been cleared."
        )

    llm_config = get_llm_config()
    llm = ChatAnthropic(
        model=llm_config.model,
        temperature=llm_config.temperature,
    )

    agent_config = get_agent_config()
    available_agents = agent_config.agents_required or _SPECIALISTS

    dataset_info = _inspect_dataset(df)
    plan = _generate_plan(llm, request, dataset_info, available_agents=available_agents)

    # Filter plan agents to only those enabled by config
    filtered_agents = [a for a in plan.agents if a in available_agents]
    if not filtered_agents:
        filtered_agents = available_agents
    filtered_parallel = [
        [a for a in group if a in available_agents]
        for group in plan.parallel
    ]
    filtered_parallel = [g for g in filtered_parallel if g]
    if not filtered_parallel:
        filtered_parallel = [filtered_agents]
    plan = plan.model_copy(update={
        "agents": filtered_agents,
        "parallel": filtered_parallel,
    })

    # HITL interrupt — graph pauses here until resumed
    human_input = interrupt({
        "plan": plan.model_dump(),
        "display": _format_plan_for_display(plan),
        "message": (
            "Please review the analysis plan above. "
            "Type 'approve' to proceed or provide feedback to revise the plan."
        ),
    })

    response = str(human_input).strip().lower()
    approved = response in ("y", "yes", "approve", "approved", "ok",
                            "looks good", "proceed")

    if approved:
        return {
            "plan": plan,
            "plan_approved": True,
            "next_action": "dispatch",
            "messages": [
                AIMessage(
                    content=f"Plan approved. Dispatching: {', '.join(plan.agents)}."
                )
            ],
        }
    else:
        return {
            "plan": None,
            "plan_approved": False,
            "messages": [
                AIMessage(content="Plan rejected. Re-planning with your feedback."),
                HumanMessage(content=str(human_input)),
            ],
        }


def _handle_dispatch_mode(state: AgentState) -> dict:
    """MODE 2 — Record which agents are being dispatched.

    Returns a dict that updates state. The actual Send() objects for
    parallel fan-out are produced by route_after_supervisor() in builder.py.
    """
    plan = state.get("plan")
    dispatched: List[str] = state.get("dispatched_agents", [])
    rerun_agent = state.get("rerun_agent")
    rerun_count = state.get("rerun_count", 0)

    # targeted rerun from critic
    if rerun_agent and rerun_count <= 1:
        agents_to_dispatch = [rerun_agent]
    else:
        # only dispatch agents not yet dispatched
        planned: List[str] = plan.agents if plan else []
        agents_to_dispatch = [a for a in planned if a not in dispatched]

    if not agents_to_dispatch:
        # nothing to dispatch — transition to route
        return {
            "next_action": "critique",
            "messages": [
                AIMessage(content="All agents already dispatched. Moving to critique.")
            ],
        }

    return {
        "dispatched_agents": agents_to_dispatch,
        "next_action": "dispatch",
        "messages": [
            AIMessage(content=f"Dispatching agents: {', '.join(agents_to_dispatch)}.")
        ],
    }


def _handle_replan_mode(state: AgentState) -> dict:
    """MODE 4 — Re-generate plan using evaluator's retry_instructions."""
    request = state["request"]
    df = get_df(request.session_id)
    if df is None:
        raise ValueError(
            f"DataFrame not found for session {request.session_id}. "
            "Registry may have been cleared."
        )

    retry_instructions: Optional[str] = None
    if state.get("evaluation"):
        retry_instructions = state["evaluation"].retry_instructions

    llm_config = get_llm_config()
    llm = ChatAnthropic(
        model=llm_config.model,
        temperature=llm_config.temperature,
    )

    agent_config = get_agent_config()
    available_agents = agent_config.agents_required or _SPECIALISTS

    dataset_info = _inspect_dataset(df)
    plan = _generate_plan(llm, request, dataset_info, retry_instructions, available_agents)

    # Filter plan agents to only those enabled by config
    filtered_agents = [a for a in plan.agents if a in available_agents]
    if not filtered_agents:
        filtered_agents = available_agents
    filtered_parallel = [
        [a for a in group if a in available_agents]
        for group in plan.parallel
    ]
    filtered_parallel = [g for g in filtered_parallel if g]
    if not filtered_parallel:
        filtered_parallel = [filtered_agents]
    plan = plan.model_copy(update={
        "agents": filtered_agents,
        "parallel": filtered_parallel,
    })

    return {
        "plan": plan,
        "plan_approved": True,
        "next_action": "dispatch",
        "rerun_agent": None,
        "messages": [
            AIMessage(
                content=(
                    f"Re-planned based on evaluator feedback. "
                    f"New focus: {plan.steps[0] if plan.steps else 'revised'}."
                )
            )
        ],
    }


def _handle_route_mode(state: AgentState) -> dict:
    """MODE 3 — Determine next serial node based on what has completed."""
    plan = state.get("plan")
    completed: List[str] = state.get("completed_agents", [])
    next_action = state.get("next_action")

    # honour explicit signal from another node first
    explicit_map = {
        "critique": "critique",
        "narrate":  "narrate",
        "evaluate": "evaluate",
        "output":   "output",
        "complete": "complete",
    }
    if next_action in explicit_map:
        action = explicit_map[next_action]
        return {
            "next_action": action,
            "messages": [AIMessage(content=f"Supervisor routing to: {action}.")],
        }

    # infer from what has completed
    planned_agents: List[str] = plan.agents if plan else []
    all_agents_done = all(a in completed for a in planned_agents)

    if all_agents_done and "insight_critic" not in completed:
        return {
            "next_action": "critique",
            "messages": [AIMessage(content="All specialists complete. Routing to critic.")],
        }

    if "insight_critic" in completed and "narrator" not in completed:
        return {
            "next_action": "narrate",
            "messages": [AIMessage(content="Critic complete. Routing to narrator.")],
        }

    if "narrator" in completed and "evaluator" not in completed:
        return {
            "next_action": "evaluate",
            "messages": [AIMessage(content="Narrator complete. Routing to evaluator.")],
        }

    if "evaluator" in completed:
        return {
            "next_action": "output",
            "messages": [AIMessage(content="Evaluator complete. Routing to output.")],
        }

    # fallback
    return {
        "next_action": "complete",
        "messages": [AIMessage(content="Supervisor: no further actions. Completing.")],
    }


def supervisor_node(state: AgentState) -> dict:
    """LangGraph node: supervisor hub.

    Runs multiple times per graph execution. Determines its own mode
    from state signals and dispatches to the appropriate handler.
    """
    mode = _determine_mode(state)
    print(
        f"[SUPERVISOR] mode={mode} "
        f"next_action={state.get('next_action')} "
        f"completed={state.get('completed_agents', [])} "
        f"dispatched={state.get('dispatched_agents', [])}"
    )

    if mode == "plan":
        return _handle_plan_mode(state)
    elif mode == "dispatch":
        return _handle_dispatch_mode(state)
    elif mode == "replan":
        return _handle_replan_mode(state)
    elif mode == "route":
        return _handle_route_mode(state)
    else:  # "wait" — should not normally occur in a well-wired graph
        return {
            "messages": [
                AIMessage(content="Supervisor waiting for agents to complete.")
            ],
        }
