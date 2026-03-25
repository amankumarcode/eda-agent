import operator
from typing import Annotated, Optional

from typing_extensions import TypedDict

from core.schema import AgentResult, AnalysisPlan, EDARequest, EvaluationResult


class AgentState(TypedDict):
    # Set at entry by the adapter, never mutated after
    request: EDARequest

    # Set by supervisor node
    plan: Optional[AnalysisPlan]
    plan_approved: bool

    # Specialist agent results — operator.add reducer enables safe parallel fan-out
    agent_results: Annotated[list[AgentResult], operator.add]

    # Critic-scored results — plain list, last-write-wins (no reducer)
    scored_results: list[AgentResult]

    # Set by insight_critic node
    rerun_agent: Optional[str]
    rerun_count: int

    # Set by narrator node
    narrative: Optional[str]
    key_insights: list[str]
    caveats: list[str]

    # Set by output_router node
    output_paths: dict[str, str]

    # Append-only message history
    messages: Annotated[list, operator.add]

    # Set by evaluator node
    evaluation: Optional[EvaluationResult]
    evaluation_count: int

    # Supervisor hub routing fields (Option B dynamic hub pattern)
    # Agents the supervisor has dispatched; append-only via operator.add
    dispatched_agents: Annotated[list[str], operator.add]
    # Agents that have reported back with a result; append-only via operator.add
    completed_agents: Annotated[list[str], operator.add]
    # Supervisor's declared intent for its next invocation
    # Valid values: "dispatch", "critique", "narrate", "evaluate",
    #               "output", "complete", "replan"
    next_action: Optional[str]
