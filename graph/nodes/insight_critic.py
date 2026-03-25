import json
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.config import get_llm_config
from core.schema import AgentResult, CriticOutput
from core.state import AgentState

CRITIC_SYSTEM_PROMPT: str = """You are an independent quality reviewer of data analysis results.

Your stance is adversarial by design: assume the analysis is incomplete until proven otherwise.
Your job is to find gaps and weaknesses — not to confirm quality.

## What to evaluate for each specialist

**Profiler**
- Did it cover every column in the dataset?
- Are nulls, distributions, and data quality all present in findings?
- Are notable_columns justified with evidence, not just listed?

**Stat analyst**
- Are correlations computed for all numeric column pairs?
- Are outlier findings actionable (specific counts, which rows, magnitude)?
- Are notable_findings specific and data-backed — not generic observations like "some correlation exists"?

**Viz agent**
- Do charts directly address the user's stated goal?
- Are there at least 3 charts?
- Is the recommended_primary_chart justified given the goal?

## Confidence scoring (assign per agent)
- 0.9–1.0: thorough, specific, well-evidenced — no meaningful gaps
- 0.7–0.8: adequate but missing some depth or specificity
- 0.5–0.6: surface level, important gaps present
- below 0.5: significant gaps — re-run strongly recommended

## Re-run recommendation
- Recommend a re-run for AT MOST ONE agent — the one with the lowest confidence if it scores below 0.6.
- Set rerun_reason as a specific, actionable instruction for that agent (e.g. "Generate at least 3 charts: a histogram for income, a scatter of age vs income, and a correlation heatmap").
- Do NOT write vague complaints like "analysis was insufficient".
- Set rerun_agent to null if ALL agents score >= 0.6.
"""

_SPECIALISTS = ["profiler", "stat_analyst", "viz_agent"]


def _format_agent_results(agent_results: List[AgentResult]) -> str:
    """Format agent results for the LLM prompt — summaries only, never full findings."""
    lines = []
    for r in agent_results:
        findings_summary = {
            k: (f"dict({len(v)} keys)" if isinstance(v, dict)
                else f"list({len(v)} items)" if isinstance(v, list)
                else str(v)[:100])
            for k, v in r.findings.items()
        }
        lines.append(
            f"Agent: {r.agent_name}\n"
            f"Success: {r.success}\n"
            f"Confidence: {r.confidence}\n"
            f"Warnings: {r.warnings}\n"
            f"Findings summary: {json.dumps(findings_summary, indent=2)}\n"
        )
    return "\n---\n".join(lines)


def _build_critic_prompt(
    request: object,
    agent_results: List[AgentResult],
    formatted_results: str,
) -> str:
    """Build the human message for the critic's structured output call."""
    goal = getattr(request, "goal", "")
    n = len(agent_results)
    parts = [
        f"User's original goal: {goal}",
        f"{n} of 3 specialists complete.",
        "",
        "Agent results:",
        formatted_results,
        "",
        "Instructions:",
        "- Score each agent's confidence (0.0–1.0) using the rubric in the system prompt.",
        "- Identify the weakest agent.",
        "- Recommend a re-run ONLY if that agent scores below 0.6.",
        "- If recommending a re-run, set rerun_reason to a specific, actionable instruction.",
        "- If all agents score >= 0.6, set rerun_agent to null and rerun_reason to null.",
    ]
    return "\n".join(parts)


def insight_critic_node(state: AgentState) -> dict:
    """LangGraph node: independent quality reviewer.

    Single LLM call — no tools, no ReAct loop.
    Scores each specialist's results, flags weak conclusions,
    and sets next_action / rerun_agent for the supervisor's next turn.
    """
    try:
        request = state["request"]
        all_results: List[AgentResult] = state.get("agent_results", [])
        relevant = [r for r in all_results if r.agent_name in _SPECIALISTS]

        if not relevant:
            return {
                "next_action": "narrate",
                "completed_agents": ["insight_critic"],
                "messages": [
                    AIMessage(
                        content="Critic: no specialist results to review, proceeding to narration."
                    )
                ],
            }

        llm_config = get_llm_config()
        llm = ChatAnthropic(
            model=llm_config.model,
            temperature=llm_config.temperature,
        )

        structured_llm = llm.with_structured_output(CriticOutput)
        formatted = _format_agent_results(relevant)
        critic_output: CriticOutput = structured_llm.invoke([
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=_build_critic_prompt(request, relevant, formatted)),
        ])

        next_action = "dispatch" if critic_output.rerun_agent is not None else "narrate"

        # Preserve original findings — only update confidence, warnings, rerun_hint
        original_map = {r.agent_name: r for r in relevant}
        updated_results = []
        for scored in critic_output.scored_results:
            original = original_map.get(scored.agent_name)
            updated_results.append(AgentResult(
                agent_name=scored.agent_name,
                success=scored.success,
                findings=original.findings if original else scored.findings,
                confidence=scored.confidence,
                warnings=scored.warnings,
                rerun_hint=(
                    critic_output.rerun_reason
                    if scored.agent_name == critic_output.rerun_agent
                    else None
                ),
            ))

        return {
            "scored_results": updated_results,
            "rerun_agent": critic_output.rerun_agent,
            "rerun_count": state.get("rerun_count", 0) + 1,
            "next_action": next_action,
            "completed_agents": ["insight_critic"],
            "messages": [
                AIMessage(
                    content=(
                        f"Critic complete. Overall quality: "
                        f"{critic_output.overall_quality:.2f}. "
                        f"Rerun: {critic_output.rerun_agent or 'none'}. "
                        f"Next: {next_action}."
                    )
                )
            ],
        }

    except Exception as e:
        return {
            "next_action": "narrate",
            "completed_agents": ["insight_critic"],
            "messages": [
                AIMessage(
                    content=f"Critic failed: {str(e)}. Proceeding to narration."
                )
            ],
        }
