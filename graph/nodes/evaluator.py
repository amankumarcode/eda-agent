import json
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.config import get_agent_config, get_llm_config
from core.schema import AgentResult, EvaluationResult, Verdict
from core.state import AgentState

EVALUATOR_SYSTEM_PROMPT: str = """You are an independent quality judge of data analysis. You are not part of the analysis team and have no stake in the outcome.

## Primary question
Did this analysis genuinely answer the user's stated goal — or did it produce generic EDA observations that happen to mention the goal?

## Scoring dimensions

**goal_coverage (weight 0.5)**
- Does the narrative directly and specifically address the user's goal?
- Are the key insights goal-relevant, or are they generic dataset observations?
- A narrative that says "income varies by region" when the goal is "find revenue drivers" scores low here.

**insight_quality (weight 0.3)**
- Are insights specific and quantified with actual numbers?
- Vague observations like "income varies widely" score low.
- Specific observations like "age explains 85% of income variance (r=0.85)" score high.

**evidence_quality (weight 0.2)**
- Is every claim in the narrative traceable to agent findings?
- Are there unsupported assertions or invented statistics?
- Claims without corresponding agent evidence score low.

## Verdict thresholds
- **strong**: overall_score >= 0.75
- **adequate**: overall_score >= 0.5
- **weak**: overall_score < 0.5

## retry_instructions
Set retry_instructions ONLY when verdict is weak. It must be specific and actionable:
- Name the exact insights that are missing
- Name the specific claims that lack evidence
- Do NOT write "do better" or "improve the analysis"

## Scoring stance
Be adversarial. A score of 0.8+ should be rare and reserved for analyses that directly and specifically answer the goal with quantified, evidence-backed insights. When in doubt, score lower.

Compute: overall_score = goal_coverage * 0.5 + insight_quality * 0.3 + evidence_quality * 0.2
"""


def _build_evaluator_prompt(request: object, state: AgentState) -> str:
    """Build the human message for the evaluator's structured output call."""
    goal = getattr(request, "goal", "")
    narrative = state.get("narrative", "") or ""
    key_insights: List[str] = state.get("key_insights", []) or []
    caveats: List[str] = state.get("caveats", []) or []
    agent_results: List[AgentResult] = state.get("agent_results", []) or []

    # Evidence summary — keys + types, not full dump
    evidence_parts = []
    confidences = []
    for result in agent_results:
        if result.success:
            confidences.append(result.confidence)
        findings_summary = {}
        for key, val in result.findings.items():
            if isinstance(val, dict):
                findings_summary[key] = f"dict({len(val)} keys)"
            elif isinstance(val, list):
                findings_summary[key] = f"list({len(val)} items)"
            else:
                findings_summary[key] = repr(val)[:80]
        evidence_parts.append({
            "agent_name": result.agent_name,
            "confidence": result.confidence,
            "warnings": result.warnings,
            "findings_summary": findings_summary,
        })
    overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    insights_numbered = "\n".join(
        f"{i + 1}. {insight}" for i, insight in enumerate(key_insights)
    )
    caveats_numbered = "\n".join(
        f"{i + 1}. {caveat}" for i, caveat in enumerate(caveats)
    )

    parts = [
        "## Section 1 — Original goal",
        goal,
        "",
        "## Section 2 — Narrator output",
        f"### Narrative\n{narrative}",
        "",
        f"### Key insights\n{insights_numbered or '(none)'}",
        "",
        f"### Caveats\n{caveats_numbered or '(none)'}",
        "",
        "## Section 3 — Evidence summary",
        f"Overall mean agent confidence: {overall_confidence:.2f}",
        json.dumps(evidence_parts, indent=2, default=str),
        "",
        "## Section 4 — Evaluation instructions",
        "Score each dimension 0.0–1.0.",
        "Compute: overall_score = goal_coverage * 0.5 + insight_quality * 0.3 + evidence_quality * 0.2",
        "Set verdict: strong (>=0.75), adequate (>=0.5), weak (<0.5).",
        "Set retry_instructions ONLY if verdict is weak — name exact missing insights or unsupported claims.",
    ]
    return "\n".join(parts)


def evaluator_node(state: AgentState) -> dict:
    """LangGraph node: LLM-as-Judge evaluator.

    Single LLM call — no tools, no ReAct loop.
    Evaluates the full analysis against the user's goal and sets
    next_action to "output" or "replan" based on verdict.
    Skips entirely (zero LLM calls) when EVALUATOR_ENABLED=false.
    """
    request = state["request"]
    evaluation_count: int = state.get("evaluation_count", 0)

    agent_config = get_agent_config()
    if not agent_config.evaluator_enabled:
        return {
            "evaluation": None,
            "evaluation_count": 0,
            "next_action": "output",
            "completed_agents": ["evaluator"],
            "messages": [
                AIMessage(
                    content="Evaluator disabled via config. Proceeding directly to output."
                )
            ],
        }

    try:
        llm_config = get_llm_config()
        llm = ChatAnthropic(
            model=llm_config.model,
            temperature=llm_config.temperature,
        )

        structured_llm = llm.with_structured_output(EvaluationResult)
        evaluation: EvaluationResult = structured_llm.invoke([
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=_build_evaluator_prompt(request, state)),
        ])

        new_count = evaluation_count + 1
        if evaluation.verdict == Verdict.WEAK and new_count < 2:
            next_action = "replan"
        else:
            next_action = "output"

        msg = (
            f"Evaluator complete. "
            f"Score: {evaluation.overall_score:.2f}. "
            f"Verdict: {evaluation.verdict}. "
            f"Goal coverage: {evaluation.goal_coverage:.2f}. "
            f"Insight quality: {evaluation.insight_quality:.2f}. "
            f"Evidence quality: {evaluation.evidence_quality:.2f}. "
            f"Next: {next_action}."
        )

        return {
            "evaluation": evaluation,
            "evaluation_count": new_count,
            "next_action": next_action,
            "completed_agents": ["evaluator"],
            "messages": [AIMessage(content=msg)],
        }

    except Exception as e:
        return {
            "evaluation": EvaluationResult(
                goal_coverage=0.0,
                insight_quality=0.0,
                evidence_quality=0.0,
                overall_score=0.0,
                strengths=[],
                gaps=[f"Evaluator node failed: {str(e)}"],
                verdict=Verdict.WEAK,
                retry_instructions="Evaluator failed — re-run full pipeline.",
            ),
            "evaluation_count": evaluation_count + 1,
            "next_action": "output",
            "completed_agents": ["evaluator"],
            "messages": [
                AIMessage(
                    content=f"Evaluator failed: {str(e)}. Defaulting to output."
                )
            ],
        }
