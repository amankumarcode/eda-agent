import json
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from core.config import get_llm_config
from core.schema import AgentResult, NarratorResult
from core.state import AgentState

NARRATOR_SYSTEM_PROMPT: str = """You are a senior data analyst presenting findings to a business stakeholder — not a technical audience.

## Primary rule
Every insight must connect back to the user's original goal. Never present findings in isolation.

## Narrative structure
1. **Executive summary** — one paragraph that directly answers the user's goal with the most important finding.
2. **Key findings** — grouped by theme (e.g. "Income drivers", "Data quality"), NOT by which agent produced them.
3. **Caveats and next steps** — close with caveats and concrete recommended follow-up analyses.

## Specificity
Use actual numbers from the findings. Never say "income is high" when you can say "median income is $74,000 with a std of $27,000". Every key insight must contain at least one number.

## Honesty
If the analysis has gaps — low confidence scores, warnings from agents — surface them in caveats rather than hiding them. The stakeholder deserves an accurate picture of what the analysis can and cannot conclude.

## key_insights format
Return 3–7 bullet-ready strings. Each must:
- Start with a strong verb (e.g. "Reveals", "Shows", "Identifies", "Confirms")
- Contain at least one specific number or percentage

## recommended_next_steps format
Return 2–5 concrete follow-up analyses specific to what was found — not generic advice like "collect more data".
"""


def _get_all_findings(state: AgentState) -> dict:
    """Extract and organise findings from all agent results in state.

    Prefers critic-scored results (scored_results) over raw agent_results
    when available. Skips any findings that are not real dicts (e.g. critic
    summaries with string values like "dict(5 keys)").
    """
    results: List[AgentResult] = (
        state.get("scored_results") or state.get("agent_results", [])
    )

    profiler_findings: dict = {}
    stat_findings: dict = {}
    viz_findings: dict = {}

    for result in results:
        findings = result.findings
        # skip non-dict findings or abbreviated critic summaries
        if not isinstance(findings, dict):
            continue
        if any(isinstance(v, str) and "keys)" in str(v)
               for v in findings.values()):
            continue

        if result.agent_name == "profiler" and result.success:
            profiler_findings = findings
        elif result.agent_name == "stat_analyst" and result.success:
            stat_findings = findings
        elif result.agent_name == "viz_agent" and result.success:
            viz_findings = findings

    confidences = [
        r.confidence for r in results
        if r.success and isinstance(r.findings, dict)
        and not any(isinstance(v, str) and "keys)" in str(v)
                    for v in r.findings.values())
    ]
    overall_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    any_warnings = [w for r in results for w in (r.warnings or [])]
    chart_count = len(
        viz_findings.get("charts", []) if isinstance(viz_findings, dict) else []
    )

    return {
        "profiler": profiler_findings,
        "stat_analyst": stat_findings,
        "viz_agent": viz_findings,
        "chart_count": chart_count,
        "overall_confidence": overall_confidence,
        "any_warnings": any_warnings,
    }


def _build_narrator_prompt(request: object, all_findings: dict) -> str:
    """Build the human message for the narrator's structured output call."""
    goal = getattr(request, "goal", "")
    confidence = all_findings["overall_confidence"]
    warnings = all_findings["any_warnings"]

    profiler = all_findings["profiler"]
    stat = all_findings["stat_analyst"]
    viz = all_findings["viz_agent"]

    # Profiler summary — extract the most relevant sub-fields
    schema_info = profiler.get("schema_info", {})
    profiler_summary = {
        "shape": schema_info.get("shape"),
        "numeric_cols": schema_info.get("numeric_cols", []),
        "categorical_cols": schema_info.get("categorical_cols", []),
        "notable_columns": profiler.get("notable_columns", []),
        "distributions": profiler.get("distributions", {}),
        "data_quality": profiler.get("data_quality", {}),
        "null_report": profiler.get("null_report", {}),
    }

    # Stat summary
    stat_summary = {
        "correlations": stat.get("correlations", {}),
        "outliers": stat.get("outliers", {}),
        "feature_ranking": stat.get("feature_ranking", []),
        "notable_findings": stat.get("notable_findings", []),
    }

    # Viz summary
    viz_summary = {
        "chart_count": all_findings["chart_count"],
        "chart_descriptions": viz.get("chart_descriptions", []),
        "recommended_primary_chart": viz.get("recommended_primary_chart", ""),
    }

    parts = [
        f"## User goal",
        f"{goal}",
        "",
        f"## Overall analysis confidence",
        f"{confidence:.2f}",
        "",
        f"## Agent warnings",
        json.dumps(warnings, indent=2, default=str) if warnings else "None",
        "",
        f"## Profiler findings",
        json.dumps(profiler_summary, indent=2, default=str),
        "",
        f"## Statistical analyst findings",
        json.dumps(stat_summary, indent=2, default=str),
        "",
        f"## Visualisation agent findings",
        json.dumps(viz_summary, indent=2, default=str),
        "",
        "## Your task",
        "Synthesise all of the above into a NarratorResult.",
        "- The narrative must directly answer the user's goal.",
        "- key_insights must each contain actual numbers from the findings.",
        "- caveats must reflect any agent warnings and low-confidence areas.",
        "- recommended_next_steps must be specific to the findings, not generic.",
    ]
    return "\n".join(parts)


def narrator_node(state: AgentState) -> dict:
    """LangGraph node: narrative synthesis.

    Single LLM call — no tools, no ReAct loop.
    Synthesises all specialist findings into a coherent narrative
    anchored to the user's original goal, then sets next_action = "evaluate".
    """
    try:
        request = state["request"]
        all_findings = _get_all_findings(state)

        llm_config = get_llm_config()
        llm = ChatAnthropic(
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=8192,
        )

        structured_llm = llm.with_structured_output(NarratorResult)
        result: NarratorResult = structured_llm.invoke([
            SystemMessage(content=NARRATOR_SYSTEM_PROMPT),
            HumanMessage(content=_build_narrator_prompt(request, all_findings)),
        ])

        return {
            "narrative": result.narrative,
            "key_insights": result.key_insights,
            "caveats": result.caveats,
            "next_action": "evaluate",
            "completed_agents": ["narrator"],
            "messages": [
                AIMessage(
                    content=(
                        f"Narrator complete. "
                        f"{len(result.key_insights)} insights. "
                        f"{len(result.caveats)} caveats. "
                        f"Confidence: {all_findings['overall_confidence']:.2f}."
                    )
                )
            ],
        }

    except Exception as e:
        return {
            "narrative": "Narration failed due to an internal error.",
            "key_insights": [],
            "caveats": [f"Narrator node error: {str(e)}"],
            "next_action": "evaluate",
            "completed_agents": ["narrator"],
            "messages": [AIMessage(content=f"Narrator failed: {str(e)}")],
        }
