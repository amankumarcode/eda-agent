import json
from typing import Any, Dict, List, Optional, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from core.config import get_llm_config
from core.schema import (
    AgentResult,
    EDARequest,
    ProfilerFindings,
    StatFindings,
    VizFindings,
)
from core.dataframe_registry import get as get_df
from core.state import AgentState
from core.tool_registry import VIZ_TOOLS, inject_dataframe

VIZ_AGENT_SYSTEM_PROMPT: str = """You are a data visualisation specialist.

Your task is to generate Plotly chart specifications that best communicate the
patterns and insights found in the dataset.

Guidelines:
- Use profiler and statistical findings to select the most informative charts.
- Always generate at least 3 charts: one distribution, one relationship, and one
  overview (correlation heatmap or summary).
- Prefer charts that directly address the user's stated goal.
- Avoid redundant charts — each chart should answer a distinct question.
- All chart specs must be JSON-serialisable dicts with 'data' and 'layout' keys.
- Express chart_descriptions as plain English captions (one per chart).
- Set recommended_primary_chart to the title of the single most important chart.
"""

VIZ_AGENT_REACT_PROMPT: str = """You are a data visualisation specialist operating in ReAct mode.

Use the available tools to generate Plotly chart specifications for the dataset.
Base chart selection on the profiler and statistical findings provided in the human message.

Generate at least 3 charts:
1. A distribution chart (histogram or boxplot) for the most important numeric column
2. A scatter plot showing the key relationship identified by the stat analyst
3. A correlation heatmap for all numeric columns

Use make_histogram_spec, make_scatter_spec, make_correlation_heatmap_spec, and
make_boxplot_spec as needed. Return all chart specs as part of your final analysis.
"""


def _get_upstream_findings(
    state: AgentState,
) -> Tuple[Optional[ProfilerFindings], Optional[StatFindings]]:
    """Extract ProfilerFindings and StatFindings from completed agent_results.

    Returns (None, None) if the upstream agents have not yet completed.
    Safe to call when agents run in parallel — absent results return None.
    """
    profiler_findings: Optional[ProfilerFindings] = None
    stat_findings: Optional[StatFindings] = None

    for result in state.get("agent_results", []):
        if result.agent_name == "profiler" and result.success:
            try:
                profiler_findings = ProfilerFindings(**result.findings)
            except Exception:
                pass
        elif result.agent_name == "stat_analyst" and result.success:
            try:
                stat_findings = StatFindings(**result.findings)
            except Exception:
                pass

    return profiler_findings, stat_findings


def _build_viz_input(
    request: EDARequest,
    profiler_findings: Optional[ProfilerFindings],
    stat_findings: Optional[StatFindings],
) -> str:
    """Build the human message content for the ReAct viz agent."""
    parts = [f"User goal: {request.goal}"]

    if profiler_findings:
        notable = profiler_findings.notable_columns
        schema_info = profiler_findings.schema_info
        numeric_cols = schema_info.get("numeric_cols", [])
        categorical_cols = schema_info.get("categorical_cols", [])
        parts.append(f"Notable columns from profiler: {notable}")
        parts.append(f"Numeric columns: {numeric_cols}")
        parts.append(f"Categorical columns: {categorical_cols}")

    if stat_findings:
        parts.append(f"Key statistical findings: {stat_findings.notable_findings}")
        if stat_findings.feature_ranking:
            top_features = [
                f["feature"] for f in stat_findings.feature_ranking[:3]
                if isinstance(f, dict) and "feature" in f
            ]
            parts.append(f"Top features by importance: {top_features}")

    parts.append(
        "\nGenerate chart specifications using the available tools. "
        "Create at least 3 charts that together tell the data story relevant to the goal."
    )

    return "\n".join(parts)


def _extract_chart_specs(agent_messages: List[Any]) -> List[Dict[str, Any]]:
    """Scan agent message history for Plotly chart specs.

    A chart spec is any dict with both 'data' and 'layout' keys.
    Never raises — malformed content is silently skipped.
    """
    charts: List[Dict[str, Any]] = []

    for message in agent_messages:
        content = getattr(message, "content", None)
        if not content:
            continue

        # Try to parse JSON blobs embedded in text
        if isinstance(content, str):
            # Look for JSON objects in the text
            depth = 0
            start = -1
            for i, ch in enumerate(content):
                if ch == "{":
                    if depth == 0:
                        start = i
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0 and start != -1:
                        candidate = content[start : i + 1]
                        try:
                            obj = json.loads(candidate)
                            if isinstance(obj, dict) and "data" in obj and "layout" in obj:
                                charts.append(obj)
                        except (json.JSONDecodeError, ValueError):
                            pass
                        start = -1

        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and "data" in item and "layout" in item:
                    charts.append(item)

    return charts


def viz_agent_node(state: AgentState) -> dict:
    """LangGraph node: data visualisation specialist.

    Always uses the ReAct pattern — chart generation benefits from
    iterative tool use to explore which charts best serve the goal.
    """
    try:
        request: EDARequest = state["request"]
        df = get_df(request.session_id)
        if df is None:
            raise ValueError(
                f"DataFrame not found for session {request.session_id}. "
                "Registry may have been cleared."
            )

        inject_dataframe(df)

        profiler_findings, stat_findings = _get_upstream_findings(state)

        llm_config = get_llm_config()
        llm = ChatAnthropic(
            model=llm_config.model,
            temperature=llm_config.temperature,
        )

        agent = create_react_agent(llm, VIZ_TOOLS)
        human_content = _build_viz_input(request, profiler_findings, stat_findings)

        response = agent.invoke({
            "messages": [
                SystemMessage(content=VIZ_AGENT_REACT_PROMPT),
                HumanMessage(content=human_content),
            ]
        })

        agent_messages = response.get("messages", [])
        final_text = agent_messages[-1].content if agent_messages else ""
        chart_specs = _extract_chart_specs(agent_messages)

        structured_llm = llm.with_structured_output(VizFindings)
        findings: VizFindings = structured_llm.invoke([
            SystemMessage(content=VIZ_AGENT_SYSTEM_PROMPT),
            HumanMessage(
                content=(
                    f"Based on this visualisation analysis:\n{final_text}\n\n"
                    f"Chart specs found: {len(chart_specs)}\n"
                    "Extract the findings into VizFindings."
                )
            ),
        ])

        # Confidence: 0.9 if ≥3 charts generated, 0.6 otherwise
        confidence = 0.9 if len(findings.charts) >= 3 else 0.6

        result = AgentResult(
            agent_name="viz_agent",
            success=True,
            findings=findings.model_dump(),
            confidence=confidence,
        )
        return {
            "agent_results": [result],
            "completed_agents": ["viz_agent"],
            "messages": [
                HumanMessage(
                    content=(
                        f"Viz agent complete (react mode). "
                        f"Generated {len(findings.charts)} chart(s)."
                    )
                )
            ],
        }

    except Exception as e:
        return {
            "agent_results": [AgentResult(
                agent_name="viz_agent",
                success=False,
                findings={},
                confidence=0.0,
                warnings=[f"Viz agent node failed: {str(e)}"],
            )],
            "completed_agents": ["viz_agent"],
            "messages": [
                HumanMessage(content=f"Viz agent failed: {str(e)}")
            ],
        }
