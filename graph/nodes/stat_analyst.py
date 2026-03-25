import json
from typing import Any, List, Optional, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from core.config import get_agent_config, get_llm_config
from core.dataframe_registry import get as get_df
from core.schema import AgentResult, EDARequest, ProfilerFindings, StatFindings
from core.state import AgentState
from core.tool_registry import (
    STATISTICAL_TOOLS,
    detect_outliers,
    get_correlation_matrix,
    get_crosstab,
    get_feature_importance_proxy,
    get_kurtosis_report,
    get_schema,
    get_skewness_report,
    get_value_counts,
    run_normality_test,
)
from graph.nodes.profiler import _safe_call

STAT_ANALYST_SYSTEM_PROMPT: str = """You are a statistical analysis specialist.

Your task is to surface statistically meaningful patterns, anomalies, and
relationships in the data by interpreting raw tool outputs provided to you.

Guidelines:
- Prioritise analysis of columns the profiler flagged as notable — these are
  listed explicitly in the human message.
- Ensure coverage of all numeric columns for: correlation analysis, outlier
  detection, normality testing, skewness, and kurtosis.
- If a target column can be inferred from the user goal, include feature
  importance proxy results in your findings.
- For the ReAct mode only: use run_python_repl for any custom statistical
  analysis that the structured tools cannot handle directly.
- Extract raw results into a StatFindings structure with notable_findings
  expressed as 3–7 plain English observations, not raw numbers.
"""

STAT_ANALYST_REACT_PROMPT: str = """You are a statistical analysis specialist operating in ReAct mode.

Reason step by step about which analyses matter most for the stated goal.
Use findings from the profiler (provided in the human message) to skip
redundant analysis and focus on the most informative columns.

Available tools cover correlations, outlier detection, normality tests,
skewness, kurtosis, value counts, and cross-tabulations. Use run_python_repl
for any analysis the structured tools cannot handle — always end REPL code
with: print(json.dumps(your_result))

Return a thorough analysis that can be structured into StatFindings with
meaningful notable_findings observations.
"""


def _get_profiler_findings(state: AgentState) -> Optional[ProfilerFindings]:
    """Extract ProfilerFindings from agent_results already in state.

    Returns None if the profiler result is absent or could not be parsed
    (stat_analyst may run in parallel before profiler completes).
    """
    for result in state.get("agent_results", []):
        if result.agent_name == "profiler" and result.success:
            try:
                return ProfilerFindings(**result.findings)
            except Exception:
                return None
    return None


def _collect_warnings(raw_results: dict) -> List[str]:
    """Collect _error entries from top-level and nested tool results."""
    warnings: List[str] = [
        f"{key}: {val['_error']}"
        for key, val in raw_results.items()
        if isinstance(val, dict) and "_error" in val
    ]
    for group_key in ("outliers", "normality", "value_counts"):
        group = raw_results.get(group_key, {})
        if isinstance(group, dict):
            for col, result in group.items():
                if isinstance(result, dict) and "_error" in result:
                    warnings.append(f"{group_key}.{col}: {result['_error']}")
    return warnings


def _run_all_stat_tools(
    df: Any,
    numeric_cols: List[str],
    categorical_cols: List[str],
    notable_columns: List[str],
    goal: str = "",
) -> dict:
    """Run every statistical tool directly and collect results.

    Never raises — individual tool failures are captured as {"_error": ...}.
    """
    raw: dict = {
        "correlation_matrix": _safe_call(get_correlation_matrix, df),
        "skewness_report":    _safe_call(get_skewness_report, df),
        "kurtosis_report":    _safe_call(get_kurtosis_report, df),
        "outliers": {
            col: _safe_call(detect_outliers, df, col)
            for col in numeric_cols
        },
        "normality": {
            col: _safe_call(run_normality_test, df, col)
            for col in numeric_cols
        },
        "value_counts": {
            col: _safe_call(get_value_counts, df, col)
            for col in categorical_cols
        },
    }

    # Infer target column from the goal string
    target = next(
        (col for col in numeric_cols if col in goal.lower()),
        None,
    )
    if target:
        raw["feature_ranking"] = _safe_call(get_feature_importance_proxy, df, target)
    else:
        raw["feature_ranking"] = {
            "ranking": [],
            "note": "no target inferred",
        }

    return raw


def _build_stat_prompt(
    request: EDARequest,
    raw_results: dict,
    profiler_findings: Optional[ProfilerFindings],
) -> str:
    """Build the human message for the structured output LLM call.

    Caps serialised output at 12000 chars to stay within token limits
    on large datasets.
    """
    notable = profiler_findings.notable_columns if profiler_findings else []
    serialised = json.dumps(raw_results, indent=2, default=str)
    if len(serialised) > 12000:
        serialised = serialised[:12000] + "\n... [truncated]"
    parts = [
        f"User goal: {request.goal}",
        f"Notable columns from profiler: {notable}",
        "\nRaw statistical tool outputs:\n" + serialised,
        "\nExtract these raw results into the StatFindings structure. "
        "Express notable_findings as 3–7 plain English observations "
        "(not raw numbers) that are relevant to the stated goal.",
    ]
    return "\n".join(parts)


def _run_direct_mode(
    df: Any,
    request: EDARequest,
    state: AgentState,
    llm: Any,
) -> Tuple[StatFindings, List[str]]:
    """Run all stat tools directly then make one structured output LLM call.

    Returns (StatFindings, warnings).
    """
    schema = get_schema(df)
    numeric_cols: List[str] = schema.get("numeric_cols", [])
    categorical_cols: List[str] = schema.get("categorical_cols", [])
    profiler_findings = _get_profiler_findings(state)
    notable = profiler_findings.notable_columns if profiler_findings else []

    raw_results = _run_all_stat_tools(
        df, numeric_cols, categorical_cols, notable, goal=request.goal
    )
    warnings = _collect_warnings(raw_results)

    structured_llm = llm.with_structured_output(StatFindings)
    findings: StatFindings = structured_llm.invoke([
        SystemMessage(content=STAT_ANALYST_SYSTEM_PROMPT),
        HumanMessage(content=_build_stat_prompt(request, raw_results, profiler_findings)),
    ])
    return findings, warnings


def _run_react_mode(
    df: Any,
    request: EDARequest,
    state: AgentState,
    llm: Any,
) -> StatFindings:
    """Run stat analysis via a ReAct agent, then structure the output."""
    profiler_findings = _get_profiler_findings(state)
    notable = profiler_findings.notable_columns if profiler_findings else []

    agent = create_react_agent(llm, STATISTICAL_TOOLS)
    human_content = (
        f"Goal: {request.goal}\n"
        f"Notable columns from profiler: {notable}\n"
        "Perform comprehensive statistical analysis on the dataset `df`. "
        "Cover correlations, outliers, normality, skewness, and kurtosis "
        "for all numeric columns. Use run_python_repl for any analysis "
        "the structured tools cannot handle."
    )
    response = agent.invoke({
        "messages": [
            SystemMessage(content=STAT_ANALYST_REACT_PROMPT),
            HumanMessage(content=human_content),
        ]
    })

    final_messages = response.get("messages", [])
    final_text = final_messages[-1].content if final_messages else ""

    structured_llm = llm.with_structured_output(StatFindings)
    findings: StatFindings = structured_llm.invoke([
        SystemMessage(content=STAT_ANALYST_SYSTEM_PROMPT),
        HumanMessage(
            content=f"Based on this analysis:\n{final_text}\n\n"
                    "Extract the findings into StatFindings."
        ),
    ])
    return findings


def stat_analyst_node(state: AgentState) -> dict:
    """LangGraph node: statistical analysis specialist.

    Runs after the profiler in the parallel fan-out. Supports two modes:
    - direct (default): deterministic tool calls + one structured LLM call
    - react: ReAct loop with STATISTICAL_TOOLS + one structured LLM call
    """
    try:
        request: EDARequest = state["request"]
        df = get_df(request.session_id)
        if df is None:
            raise ValueError(
                f"DataFrame not found for session {request.session_id}. "
                "Registry may have been cleared."
            )

        schema = get_schema(df)
        numeric_cols: List[str] = schema.get("numeric_cols", [])

        llm_config = get_llm_config()
        agent_config = get_agent_config()
        n_cols = len(numeric_cols)
        dynamic_max_tokens = min(8192, 4096 + n_cols * 200)
        llm = ChatAnthropic(
            model=llm_config.model,
            temperature=llm_config.temperature,
            max_tokens=dynamic_max_tokens,
        )

        if agent_config.stat_analyst_use_react:
            findings = _run_react_mode(df, request, state, llm)
            warnings: List[str] = []
            mode = "react"
        else:
            findings, warnings = _run_direct_mode(df, request, state, llm)
            mode = "direct"

        confidence = max(0.1, 1.0 - len(warnings) * 0.1)

        result = AgentResult(
            agent_name="stat_analyst",
            success=True,
            findings=findings.model_dump(),
            confidence=confidence,
            warnings=warnings,
        )
        return {
            "agent_results": [result],
            "completed_agents": ["stat_analyst"],
            "messages": [
                HumanMessage(
                    content=(
                        f"Stat analyst complete ({mode} mode). "
                        f"Analysed {len(numeric_cols)} numeric columns."
                    )
                )
            ],
        }

    except Exception as e:
        return {
            "agent_results": [AgentResult(
                agent_name="stat_analyst",
                success=False,
                findings={},
                confidence=0.0,
                warnings=[f"Stat analyst node failed: {str(e)}"],
            )],
            "completed_agents": ["stat_analyst"],
            "messages": [
                HumanMessage(content=f"Stat analyst failed: {str(e)}")
            ],
        }
