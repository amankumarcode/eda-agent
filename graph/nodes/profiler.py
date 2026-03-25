import json
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from core.config import get_llm_config
from core.dataframe_registry import get as get_df
from core.schema import AgentResult, EDARequest, ProfilerFindings
from core.state import AgentState
from core.tool_registry import (
    get_cardinality,
    get_categorical_summary,
    get_constant_columns,
    get_distribution_stats,
    get_duplicate_report,
    get_high_cardinality_columns,
    get_null_report,
    get_sample_rows,
    get_schema,
    inject_dataframe,
)

PROFILER_SYSTEM_PROMPT: str = """You are a structured data profiling specialist.

Your task is to interpret raw profiling tool outputs and extract them into a
validated ProfilerFindings structure. Do NOT perform any new analysis — only
structure and interpret the raw results that have been provided to you.

Requirements:
- Every column in the dataset must appear in your findings. Do not skip or
  summarise away any column, no matter how simple or uninteresting it appears.
- If a tool returned an error or empty result for a column, reflect that
  honestly in the warnings field rather than omitting the column.
- Populate notable_columns with column names that warrant closer attention
  (high null rate, high skew, unusual cardinality, data quality issues, etc.).
- Populate data_quality with a summary of duplicates, constant columns, and
  high-cardinality columns from the tool outputs provided.
- Be precise: use numbers and field names from the raw tool outputs rather
  than inventing or estimating values.
"""


def _safe_call(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call fn(*args, **kwargs). On any exception return {"_error": str(e)}."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        return {"_error": str(e)}


def _run_all_profiling_tools(df: Any) -> dict:
    """Run every profiling tool against df and collect all results.

    Never raises — individual tool failures are captured as {"_error": ...}.
    """
    schema = get_schema(df)
    numeric_cols = schema.get("numeric_cols", [])
    categorical_cols = schema.get("categorical_cols", [])

    raw = {
        "schema":                  schema,
        "null_report":             _safe_call(get_null_report, df),
        "sample_rows":             _safe_call(get_sample_rows, df, 5),
        "duplicate_report":        _safe_call(get_duplicate_report, df),
        "constant_columns":        _safe_call(get_constant_columns, df),
        "high_cardinality_columns": _safe_call(get_high_cardinality_columns, df),
        "categorical_summary":     _safe_call(get_categorical_summary, df),
        "distributions": {
            col: _safe_call(get_distribution_stats, df, col)
            for col in numeric_cols
        },
        "cardinality": {
            col: _safe_call(get_cardinality, df, col)
            for col in categorical_cols
        },
    }
    return raw


def _build_profiler_prompt(request: EDARequest, raw_results: dict) -> str:
    """Build the human message for the structured output LLM call.

    Truncates distributions and cardinality for large datasets to keep
    the prompt within token limits.
    """
    truncated = raw_results.copy()

    if "distributions" in truncated:
        dists = truncated["distributions"]
        if len(dists) > 20:
            keys = list(dists.keys())[:20]
            truncated["distributions"] = {k: dists[k] for k in keys}
            truncated["distributions"]["_truncated"] = (
                f"{len(dists) - 20} additional columns omitted"
            )

    if "cardinality" in truncated:
        card = truncated["cardinality"]
        if len(card) > 10:
            keys = list(card.keys())[:10]
            truncated["cardinality"] = {k: card[k] for k in keys}

    serialised = json.dumps(truncated, indent=2, default=str)
    if len(serialised) > 12000:
        serialised = serialised[:12000] + "\n... [truncated]"

    return (
        f"User goal: {request.goal}\n\n"
        f"Dataset: {request.metadata.get('filename', 'unknown')} "
        f"shape={request.metadata.get('shape', 'unknown')}\n\n"
        f"Raw profiling results:\n{serialised}\n\n"
        "Extract into ProfilerFindings. Be complete but prioritise "
        "columns most relevant to the goal. Use field name 'schema_info' "
        "(not 'schema') for the dataset schema dict."
    )


def profiler_node(state: AgentState) -> dict:
    """LangGraph node: deterministic data profiler.

    Runs all profiling tools directly (no ReAct loop), then makes one
    structured LLM call to interpret and package the raw outputs.
    """
    try:
        request: EDARequest = state["request"]
        df = get_df(request.session_id)
        if df is None:
            raise ValueError(
                f"DataFrame not found for session {request.session_id}. "
                "Registry may have been cleared."
            )

        # Inject df into REPL namespace for any downstream REPL usage
        inject_dataframe(df)

        # Run all tools
        raw_results = _run_all_profiling_tools(df)
        schema = raw_results["schema"]
        numeric_cols: list = schema.get("numeric_cols", [])
        categorical_cols: list = schema.get("categorical_cols", [])

        # Collect tool errors from top-level keys
        warnings: list[str] = [
            f"{key}: {val['_error']}"
            for key, val in raw_results.items()
            if isinstance(val, dict) and "_error" in val
        ]
        # Collect tool errors from nested dicts (distributions, cardinality)
        for group_key in ("distributions", "cardinality"):
            group = raw_results.get(group_key, {})
            if isinstance(group, dict):
                for col, result in group.items():
                    if isinstance(result, dict) and "_error" in result:
                        warnings.append(f"{group_key}.{col}: {result['_error']}")

        # One structured LLM call to interpret raw outputs
        config = get_llm_config()
        n_cols = len(df.columns)
        dynamic_max_tokens = min(8192, 4096 + n_cols * 200)
        llm = ChatAnthropic(
            model=config.model,
            temperature=config.temperature,
            max_tokens=dynamic_max_tokens,
        )
        structured_llm = llm.with_structured_output(ProfilerFindings)
        findings: ProfilerFindings = structured_llm.invoke([
            SystemMessage(content=PROFILER_SYSTEM_PROMPT),
            HumanMessage(content=_build_profiler_prompt(request, raw_results)),
        ])

        confidence = max(0.1, 1.0 - len(warnings) * 0.1)

        result = AgentResult(
            agent_name="profiler",
            success=True,
            findings=findings.model_dump(),
            confidence=confidence,
            warnings=warnings,
        )
        return {
            "agent_results": [result],
            "completed_agents": ["profiler"],
            "messages": [
                HumanMessage(
                    content=(
                        f"Profiler complete. "
                        f"Analysed {len(df.columns)} columns "
                        f"({len(numeric_cols)} numeric, "
                        f"{len(categorical_cols)} categorical). "
                        f"Warnings: {len(warnings)}"
                    )
                )
            ],
        }

    except Exception as e:
        return {
            "agent_results": [AgentResult(
                agent_name="profiler",
                success=False,
                findings={},
                confidence=0.0,
                warnings=[f"Profiler node failed: {str(e)}"],
            )],
            "completed_agents": ["profiler"],
            "messages": [HumanMessage(content=f"Profiler failed: {str(e)}")],
        }
