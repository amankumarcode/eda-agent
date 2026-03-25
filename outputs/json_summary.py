import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from core.dataframe_registry import get as get_df
from core.schema import AgentResult, EvaluationResult
from core.state import AgentState


def _compute_mean_confidence(agent_results: List[AgentResult]) -> float:
    """Mean confidence across successful agents. Returns 0.0 for empty/all-failed."""
    scores = [r.confidence for r in agent_results if r.success]
    return sum(scores) / len(scores) if scores else 0.0


def _serialize_agent_results(agent_results: List[AgentResult]) -> list:
    """Convert AgentResult list to JSON-serialisable list of dicts."""
    try:
        return [r.model_dump() for r in agent_results]
    except Exception:
        return []


def _serialize_evaluation(evaluation: Optional[EvaluationResult]) -> Optional[dict]:
    """Convert EvaluationResult to dict (with enum values as strings), or None."""
    if evaluation is None:
        return None
    try:
        data = evaluation.model_dump()
        # Ensure verdict enum is serialised as its string value
        if "verdict" in data and hasattr(data["verdict"], "value"):
            data["verdict"] = data["verdict"].value
        return data
    except Exception:
        return None


def _merge_results(state: AgentState) -> List[AgentResult]:
    """Merge agent_results with scored_results, preferring scored confidence."""
    scored_map = {r.agent_name: r for r in state.get("scored_results", [])}
    results = []
    for r in state.get("agent_results", []):
        if r.agent_name in scored_map:
            results.append(scored_map[r.agent_name])
        else:
            results.append(r)
    return results


def _build_summary(state: AgentState) -> dict:
    """Build the full summary dict from AgentState."""
    request = state["request"]
    plan = state.get("plan")

    return {
        "metadata": {
            "session_id": request.session_id,
            "goal": request.goal,
            "generated_at": datetime.utcnow().isoformat(),
            "dataset_shape": request.metadata.get("shape", []),
            "dataset_columns": request.metadata.get("columns", []),
            "output_formats": [f.value for f in request.output_formats],
        },
        "plan": plan.model_dump() if plan else None,
        "analysis": {
            "agent_results": _serialize_agent_results(_merge_results(state)),
            "overall_confidence": _compute_mean_confidence(_merge_results(state)),
        },
        "narrative": {
            "summary": state.get("narrative", ""),
            "key_insights": state.get("key_insights", []),
            "caveats": state.get("caveats", []),
        },
        "evaluation": _serialize_evaluation(state.get("evaluation")),
        "output_paths": state.get("output_paths", {}),
    }


def generate_json(state: AgentState) -> str:
    """Generate a structured JSON summary file from the final AgentState.

    Called by output_router_node when OutputType.JSON is requested.
    Writes to OUTPUT_DIR/<session_id>/summary_<session_id>.json.

    Returns the absolute path to the generated file.
    Raises RuntimeError (wrapping the original exception) on failure.
    """
    try:
        request = state["request"]
        session_id = request.session_id

        output_dir = Path(os.getenv("OUTPUT_DIR", "./output")) / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / f"summary_{session_id}.json"
        summary = _build_summary(state)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        return str(output_path)

    except Exception as e:
        raise RuntimeError(f"generate_json failed: {e}") from e
