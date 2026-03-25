import os
from datetime import datetime
from pathlib import Path
from typing import List

from plotly.graph_objects import Figure
from plotly.io import to_html

from core.schema import AgentResult, EvaluationResult, Verdict
from core.state import AgentState


def _get_viz_findings(state: AgentState) -> dict:
    """Return viz_agent findings dict, or {} if not found or failed."""
    for result in state.get("agent_results", []):
        if result.agent_name == "viz_agent" and result.success:
            return result.findings
    return {}


def _reconstruct_charts(viz_findings: dict) -> List[str]:
    """Convert Plotly chart dicts to embeddable HTML div strings."""
    charts = viz_findings.get("charts", [])
    divs: List[str] = []
    for chart_dict in charts:
        try:
            fig = Figure(chart_dict)
            html_div = to_html(fig, full_html=False, include_plotlyjs=False)
            divs.append(html_div)
        except Exception:
            pass
    return divs


def _verdict_badge(verdict) -> str:
    """Return an inline-styled HTML pill for the evaluation verdict."""
    styles = {
        Verdict.STRONG:   ("#22c55e", "strong"),
        Verdict.ADEQUATE: ("#f59e0b", "adequate"),
        Verdict.WEAK:     ("#ef4444", "weak"),
    }
    if verdict in styles:
        color, label = styles[verdict]
        return (
            f'<span style="background:{color};color:#fff;'
            f'padding:4px 12px;border-radius:9999px;'
            f'font-weight:600;font-size:0.85rem;">{label}</span>'
        )
    return (
        '<span style="background:#94a3b8;color:#fff;'
        'padding:4px 12px;border-radius:9999px;'
        'font-weight:600;font-size:0.85rem;">Not evaluated</span>'
    )


def _build_html(state: AgentState, chart_divs: List[str]) -> str:  # noqa: C901
    """Build a complete self-contained HTML report from AgentState."""
    request = state["request"]
    narrative: str = state.get("narrative") or ""
    key_insights: List[str] = state.get("key_insights") or []
    caveats: List[str] = state.get("caveats") or []
    agent_results: List[AgentResult] = state.get("agent_results") or []
    evaluation: EvaluationResult = state.get("evaluation")
    goal = request.goal
    session_id = request.session_id
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    shape = request.metadata.get("shape", [0, 0])
    columns = request.metadata.get("columns", [])

    # ---- CSS ----
    css = (
        "body{max-width:900px;margin:40px auto;padding:0 24px;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "color:#1e293b;line-height:1.6;}"
        "h1{font-size:2rem;margin-bottom:4px;}"
        "h2{font-size:1.25rem;margin-top:2rem;margin-bottom:0.5rem;"
        "border-bottom:2px solid #e2e8f0;padding-bottom:6px;}"
        "ul{padding-left:1.5rem;}"
        "li{margin-bottom:4px;}"
        "table{border-collapse:collapse;width:100%;}"
        "td,th{padding:8px 12px;border:1px solid #e2e8f0;text-align:left;}"
        "th{background:#f8fafc;font-weight:600;}"
        ".meta{color:#64748b;font-size:0.875rem;margin-top:4px;}"
        ".warning{color:#b45309;font-size:0.85rem;}"
    )

    # ---- head ----
    head = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='UTF-8'>\n"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
        f"<title>EDA Report — {session_id}</title>\n"
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>\n"
        f"<style>{css}</style>\n"
        "</head>\n<body>\n"
    )

    body = ""

    # 1. Header
    body += (
        f"<h1>EDA Report</h1>\n"
        f"<p class='meta'><strong>Goal:</strong> {goal}</p>\n"
        f"<p class='meta'><strong>Session:</strong> {session_id} "
        f"&nbsp;|&nbsp; <strong>Generated:</strong> {timestamp}</p>\n"
    )

    # 2. Executive Summary
    if narrative:
        body += f"<h2>Executive Summary</h2>\n<p>{narrative}</p>\n"

    # 3. Key Insights
    if key_insights:
        items = "".join(f"<li>{i}</li>\n" for i in key_insights)
        body += f"<h2>Key Insights</h2>\n<ul>\n{items}</ul>\n"

    # 4. Visualisations
    if chart_divs:
        body += "<h2>Visualisations</h2>\n"
        for div in chart_divs:
            body += f"<div style='margin-bottom:24px;'>{div}</div>\n"

    # 5. Analysis Quality
    if evaluation is not None:
        badge = _verdict_badge(evaluation.verdict)
        body += (
            "<h2>Analysis Quality</h2>\n"
            f"<p>{badge}</p>\n"
            "<table>\n"
            f"<tr><th>Metric</th><th>Score</th></tr>\n"
            f"<tr><td>Overall</td><td>{evaluation.overall_score:.2f}</td></tr>\n"
            f"<tr><td>Goal coverage</td><td>{evaluation.goal_coverage:.2f}</td></tr>\n"
            f"<tr><td>Insight quality</td><td>{evaluation.insight_quality:.2f}</td></tr>\n"
            f"<tr><td>Evidence quality</td><td>{evaluation.evidence_quality:.2f}</td></tr>\n"
            "</table>\n"
        )
        if evaluation.strengths:
            items = "".join(f"<li>{s}</li>\n" for s in evaluation.strengths)
            body += f"<p><strong>Strengths:</strong></p><ul>\n{items}</ul>\n"
        if evaluation.gaps:
            items = "".join(f"<li>{g}</li>\n" for g in evaluation.gaps)
            body += f"<p><strong>Gaps:</strong></p><ul>\n{items}</ul>\n"
        if evaluation.verdict == Verdict.WEAK and evaluation.retry_instructions:
            body += (
                f"<p><strong>Improvement guidance:</strong> "
                f"{evaluation.retry_instructions}</p>\n"
            )

    # 6. Agent Confidence
    if agent_results:
        body += "<h2>Agent Confidence</h2>\n<table>\n"
        body += "<tr><th>Agent</th><th>Confidence</th><th>Status</th><th>Warnings</th></tr>\n"
        for r in agent_results:
            status = "✓" if r.success else "✗"
            pct = f"{r.confidence * 100:.0f}%"
            warns = (
                "<br>".join(f'<span class="warning">{w}</span>' for w in r.warnings)
                if r.warnings else "—"
            )
            body += (
                f"<tr><td>{r.agent_name}</td><td>{pct}</td>"
                f"<td>{status}</td><td>{warns}</td></tr>\n"
            )
        body += "</table>\n"

    # 7. Caveats
    if caveats:
        items = "".join(f"<li>{c}</li>\n" for c in caveats)
        body += f"<h2>Caveats</h2>\n<ul>\n{items}</ul>\n"

    # 8. Dataset Info
    rows_n = shape[0] if len(shape) > 0 else 0
    cols_n = shape[1] if len(shape) > 1 else 0
    body += (
        "<h2>Dataset Info</h2>\n"
        f"<p><strong>Shape:</strong> {rows_n} rows × {cols_n} columns</p>\n"
        f"<p><strong>Columns:</strong> {', '.join(str(c) for c in columns)}</p>\n"
    )

    return head + body + "</body>\n</html>"


def generate_report(state: AgentState) -> str:
    """Generate a self-contained HTML report from AgentState.

    Called by output_router_node when OutputType.REPORT is requested.
    Writes to OUTPUT_DIR/<session_id>/report_<session_id>.html.

    Returns the absolute path to the generated file.
    Raises RuntimeError on failure.
    """
    try:
        request = state["request"]
        session_id = request.session_id

        output_dir = Path(os.getenv("OUTPUT_DIR", "./output")) / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        chart_divs = _reconstruct_charts(_get_viz_findings(state))
        html = _build_html(state, chart_divs)

        output_path = output_dir / f"report_{session_id}.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return str(output_path)

    except Exception as e:
        raise RuntimeError(f"generate_report failed: {e}") from e
