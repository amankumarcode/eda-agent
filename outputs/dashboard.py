import os
from datetime import datetime
from pathlib import Path
from typing import List

from plotly.graph_objects import Figure
from plotly.io import to_html

from core.schema import AgentResult
from core.state import AgentState


def _get_viz_findings(state: AgentState) -> dict:
    """Return viz_agent findings dict, or {} if not found or failed."""
    for result in state.get("agent_results", []):
        if result.agent_name == "viz_agent" and result.success:
            return result.findings
    return {}


def _reconstruct_charts(viz_findings: dict) -> List[str]:
    """Convert Plotly chart dicts to standalone HTML div strings (no CDN script)."""
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


def _get_chart_descriptions(viz_findings: dict) -> List[str]:
    """Return chart_descriptions list from viz findings, or empty list."""
    return viz_findings.get("chart_descriptions", [])


def _build_dashboard_html(state: AgentState, chart_divs: List[str]) -> str:
    """Build a standalone Plotly HTML dashboard from AgentState."""
    request = state["request"]
    goal = request.goal
    session_id = request.session_id
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    narrative: str = state.get("narrative") or ""
    key_insights: List[str] = state.get("key_insights") or []
    agent_results: List[AgentResult] = state.get("agent_results") or []

    viz_findings = _get_viz_findings(state)
    chart_descriptions = _get_chart_descriptions(viz_findings)
    recommended = viz_findings.get("recommended_primary_chart", "")

    # ---- CSS ----
    css = (
        "*{box-sizing:border-box;margin:0;padding:0;}"
        "body{background:#0f172a;color:#e2e8f0;"
        "font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;"
        "padding:24px;}"
        "header{max-width:1200px;margin:0 auto 24px;}"
        "h1{font-size:1.75rem;color:#f1f5f9;margin-bottom:6px;}"
        "h2{font-size:1.1rem;color:#94a3b8;margin-bottom:16px;}"
        ".meta{font-size:0.8rem;color:#64748b;margin-top:4px;}"
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));"
        "gap:20px;max-width:1200px;margin:0 auto;}"
        ".card{background:#1e293b;border-radius:12px;padding:20px;"
        "border:1px solid #334155;}"
        ".card-title{font-size:0.8rem;color:#94a3b8;text-transform:uppercase;"
        "letter-spacing:0.05em;margin-bottom:12px;font-weight:600;}"
        ".chart-caption{font-size:0.8rem;color:#64748b;margin-top:8px;"
        "font-style:italic;}"
        ".primary-badge{display:inline-block;background:#3b82f6;color:#fff;"
        "font-size:0.7rem;padding:2px 8px;border-radius:9999px;margin-left:8px;"
        "font-weight:600;vertical-align:middle;}"
        ".insight-list{list-style:none;}"
        ".insight-list li{padding:6px 0;border-bottom:1px solid #334155;"
        "font-size:0.9rem;}"
        ".insight-list li:last-child{border-bottom:none;}"
        ".insight-list li::before{content:'▸ ';color:#3b82f6;}"
        ".agent-row{display:flex;align-items:center;justify-content:space-between;"
        "padding:6px 0;border-bottom:1px solid #334155;font-size:0.85rem;}"
        ".agent-row:last-child{border-bottom:none;}"
        ".conf-bar-bg{width:100px;height:8px;background:#334155;border-radius:4px;"
        "overflow:hidden;margin-left:8px;display:inline-block;vertical-align:middle;}"
        ".conf-bar{height:100%;background:#3b82f6;border-radius:4px;}"
        ".narrative{font-size:0.9rem;line-height:1.7;color:#cbd5e1;}"
        "footer{max-width:1200px;margin:24px auto 0;"
        "font-size:0.75rem;color:#475569;text-align:center;}"
    )

    # ---- head ----
    head = (
        "<!DOCTYPE html>\n<html lang='en'>\n<head>\n"
        "<meta charset='UTF-8'>\n"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>\n"
        f"<title>EDA Dashboard — {session_id}</title>\n"
        "<script src='https://cdn.plot.ly/plotly-latest.min.js'></script>\n"
        f"<style>{css}</style>\n"
        "</head>\n<body>\n"
    )

    body = ""

    # ---- header ----
    body += (
        "<header>\n"
        f"<h1>EDA Dashboard</h1>\n"
        f"<h2>{goal}</h2>\n"
        f"<p class='meta'>Session: {session_id} &nbsp;|&nbsp; Generated: {timestamp}</p>\n"
        "</header>\n"
        "<div class='grid'>\n"
    )

    # ---- charts ----
    for i, div in enumerate(chart_divs):
        caption = chart_descriptions[i] if i < len(chart_descriptions) else ""
        is_primary = recommended and caption and recommended in caption

        title_html = f"<span class='card-title'>Chart {i + 1}</span>"
        if is_primary:
            title_html += "<span class='primary-badge'>Primary</span>"

        body += (
            "<div class='card'>\n"
            f"{title_html}\n"
            f"{div}\n"
        )
        if caption:
            body += f"<p class='chart-caption'>{caption}</p>\n"
        body += "</div>\n"

    # ---- narrative card ----
    if narrative:
        body += (
            "<div class='card'>\n"
            "<p class='card-title'>Summary</p>\n"
            f"<p class='narrative'>{narrative}</p>\n"
            "</div>\n"
        )

    # ---- key insights card ----
    if key_insights:
        items = "".join(f"<li>{i}</li>\n" for i in key_insights)
        body += (
            "<div class='card'>\n"
            "<p class='card-title'>Key Insights</p>\n"
            f"<ul class='insight-list'>\n{items}</ul>\n"
            "</div>\n"
        )

    # ---- agent confidence card ----
    if agent_results:
        rows = ""
        for r in agent_results:
            pct_int = int(r.confidence * 100)
            status_color = "#22c55e" if r.success else "#ef4444"
            status_symbol = "✓" if r.success else "✗"
            rows += (
                "<div class='agent-row'>\n"
                f"<span>{r.agent_name}</span>\n"
                "<span style='display:flex;align-items:center;gap:6px;'>\n"
                f"<span style='color:{status_color};'>{status_symbol}</span>\n"
                f"<span>{pct_int}%</span>\n"
                f"<span class='conf-bar-bg'>"
                f"<span class='conf-bar' style='width:{pct_int}%;'></span>"
                f"</span>\n"
                "</span>\n"
                "</div>\n"
            )
        body += (
            "<div class='card'>\n"
            "<p class='card-title'>Agent Confidence</p>\n"
            f"{rows}"
            "</div>\n"
        )

    body += "</div>\n"

    # ---- footer ----
    body += (
        "<footer>\n"
        f"<p>Generated by EDA Agent &nbsp;|&nbsp; Session: {session_id}</p>\n"
        "</footer>\n"
    )

    return head + body + "</body>\n</html>"


def generate_dashboard(state: AgentState) -> str:
    """Generate a standalone Plotly HTML dashboard from AgentState.

    Called by output_router_node when OutputType.DASHBOARD is requested.
    Writes to OUTPUT_DIR/<session_id>/dashboard_<session_id>.html.

    Returns the absolute path to the generated file.
    Raises RuntimeError on failure.
    """
    try:
        request = state["request"]
        session_id = request.session_id

        output_dir = Path(os.getenv("OUTPUT_DIR", "./output")) / session_id
        output_dir.mkdir(parents=True, exist_ok=True)

        chart_divs = _reconstruct_charts(_get_viz_findings(state))
        html = _build_dashboard_html(state, chart_divs)

        output_path = output_dir / f"dashboard_{session_id}.html"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        return str(output_path)

    except Exception as e:
        raise RuntimeError(f"generate_dashboard failed: {e}") from e
