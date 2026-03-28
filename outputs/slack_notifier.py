import json
import os
from urllib import request, error

from core.state import AgentState


def notify_slack(state: AgentState) -> str | None:
    """Post an EDA summary to a Slack webhook.

    Returns the webhook URL on success, None if the env var is unset.
    Failures are silenced — Slack notification is best-effort and must
    never block the pipeline.
    """
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    if not webhook_url:
        return None

    session_id = state["request"].session_id
    goal = state["request"].goal
    insights = state.get("key_insights") or []
    output_paths = state.get("output_paths") or {}

    bullets = "\n".join(f"• {i}" for i in insights[:5]) or "_No insights generated._"
    files = ", ".join(f"`{k}`" for k in output_paths) or "_none_"

    text = (
        f"*EDA Complete* — `{session_id}`\n"
        f"*Goal:* {goal}\n\n"
        f"*Key Insights:*\n{bullets}\n\n"
        f"*Outputs generated:* {files}"
    )

    payload = json.dumps({"text": text}).encode()
    req = request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        request.urlopen(req, timeout=5)
    except error.URLError:
        pass

    return webhook_url
