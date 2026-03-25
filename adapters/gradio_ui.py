import os
import uuid
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.types import Command

from adapters.cli import _load_dataframe, _parse_output_formats
from core.config import get_llm_config
from core.dataframe_registry import register as register_df
from core.schema import EDARequest, OutputType
from core.state import AgentState
from memory.checkpointer import make_config


QA_SYSTEM_PROMPT: str = """You are a data analyst answering follow-up questions about a completed EDA analysis.

## Context available to you
You have access to:
- The user's original analysis goal
- A narrative summary of the full analysis
- Key insights (specific, number-backed observations)
- Caveats and data quality warnings
- Agent findings: profiler results, statistical analysis, confidence scores
- Dataset schema: column names, types, shape

## How to answer
- Answer specifically using numbers and facts from the findings — never make generic observations
- If you reference a number, it must come from the context provided
- Keep answers concise: 3–5 sentences unless detail is genuinely needed
- If the question goes beyond what the analysis found, say clearly:
  "I don't have enough information from this analysis to answer that."
  Never hallucinate findings that aren't in the context.
- When a more precise answer would require additional analysis, suggest the specific technique:
  e.g. "To answer that precisely you would need to run a regression analysis on income ~ age + score."
"""


def _format_plan_as_markdown(plan_dict: dict, message: str = "") -> str:
    """Format an AnalysisPlan dict as a markdown string for the chat bubble."""
    goal = plan_dict.get("goal", "")
    steps = plan_dict.get("steps", [])
    agents = ", ".join(plan_dict.get("agents", []))
    parallel = plan_dict.get("parallel", [])
    formats = ", ".join(
        f.value if hasattr(f, "value") else str(f)
        for f in plan_dict.get("output_formats", [])
    )

    numbered_steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))

    msg_section = f"\n---\n{message}\n" if message else "\n---\n"

    return (
        f"### Analysis Plan\n\n"
        f"**Goal:** {goal}\n\n"
        f"**Steps:**\n{numbered_steps}\n\n"
        f"**Agents:** {agents}\n"
        f"**Parallel groups:** {parallel}\n"
        f"**Output formats:** {formats}\n"
        f"{msg_section}\n"
        f"Type `approve` to proceed or provide feedback to revise."
    )


def _format_outputs_as_markdown(output_paths: dict) -> str:
    """Format output_paths dict as markdown for the final chat message."""
    lines = ["### Analysis Complete\n", "**Output files generated:**"]
    for fmt, path in output_paths.items():
        lines.append(f"- **{fmt}** → `{path}`")
    lines.append("\nDownload the files below.")
    return "\n".join(lines)


def _extract_progress_message(node_name: str, node_output: dict) -> Optional[str]:
    """Extract a progress string from a node output dict.

    Returns None if no meaningful message to show.
    """
    if not isinstance(node_output, dict):
        return None
    messages = node_output.get("messages")
    if not messages:
        return None
    last = messages[-1]
    content = getattr(last, "content", str(last))
    if len(content) > 200:
        content = content[:200] + "..."
    return f"**[{node_name}]** {content}"


def _is_pipeline_complete(state: dict) -> bool:
    """Return True when the pipeline has finished and Q&A mode is active."""
    if not state:
        return False
    if state.get("awaiting_hitl"):
        return False
    return "output_router" in state.get("completed_agents", [])


def _build_qa_context(full_state: dict) -> str:
    """Build the context string injected into every Q&A LLM call."""
    request = full_state.get("request")
    goal = getattr(request, "goal", "") if request else ""
    metadata = getattr(request, "metadata", {}) if request else {}

    shape = metadata.get("shape", [])
    columns = metadata.get("columns", [])

    narrative = full_state.get("narrative") or ""
    key_insights = full_state.get("key_insights") or []
    caveats = full_state.get("caveats") or []

    # Use scored_results if available, else agent_results
    scored = full_state.get("scored_results") or []
    raw = full_state.get("agent_results") or []
    results = scored if scored else raw

    confidence_lines = []
    notable_findings = []
    for r in results:
        agent_name = getattr(r, "agent_name", "")
        confidence = getattr(r, "confidence", None)
        findings = getattr(r, "findings", {})
        if confidence is not None:
            confidence_lines.append(f"- {agent_name}: {confidence * 100:.0f}%")
        if agent_name == "stat_analyst" and isinstance(findings, dict):
            nf = findings.get("notable_findings", [])
            if isinstance(nf, list):
                notable_findings.extend(nf)

    insights_text = "\n".join(
        f"{i + 1}. {ins}" for i, ins in enumerate(key_insights)
    ) or "None"
    caveats_text = "\n".join(
        f"{i + 1}. {c}" for i, c in enumerate(caveats)
    ) or "None"
    confidence_text = "\n".join(confidence_lines) or "Not available"
    notable_text = "\n".join(f"- {f}" for f in notable_findings) or "None"

    parts = [
        f"## Original goal\n{goal}",
        f"## Dataset\nShape: {shape}\nColumns: {columns}",
        f"## Narrative summary\n{narrative[:1000]}",
        f"## Key insights\n{insights_text}",
        f"## Caveats\n{caveats_text}",
        f"## Agent confidence scores\n{confidence_text}",
        f"## Notable statistical findings\n{notable_text}",
        "Answer the user's question using only the information above.",
    ]
    return "\n\n".join(parts)


def _answer_question(
    question: str,
    state: dict,
    graph,
    history: list,
) -> list:
    """Handle a Q&A turn. Returns updated history. Never raises."""
    try:
        config = make_config(state["session_id"])
        full_state = graph.get_state(config).values

        qa_context = _build_qa_context(full_state)

        llm_config = get_llm_config()
        llm = ChatAnthropic(
            model=llm_config.model,
            temperature=0.3,
        )

        response = llm.invoke([
            SystemMessage(content=QA_SYSTEM_PROMPT),
            HumanMessage(content=(
                f"Context about this analysis:\n{qa_context}\n\n"
                f"User question: {question}"
            )),
        ])

        history.append({
            "role": "assistant",
            "content": response.content,
        })

    except Exception as e:
        history.append({
            "role": "assistant",
            "content": f"Sorry, I couldn't answer that: {str(e)}",
        })

    return history


def build_gradio_app(graph, checkpointer) -> gr.Blocks:
    """Build and return the Gradio Blocks app.

    graph and checkpointer are passed in from run.py.
    This function never builds the graph itself.
    """

    def chat_fn(message, history, file, output_formats_selected, state):
        # Q&A mode — pipeline already complete (must be FIRST check)
        if _is_pipeline_complete(state):
            history = history + [{"role": "user", "content": message}]
            yield history, state, gr.update(visible=False), gr.update(
                value="💬 Q&A mode — ask anything about the analysis"
            )

            history = _answer_question(message, state, graph, history)
            yield history, state, gr.update(visible=False), gr.update(
                value="💬 Q&A mode — ask anything about the analysis"
            )
            return

        # Step 1: Add user message to history
        history = history + [{"role": "user", "content": message}]
        yield history, state, gr.update(visible=False), gr.update(
            value="⚙️ Pipeline running..."
        )

        # Step 2: Check if this is a resume (HITL response)
        if state.get("awaiting_hitl"):
            session_id = state["session_id"]
            config = make_config(session_id)

            history.append({
                "role": "assistant",
                "content": f"Received: *{message}*. Resuming analysis...",
            })
            yield history, state, gr.update(visible=False), gr.update(
                value="⚙️ Pipeline running..."
            )

            state["awaiting_hitl"] = False
            initial_input = Command(resume=message)

        else:
            # Step 3: New analysis — build EDARequest
            if file is None:
                history.append({
                    "role": "assistant",
                    "content": "Please upload a CSV or Excel file first.",
                })
                yield history, state, gr.update(visible=False), gr.update(
                    value="Upload a file and describe your goal to begin."
                )
                return

            try:
                df = _load_dataframe(file.name)
            except Exception as e:
                history.append({
                    "role": "assistant",
                    "content": f"Failed to load file: {str(e)}",
                })
                yield history, state, gr.update(visible=False), gr.update(
                    value="❌ An error occurred"
                )
                return

            session_id = str(uuid.uuid4())[:8]
            state["session_id"] = session_id

            register_df(session_id, df)

            output_formats = _parse_output_formats(
                ",".join(output_formats_selected)
            )

            request = EDARequest(
                goal=message,
                session_id=session_id,
                output_formats=output_formats,
                metadata={
                    "filename": Path(file.name).name,
                    "source": "gradio",
                    "shape": list(df.shape),
                    "columns": list(df.columns),
                },
            )

            initial_input = {
                "request": request,
                "plan": None,
                "plan_approved": False,
                "agent_results": [],
                "scored_results": [],
                "rerun_agent": None,
                "rerun_count": 0,
                "narrative": None,
                "key_insights": [],
                "caveats": [],
                "output_paths": {},
                "messages": [],
                "evaluation": None,
                "evaluation_count": 0,
                "dispatched_agents": [],
                "completed_agents": [],
                "next_action": None,
            }

            config = make_config(session_id)
            state["config"] = config

            history.append({
                "role": "assistant",
                "content": (
                    f"Starting analysis | Session: `{session_id}`\n"
                    f"Goal: *{message}*\n"
                    f"Dataset: {df.shape[0]} rows × {df.shape[1]} columns"
                ),
            })
            yield history, state, gr.update(visible=False), gr.update(
                value="⚙️ Pipeline running..."
            )

        # Step 4: Stream graph
        config = make_config(state["session_id"])
        interrupted = False

        try:
            for event in graph.stream(
                initial_input,
                config={**config, "recursion_limit": 100},
                stream_mode="updates",
            ):
                # Check for HITL interrupt
                if "__interrupt__" in event:
                    interrupt_data = event["__interrupt__"][0].value
                    plan_md = _format_plan_as_markdown(
                        interrupt_data.get("plan", {}),
                        interrupt_data.get("message", ""),
                    )
                    history.append({
                        "role": "assistant",
                        "content": plan_md,
                    })
                    state["awaiting_hitl"] = True
                    interrupted = True
                    yield history, state, gr.update(visible=False), gr.update(
                        value="⏸ Waiting for your approval..."
                    )
                    return  # pause — wait for next user message

                # Stream progress updates
                for node_name, node_output in event.items():
                    progress = _extract_progress_message(node_name, node_output)
                    if progress:
                        history.append({
                            "role": "assistant",
                            "content": progress,
                        })
                        yield history, state, gr.update(visible=False), gr.update(
                            value="⚙️ Pipeline running..."
                        )

        except Exception as e:
            history.append({
                "role": "assistant",
                "content": f"Pipeline error: {str(e)}",
            })
            yield history, state, gr.update(visible=False), gr.update(
                value="❌ An error occurred"
            )
            return

        # Step 5: Pipeline complete — get outputs
        if not interrupted:
            try:
                final_state = graph.get_state(config)
                output_paths = final_state.values.get("output_paths", {})
                completed = final_state.values.get("completed_agents", [])
                state["completed_agents"] = completed

                if output_paths:
                    output_md = _format_outputs_as_markdown(output_paths)
                    history.append({
                        "role": "assistant",
                        "content": output_md,
                    })
                    history.append({
                        "role": "assistant",
                        "content": (
                            "Analysis complete. You can now ask me questions about "
                            "the findings — for example:\n"
                            "- *What is the strongest relationship in this data?*\n"
                            "- *Are there any data quality concerns?*\n"
                            "- *What would you recommend analyzing next?*"
                        ),
                    })
                    file_paths = [
                        p for p in output_paths.values() if Path(p).exists()
                    ]
                    yield (
                        history,
                        state,
                        gr.update(visible=True, value=file_paths),
                        gr.update(value="💬 Q&A mode — ask anything about the analysis"),
                    )
                    return
            except Exception as e:
                history.append({
                    "role": "assistant",
                    "content": f"Could not retrieve output files: {str(e)}",
                })

            yield history, state, gr.update(visible=False), gr.update(
                value="⚙️ Pipeline running..."
            )

    with gr.Blocks(title="EDA Agent") as app:
        session_state = gr.State({})

        gr.Markdown("# EDA Agent\nUpload a dataset and describe your goal.")

        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    label="Upload dataset (CSV or Excel)",
                    file_types=[".csv", ".xlsx", ".xls"],
                )
                output_format_input = gr.CheckboxGroup(
                    choices=["report", "json", "email"],
                    value=["report", "json"],
                    label="Output formats",
                )

            with gr.Column(scale=2):
                chatbot = gr.Chatbot(
                    label="EDA Agent",
                    height=500,
                    type="messages",
                )
                msg_input = gr.Textbox(
                    placeholder="Describe your analysis goal...",
                    label="Goal / Response",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Run Analysis", variant="primary")
                    clear_btn = gr.Button("Clear")

        output_files = gr.Files(label="Download outputs", visible=False)
        status_label = gr.Markdown(
            "Upload a file and describe your goal to begin.",
            elem_id="status",
        )

        submit_btn.click(
            chat_fn,
            inputs=[msg_input, chatbot, file_input,
                    output_format_input, session_state],
            outputs=[chatbot, session_state, output_files, status_label],
            api_name=False,
        )
        msg_input.submit(
            chat_fn,
            inputs=[msg_input, chatbot, file_input,
                    output_format_input, session_state],
            outputs=[chatbot, session_state, output_files, status_label],
            api_name=False,
        )
        clear_btn.click(
            lambda: ([], {}, gr.update(visible=False),
                     "Upload a file and describe your goal to begin."),
            outputs=[chatbot, session_state, output_files, status_label],
            api_name=False,
        )

    return app
