"""Command line: dva run | chat | review.

    dva run --files sales.xlsx products.csv stores.csv --data-model retail_sales --session cust-418
    dva chat --session cust-418 --data-model retail_sales
    dva review list [--session cust-418]
    dva review show <draft_id>
    dva review mark <draft_id> approved|rejected

Nothing is ever sent: review only changes a draft's status.
"""

from __future__ import annotations

import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import ValidationError

from dva.config import Role, Settings, configure_logging, load_settings
from dva.data_model import DataModelNotFoundError, make_provider
from dva.llm import create_chat_model
from dva.orchestrator.graph import (
    build_graph,
    chat_input,
    final_reply,
    open_checkpointer,
    run_batch,
    thread_config,
)
from dva.orchestrator.tools import GraphDeps
from dva.review_queue import DraftNotFoundError, ReviewQueue, iter_queues
from dva.workspace import session_workspace

MULTI_VALUE_OPTIONS = ("--files",)
CHAT_HELP = (
    "Chat about your data. /upload <path> [<path> ...] adds files (sent with your next "
    "message); /quit exits."
)

FilesOption = Annotated[list[Path], typer.Option(
    "--files", exists=True, dir_okay=False, resolve_path=True,
    help="Customer files (CSV, Excel, .eml).")]  # fmt: skip
SessionOption = Annotated[str, typer.Option("--session", help="Session id, e.g. cust-418.")]
OptionalSession = Annotated[str | None, typer.Option("--session", help="Limit to one session.")]
DataModelOption = Annotated[str | None, typer.Option("--data-model", help="Data model name.")]

app = typer.Typer(help="Data validation agent.", no_args_is_help=True, add_completion=False)
review_app = typer.Typer(help="Review queue: list, show and mark drafts. Nothing is ever sent.",
                         no_args_is_help=True)  # fmt: skip
app.add_typer(review_app, name="review")


def expand_multi_value(
    argv: list[str], options: tuple[str, ...] = MULTI_VALUE_OPTIONS
) -> list[str]:
    """Turn ``--files a b c`` into ``--files a --files b --files c`` for the parser."""
    out: list[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        i += 1
        if token not in options:
            out.append(token)
            continue
        values = []
        while i < len(argv) and not argv[i].startswith("-"):
            values.append(argv[i])
            i += 1
        out += [part for value in values for part in (token, value)] or [token]
    return out


def main() -> None:
    """Console entry point (``dva``)."""
    app(args=expand_multi_value(sys.argv[1:]), prog_name="dva")


# --- shared setup -----------------------------------------------------------------------


def build_deps(settings: Settings, data_model: str | None) -> GraphDeps:
    """Graph dependencies with one real chat model per role, created on first use."""
    model = make_provider(settings).get_model(data_model or settings.data_model.default)
    models: dict[str, Any] = {}

    def factory(role: Role) -> Any:
        if role not in models:
            models[role] = create_chat_model(settings, role)
        return models[role]

    return GraphDeps(settings=settings, data_model=model, llm_factory=factory)


def _setup(data_model: str | None) -> tuple[Settings, GraphDeps]:
    settings = load_settings()
    configure_logging(settings)
    try:
        return settings, build_deps(settings, data_model)
    except DataModelNotFoundError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc


def _check_session(settings: Settings, session: str) -> None:
    try:
        session_workspace(settings.paths.workspace_root, session)
    except ValidationError as exc:
        typer.echo(f"Error: invalid --session {session!r}: {exc.errors()[0]['msg']}", err=True)
        raise typer.Exit(2) from exc


def run_footer(state: dict[str, Any], elapsed: float | None = None) -> str:
    """One line with stop reason, tools, tokens and (optionally) time."""
    tools = ", ".join(state.get("tool_history") or []) or "none"
    timing = f"; {elapsed:.1f} s" if elapsed is not None else ""
    return (f"[stop: {state.get('stop_reason')}; tools: {tools}; "
            f"tokens: {state.get('tokens_used', 0)}{timing}]")  # fmt: skip


# --- dva run ----------------------------------------------------------------------------


@app.command()
def run(
    files: FilesOption,
    session: SessionOption,
    data_model: DataModelOption = None,
) -> None:
    """Batch mode: files in, report and drafts out (drafts wait in the review queue)."""
    settings, deps = _setup(data_model)
    _check_session(settings, session)
    started = time.monotonic()
    state = run_batch(deps, session, [str(f) for f in files])
    typer.echo(f"Report: {state.get('report_ref') or 'none'}")
    typer.echo(f"Summary: {state.get('report_summary') or final_reply(state)}")
    typer.echo(f"Stop reason: {state.get('stop_reason')}")
    typer.echo(run_footer(state, time.monotonic() - started))
    raise typer.Exit(0 if state.get("stop_reason") == "task_complete" else 1)


# --- dva chat ---------------------------------------------------------------------------


def handle_upload(line: str, pending: list[str]) -> str:
    """Add the files named after /upload to ``pending``; return a message for the user."""
    paths = shlex.split(line)[1:]
    missing = [p for p in paths if not Path(p).is_file()]
    if not paths or missing:
        return f"Not found: {', '.join(missing)}" if missing else "Usage: /upload <path> [...]"
    pending += [str(Path(p).resolve()) for p in paths]
    return f"{len(pending)} file(s) ready; they are sent with your next message."


def chat_loop(graph: Any, deps: GraphDeps, session: str, read: Callable[[], str],
              write: Callable[[str], None]) -> None:  # fmt: skip
    """Read messages until /quit or end of input; one graph invocation per message."""
    pending: list[str] = []
    write(CHAT_HELP)
    while True:
        try:
            line = read().strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line in ("/quit", "/exit"):
            break
        if line.startswith("/upload"):
            write(handle_upload(line, pending))
            continue
        if not line:
            continue
        state = graph.invoke(chat_input(deps, session, line, pending or None),
                             thread_config(session))  # fmt: skip
        pending = []
        write(f"agent> {final_reply(state)}")
        write(run_footer(state))


@app.command()
def chat(
    session: SessionOption,
    data_model: DataModelOption = None,
) -> None:
    """Chat mode: a conversation about the uploaded data, kept across turns and runs."""
    settings, deps = _setup(data_model)
    _check_session(settings, session)
    with open_checkpointer(settings.paths.checkpoint_db) as saver:
        chat_loop(build_graph(deps, saver), deps, session, lambda: input("you> "), typer.echo)


# --- dva review -------------------------------------------------------------------------


def _queues(settings: Settings, session: str | None) -> list[ReviewQueue]:
    if session:
        return [ReviewQueue(session_workspace(settings.paths.workspace_root, session))]
    return list(iter_queues(settings.paths.workspace_root))


def _find(settings: Settings, draft_id: str, session: str | None) -> ReviewQueue:
    for queue in _queues(settings, session):
        try:
            queue.get(draft_id)
            return queue
        except DraftNotFoundError:
            continue
    typer.echo(f"Error: no draft {draft_id!r} found", err=True)
    raise typer.Exit(2)


@review_app.command("list")
def review_list(session: OptionalSession = None) -> None:
    """List drafts with their status."""
    rows = [(q.session_id, d) for q in _queues(load_settings(), session) for d in q.drafts()]
    if not rows:
        typer.echo("No drafts.")
    for session_id, d in rows:
        typer.echo(f"{session_id:16} {d.draft_id:20} {d.kind:6} {d.status.value:15} {d.title}")


@review_app.command("show")
def review_show(draft_id: str, session: OptionalSession = None) -> None:
    """Print one draft."""
    draft = _find(load_settings(), draft_id, session).get(draft_id)
    typer.echo(f"{draft.kind.upper()} {draft.draft_id} [{draft.status.value}]\n{draft.title}\n")
    typer.echo(draft.body)


@review_app.command("mark")
def review_mark(draft_id: str, status: str, session: OptionalSession = None) -> None:
    """Mark a draft approved or rejected. Only the status changes; nothing is sent."""
    queue = _find(load_settings(), draft_id, session)
    try:
        draft = queue.mark(draft_id, status)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(2) from exc
    typer.echo(f"{draft.draft_id}: {draft.status.value} (nothing was sent)")
