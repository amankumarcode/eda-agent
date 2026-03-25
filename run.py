"""
EDA Agent — unified entrypoint

Usage:
  python run.py                                         # launch Gradio UI
  python run.py --mode gradio                           # launch Gradio UI
  python run.py --mode cli \
      --file data.csv \
      --goal "find revenue drivers" \
      --output report,json
"""
from dotenv import load_dotenv
load_dotenv()  # must be first — loads .env before any module
               # reads ANTHROPIC_API_KEY, DB_PATH, OUTPUT_DIR etc.

import os
import sys

from memory.checkpointer import get_checkpointer
from graph.builder import get_graph
from adapters.cli import build_arg_parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    # bootstrap — order matters:
    # checkpointer first (creates DB file/dirs if needed)
    # graph second (imports all nodes, validates wiring)
    checkpointer = get_checkpointer()
    graph = get_graph(checkpointer)

    if args.mode == "cli":
        # validate required cli args
        if not args.file:
            print("Error: --file is required in cli mode.")
            sys.exit(1)
        if not args.goal:
            print("Error: --goal is required in cli mode.")
            sys.exit(1)

        from adapters.cli import run_cli
        run_cli(graph, args)

    else:
        # default: gradio
        from adapters.gradio_ui import build_gradio_app
        app = build_gradio_app(graph, checkpointer)
        app.launch(
            server_name=os.getenv("GRADIO_HOST", "0.0.0.0"),
            server_port=int(os.getenv("GRADIO_PORT", "7860")),
            share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        )


if __name__ == "__main__":
    main()
