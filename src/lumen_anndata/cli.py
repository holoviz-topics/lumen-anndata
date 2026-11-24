"""Command-line interface for lumen-anndata."""

import argparse
import sys

from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lumen-anndata",
        description="Explore AnnData with Lumen AI, or with --flowdash in Lumen's FlowDash app.",
    )
    parser.add_argument(
        "data", nargs="*",
        help="With --flowdash, .h5ad files or a scanpy example dataset (pbmc68k_reduced, the default).",
    )
    parser.add_argument("--flowdash", action="store_true", help="Serve the FlowDash app (experimental).")
    parser.add_argument(
        "--agent-model", action="append", dest="agent_models", metavar="ID[=LABEL]",
        help="With --flowdash, a pydantic-ai model for the agent, e.g. openai:gpt-6-luna; repeat to offer several.",
    )
    parser.add_argument(
        "--db-path", help="With --flowdash, the SQLite database dashboards are saved in (default: dashboards.db)."
    )
    parser.add_argument("--port", type=int, default=5006, help="Port to serve on (default: 5006).")
    parser.add_argument("--show", action="store_true", help="Open the app in a browser.")
    return parser


def main(argv=None):
    """Serve the lumen-anndata Panel application."""
    args = _parser().parse_args(sys.argv[1:] if argv is None else argv)
    if args.flowdash:
        from .flowdash.app import serve

        serve(
            args.data or ["pbmc68k_reduced"], port=args.port, show=args.show,
            agent_model=args.agent_models, db_path=args.db_path,
        )
        return

    import panel.command as pn_cmd

    app_path = Path(__file__).parent / "app.py"
    sys.argv = ["panel", "serve", str(app_path), "--port", str(args.port), *(["--show"] if args.show else [])]
    pn_cmd.main()


if __name__ == "__main__":
    main()
