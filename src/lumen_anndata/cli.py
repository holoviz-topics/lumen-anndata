"""Command-line interface for lumen-anndata."""

import sys

from pathlib import Path


def main():
    """Serve the lumen-anndata Panel application."""
    import panel.command as pn_cmd

    # Get the path to the app.py file
    app_path = Path(__file__).parent / "app.py"

    # Construct the arguments for panel serve
    sys.argv = ["panel", "serve", str(app_path), "--show"]

    # Run panel serve
    pn_cmd.main()


if __name__ == "__main__":
    main()
