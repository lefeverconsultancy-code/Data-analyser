"""Start the local app:  python -m analyser

Extra arguments are passed to Streamlit, e.g.  python -m analyser --server.port 8600
The app listens on localhost only, sends no usage statistics and hides Streamlit's Deploy button.
"""

from __future__ import annotations

import sys
from pathlib import Path

APP = Path(__file__).parent / "app" / "streamlit_app.py"


def streamlit_args(extra: list[str] | None = None) -> list[str]:
    return [
        "streamlit", "run", str(APP),
        "--server.address=localhost",
        "--browser.gatherUsageStats=false",
        "--client.toolbarMode=viewer",
        *(extra or []),
    ]


def main(argv: list[str] | None = None) -> None:
    from streamlit.web import cli as stcli

    sys.argv = streamlit_args(list(sys.argv[1:] if argv is None else argv))
    sys.exit(stcli.main())


if __name__ == "__main__":
    main()
