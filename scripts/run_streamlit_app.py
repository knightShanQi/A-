from __future__ import annotations

from pathlib import Path

from streamlit.web import bootstrap

from a_share_predictor.database_source import load_env_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"


def main() -> None:
    load_env_file()
    bootstrap.run(
        str(APP_PATH),
        is_hello=False,
        args=[],
        flag_options={
            "server.headless": True,
            "server.port": 8501,
            "server.fileWatcherType": "none",
        },
    )


if __name__ == "__main__":
    main()
