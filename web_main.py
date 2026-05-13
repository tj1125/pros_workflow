"""Run the repo-local FastAPI chat UI."""

import logging
import os

import click
from dotenv import load_dotenv


class _SuppressFindCandidatesAccessLog(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args if isinstance(record.args, tuple) else ()
        return not any(
            isinstance(arg, str) and "/find-candidates" in arg
            for arg in args
        )


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True, type=int)
@click.option("--reload", is_flag=True, default=False)
def main(host: str, port: int, reload: bool) -> None:
    """Start the VLM-RL web chat server."""
    load_dotenv(override=True)
    logging.getLogger("uvicorn.access").addFilter(_SuppressFindCandidatesAccessLog())

    import uvicorn

    uvicorn.run(
        "commander.web_server:app",
        host=os.getenv("WEB_HOST", host),
        port=int(os.getenv("WEB_PORT", str(port))),
        reload=reload,
        timeout_keep_alive=int(os.getenv("WEB_KEEP_ALIVE_TIMEOUT", "1")),
        timeout_graceful_shutdown=int(os.getenv("WEB_GRACEFUL_SHUTDOWN_TIMEOUT", "1")),
    )


if __name__ == "__main__":
    main()
