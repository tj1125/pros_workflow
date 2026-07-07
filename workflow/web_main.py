"""Run the repo-local FastAPI chat UI."""

import os

import click

from commander import load_project_env


@click.command()
@click.option("--host", default="0.0.0.0", show_default=True)
@click.option("--port", default=8080, show_default=True, type=int)
@click.option("--reload", "reload_enabled", is_flag=True, default=False)
@click.option(
    "--keep-logs",
    "keep_logs",
    is_flag=True,
    default=False,
    help="Keep the logs/ folder after shutdown. Default: off (logs/ is deleted on exit).",
)
def main(host: str, port: int, reload_enabled: bool, keep_logs: bool) -> None:
    """Start the VLM grasp web chat server."""
    load_project_env()

    if keep_logs:
        os.environ["WEB_KEEP_LOGS"] = "1"

    import uvicorn

    uvicorn.run(
        "commander.web.app:app",
        host=os.getenv("WEB_HOST", host),
        port=int(os.getenv("WEB_PORT", str(port))),
        reload=reload_enabled,
        timeout_keep_alive=int(os.getenv("WEB_KEEP_ALIVE_TIMEOUT", "1")),
        timeout_graceful_shutdown=int(os.getenv("WEB_GRACEFUL_SHUTDOWN_TIMEOUT", "1")),
    )


if __name__ == "__main__":
    main()
