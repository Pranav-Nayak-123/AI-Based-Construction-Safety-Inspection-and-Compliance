from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("status")
def status() -> None:
    """Print scaffold status for the data/tooling CLI."""
    typer.echo("safety-tools 0.1.0 (scaffold)")
    typer.echo("Existing scripts: tools/cctv.py, fetch_sard.py, profile_sard.py, profile_yolo.py")


@app.command("version")
def version() -> None:
    """Print the package version."""
    typer.echo("0.1.0")


DEFAULT_MODELS = ["yolo26s.pt", "yolo26n-pose.pt"]


@app.command("fetch-models")
def fetch_models(
    names: Annotated[
        list[str] | None, typer.Argument(help="Ultralytics asset names, e.g. yolo26m.pt")
    ] = None,
    destination: Annotated[Path, typer.Option(help="Target folder")] = Path(
        "artifacts/models/pretrained"
    ),
) -> None:
    """Download pretrained Ultralytics weights and print their SHA-256."""
    # Imported here so the rest of the tools CLI works without the vision extra.
    from ultralytics.utils.downloads import attempt_download_asset

    destination.mkdir(parents=True, exist_ok=True)
    for name in names or DEFAULT_MODELS:
        path = Path(attempt_download_asset(destination / name))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        typer.echo(f"{path}  {path.stat().st_size / 1e6:.1f} MB  sha256={digest}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
