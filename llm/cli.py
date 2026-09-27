from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command("status")
def status() -> None:
    """Print scaffold status for the LLM experiment CLI."""
    typer.echo("safety-llm 0.1.0 (scaffold)")
    typer.echo("Fine-tuning is a graded experiment, not the authority for rules or citations.")


@app.command("version")
def version() -> None:
    """Print the package version."""
    typer.echo("0.1.0")


@app.command("narrate")
def narrate(
    run_dir: Annotated[
        Path, typer.Argument(exists=True, file_okay=False, help="Run bundle: output/<run_id>")
    ],
    base: Annotated[Path, typer.Option(help="Base model")] = Path(
        "artifacts/models/llm/Qwen2.5-1.5B-Instruct"
    ),
    adapter: Annotated[Path, typer.Option(help="LoRA adapter")] = Path(
        "artifacts/adapters/narration-qwen2.5-1.5b"
    ),
) -> None:
    """Narrate a run's incidents with the fine-tuned model behind the L5 guardrail."""
    from llm.runtime.narrator import Narrator, narrate_run

    summary = narrate_run(run_dir, Narrator(base=base, adapter=adapter))
    typer.echo(
        f"{summary['model']} of {summary['incidents']} narrations from the model, "
        f"{summary['template_fallback']} from the template "
        f"(failed checks: {summary['failed_checks'] or 'none'}) in {summary['seconds']} s"
    )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
