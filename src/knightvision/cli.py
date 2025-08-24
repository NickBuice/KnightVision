from pathlib import Path
from typing import Optional

import typer

from .io import default_pgn_out_path
from .assets import CACHE_DIR, DEFAULT_MODELS, download_models, find_model, valid_model
from .pipeline import run_video_to_pgn


app = typer.Typer(no_args_is_help=True, add_completion=False, help="KnightVision CLI")
models_app = typer.Typer(help="Manage KnightVision models")
app.add_typer(models_app, name="models")


@models_app.command(help="Download model(s) from latest GitHub release")
def download(
    which: str = typer.Argument(..., help="'board', 'pieces', or 'all'"),
    dest: Path = typer.Option(CACHE_DIR / "models", help="Destination directory"),
):
    """
    Download model assets from the latest GitHub release into a local directory.
    """
    valid = list(DEFAULT_MODELS) + ["all"]
    if which not in valid:
        raise typer.BadParameter(f"Must be one of {valid}")
    names = list(DEFAULT_MODELS) if which == "all" else [which]
    msg = f"Download {', '.join(names)} to {dest}?"
    typer.confirm(msg, abort=True)
    for n in names:
        filename = DEFAULT_MODELS[n]
        download_models(filename, dest / filename)


@models_app.command(help="Search for models")
def locate():
    """
    Shows where each model would be resolved from by default.
    """
    for n in DEFAULT_MODELS:
        p = find_model(n)
        status = str(p) if p else "(not found)"
        typer.echo(f"{n} model: {status}")


@app.command(help="Process a video into PGN")
def run(
    video: Path = typer.Option(..., "--video", exists=True, readable=True, help="Path to input video"),
    board_model: Optional[Path] = typer.Option(None, "--board-model", help="Path to the board model"),
    piece_model: Optional[Path] = typer.Option(None, "--piece-model", help="Path to the piece model"),
    out: Optional[Path] = typer.Option(None, "--out", help="Path for output PGN"),
    show: bool = typer.Option(False, "--show/--no-show", help="Display windows"),
):
    """
    Runs video inferencing and outputs PGN.
    """
    out_path = out if out else default_pgn_out_path(video)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    board_path = board_model if valid_model(model=board_model) else find_model('board')
    pieces_path = piece_model if valid_model(model=piece_model) else find_model('pieces')
    typer.echo(f"Using models:  {board_path}, {pieces_path}")

    run_video_to_pgn(video=str(video),
                     out=str(out_path),
                     board_model=str(board_path),
                     piece_model=str(pieces_path),
                     show=show)
    typer.echo(f"PGN saved at {out_path}")

if __name__ == "__main__":
    app()
