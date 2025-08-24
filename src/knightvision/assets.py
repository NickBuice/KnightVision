import shutil
import urllib.request
from pathlib import Path
from typing import Optional

CACHE_DIR = Path.home() / ".knightvision"

DEFAULT_MODELS = {
    "board":  "board-detector.onnx",
    "pieces": "piece-detector.onnx",
}

def _model_url(filename: str) -> str:
    """
    Most recent repo release url.
    """
    return f"https://github.com/nickbuice/knightvision/releases/latest/download/{filename}"

def download_models(filename: str, dest: Path, timeout: int = 60) -> None:
    """
    Downloads file from url.
    """
    url = _model_url(filename)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)

def find_model(name: str) -> Optional[Path]:
    """
    Searches for pre-existing models.
    """
    if name not in DEFAULT_MODELS:
        raise ValueError(f"Unknown model name '{name}'. Known: {list(DEFAULT_MODELS)}")
    filename = DEFAULT_MODELS[name]
    candidates: list[Path] = [Path.cwd() / "models" / filename, CACHE_DIR / "models" / filename]
    for p in candidates:
        if p.exists():
            return p
    return None

def valid_model(model: Optional[Path]) -> bool:
    """
    Verifies existence of model path.
    """
    if model:
        if model.exists():
            return True
    return False
