from pathlib import Path
from typing import Iterator, Union

import cv2

from .processing import image_resize


def iterate_frames(source: Union[str, int], image_size: int = 640) -> Iterator[tuple[int, cv2.typing.MatLike]]:
    """
    Takes in video source and desired frame size and yields frames.
    """
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open source: {source}")
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = image_resize(frame, image_size)
            yield index, frame
            index += 1
    finally:
        cap.release()

def default_pgn_out_path(video: Union[Path, str]) -> Path:
    """
    Sets and creates default output path.
    """
    if isinstance(video, Path):
        out = video.stem
    else:
        out = "out"
    runs = Path.cwd() / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    return runs / f"{out}.pgn"
