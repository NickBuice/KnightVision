from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import onnxruntime as ort


def _pick_device(device: str = "CPU") -> list[str]:
    """
    Attempts to run on GPU if selected then falls back to CPU inferencing
    """
    if device == "GPU":
        if ort.get_device() == "GPU":
            ort.preload_dlls()
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            print("GPU setup failed, falling back to CPU")
            return ["CPUExecutionProvider"]
    else:
        return ["CPUExecutionProvider"]


def _must_exist(path: Path) -> Path:
    """
    Checks existence of model path.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}\n")
    return path


@dataclass
class Models:
    """
    Thin container around onnx models.
    """
    board: ort.InferenceSession
    color: ort.InferenceSession
    board_inputs: Sequence[ort.NodeArg]
    color_inputs: Sequence[ort.NodeArg]

    def predict_board(self, img_data: np.ndarray):
        """
        Runs the board model on a frame.
        """
        return self.board.run(None, {self.board_inputs[0].name: img_data})

    def predict_color(self, img_data: np.ndarray):
        """
        Runs the piece color classifier on a frame.
        """
        return self.color.run(None, {self.color_inputs[0].name: img_data})


def load_models(board_model: str, color_model: str, device: str = "CPU") -> tuple[Models, tuple[int, int], tuple[int, int], bool, bool]:
    """
    Load both models once.
    """
    providers=_pick_device(device)

    board_path = Path(board_model)
    color_path = Path(color_model)
    board = ort.InferenceSession(str(_must_exist(board_path)), providers=providers)
    color = ort.InferenceSession(str(_must_exist(color_path)), providers=providers)

    # Get the model inputs
    board_inputs: Sequence[ort.NodeArg] = board.get_inputs()
    color_inputs: Sequence[ort.NodeArg] = color.get_inputs()

    # Get the model precision
    board_half = board_inputs[0].type == "tensor(float16)"
    color_half = color_inputs[0].type == "tensor(float16)"

    # Store the shape of the input for later use
    board_input_shape: tuple = board_inputs[0].shape
    board_input_height: int = board_input_shape[2]
    board_input_width: int = board_input_shape[3]
    color_input_shape: tuple = color_inputs[0].shape
    color_input_height: int = color_input_shape[2]
    color_input_width: int = color_input_shape[3]

    models = Models(board=board, color=color, board_inputs=board_inputs, color_inputs=color_inputs)

    return models, (board_input_height, board_input_width), (color_input_height, color_input_width), board_half, color_half
