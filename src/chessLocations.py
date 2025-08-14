import numpy as np
import cv2
import chess
from typing import Any, Optional


def adjust_for_angle(y: float, height: float) -> float:
    """
    Shifts y value of piece location down by 'magic number' percent of box height.
    """
    return y + 0.30 * height # Magic number


def map_points(num: int, board_size: int = 400) -> int:
    """
    Adjusts transformed coordinates to 8x8 grid.
    """
    return int(np.floor(num * 8 / board_size))


def orient_board(piece_results_data: Any, transformation_matrix: Optional[np.ndarray]) -> str:
    """
    Takes piece model predictions and outputs flag for board orientation.
    """
    rotation = "NONE"
    white_x, white_y, black_x, black_y = [], [], [], []
    result = piece_results_data[0]
    if transformation_matrix is not None:
        for box in result.boxes.data.cpu().numpy():
            real_x, real_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            real_y = adjust_for_angle(real_y, height=abs(box[3] - box[1]))
            ideal = np.matmul(transformation_matrix, [real_x, real_y, 1])
            if 0 <= ideal[0] / ideal[2] < 400 and 0 <= ideal[1] / ideal[2] < 400:
                raw_name = int(box[5])
                ideal_x, ideal_y = map_points(ideal[1] / ideal[2]), map_points(ideal[0] / ideal[2])
                if raw_name < 6:
                    black_x.append(ideal_x), black_y.append(ideal_y)
                else:
                    white_x.append(ideal_x), white_y.append(ideal_y)
        white_x, white_y = np.average(white_x), np.average(white_y)
        black_x, black_y = np.average(black_x), np.average(black_y)
        if white_x > 5 > white_y > 2 and black_x < 2 < black_y < 5:
            rotation =  "ROTATE 90 COUNTERCLOCKWISE"
        elif black_x > 5 > black_y > 2 and white_x < 2 < white_y < 5:
            rotation = "ROTATE 90 CLOCKWISE"
        elif black_y > 5 > black_x > 2 and white_y < 2 < white_x < 5:
            rotation = "ROTATE 180"
        elif white_y > 5 > white_x > 2 and black_y < 2 < black_x < 5:
            rotation = "NONE"
    else:
        print("*Board Not Detected*")
    return rotation


def locate_pieces(results_img: cv2.typing.MatLike, piece_results_data: Any, transformation_matrix: Optional[np.ndarray],
                  rotate_board: str) -> tuple[cv2.typing.MatLike, dict]:
    """
    Transforms chess piece image data into raw numpy board.
    """
    out_board_map = dict()
    piece_names, mapped_pts = [], []
    result = piece_results_data[0]
    results_img = result.plot(img=results_img)
    if transformation_matrix is not None:
        for box in result.boxes.data.cpu().numpy():
            real_x, real_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            real_y = adjust_for_angle(real_y, height=abs(box[3] - box[1]))
            ideal = np.matmul(transformation_matrix, [real_x, real_y, 1])
            if 0 <= ideal[0] / ideal[2] < 400 and 0 <= ideal[1] / ideal[2] < 400:
                piece_names.append(int(box[5]))  # change (int(box[5])) to (int(box[5]) // 6) for ClassifyByType model
                mapped_pts.append((map_points(ideal[1] / ideal[2]), map_points(ideal[0] / ideal[2])))
        if rotate_board == "ROTATE 90 CLOCKWISE":
            mapped_pts = [(7 - pt[1], pt[0]) for pt in mapped_pts]
        elif rotate_board == "NONE":
            mapped_pts = [(7 - pt[0], 7 - pt[1]) for pt in mapped_pts]
        elif rotate_board == "ROTATE 90 COUNTERCLOCKWISE":
            mapped_pts = [(pt[1], 7 - pt[0]) for pt in mapped_pts]
        for index, (row, column) in enumerate(mapped_pts):
            out_board_map[chess.square(row, column)] = piece_names[index]
    return results_img, out_board_map
