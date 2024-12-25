import numpy as np
import cv2
import chess
from typing import Any, Optional


def adjust_for_angle(y: float, height: float) -> float:
    """
    Shifts y value of piece location down by 'magic number' percent of box height.
    """
    return y + 0.30 * height # Magic number


# def round_square(num: int, board_size: int = 400) -> int:
#     """
#     Adjusts transformed coordinates to 8x8 grid
#     """
#     square_size = board_size / 8
#     shift = square_size / 2
#     value = square_size * round((num - shift) / square_size) + shift
#     if value < shift:
#         value = shift
#     elif value > board_size - shift:
#         value = board_size - shift
#     return int(value)

def new_round_square(num: int, board_size: int = 400) -> int:
    """
    Adjusts transformed coordinates to 8x8 grid.
    """
    return int(np.floor(num * 8 / board_size))


def locate_pieces(results_img: cv2.typing.MatLike, piece_results_data: Any, transformation_matrix: Optional[np.ndarray],
                  rotate_board: bool) -> tuple[cv2.typing.MatLike, np.ndarray]:
    """
    Transforms chess piece image data into raw numpy board.
    """
    key = {1: 'b', 2: 'k', 3: 'n', 4: 'p', 5: 'q', 6: 'r', 7: 'B', 8: 'K', 9: 'N', 10: 'P', 11: 'Q', 12: 'R'}
    out_board = chess.Board(fen="8/8/8/8/8/8/8/8")
    raw_board = np.zeros((8, 8))
    piece_names, mapped_pts = [], []
    result = piece_results_data[0]
    results_img = result.plot(img=results_img)
    if transformation_matrix is not None:
        for box in result.boxes.data.cpu().numpy():
            real_x, real_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            real_y = adjust_for_angle(real_y, height=abs(box[3] - box[1]))
            ideal = np.matmul(transformation_matrix, [real_x, real_y, 1])
            if 0 <= ideal[0] / ideal[2] < 400 and 0 <= ideal[1] / ideal[2] < 400:
                piece_names.append(int(box[5] + 1))
                mapped_pts.append((7 - new_round_square(ideal[1] / ideal[2]), new_round_square(ideal[0] / ideal[2])))
        if rotate_board:
            mapped_pts = [(7 - pt[1], pt[0]) for pt in mapped_pts]  # white on right
            #  mapped_pts = [(pt[1],  7 - pt[0]) for pt in mapped_pts]  # white on left
        for index, (row, column) in enumerate(mapped_pts):
            raw_board[column, row] = piece_names[index]
            out_board.set_piece_at(chess.square(row, column), chess.Piece.from_symbol(key[piece_names[index]]))
    else:
        print("*Board Not Detected*")
    return results_img, raw_board

def new_locate_pieces(results_img: cv2.typing.MatLike, piece_results_data: Any, transformation_matrix: Optional[np.ndarray],
                  rotate_board: bool) -> tuple[cv2.typing.MatLike, np.ndarray]:
    """
    Transforms chess piece image data into raw numpy board.
    """
    key = {'0': 0, 'b': 1, 'k': 2, 'n': 3, 'p': 4, 'q': 5, 'r': 6, 'B': 7, 'K': 8, 'N': 9, 'P': 10, 'Q': 11, 'R': 12}
    out_board = chess.Board()
    piece_names, mapped_pts = [], []
    result = piece_results_data[0]
    results_img = result.plot(img=results_img)
    if transformation_matrix is not None:
        for box in result.boxes.data.cpu().numpy():
            real_x, real_y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            real_y = adjust_for_angle(real_y, height=abs(box[3] - box[1]))
            ideal = np.matmul(transformation_matrix, [real_x, real_y, 1])
            if 0 <= ideal[0] / ideal[2] < 400 and 0 <= ideal[1] / ideal[2] < 400:
                piece_names.append(int(box[5] + 1))
                mapped_pts.append((new_round_square(ideal[1] / ideal[2]), 7 - new_round_square(ideal[0] / ideal[2])))
        if rotate_board:
            mapped_pts = [(7 - pt[1], pt[0]) for pt in mapped_pts]  # white on right
            #  mapped_pts = [(pt[1],  7 - pt[0]) for pt in mapped_pts]  # white on left
        for index, (row, column) in enumerate(mapped_pts):
            out_board.set_piece_at(chess.square(row, column), chess.Piece.from_symbol(key[piece_names[index]]))
    else:
        print("*Board Not Detected*")
    return results_img, out_board
