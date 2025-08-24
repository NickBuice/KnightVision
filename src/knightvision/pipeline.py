from typing import Union

import cv2

from .io import iterate_frames
from .processing import find_chessboard_corners, orient_board, locate_pieces, preprocess, color_postprocess, board_postprocess
from .models import load_models
from .game import StartChessGame


def run_video_to_pgn(
    video: Union[str, int],
    board_model: str,
    piece_model: str,
    out: str = "out.pgn",
    show: bool = False,
    device: str = "CPU",
) -> None:
    """
    Loads in models, iterates through frames while inferencing chess game state, and writes to PGN.
    """
    models, board_input_shapes, piece_input_shapes, board_precision, piece_precision = load_models(board_model, piece_model, device=device)
    game = StartChessGame(board_delay=40)

    for index, frame in iterate_frames(video, 640):
        piece_name_processed_frame, piece_pad = preprocess(frame, piece_input_shapes, piece_precision)
        piece_outputs = models.predict_color(piece_name_processed_frame)
        piece_results = color_postprocess(frame, piece_outputs, piece_pad, piece_input_shapes)
        if index == 0:
            board_processed_frame, board_pad = preprocess(frame, board_input_shapes, board_precision)
            board_outputs = models.predict_board(board_processed_frame)
            corner_results = board_postprocess(frame, board_outputs, board_pad, board_input_shapes)
            conversion_matrix, corners = find_chessboard_corners(corner_results)
            rotation = orient_board(piece_results, conversion_matrix)
        else:
            raw_map = locate_pieces(piece_results, conversion_matrix, rotation)
            game.update_board_stack(raw_map)
            game.update_move_stack()
            if game.board_has_changed():
                mismatched_squares = game.find_mismatches()
                game.generate_moves_from_mismatches(mismatched_squares)
                game.validate_moves_and_push()
                game.find_and_validate_takebacks_and_push()
                out_of_turn_moves = game.find_out_of_turn_moves()
                potentially_skipped_moves = game.find_potentially_skipped_moves(out_of_turn_moves)
                game.validate_potentially_skipped_moves_and_push(potentially_skipped_moves)

        if not show:
            continue
        for piece in piece_results:
            color = (0, 255, 0) if piece[4] == 0 else (0, 0, 255)
            cv2.circle(frame, (piece[0], piece[1]), 1, color, 5)
        if corners:
            for corner in corners:
                cv2.circle(frame, corner, 1, (255, 0, 0), 5)
        cv2.imshow("video", frame)
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    game.write_pgn(out)
