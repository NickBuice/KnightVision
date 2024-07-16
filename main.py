import cv2
import chess.pgn
from ultralytics import YOLO
import numpy as np
import concurrent.futures
import logging
import chessCorners
import chessLocations
import chessGeneral



pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # double check max_workers count needed
file = "C:/Users/nbuic/OneDrive/Desktop/TestingPGN/TEST.pgn"
board_output_size, image_output_size, image_size = 600, 640, 640
conversion_matrix, rotation = None, False
np_board = np.zeros((8, 8))
chess_game, chess_game_node, chess_board, chess_moves, waiting_moves = chessGeneral.chess_board_and_game_startup(white="Richard Reti", black="Savielly Tartakower")
logging.basicConfig(filename='./misc/example.log', filemode='w', level=logging.DEBUG)
video_capture = cv2.VideoCapture(1)
corner_prediction_model = YOLO('C://PythonStuff/ChessScanner/CloudModels/BoardModels/NanoA100_BEST/train/weights/best.pt')
piece_prediction_model = YOLO('C://PythonStuff/ChessScanner/CloudModels/BestYet/train/weights/best.pt')
on_startup = True
while video_capture.isOpened():  # rework this logic to be cleaner
    _, frame = video_capture.read()
    corrected_frame = chessGeneral.image_resize(frame, image_size)
    piece_results = piece_prediction_model.predict(corrected_frame, imgsz=image_size, agnostic_nms=True, verbose=False)
    key_press = cv2.waitKey(1)
    if key_press == ord(" ") or on_startup:
        print("--- Resetting Calibration ---")
        corner_results = corner_prediction_model.predict(corrected_frame, imgsz=image_size, verbose=False)
        conversion_matrix, rotation = chessCorners.find_chessboard_corners(corrected_frame, corner_results)
        old_np_board = np_board
        img, np_board = chessLocations.locate_pieces(corrected_frame, piece_results, conversion_matrix, rotation, old_np_board)
        chess_game, chess_game_node, chess_board, chess_moves, waiting_moves = chessGeneral.chess_board_and_game_startup(white="Richard Reti", black="Savielly Tartakower", raw_board=np_board)
        pool.submit(chessGeneral.show_svg_display, chessGeneral.write_fen(np_board))
        pool.submit(chessGeneral.write_pgn_to_file, file, chess_game)
        logging.info("--- Resetting calibration ---")
        on_startup = False
        print(chess_game)
    else:
        old_np_board = np_board
        img, np_board = chessLocations.locate_pieces(corrected_frame, piece_results, conversion_matrix, rotation, old_np_board)
        if not np.array_equal(np_board, old_np_board):
            np_board, waiting_moves = chessGeneral.update_board_and_waiting_move_stack(np_board, old_np_board, waiting_moves)
        chess_moves = chessGeneral.update_move_stack(chess_moves, waiting_moves)
        if chess_move_to_be_pushed := chess_moves[0]:
            pool.submit(chessGeneral.show_svg_display, chessGeneral.write_fen(np_board))
            if chess_move_to_be_pushed in [chess.Move.uci(legal_move) for legal_move in chess_board.legal_moves]:
                chess_game_node = chess_game_node.add_variation(chess.Move.from_uci(chess_move_to_be_pushed))
                chess_board.push_uci(chess_move_to_be_pushed)
                pool.submit(chessGeneral.write_pgn_to_file, file, chess_game)
                logging.info("PGN:\n%s", chess_game)
                print(chess_game)
    if key_press == ord("q"):
        print("Exit key pressed")
        break
    cv2.imshow("Capture", img)
pool.shutdown(wait=True)
video_capture.release()
cv2.destroyAllWindows()
