import cv2
import chess.pgn
from ultralytics import YOLO  # type: ignore
import logging
from src import chessCorners, chessGeneral, chessLocations
import time


base_path = "C://PythonStuff/Projects/ChessScanner"
board_output_size, image_output_size, image_size = 600, 640, 640  # Magic Numbers
conversion_matrix, rotation = None, False
chess_game = chessGeneral.StartChessGame()
logging.basicConfig(filename='../misc/example.log', filemode='w', level=logging.DEBUG)
video_capture = cv2.VideoCapture(1)
corner_prediction_model = YOLO(f'{base_path}/CloudModels/BoardModels/NanoA100_BEST/train/weights/best.pt')
piece_prediction_model = YOLO(f'{base_path}/CloudModels/BestYet/train/weights/best.pt')
on_startup = True
start, count = time.time(), 0
while video_capture.isOpened():  # todo rework this logic to be cleaner
    count += 1
    _, frame = video_capture.read()
    corrected_frame = chessGeneral.image_resize(frame, image_size)
    piece_results = piece_prediction_model.predict(corrected_frame, imgsz=image_size, agnostic_nms=True, verbose=False)
    key_press = cv2.waitKey(1)
    if key_press == ord(" ") or on_startup:
        start, count = time.time(), 0
        corner_results = corner_prediction_model.predict(corrected_frame, imgsz=image_size, verbose=False)
        conversion_matrix, rotation = chessCorners.find_chessboard_corners(corrected_frame, corner_results)
        chess_game = chessGeneral.StartChessGame()
        img, chess_game.old_np_board = chessLocations.locate_pieces(corrected_frame, piece_results, conversion_matrix,
                                                                    rotation, chess_game.new_np_board)
        chessGeneral.pool.submit(chessGeneral.show_svg_display, chess_game.chessboard, 600)
        chessGeneral.pool.submit(chessGeneral.write_pgn_to_file, chess_game.pgn_file, chess_game.game)
        logging.info("--- Resetting calibration ---")
        on_startup = False
    else:
        img, chess_game.old_np_board = chessLocations.locate_pieces(corrected_frame, piece_results, conversion_matrix,
                                                                    rotation, chess_game.new_np_board)
        if chess_game.board_has_changed():
            chess_game.update_board_and_waiting_move_stack()
        else:
            chessGeneral.pool.submit(chessGeneral.show_same_display)
    if key_press == ord("q"):
        break
    cv2.imshow("Capture", chessGeneral.image_resize(img, image_output_size))
logging.info("AVERAGE LOOP TIME: %s", (time.time() - start)/count)
chessGeneral.pool.shutdown(wait=True)
video_capture.release()
cv2.destroyAllWindows()
