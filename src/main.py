import cv2
from ultralytics import YOLO
import logging
from src import chessCorners, chessGeneral, chessLocations
import time


on_startup, image_output_size, image_size = True, 640, 640  # Magic Numbers
logging.basicConfig(filename='../misc/example.log', filemode='w', level=logging.DEBUG)
video_capture = cv2.VideoCapture(0)
corner_prediction_model = YOLO('../models/BoardPredictionModels/best.pt')
piece_prediction_model = YOLO('../models/PiecePredictionModels/best.pt')
while video_capture.isOpened():
    successful_capture, frame = video_capture.read()
    if not successful_capture:
        break
    img = chessGeneral.image_resize(frame, image_size)
    piece_results = piece_prediction_model.predict(img, imgsz=image_size, agnostic_nms=True, verbose=False)
    key_press = cv2.waitKey(1)
    if key_press == ord(" ") or on_startup:
        start, count, on_startup = time.time(), 0, False
        corner_results = corner_prediction_model.predict(img, imgsz=image_size, verbose=False)
        conversion_matrix, rotation = chessCorners.find_chessboard_corners(img, corner_results)
        chess_game = chessGeneral.StartChessGame(board_delay=20)
        chess_game.show_chessboard(force=True)
    else:
        img, chess_game.raw_board = chessLocations.locate_pieces(img, piece_results, conversion_matrix, rotation)
        chess_game.update_board_stack(), chess_game.update_move_stack()
        if chess_game.board_has_changed():
            chess_game.update_chessboard()
            chess_game.skipped_move_search()
            if fixes:= chess_game.fix_skipped_move():
                chess_game.push_fix(fixes)
        chess_game.show_chessboard()
    if key_press == ord("q"):
        break
    cv2.imshow("Capture", chessGeneral.image_resize(img, image_output_size))
    count += 1
logging.info("AVERAGE LOOP TIME: %s", (time.time() - start)/count)
chessGeneral.pool.shutdown(wait=True)
video_capture.release()
cv2.destroyAllWindows()
