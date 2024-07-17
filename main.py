import cv2
import chess.pgn
from ultralytics import YOLO
import logging
import chessCorners
import chessLocations
import chessGeneral
import time




pgn_file = "C:/Users/nbuic/OneDrive/Desktop/TestingPGN/TEST.pgn"
board_output_size, image_output_size, image_size = 600, 640, 640  # Magic Numbers
conversion_matrix, rotation = None, False
game = chessGeneral.StartChessGame()
logging.basicConfig(filename='./misc/example.log', filemode='w', level=logging.DEBUG)
video_capture = cv2.VideoCapture(1)
corner_prediction_model = YOLO('C://PythonStuff/ChessScanner/CloudModels/BoardModels/NanoA100_BEST/train/weights/best.pt')
piece_prediction_model = YOLO('C://PythonStuff/ChessScanner/CloudModels/BestYet/train/weights/best.pt')
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
        game = chessGeneral.StartChessGame()
        img, game.old_np_board = chessLocations.locate_pieces(corrected_frame, piece_results, conversion_matrix, rotation, game.np_board)
        chessGeneral.pool.submit(chessGeneral.show_svg_display, chessGeneral.write_fen(game.np_board), 600)
        chessGeneral.pool.submit(chessGeneral.write_pgn_to_file, pgn_file, game.game)
        logging.info("--- Resetting calibration ---")
        on_startup = False
    else:
        img, game.old_np_board = chessLocations.locate_pieces(corrected_frame, piece_results, conversion_matrix, rotation, game.np_board)
        if game.board_has_changed():
            game.update_board_and_waiting_move_stack()
        else:
            chessGeneral.pool.submit(chessGeneral.show_same_display)
        game.update_move_stack()
        if chess_move_to_be_pushed := game.moves[0]:
            if chess_move_to_be_pushed in [chess.Move.uci(legal_move) for legal_move in game.chessboard.legal_moves]:
                game.node = game.node.add_variation(chess.Move.from_uci(chess_move_to_be_pushed))
                game.chessboard.push_uci(chess_move_to_be_pushed)
                chessGeneral.pool.submit(chessGeneral.write_pgn_to_file, pgn_file, game.game)
                logging.info("PGN:\n%s", game.game)
    if key_press == ord("q"):
        break
    cv2.imshow("Capture", chessGeneral.image_resize(img, image_output_size))
logging.info("AVERAGE LOOP TIME: %s", (time.time() - start)/count)
chessGeneral.pool.shutdown(wait=True)
video_capture.release()
cv2.destroyAllWindows()
