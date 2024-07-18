# import cv2
# from ultralytics import YOLO
# import onnxruntime
# import numpy as np
# import chessCorners
# import chessGeneral
# import time
import chess
import chess.svg
import cairo
import rsvg
import tkinter as tk
import tksvg
#
#
#
#
# board_output_size, image_output_size, image_size = 600, 640, 640
# video_capture = cv2.VideoCapture(1)
# fps = video_capture.get(cv2.CAP_PROP_FPS)
# print(fps)
# corner_prediction_model = YOLO('C://PythonStuff/ChessScanner/CloudModels/BoardModels/NanoA100_BEST/train/weights/best.pt')
# # piece_prediction_model = YOLO('C://PythonStuff/ChessScanner/CloudModels/ColorOnly/train2/weights/best.onnx')
# session = onnxruntime.InferenceSession('C://PythonStuff/ChessScanner/CloudModels/ColorOnly/train2/weights/best.onnx', providers=['CPUExecutionProvider'])
# count = 0
# start = time.time()
# while video_capture.isOpened():
#     count += 1
#     _, frame = video_capture.read()
#     corrected_frame = chessGeneral.image_resize(frame, image_size)
#     X = np.asarray(corrected_frame)
#     ortvalue = onnxruntime.OrtValue.ortvalue_from_numpy(X.astype(np.float32))
#     print(ortvalue.device_name())
#     input_name = session.get_inputs()[0].name
#     piece_results = session.run_with_ort_values(None, {input_name: ortvalue})
#     print(piece_results[0])
#     key_press = cv2.waitKey(1)
#     # corner_results = corner_prediction_model.predict(corrected_frame, imgsz=image_size, verbose=False)
#     # conversion_matrix, rotation = chessCorners.find_chessboard_corners(corrected_frame, corner_results)
#     if key_press == ord(" "):
#         count = 0
#         start = time.time()
#         print("reset")
#     if key_press == ord("q"):
#         print("Exit key pressed")
#         break
#     cv2.imshow("Capture", corrected_frame)
# print("AVERAGE LOOP TIME:", (time.time() - start)/count)
# # print(piece_results[0])
# chessGeneral.pool.shutdown(wait=True)
# video_capture.release()
# cv2.destroyAllWindows()

board = chess.Board("8/8/8/8/4N3/8/8/8 w - - 0 1")

chessboard = chess.svg.board(
    board,
    fill=dict.fromkeys(board.attacks(chess.E4), "#cc0000cc"),
    arrows=[chess.svg.Arrow(chess.E4, chess.F6, color="#0000cccc")],
    squares=chess.SquareSet(chess.BB_DARK_SQUARES & chess.BB_FILE_B),
    size=350,
    )
