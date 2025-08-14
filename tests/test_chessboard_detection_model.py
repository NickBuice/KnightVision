import cv2
from ultralytics import YOLO
import chessCorners_SHOW
from src import chessGeneral


on_startup, image_output_size, image_size = True, 640, 640  # Magic Numbers
video_capture = cv2.VideoCapture(0)
corner_prediction_model = YOLO('../models/BoardPredictionModels/best.pt')
while video_capture.isOpened():
    successful_capture, frame = video_capture.read()
    if not successful_capture:
        break
    img = chessGeneral.image_resize(frame, image_size)
    key_press = cv2.waitKey(1)
    corner_results = corner_prediction_model.predict(img, imgsz=image_size, verbose=False)
    img = corner_results[0].plot(img=img)
    if key_press == ord(" "):
        img_painted, conversion_matrix = chessCorners_SHOW.find_chessboard_corners(img, corner_results)
        print(conversion_matrix)
        cv2.imshow("Painted", chessGeneral.image_resize(img_painted, image_output_size))
    if key_press == ord("q"):
        break
    cv2.imshow("Capture", chessGeneral.image_resize(img, image_output_size))
chessGeneral.pool.shutdown(wait=True)
video_capture.release()
cv2.destroyAllWindows()
