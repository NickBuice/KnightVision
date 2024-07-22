import cv2
from ultralytics import YOLO
import onnxruntime as ort
import numpy as np
from src import chessGeneral

#
# frame = cv2.imread("C:/PythonStuff/Projects/ChessScanner/ChessBoardImages/board2.jpg")
# image = chessGeneral.image_resize(frame, 352)
# image = cv2.copyMakeBorder(image, 64, 64, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0])
# image_array = np.array(image).astype(np.float16)
# print(image.shape)
# corrected_frame = image_array.reshape((1, 3, 288, 480))
# print(corrected_frame.shape)
# session = ort.InferenceSession('C://PythonStuff/Projects/ChessScanner/CloudModels/PbatchModel/Pbatch.onnx',
#                                providers=['CPUExecutionProvider'])
# piece_results = session.run(None, {'images': corrected_frame})
# print(np.shape(piece_results))
# print(piece_results[0][0][0])
# cv2.imshow("Capture", image)
# key_press = cv2.waitKey(0)
# cv2.destroyAllWindows()

model = YOLO('C://PythonStuff/Projects/ChessScanner/CloudModels/BestYet/train/weights/best.onnx')

im = cv2.imread("C:/PythonStuff/Projects/ChessScanner/ChessBoardImages/board2.jpg")
im = chessGeneral.image_resize(im, 640)
results = model.predict(source=im, imgsz=640, device='CPU')

for result in results:
    boxes = result.boxes  # Boxes object for bbox outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    probs = result.probs  # Class probabilities for classification outputs
print(boxes.data)
