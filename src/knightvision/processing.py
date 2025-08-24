from typing import Any, Optional, cast, List

import numpy as np
import cv2
import chess


def image_resize(image: cv2.typing.MatLike, new_size: int) -> cv2.typing.MatLike:
    """
    Resizes longest image edge to new size while maintaining aspect ratio.
    """
    h, w = image.shape[0], image.shape[1]
    if w > h:
        scale = new_size / w
        height = int((np.ceil(h * scale / 32)) * 32)
        dim = (new_size, height)
    else:
        scale = new_size / h
        width = int((np.ceil(w * scale / 32)) * 32)
        dim = (width, new_size)
    return cv2.resize(image, dim)

def _letterbox(img: np.ndarray, new_shape: tuple[int, int] = (640, 640)) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Resize and reshape images while maintaining aspect ratio by adding padding.
    """
    shape = img.shape[:2]
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    return img, (top, left)

def preprocess(input_img: cv2.typing.MatLike, input_shapes: tuple[int, int], half: bool = False) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Adjusts input image to expected input for model.
    """
    img = input_img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img, pad = _letterbox(img, input_shapes)
    image_data = np.array(img) / 255.0
    image_data = np.transpose(image_data, (2, 0, 1))  # Channel first
    ndtype = np.half if half else np.single
    image_data = np.expand_dims(image_data, axis=0).astype(ndtype)
    return image_data, pad

def color_postprocess(img: cv2.typing.MatLike, output: list[np.ndarray], pad: tuple[int, int], input_shapes: tuple[int, int]) -> list[list[int]]:
    """
    Post-process the chess pieces prediction.
    """
    img_height, img_width = img.shape[:2]
    input_height, input_width = input_shapes
    confidence_thres, iou_thres = 0.4, 0.7
    outputs = np.transpose(np.squeeze(output[0]))
    rows = outputs.shape[0]
    centers = []
    boxes = []
    scores = []
    class_ids = []
    gain = min(input_height / img_height, input_width / img_width)
    outputs[:, 0] -= pad[1]
    outputs[:, 1] -= pad[0]
    for i in range(rows):
        classes_scores = outputs[i][4:]
        max_score = np.amax(classes_scores)
        if max_score >= confidence_thres:
            class_id = np.argmax(classes_scores)
            x, y, w, h = outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]
            left = int((x - w / 2) / gain)
            top = int((y - h / 2) / gain)
            width = int(w / gain)
            height = int(h / gain)
            x = int(x / gain)
            y = int(y / gain)
            class_ids.append(class_id)
            scores.append(max_score)
            boxes.append([left, top, width, height])
            centers.append([x, y, width, height])
    indices = cv2.dnn.NMSBoxes(boxes, scores, confidence_thres, iou_thres)
    piece_data = []
    for i in indices:
        center = centers[i]
        class_id = class_ids[i]
        piece_data.append([int(center[0]), int(center[1]), int(center[2]), int(center[3]), int(class_id)])
    return piece_data

def board_postprocess(img: cv2.typing.MatLike, output: list[np.ndarray], pad: tuple[int, int], input_shapes: tuple[int, int]) -> Any:
    """
    Post-process the chessboard prediction.
    """
    x, protos = output[0], output[1]  # Two outputs: predictions and protos
    pad_h, pad_w = pad
    img_height, img_width = img.shape[:2]
    input_width, input_height = input_shapes
    ratio = min(input_height / img_height, input_width / img_width)
    nm = 32
    conf_threshold = .4
    iou_threshold = .7
    x = np.einsum("bcn->bnc", x)
    x = x[np.amax(x[..., 4:-nm], axis=-1) > conf_threshold]
    x = np.c_[x[..., :4], np.amax(x[..., 4:-nm], axis=-1), np.argmax(x[..., 4:-nm], axis=-1), x[..., -nm:]]
    x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]
    if len(x) > 0:
        x[..., [0, 1]] -= x[..., [2, 3]] / 2
        x[..., [2, 3]] += x[..., [0, 1]]
        x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
        x[..., :4] /= ratio
        x[..., [0, 2]] = x[:, [0, 2]].clip(0, img.shape[1])
        x[..., [1, 3]] = x[:, [1, 3]].clip(0, img.shape[0])
        masks = process_mask(protos[0], x[:, 6:], x[:, :4], img.shape)
        return masks
    else:
        return []

def crop_mask(masks, boxes):
    """
    Takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
    r = np.arange(w, dtype=x1.dtype)[None, None, :]
    c = np.arange(h, dtype=x1.dtype)[None, :, None]
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

def process_mask(protos, masks_in, bboxes, im0_shape):
    """
    Takes the output of the mask head, and applies the mask to the bounding boxes.
    """
    c, mh, mw = protos.shape
    masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
    masks = np.ascontiguousarray(masks)
    masks = scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
    masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
    masks = crop_mask(masks, bboxes)
    return np.greater(masks, 0.5)

def scale_mask(masks, im0_shape, ratio_pad=None):
    """
    Takes a mask, and resizes it to the original image size
    """
    im1_shape = masks.shape[:2]
    if ratio_pad is None:
        gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
        pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
    else:
        pad = ratio_pad[1]
    top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
    bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
    masks = masks[top:bottom, left:right]
    masks = cv2.resize(masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR)
    if len(masks.shape) == 2:
        masks = masks[:, :, None]
    return masks

def _mark_corners(segmentation: np.ndarray) -> list[list[int]]:
    """
    Estimates corners from segmentation.
    """
    mask = segmentation.astype(np.uint8)
    mask *= 255
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    approx = cv2.approxPolyDP(hull, epsilon=50, closed=True)  # Epsilon magic number
    corners = cast(List[List[int]], approx.reshape(-1, 2).astype(int).tolist())
    return corners


def _sort_clockwise(raw_pts: list[list[int]]) -> list[list[int]]:
    """
    Sorts list of points clockwise.
    """
    center = np.mean(raw_pts, axis=0)
    angles = np.arctan2([p[1] - center[1] for p in raw_pts], [p[0] - center[0] for p in raw_pts])
    return [pt for _, pt in sorted(zip(angles, raw_pts))]


def find_chessboard_corners(corner_results_data: list[np.ndarray]) -> tuple[Optional[np.ndarray], Optional[list]]:
    """
    Uses predicted corners to create conversion matrix.
    """
    transformation_matrix = None
    corners = None
    for mask in corner_results_data:
        corners = _mark_corners(mask)
        if len(corners) == 4:
            real = np.array(_sort_clockwise(corners), dtype=np.float32)
            ideal = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)
            transformation_matrix = cv2.getPerspectiveTransform(real, ideal)
    return transformation_matrix, corners


def _adjust_for_angle(y: float, height: float) -> int:
    """
    Shifts y value of piece location down by 'magic number' percent of box height.
    """
    return int(y + 0.30 * height) # Magic number


def _map_points(num: int, board_size: int = 400) -> int:
    """
    Adjusts transformed coordinates to 8x8 grid.
    """
    return int(np.floor(num * 8 / board_size))


def orient_board(piece_results_data: list[list[int]], transformation_matrix: Optional[np.ndarray]) -> str:
    """
    Takes piece model predictions and outputs flag for board orientation.
    """
    rotation = "NONE"
    white_x, white_y, black_x, black_y = [], [], [], []
    if transformation_matrix is not None:
        for piece in piece_results_data:
            real_x, real_y = piece[0], piece[1]
            real_y = _adjust_for_angle(real_y, height=abs(piece[3]))
            ideal = np.matmul(transformation_matrix, [real_x, real_y, 1])
            if 0 <= ideal[0] / ideal[2] < 400 and 0 <= ideal[1] / ideal[2] < 400:
                raw_name = int(piece[4])
                ideal_x, ideal_y = _map_points(ideal[1] / ideal[2]), _map_points(ideal[0] / ideal[2])
                if raw_name < 1:
                    black_x.append(ideal_x)
                    black_y.append(ideal_y)
                else:
                    white_x.append(ideal_x)
                    white_y.append(ideal_y)
        average_white_x, average_white_y = np.average(white_x), np.average(white_y)
        average_black_x, average_black_y = np.average(black_x), np.average(black_y)
        if average_white_x > 5 > average_white_y > 2 and average_black_x < 2 < average_black_y < 5:
            rotation =  "ROTATE 90 COUNTERCLOCKWISE"
        elif average_black_x > 5 > average_black_y > 2 and average_white_x < 2 < average_white_y < 5:
            rotation = "ROTATE 90 CLOCKWISE"
        elif average_black_y > 5 > average_black_x > 2 and average_white_y < 2 < average_white_x < 5:
            rotation = "ROTATE 180"
        elif average_white_y > 5 > average_white_x > 2 and average_black_y < 2 < average_black_x < 5:
            rotation = "NONE"
    return rotation


def locate_pieces(piece_results_data: list[list[int]], transformation_matrix: Optional[np.ndarray], rotate_board: str) -> dict[chess.Square, int]:
    """
    Transforms chess piece image data into raw numpy board.
    """
    out_board_map = dict()
    piece_names, mapped_pts = [], []
    if transformation_matrix is not None:
        for piece in piece_results_data:
            real_x, real_y = piece[0], piece[1]
            real_y = _adjust_for_angle(real_y, height=abs(piece[3]))
            ideal = np.matmul(transformation_matrix, [real_x, real_y, 1])
            if 0 <= ideal[0] / ideal[2] < 400 and 0 <= ideal[1] / ideal[2] < 400:
                piece_names.append(piece[4])
                mapped_pts.append((_map_points(ideal[1] / ideal[2]), _map_points(ideal[0] / ideal[2])))
        if rotate_board == "ROTATE 90 CLOCKWISE":
            mapped_pts = [(7 - pt[1], pt[0]) for pt in mapped_pts]
        elif rotate_board == "NONE":
            mapped_pts = [(7 - pt[0], 7 - pt[1]) for pt in mapped_pts]
        elif rotate_board == "ROTATE 90 COUNTERCLOCKWISE":
            mapped_pts = [(pt[1], 7 - pt[0]) for pt in mapped_pts]
        for index, (row, column) in enumerate(mapped_pts):
            out_board_map[chess.square(row, column)] = piece_names[index]
    return out_board_map
