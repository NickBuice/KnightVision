import numpy as np
import cv2
from typing import Any


def mark_corners(segmentation: np.ndarray) -> list[tuple[int, int]]:
    """
    Estimates corners from YOLO segmentation.
    """
    segmentation *= 255
    mask = segmentation.astype(np.uint8)
    contours, hierarchy = cv2.findContours(mask, 1, 2)
    cnt = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(cnt)
    approx = cv2.approxPolyDP(hull, epsilon=50, closed=True)  # Epsilon magic number
    corners = [tuple(pt[0]) for pt in approx]
    return corners

def sort_clockwise(raw_pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Sorts list of points clockwise.
    """
    center = np.mean(raw_pts, axis=0)
    angles = np.arctan2([p[1] - center[1] for p in raw_pts], [p[0] - center[0] for p in raw_pts])
    return [pt for _, pt in sorted(zip(angles, raw_pts))]


def find_chessboard_corners(corner_results_data: Any) -> np.ndarray:
    """
    Uses predicted corners to create conversion matrix.
    """
    transformation_matrix = None
    for result in corner_results_data:
        result_masks = result.masks
        if result_masks is not None:
            for mask in result_masks.data:
                corners = mark_corners(mask.cpu().numpy())
                if len(corners) == 4:
                    real = np.array(sort_clockwise(corners), dtype=np.float32)
                    ideal = np.array([[0, 0], [400, 0], [400, 400], [0, 400]], dtype=np.float32)
                    transformation_matrix = cv2.getPerspectiveTransform(real, ideal)
    return transformation_matrix
