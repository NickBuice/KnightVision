import numpy as np
import cv2




def find_intersections(line_endpoints):
    intersections = []
    parallel_threshold = 45 * np.pi/180 # Magic Number
    slopes = [(pt[1][1] - pt[0][1]) / ((pt[1][0] - pt[0][0]) + .0001) for pt in line_endpoints]  # Magic number
    y_intercepts = [pt[0][1] - slope*pt[0][0] for slope, pt in zip(slopes, line_endpoints)]
    for i, (slope1, y_intercept1) in enumerate(zip(slopes, y_intercepts)):
        for j in range(i + 1, len(slopes)):
            slope2, y_intercept2 = slopes[j], y_intercepts[j]
            nearly_perpendicular = abs(np.arctan(abs((slope2 - slope1)/(1 + slope2*slope1)))) > parallel_threshold
            if nearly_perpendicular:
                output_point_x = (y_intercept2 - y_intercept1) / (slope1 - slope2)
                output_point_y = slope1 * output_point_x + y_intercept1
                intersections.append([int(output_point_x), int(output_point_y)])
    return intersections


def cluster(points):
    clustered_points = []
    for i, pt in enumerate(points):
        x1, y1 = pt[0], pt[1]
        temp_x = [x1]
        temp_y = [y1]
        for j in reversed(range(i + 1, len(points))):
            x2, y2 = points[j][0], points[j][1]
            cluster_threshold = 100  # Magic Number
            if abs(x2 - x1) <= cluster_threshold and abs(y2 - y1) <= cluster_threshold:
                temp_x.append(x2)
                temp_y.append(y2)
                del points[j]
        clustered_points.append([int(np.average(temp_x)), int(np.average(temp_y))])
    return clustered_points


def mark_corners(segmentation):
    potential_corners = []
    segmentation *= 255
    thresh = segmentation.astype(np.uint8)
    gray = cv2.Canny(thresh, 50, 200, None, 3)
    lines = cv2.HoughLines(gray, 1, np.pi / 180, 50, None, 0, 0)  # Magic Number 50
    line_points = []
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = [int(x0 + 10000 * (-b)), int(y0 + 10000 * a)] # Magic Number 10000
            pt2 = [int(x0 - 10000 * (-b)), int(y0 - 10000 * a)] # Magic Number 10000
            line_points.append([pt1, pt2])
        many_potential_corners = find_intersections(line_points)
        potential_corners = cluster(many_potential_corners)
    return potential_corners


def sort_clockwise(raw_pts):
    center = [sum(pt[0] for pt in raw_pts) / len(raw_pts), sum(pt[1] for pt in raw_pts) / len(raw_pts)]
    adjusted = [[pt[0] - center[0], pt[1] - center[1]] for pt in raw_pts]
    angles = []
    for pt in adjusted:
        if pt[0] > 0:
            angles.append(np.arctan(pt[1]/pt[0]))
        elif pt[0] < 0:
            angles.append(np.arctan(pt[1]/pt[0]) + np.pi)
        else:
            angles.append(np.pi / 2)
    return [pt for _, pt in sorted(zip(angles, raw_pts))]


def board_rotation(birdseye_img, size=400):
    gray_birdseye_img = cv2.cvtColor(birdseye_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray_birdseye_img, 150, 255, cv2.THRESH_BINARY)
    fake = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if (i // (size/8)) % 2 == (j // (size/8)) % 2:
                fake[i][j] = 255
    diff = abs(fake - thresh)  # play with threshold difference values due to board rotating only sometimes
    return np.average(diff) > np.average(thresh)


def find_chessboard_corners(results_img, corner_results_data):
    top_view = np.zeros((400, 400, 3)).astype(np.uint8)
    transformation_matrix = None
    for result in corner_results_data:
        result_masks = result.masks
        if result_masks is not None:
            for mask in result_masks.data:
                corners = mark_corners(mask.cpu().numpy())
                if len(corners) == 4:
                    real = np.float32(sort_clockwise(corners))
                    ideal = np.float32([[0, 0], [400, 0], [400, 400], [0, 400]])
                    transformation_matrix = cv2.getPerspectiveTransform(real, ideal)
                    top_view = cv2.warpPerspective(results_img, transformation_matrix, (400, 400))
                    top_view = cv2.rotate(top_view, cv2.ROTATE_90_CLOCKWISE)
    rotate_board = board_rotation(top_view, 400)
    return transformation_matrix, rotate_board
