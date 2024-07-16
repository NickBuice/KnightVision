import cv2
import chess.svg
import chess.pgn
from ultralytics import YOLO
import os
os.environ['path'] = 'C://Program Files/UniConvertor-2.0rc5/dlls'
import cairosvg
import numpy as np
import statistics
import time
import concurrent.futures
import logging

def timer_func(func):
   def function_timer(*args, **kwargs):
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "{func} took {time} seconds to complete its execution."
        logging.info(msg.format(func = func.__name__,time = runtime))
        return value
   return function_timer

# @timer_func
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
# @timer_func
def resize(image, new_size):
    h, w = image.shape[0], image.shape[1]
    if w > h:
        scale = new_size / w
        height = int((np.ceil(h * scale / 32))* 32)
        dim = (new_size, height)
    else:
        scale = new_size / h
        width = int((np.ceil(w * scale / 32)) * 32)
        dim = (width, new_size)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
# @timer_func
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
# @timer_func
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
# @timer_func
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
# @timer_func
def round_square(num, board_size=400):
    square_size = board_size / 8
    shift = square_size / 2
    value = square_size * round((num - shift) / square_size) + shift
    if value < shift:
        value = shift
    elif value > board_size - shift:
        value = board_size - shift
    return int(value)
# @timer_func
def create_fen(raw_board):
    fen_string = ""
    value_names = {1: 'b', 2: 'k', 3: 'n', 4: 'p', 5: 'q', 6: 'r', 7: 'B', 8: 'K', 9: 'N', 10: 'P', 11: 'Q', 12: 'R'}
    for row_number, row in enumerate(raw_board):
        zeros = 0
        for square in row:
            if square == 0:
                zeros += 1
            else:
                if zeros != 0:
                    fen_string += str(zeros)
                fen_string += str(value_names[square])
                zeros = 0
        if zeros != 0:
            fen_string += str(zeros)
        if row_number != 7:
            fen_string += '/'
    return fen_string
# @timer_func
def cleanup_moves(move_stack):
    for move_number, move in enumerate(move_stack):
        for comparison_move in move_stack[move_number:]:
            if move and comparison_move and move[2:] == comparison_move[:2]:
                move_stack[move_number] = None
                move_stack[move_stack.index(comparison_move)] = move[:2] + comparison_move[2:]
                logging.info("Move Combination Detected")
    return move_stack
# @timer_func
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
# @timer_func
def adjust_for_angle(x, y, height):
    y += .30 * height # Magic number
    return x, y
# @timer_func
def cluster_pts_names(pts_names_info):
    pts_info = [pts_names[0] for pts_names in pts_names_info]
    names_info = [pts_names[1] for pts_names in pts_names_info]
    pts_names_dict = dict()
    for pts_i in pts_info:
        for pt in pts_i:
            if pt not in pts_names_dict:
                pts_names_dict[pt] = []
    for i, names_i in enumerate(names_info):
        for j, name in enumerate(names_i):
            pt_key = pts_info[i][j]
            pts_names_dict[pt_key].append(name)
    pts_output, names_output = [], []
    for pt_keys, names_values in pts_names_dict.items():
        pts_output.append(pt_keys)
        names_output.append(statistics.mode(names_values))
    raw_board = np.zeros((8, 8))
    for j in range(len(pts_output)):
        raw_board[pts_output[j][1], pts_output[j][0]] = names_output[j]
    return raw_board
# @timer_func
def update_board_and_move_stack(raw_board, old_raw_board, move_wait_list):  # fix for en passant
    output_raw_board = old_raw_board.copy()
    file_names = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    unique, counts = np.unique(raw_board, return_counts=True)
    old_unique, old_counts = np.unique(output_raw_board, return_counts=True)
    counts = dict(zip(unique, counts))
    old_counts = dict(zip(old_unique, old_counts))
    for piece_index in range(0, 13):
        if piece_index not in counts.keys():
            counts[piece_index] = 0
        if piece_index not in old_counts.keys():
            old_counts[piece_index] = 0
    removed, added, moves = [], [], []
    for i, (raw_board_row, old_raw_board_row) in enumerate(zip(raw_board, output_raw_board)):
        for j, (raw_board_value, old_raw_board_value) in enumerate(zip(raw_board_row, old_raw_board_row)):
            color = 1 <= raw_board_value < 7
            old_color = 1 <= old_raw_board_value < 7
            if raw_board_value != 0 and old_raw_board_value != 0:
                if color != old_color:
                    added.append((i, j, raw_board_value, True))
            if raw_board_value != 0 and old_raw_board_value == 0:
                added.append((i, j, raw_board_value, False))
            if raw_board_value == 0 and old_raw_board_value != 0:
                removed.append((i, j, old_raw_board_value))
    for (old_i, old_j, old_piece) in removed:  # fix for castling, en passant, promotion, and under promotion
        for (i, j, piece, capture) in added:
            if old_piece == piece:
                if counts[0] == (old_counts[0] + 1 if capture else old_counts[0]):
                    if counts[piece] == old_counts[piece]:
                        output_raw_board[i][j] = old_raw_board[old_i][old_j]
                        output_raw_board[old_i][old_j] = 0
                        move_wait_list.append(file_names[old_j] + str(8 - old_i) + file_names[j] + str(8 - i))
                        logging.info("Removed: %s, Added: %s, Move: %s", (old_i, old_j, old_piece), (i, j, piece, capture), move_wait_list[-1])
    if not np.array_equal(output_raw_board, old_raw_board):
        logging.info("Board:\n%s", output_raw_board)
    return output_raw_board, move_wait_list
# @timer_func
def update_display(fen):
    digital_chessboard = chess.Board(fen)
    digital_display = chess.svg.board(digital_chessboard, size=600)
    cairosvg.svg2png(bytestring=digital_display, write_to='./misc/test.png')
    chessboard_img = cv2.imread('./misc/test.png')
    cv2.imshow("Chessboard", chessboard_img)
    cv2.waitKey(1)
# @timer_func
def write_pgn(pgn_file_name, game):
    pgn_string = game.accept(chess.pgn.StringExporter(headers=True))
    with open(pgn_file_name, 'w') as pgn_file:
        pgn_file.write(pgn_string)
# @timer_func
def find_chessboard_corners(results_img, corner_results_data):
    top_view = np.zeros((400, 400, 3)).astype(np.uint8)
    transformation_matrix = None
    for result in corner_results_data:  # can make into separate function
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
                    # for corner in corners:  # does nothing for now
                    #     cv2.circle(results_img, corner, radius=0, color=(255, 0, 0), thickness=10)
    rotate_board = board_rotation(top_view, 400)
    return transformation_matrix, rotate_board

# @timer_func
def find_ideal_piece_locations(results_img, piece_results_data, transformation_matrix, rotate_board, raw_board):  # use transformation matrix as input
    board_error = 25  # MAGIC NUMBER
    piece_names, ideal_pts = [], []
    results_img = piece_results_data[0].plot(img=results_img)
    if transformation_matrix is not None:
        result = piece_results_data[0]
        piece_data = result.boxes.data.cpu().numpy()
        for piece in piece_data:
            real_x, real_y = (piece[0] + piece[2]) / 2, (piece[1] + piece[3]) / 2
            box_width, box_height = abs(piece[2] - piece[0]), abs(piece[3] - piece[1])
            adjusted_real_x, adjusted_real_y = adjust_for_angle(real_x, real_y, box_height)
            real = [adjusted_real_x, adjusted_real_y, 1]
            ideal = np.matmul(transformation_matrix, real)
            if 0 - board_error <= ideal[0] / ideal[2] <= 400 + board_error and 0 - board_error <= ideal[1] / ideal[2] <= 400 + board_error:
                piece_names.append(int(piece[5] + 1))
                column = round_square(ideal[0] / ideal[2])
                row = 400 - round_square(ideal[1] / ideal[2])  # Magic Rotation/Flipping??? why is this line needed
                ideal_pts.append((int((row - 25) / 50), int((column - 25) / 50)))
        if rotate_board:
            ideal_pts = [(7 - pt[1], pt[0]) for pt in ideal_pts]  # clockwise 90 degrees
            # ideal_pts = [(pt[1],  7 - pt[0]) for pt in ideal_pts] # counterclockwise 90 degrees
        raw_board = np.zeros((8, 8))
        for j in range(len(ideal_pts)):
            raw_board[ideal_pts[j][1], ideal_pts[j][0]] = piece_names[j]
    return results_img, ideal_pts, piece_names, raw_board

pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)  # double check max_workers count needed
FEN_memory, calibration_count, calibration_limit = [], 0, 10 # Magic number
file = "C:/Users/nbuic/OneDrive/Desktop/TestingPGN/TEST.pgn"
board_output_size, image_output_size, image_size = 600, 640, 640
matrix, rotation = None, False
np_board = np.zeros((8, 8))
frame_delay = -5  # Magic number
chess_moves, waiting_moves = [None for _ in range(-1 * frame_delay)], []
logging.basicConfig(filename='./misc/example.log', filemode='w', level=logging.DEBUG)
print("Loading video capture...")
video_capture = cv2.VideoCapture(1)
print("Loading Models...")
corner_prediction_model = YOLO('C://PythonStuff/ChessScanner/CloudModels/BoardModels/NanoA100_BEST/train/weights/best.pt')
piece_prediction_model = YOLO('C://PythonStuff/ChessScanner/CloudModels/BestYet/train/weights/best.pt')  # create classless piece model
print("Calibrating...")
while video_capture.isOpened():  # rework this logic to be cleaner
    grabbed, frame = video_capture.read()
    if not grabbed:
        print("Failed to grab frame")
        break
    corrected_frame = resize(frame, image_size)
    start_piece = time.time()
    piece_results = piece_prediction_model.predict(corrected_frame, imgsz=image_size, agnostic_nms=True, verbose=False)
    old_np_board = np_board
    img, pts, names, np_board = find_ideal_piece_locations(corrected_frame, piece_results, matrix, rotation, old_np_board)
    cv2.imshow("Capture", resize(img, image_output_size))
    key_press = cv2.waitKey(1)
    if key_press == ord(" "):
        print("Calibrating...")
        FEN_memory, calibration_count = [], 0
    if calibration_count < calibration_limit:
        corner_results = corner_prediction_model.predict(corrected_frame, imgsz=image_size, verbose=False)
        matrix, rotation = find_chessboard_corners(img, corner_results)
        FEN_memory.append([pts, names])
        np_board = np.zeros((8, 8))
        calibration_count += 1
        if calibration_count == calibration_limit:
            np_board = cluster_pts_names(FEN_memory)
            chess_game = chess.pgn.Game.without_tag_roster()
            chess_game.headers["White"] = "Chad"
            chess_game.headers["Black"] = "Some Scrub"
            # game.setup(create_fen(np_board))
            chess_board = chess.Board()
            chess_moves, waiting_moves = [None for _ in range(-1 * frame_delay)], []
            chess_game_node = chess_game
            pool.submit(update_display, create_fen(np_board))
            pool.submit(write_pgn, file, chess_game)
            print("...Done Calibrating")
            logging.info("--- Resetting calibration ---")
    else:
        if not np.array_equal(np_board, old_np_board):
            np_board, waiting_moves = update_board_and_move_stack(np_board, old_np_board, waiting_moves)
        chess_move = None if not waiting_moves else waiting_moves.pop(0)
        del chess_moves[0]
        chess_moves.append(chess_move)
        chess_moves = cleanup_moves(chess_moves)
        legal_moves = [chess.Move.uci(legal_move) for legal_move in chess_board.legal_moves]
        if chess_move_to_be_pushed := chess_moves[0]:
            pool.submit(update_display, create_fen(np_board))
            if chess_move_to_be_pushed in legal_moves:
                chess_game_node = chess_game_node.add_variation(chess.Move.from_uci(chess_move_to_be_pushed))
                chess_board.push_uci(chess_move_to_be_pushed)
                pool.submit(write_pgn, file, chess_game)
                logging.info("%s", chess_game)
                print(chess_game)
    if key_press == ord("q"):
        print("Exit key pressed")
        break
pool.shutdown(wait=True)
video_capture.release()
cv2.destroyAllWindows()
