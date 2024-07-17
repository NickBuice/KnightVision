import numpy as np


def adjust_for_angle(x, y, height):
    y += .30 * height  # Magic number
    return x, y


def round_square(num, board_size=400):
    square_size = board_size / 8
    shift = square_size / 2
    value = square_size * round((num - shift) / square_size) + shift
    if value < shift:
        value = shift
    elif value > board_size - shift:
        value = board_size - shift
    return int(value)


def locate_pieces(results_img, piece_results_data, transformation_matrix, rotate_board, raw_board):
    board_error = 15  # MAGIC NUMBER
    piece_names, ideal_pts, old_raw_board = [], [], raw_board.copy()
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
            upper_bound, lower_bound = 0 - board_error, 400 + board_error
            if upper_bound <= ideal[0] / ideal[2] <= lower_bound and upper_bound <= ideal[1] / ideal[2] <= lower_bound:
                piece_names.append(int(piece[5] + 1))
                column = round_square(ideal[0] / ideal[2])
                row = 400 - round_square(ideal[1] / ideal[2])  # Magic Rotation/Flipping??? why is this line needed
                ideal_pts.append((int((row - 25) / 50), int((column - 25) / 50)))
        if rotate_board:
            ideal_pts = [(7 - pt[1], pt[0]) for pt in ideal_pts]  # clockwise 90 degrees
            #  ideal_pts = [(pt[1],  7 - pt[0]) for pt in ideal_pts] # counterclockwise 90 degrees
        for i in range(8):
            for j in range(8):
                raw_board[i][j] = 0
        for j in range(len(ideal_pts)):
            raw_board[ideal_pts[j][1], ideal_pts[j][0]] = piece_names[j]
    else:
        print("*Board Not Detected*")
    return results_img, old_raw_board
