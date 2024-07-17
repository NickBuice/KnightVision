import numpy as np
import cv2
import logging
import chess
import chess.svg
import chess.pgn
import os
os.environ['path'] = 'C://Program Files/UniConvertor-2.0rc5/dlls'
import cairosvg
import concurrent.futures



def image_resize(image, new_size):
    h, w = image.shape[0], image.shape[1]
    if w > h:
        scale = new_size / w
        height = int((np.ceil(h * scale / 32))* 32)
        dim = (new_size, height)
    else:
        scale = new_size / h
        width = int((np.ceil(w * scale / 32)) * 32)
        dim = (width, new_size)
    return cv2.resize(image, dim)


def write_fen(raw_board):
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


def show_same_display():
    # chessboard_img = cv2.imread('./misc/test.png')
    # cv2.imshow("Chessboard", chessboard_img)
    cv2.waitKey(1)

def show_svg_display(fen, board_size=600):
    digital_chessboard = chess.Board(fen)
    digital_display = chess.svg.board(digital_chessboard, size=board_size)
    cairosvg.svg2png(bytestring=digital_display, write_to='./misc/test.png')
    chessboard_img = cv2.imread('./misc/test.png')
    cv2.imshow("Chessboard", chessboard_img)
    cv2.waitKey(1)


def write_pgn_to_file(pgn_file_name, game):
    pgn_string = game.accept(chess.pgn.StringExporter(headers=True))
    with open(pgn_file_name, 'w') as pgn_file:
        pgn_file.write(pgn_string)


class StartChessGame:
    def __init__(self, white="Player 1", black="Player 2", from_position=False, raw_board=np.zeros((8, 8)), pgn_delay=-5, board_delay=6):
        self.game = chess.pgn.Game.without_tag_roster()
        if from_position:
            self.game.setup(write_fen(raw_board))
        self.chessboard = chess.Board()
        self.game.headers["White"], self.game.headers["Black"] = white, black
        self.node = self.game
        self.waiting_moves = []
        self.np_board, self.old_np_board = np.zeros((8, 8)), np.zeros((8, 8))
        self.board_stack, self.moves = [[] for _ in range(board_delay)], [None for _ in range(-1 * pgn_delay)]

    def update_move_stack(self):  # TODO error a1a1 not in move_stack[move_stack.index(comparison_move)]
        del self.moves[0]
        self.moves.append(None if not self.waiting_moves else self.waiting_moves.pop(0))
        for move_number, move in enumerate(self.moves[:-1]):
            for comparison_move in self.moves[move_number + 1:]:
                if move and comparison_move and move[2:] == comparison_move[:2]:
                    self.moves[move_number] = None
                    self.moves[self.moves.index(comparison_move)] = move[:2] + comparison_move[2:]
                    logging.info("Move Combination Detected")

    def board_has_changed(self):
        return not np.array_equal(self.np_board, self.old_np_board)

    def update_board_and_waiting_move_stack(self):  # Best so far
        saved_old_np_board = self.old_np_board.copy()
        self.board_stack.pop(0), self.board_stack.append([])
        file_names = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        replaced = []
        for i, (raw_board_row, old_raw_board_row) in enumerate(zip(self.np_board, self.old_np_board)):
            for j, (raw_board_value, old_raw_board_value) in enumerate(zip(raw_board_row, old_raw_board_row)):
                color = 2 if raw_board_value > 6 else (1 if 0 < raw_board_value < 7 else 0)
                old_color = 2 if old_raw_board_value > 6 else (1 if 0 < old_raw_board_value < 7 else 0)
                if color != old_color:
                    replaced.append((i, j, raw_board_value, old_raw_board_value, color != 0 and old_color != 0))
        for index, (i, j, new_piece, old_piece, capture) in enumerate(replaced[:-1]):
            for (i_, j_, new_piece_, old_piece_, capture_) in replaced[index + 1:]:
                if (capture_ or new_piece == old_piece_) and (new_piece_ == old_piece or capture) and not (capture_ and capture):
                    self.board_stack[-1].append((j_, i_, j, i, capture_, capture) if new_piece_ == 0 else (j, i, j_, i_, capture, capture_))
        for raw_move in self.board_stack[-1]:
            if sum([raw_move in board for board in self.board_stack]) >= 3:  # magic number
                old_j, old_i, new_j, new_i, capture, capture_ = raw_move
                move = file_names[old_j] + str(8 - old_i) + file_names[new_j] + str(8 - new_i)
                logging.info("Move %s, LatestBoardStack: %s", move, self.board_stack[-1])
                if move in [chess.Move.uci(legal_move) for legal_move in self.chessboard.legal_moves]:
                    swap_variable = 0 if capture else self.old_np_board[old_i][old_j]
                    self.old_np_board[old_i][old_j] = 0 if capture_ else self.old_np_board[new_i][new_j]
                    self.old_np_board[new_i][new_j] = swap_variable
                    self.waiting_moves.append(move)
                    logging.info("MOVE  %s PLAYED", move)
        if not np.array_equal(self.old_np_board, saved_old_np_board):
            pool.submit(show_svg_display, write_fen(self.old_np_board), 600)
            logging.info("Board:\n%s", self.old_np_board)
        else:
            pool.submit(show_same_display)
        self.np_board = self.old_np_board

pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
