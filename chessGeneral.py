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


def update_move_stack(move_stack, waiting_move_stack):  # error a1a1 not in move_stack[move_stack.index(comparison_move)]
    del move_stack[0]
    move_stack.append(None if not waiting_move_stack else waiting_move_stack.pop(0))
    for move_number, move in enumerate(move_stack):
        for comparison_move in move_stack[move_number:]:
            if move and comparison_move and move[2:] == comparison_move[:2] and move_number != move_stack.index(comparison_move):
                move_stack[move_number] = None
                move_stack[move_stack.index(comparison_move)] = move[:2] + comparison_move[2:]
                logging.info("Move Combination Detected")
    return move_stack


def count_pieces(raw_board):
    unique, counts = np.unique(raw_board, return_counts=True)
    counts = dict(zip(unique, counts))
    for piece_index in range(0, 13):
        if piece_index not in counts.keys():
            counts[piece_index] = 0
    return counts


def update_board_and_waiting_move_stack(raw_board, old_raw_board, move_wait_list):
    output_raw_board = old_raw_board.copy()
    file_names = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    counts = count_pieces(raw_board)
    old_counts = count_pieces(old_raw_board)
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
    for (old_i, old_j, old_piece) in removed:  # fix for promotion
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
    # TO DO: sort move_wait_list before returning?
    return output_raw_board, move_wait_list


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


def chess_board_and_game_startup(white="Player 1", black="Player 2", from_position=False, raw_board=np.zeros((8, 8)), delay=-5):
    game = chess.pgn.Game.without_tag_roster()
    game.headers["White"] = white
    game.headers["Black"] = black
    if from_position:
        game.setup(write_fen(raw_board))
    return game, game, chess.Board(), [None for _ in range(-1 * delay)], []


class StartChessGame:
    def __init__(self, white="Player 1", black="Player 2", from_position=False, raw_board=np.zeros((8, 8)), pgn_delay=-5, board_delay=6):
        self.game = chess.pgn.Game.without_tag_roster()
        self.game.headers["White"] = white
        self.game.headers["Black"] = black
        if from_position:
            self.game.setup(write_fen(raw_board))
        self.node = self.game
        self.chessboard = chess.Board()
        self.moves = [None for _ in range(-1 * pgn_delay)]
        self.waiting_moves = []
        self.np_board = np.zeros((8, 8))
        self.old_np_board = np.zeros((8, 8))
        self.saved_counts = [0 for _ in range(5)]

        self.board_stack = [[] for _ in range(board_delay)]

    def update_move_stack(self):  # error a1a1 not in move_stack[move_stack.index(comparison_move)]
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

    def update_saved_counts(self):
        del self.saved_counts[0]
        self.saved_counts.append(count_pieces(self.np_board)[0])

    # def sort_move_stack(self):
    #     legal_moves = self.chessboard.legal_moves


    def update_board_and_waiting_move_stack(self):  # relies on all pieces being visible
        file_names = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        counts = count_pieces(self.np_board)
        old_counts = count_pieces(self.old_np_board)
        removed, added = [], []
        for i, (raw_board_row, old_raw_board_row) in enumerate(zip(self.np_board, self.old_np_board)):
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
        for (old_i, old_j, old_piece) in removed:  # fix for promotion
            for (i, j, piece, capture) in added:
                if old_piece == piece:
                    if counts[0] == (old_counts[0] + 1 if capture else old_counts[0]):
                        if counts[piece] == old_counts[piece]:
                            self.old_np_board[i][j] = self.old_np_board[old_i][old_j]
                            self.old_np_board[old_i][old_j] = 0
                            self.waiting_moves.append(file_names[old_j] + str(8 - old_i) + file_names[j] + str(8 - i))
                            logging.info("Removed: %s, Added: %s, Move: %s", (old_i, old_j, old_piece),
                                         (i, j, piece, capture), self.waiting_moves[-1])
        self.np_board = self.old_np_board
        if self.board_has_changed():
            logging.info("Board:\n%s", self.old_np_board)
        self.np_board = self.old_np_board

    def update_board_and_waiting_move_stack_replaced(self): # triangle problem is most apparent
        saved_old_np_board = self.old_np_board.copy()
        file_names = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        counts = count_pieces(self.np_board)
        old_counts = count_pieces(self.old_np_board)
        removed, added, replaced = [], [], []
        for i, (raw_board_row, old_raw_board_row) in enumerate(zip(self.np_board, self.old_np_board)):
            for j, (raw_board_value, old_raw_board_value) in enumerate(zip(raw_board_row, old_raw_board_row)):
                color = 1 <= raw_board_value < 7
                old_color = 1 <= old_raw_board_value < 7
                if raw_board_value != 0 and old_raw_board_value != 0:
                    if color != old_color:
                        added.append((i, j, raw_board_value, True))
                    elif raw_board_value != old_raw_board_value:
                        replaced.append((i, j, raw_board_value, old_raw_board_value))
                if raw_board_value != 0 and old_raw_board_value == 0:
                    added.append((i, j, raw_board_value, False))
                if raw_board_value == 0 and old_raw_board_value != 0:
                    removed.append((i, j, old_raw_board_value))
        for (old_i, old_j, old_piece) in removed:  # fix for promotion
            for (i, j, piece, capture) in added:
                if old_piece == piece:
                    if counts[piece] == old_counts[piece]:
                        swap_variable = 0 if capture else self.old_np_board[i][j]
                        # swap_variable = self.old_np_board[i][j]
                        self.old_np_board[i][j] = self.old_np_board[old_i][old_j]
                        self.old_np_board[old_i][old_j] = swap_variable
                        self.waiting_moves.append(file_names[old_j] + str(8 - old_i) + file_names[j] + str(8 - i))
                        logging.info("Removed: %s, Added: %s, Move: %s", (old_i, old_j, old_piece),
                                     (i, j, piece, capture), self.waiting_moves[-1])
        for k, (i, j, new_piece, old_piece) in enumerate(replaced[:-1]):
            for (i_, j_, new_piece_, old_piece_) in replaced[k + 1:]:
                print(new_piece == old_piece_ and new_piece_ == old_piece)
                if new_piece == old_piece_ and new_piece_ == old_piece:
                    self.old_np_board[i][j] = new_piece
                    self.old_np_board[i_][j_] = new_piece_
                    logging.info("Replaced: %s %s", (i, j, new_piece, old_piece), (i_, j_, new_piece_, old_piece_))
        self.np_board = self.old_np_board
        if not np.array_equal(self.np_board, saved_old_np_board):
            logging.info("Board:\n%s", self.old_np_board)
        # TO DO: sort move_wait_list?

    def update_board_and_waiting_move_stack_swap(self):  # PROBLEM: pieces disappear then reappear, captures buggy
        saved_old_np_board = self.old_np_board.copy()
        file_names = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        counts = count_pieces(self.np_board)
        old_counts = count_pieces(self.old_np_board)
        replaced = []
        for i, (raw_board_row, old_raw_board_row) in enumerate(zip(self.np_board, self.old_np_board)):
            for j, (raw_board_value, old_raw_board_value) in enumerate(zip(raw_board_row, old_raw_board_row)):
                color = "EMPTY" if raw_board_value == 0 else ("WHITE" if raw_board_value > 6 else "BLACK")
                old_color = "EMPTY" if old_raw_board_value == 0 else ("WHITE" if old_raw_board_value > 6 else "BLACK")
                nonzero = raw_board_value != 0 and old_raw_board_value != 0
                if raw_board_value != old_raw_board_value and color != old_color:
                    replaced.append((i, j, raw_board_value, old_raw_board_value, color != old_color and nonzero, nonzero))
        for (i, j, new_piece, old_piece, capture, swap) in replaced:
            for (i_, j_, new_piece_, old_piece_, capture_, swap_) in replaced:
                if (i, j, new_piece, old_piece, capture, swap) != (i_, j_, new_piece_, old_piece_, capture_, swap_):
                    if (swap_ or new_piece == old_piece_) and (new_piece_ == old_piece or swap):
                        if new_piece == 0 and counts[new_piece_] == old_counts[new_piece_]:
                            self.waiting_moves.append(file_names[j] + str(8 - i) + file_names[j_] + str(8 - i_))
                            self.old_np_board[i][j] = new_piece if capture_ else old_piece_
                            self.old_np_board[i_][j_] = old_piece
                            logging.info("Replaced: %s %s Move: %s", (i, j, new_piece, old_piece, capture, swap),
                                         (i_, j_, new_piece_, old_piece_, capture_, swap_), self.waiting_moves[-1])
                            del replaced[replaced.index((i_, j_, new_piece_, old_piece_, capture_, swap_))]
                            break
                        elif new_piece_ == 0 and counts[new_piece] == old_counts[new_piece]:
                            self.waiting_moves.append(file_names[j_] + str(8 - i_) + file_names[j] + str(8 - i))
                            self.old_np_board[i][j] = old_piece_
                            self.old_np_board[i_][j_] = new_piece_ if capture else old_piece
                            logging.info("Replaced: %s %s Move: %s", (i, j, new_piece, old_piece, capture, swap),
                                         (i_, j_, new_piece_, old_piece_, capture_, swap_), self.waiting_moves[-1])
                            del replaced[replaced.index((i_, j_, new_piece_, old_piece_, capture_, swap_))]
                            break
                        elif swap and swap_ and counts[old_piece] == old_counts[old_piece] and counts[old_piece_] == old_counts[old_piece_] and not capture and not capture_:
                            self.old_np_board[i][j] = old_piece_
                            self.old_np_board[i_][j_] = old_piece
                            logging.info("Swapped: %s %s %s", (i, j, new_piece, old_piece, capture, swap),
                                         (i_, j_, new_piece_, old_piece_, capture_, swap_), file_names[j_] + str(8 - i_) + file_names[j] + str(8 - i))
                            del replaced[replaced.index((i_, j_, new_piece_, old_piece_, capture_, swap_))]
                            break
            del replaced[replaced.index((i, j, new_piece, old_piece, capture, swap))]
        if not np.array_equal(self.old_np_board, saved_old_np_board):
            pool.submit(show_svg_display, write_fen(self.old_np_board), 600)
            logging.info("Board:\n%s", self.old_np_board)
        else:
            pool.submit(show_same_display)
        self.np_board = self.old_np_board
        # TO DO: sort move_wait_list?

    def update_board_and_waiting_move_stack_classless_one(self):  # chaos
        saved_old_np_board = self.old_np_board.copy()
        file_names = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        counts = count_pieces(self.np_board)
        old_counts = count_pieces(self.old_np_board)
        removed, added, replaced = [], [], []
        for i, (raw_board_row, old_raw_board_row) in enumerate(zip(self.np_board, self.old_np_board)):
            for j, (raw_board_value, old_raw_board_value) in enumerate(zip(raw_board_row, old_raw_board_row)):
                color = 1 <= raw_board_value < 7
                old_color = 1 <= old_raw_board_value < 7
                if raw_board_value != 0 and old_raw_board_value != 0:
                    if color != old_color:
                        added.append((i, j, color, True))
                    # elif raw_board_value != old_raw_board_value:
                    #     replaced.append((i, j, raw_board_value, old_raw_board_value))
                if raw_board_value != 0 and old_raw_board_value == 0:
                    added.append((i, j, color, False))
                if raw_board_value == 0 and old_raw_board_value != 0:
                    removed.append((i, j, old_color))
        for (old_i, old_j, old_piece) in removed:  # fix for promotion
            for (i, j, piece, capture) in added:
                if old_piece == piece:
                    swap_variable = 0 if capture else self.old_np_board[i][j]
                    # swap_variable = self.old_np_board[i][j]
                    self.old_np_board[i][j] = self.old_np_board[old_i][old_j]
                    self.old_np_board[old_i][old_j] = swap_variable
                    self.waiting_moves.append(file_names[old_j] + str(8 - old_i) + file_names[j] + str(8 - i))
                    logging.info("Removed: %s, Added: %s, Move: %s", (old_i, old_j, old_piece),
                                 (i, j, piece, capture), self.waiting_moves[-1])
        # for k, (i, j, new_piece, old_piece) in enumerate(replaced[:-1]):
        #     for (i_, j_, new_piece_, old_piece_) in replaced[k + 1:]:
        #         print(new_piece == old_piece_ and new_piece_ == old_piece)
        #         if new_piece == old_piece_ and new_piece_ == old_piece:
        #             self.old_np_board[i][j] = new_piece
        #             self.old_np_board[i_][j_] = new_piece_
        #             logging.info("Replaced: %s %s", (i, j, new_piece, old_piece), (i_, j_, new_piece_, old_piece_))
        self.np_board = self.old_np_board
        if not np.array_equal(self.np_board, saved_old_np_board):
            logging.info("Board:\n%s", self.old_np_board)
        # TO DO: sort move_wait_list?

    def update_board_and_waiting_move_stack_classless_two(self):  # decent but mixes same color pieces sometimes
        saved_old_np_board = self.old_np_board.copy()
        self.board_stack.pop(0), self.board_stack.append([])
        file_names = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        replaced = []
        for i, (raw_board_row, old_raw_board_row) in enumerate(zip(self.np_board, self.old_np_board)):
            for j, (raw_board_value, old_raw_board_value) in enumerate(zip(raw_board_row, old_raw_board_row)):
                color = 2 if raw_board_value > 6 else (1 if 0 < raw_board_value < 7 else 0)
                old_color = 2 if old_raw_board_value > 6 else (1 if 0 < old_raw_board_value < 7 else 0)
                if color != old_color:
                    replaced.append((i, j, color, old_color, color != 0 and old_color != 0))
        for index, (i, j, color, old_color, capture) in enumerate(replaced[:-1]):
            for (i_, j_, color_, old_color_, capture_) in replaced[index + 1:]:
                if (capture_ or color == old_color_) and (color_ == old_color or capture) and not (capture_ and capture):
                    self.board_stack[-1].append((j_, i_, j, i, capture_, capture) if color_ == 0 else (j, i, j_, i_, capture, capture_))
        for raw_move in self.board_stack[-1]:
            if sum([raw_move in board for board in self.board_stack]) > 10:
                old_j, old_i, new_j, new_i, capture, capture_ = raw_move
                move = file_names[old_j] + str(8 - old_i) + file_names[new_j] + str(8 - new_i)
                swap_variable = 0 if capture else self.old_np_board[old_i][old_j]
                self.old_np_board[old_i][old_j] = 0 if capture_ else self.old_np_board[new_i][new_j]
                self.old_np_board[new_i][new_j] = swap_variable
                self.waiting_moves.append(move)
                logging.info("Move %s", move)
        if not np.array_equal(self.old_np_board, saved_old_np_board):
            pool.submit(show_svg_display, write_fen(self.old_np_board), 600)
            logging.info("Board:\n%s", self.old_np_board)
        else:
            pool.submit(show_same_display)
        self.np_board = self.old_np_board


    def update_board_and_waiting_move_stack_two(self):  # decent but mixes same color pieces sometimes
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
        # else:
        #     pool.submit(show_same_display)
        self.np_board = self.old_np_board


pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
