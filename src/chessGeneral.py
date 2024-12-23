import numpy as np
import cv2
import logging
import chess
import chess.svg
import chess.pgn
import cairosvg
import concurrent.futures
from typing import Any


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


class StartChessGame:
    """
    Initializes all information needed to model game and display raw inputs.  Contains method
    to translate raw board state to UCI move.
    """
    def __init__(self, white: str = "Player 1", black: str = "Player 2", board_delay: int = 10) -> None:
        """
        Initializes game model, raw inputs, and stacks for in place mutation.
        """
        self.pgn_file: str = "../misc/TEST.pgn"
        self.game: chess.pgn.Game = chess.pgn.Game.without_tag_roster()
        self.chessboard: chess.Board = chess.Board()
        self.old_chessboard: chess.Board = chess.Board()
        self.game.headers["White"], self.game.headers["Black"] = white, black
        self.node: chess.pgn.GameNode = self.game
        self.raw_board: np.ndarray = np.zeros((8, 8))
        self.board_stack: list[list[Any]] = [[] for _ in range(board_delay)]

    def board_has_changed(self) -> bool:
        """
        Outputs equality of new and old raw numpy board states.
        """
        return not np.array_equal(self.raw_board, self.create_chessboard_to_raw())

    def create_chessboard_to_raw(self) -> np.ndarray:
        """
        Creates numpy array from most recent chessboard chess.Board object.
        """
        key = {'0': 0, 'b': 1, 'k': 2, 'n': 3, 'p': 4, 'q': 5, 'r': 6, 'B': 7, 'K': 8, 'N': 9, 'P': 10, 'Q': 11, 'R': 12}
        fen = self.chessboard.fen().split()[0]
        for value in fen:
            if value.isnumeric():
                fen = fen.replace(value, int(value)*'0')
        raw_board_rows = fen.split('/')
        raw_board = np.zeros((8, 8))
        for rank in range(0, 8):
            for file in range(0, 8):
                raw_board[rank][file] = key[raw_board_rows[rank][file]]
        return raw_board

    def update_chessboard(self) -> None:
        """
        Compares new and old raw numpy board states for inequalities, then
        matches inequalities to check for repetitive detection, then pushes
        UCI move to chessboard chess.Board object.
        """
        replaced = []
        for i, (raw_board_row, old_raw_board_row) in enumerate(zip(self.raw_board, self.create_chessboard_to_raw())):
            for j, (raw_board_value, old_raw_board_value) in enumerate(zip(raw_board_row, old_raw_board_row)):
                color = "WHITE" if raw_board_value > 6 else ("BLACK" if 0 < raw_board_value < 7 else 0)
                old_color = "WHITE" if old_raw_board_value > 6 else ("BLACK" if 0 < old_raw_board_value < 7 else 0)
                if color != old_color:
                    replaced.append((i, j, raw_board_value, old_raw_board_value, color != 0 and old_color != 0))

        self.board_stack.pop(0)
        self.board_stack.append([])
        for index, (i, j, new_piece, old_piece, capture) in enumerate(replaced[:-1]):
            for index_, (i_, j_, new_piece_, old_piece_, capture_) in enumerate(replaced[index + 1:]):
                if (capture_ or new_piece == old_piece_) and (new_piece_ == old_piece or capture):
                    if not (capture_ and capture):
                        self.board_stack[-1].append((j_, i_, j, i) if new_piece_ == 0 else (j, i, j_, i_))

        file_names = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
        for raw_move in self.board_stack[-1]:
            if sum([raw_move in board for board in self.board_stack]) >= int(len(self.board_stack) * 0.75):  # magic number
                old_j, old_i, new_j, new_i = raw_move
                move = file_names[old_j] + str(8 - old_i) + file_names[new_j] + str(8 - new_i)
                logging.info("Move %s, LatestBoardStack: %s", move, self.board_stack[-1])
                if move in [chess.Move.uci(legal_move) for legal_move in self.chessboard.legal_moves]:
                    self.node = self.node.add_variation(chess.Move.from_uci(move))
                    self.chessboard.push_uci(move)
                    logging.info("MOVE  %s PLAYED", move)
                if self.chessboard.fen() != chess.STARTING_FEN and self.node.parent:
                    if (move := move[2:] + move[:2]) == self.chessboard.peek().uci()[:4]:
                        self.node.parent.variations.remove(self.node)
                        self.node = self.node.parent
                        self.chessboard.pop()
                        logging.info("MOVE  %s UNDONE", move)

    def show_chessboard(self, force: bool = False) -> None:
        """
        Displays digital chessboard.
        """
        def show_svg_display() -> None:
            """
            Takes chess.Board object and displays image of chess.svg board render.
            """
            digital_display = chess.svg.board(self.chessboard, size=600)
            png_data = cairosvg.svg2png(bytestring=digital_display)
            image_array = np.frombuffer(png_data, dtype=np.uint8)
            chessboard_img = cv2.imdecode(image_array, cv2.IMREAD_UNCHANGED)
            cv2.imshow("Chessboard", chessboard_img)
            cv2.waitKey(1)

        def write_pgn_to_file() -> None:
            """
            Overwrites current PGN in external file.
            """
            with open(self.pgn_file, 'w') as pgn_file:
                pgn_file.write(self.game.accept(chess.pgn.StringExporter(headers=True)))

        if self.old_chessboard.fen() != self.chessboard.fen() or force:
            pool.submit(show_svg_display)
            pool.submit(write_pgn_to_file)
            logging.info("PGN:\n%s", self.game)
        else:
            pool.submit(cv2.waitKey, 1)
        self.old_chessboard = self.chessboard.copy()

pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
