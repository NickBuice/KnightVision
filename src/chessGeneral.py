import numpy as np
import cv2
import logging
import chess
import chess.svg
import chess.pgn
import cairosvg
import concurrent.futures


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
    def __init__(self, white: str = "Player 1", black: str = "Player 2", board_delay: int = 20) -> None:
        """
        Initializes game model, raw inputs, and stacks for in place mutation.
        """
        logging.info("--- STARTING GAME ---")
        self.pgn_file: str = "C:/Users/User/Desktop/LichessPGN/LichessTEST.pgn"
        self.game: chess.pgn.Game = chess.pgn.Game.without_tag_roster()
        self.chessboard: chess.Board = chess.Board()
        self.old_chessboard: chess.Board = chess.Board()
        self.game.headers["White"], self.game.headers["Black"] = white, black
        self.node: chess.pgn.GameNode = self.game
        self.raw_map: dict = dict()
        self.move_stack: list[list[chess.Move]] = [[] for _ in range(board_delay)]
        self.board_stack: list[dict] = [self.raw_map.copy() for _ in range(board_delay)]
        self.future_moves = []

    def board_has_changed(self) -> bool:
        """
        Outputs equality of new and old raw numpy board states.
        """
        return  self.raw_map.keys() != self.chessboard.piece_map().keys()

    def update_move_stack(self) -> None:
        """
        Removes first index of move stack and adds empty list to top of stack.
        """
        self.move_stack.pop(0)
        self.move_stack.append([])

    def update_board_stack(self) -> None:
        """
        Removes first index of board stack and adds latest raw board
         chess.Board object to top of stack.
        """
        self.board_stack.pop(0)
        self.board_stack.append(self.raw_map)

    def update_chessboard(self) -> None:
        """
        Compares new and old raw numpy board states for inequalities, then
        matches inequalities to check for repetitive detection, then pushes
        UCI move to chessboard chess.Board object.
        """
        replaced = []
        chessboard_map = self.chessboard.piece_map()
        for square in chess.SQUARES:
            if square in self.raw_map and square in chessboard_map:
                if self.raw_map[square] != chessboard_map[square].color:
                    replaced.append((square, self.raw_map[square], chessboard_map[square].color, True, chessboard_map[square]))
            elif square in self.raw_map and square not in chessboard_map:
                replaced.append((square, self.raw_map[square], None, False, None))
            elif square not in self.raw_map and square in chessboard_map:
                replaced.append((square, None, chessboard_map[square].color, False, chessboard_map[square]))

        for index, (square, new_color, old_color, capture, piece) in enumerate(replaced[:-1]):
            for index_, (square_, new_color_, old_color_, capture_, piece_) in enumerate(replaced[index + 1:]):
                if not (capture and capture_) and (new_color is None or new_color_ is None) and new_color != new_color_:
                    white_pawn = piece == chess.Piece.from_symbol('P') or piece_ == chess.Piece.from_symbol('P')
                    white_rank = square // 8 == 6 and square_ // 8 == 7 or square // 8 == 7 and square_ // 8 == 6
                    black_pawn = piece == chess.Piece.from_symbol('p') or piece_ == chess.Piece.from_symbol('p')
                    black_rank = square // 8 == 1 and square_ // 8 == 0 or square // 8 == 1 and square_ // 8 == 0
                    if white_pawn and white_rank or black_pawn and black_rank:
                        from_square, to_square, promotion = (square_, square, chess.QUEEN) if (new_color_ is None) else (square, square_, chess.QUEEN)
                        self.move_stack[-1].append(chess.Move(from_square=from_square, to_square=to_square, promotion=promotion))
                    elif (capture_ or new_color == old_color_) and (new_color_ == old_color or capture):
                        from_square, to_square = (square_, square) if (new_color_ is None) else (square, square_)
                        self.move_stack[-1].append(chess.Move(from_square=from_square, to_square=to_square))

        goal, target = int(len(self.move_stack) / 2), int(len(self.move_stack) * 0.8 / 2) # magic number
        for move in self.move_stack[-1]:
            if sum([move in moves for moves in self.move_stack[goal:]]) >= target:
                logging.info("Move %s, LatestBoardStack: %s", move, self.move_stack[-1])
                if move in self.chessboard.legal_moves:
                    self.node = self.node.add_variation(move)
                    self.chessboard.push(move)
                    logging.info("MOVE  %s PLAYED", move)
                if self.chessboard.fen() != chess.STARTING_FEN and self.node.parent:
                    tri_source = self.chessboard.peek().uci()[2:4] == move.uci()[:2]
                    tri_target = self.chessboard.peek().uci()[:2] == move.uci()[2:4]
                    # source_check = True
                    if move.uci()[:2] != self.chessboard.peek().uci()[2:4]:
                        target_check = sum([chess.Move.from_uci(move.uci()[:2] + self.chessboard.peek().uci()[2:4]) in moves for moves in self.move_stack])
                    else:
                        target_check = False
                    if tri_source or (tri_target and target_check):
                        check = self.chessboard.peek()
                        self.node.parent.variations.remove(self.node)
                        self.node = self.node.parent
                        self.chessboard.pop()
                        logging.info("MOVE  %s UNDONE %s %s", check, move, target_check)

    def skipped_move_search(self) -> None:
        """
        Searches for moves that likely skipped a turn.
        """
        self.future_moves = []
        for move in self.move_stack[-1]:
            if sum([move in moves for moves in self.move_stack]) >= len(self.move_stack) * .8:
                if self.chessboard.turn != self.chessboard.color_at(move.from_square):
                    self.future_moves.append(move)
                    logging.info("DETECTED FUTURE MOVE: %s", self.future_moves[-1])

    def fix_skipped_move(self) -> list[chess.Move]:
        """
        Finds potential moves to fix the board.
        """
        potential_fixes = []
        for future_move in self.future_moves:
            for move in self.chessboard.legal_moves:
                dummy_board = self.chessboard.copy()
                dummy_board.push(move)
                if future_move in dummy_board.legal_moves:
                    potential_fixes.append(move)
                    logging.info("POTENTIAL FIX: %s", move)
        return potential_fixes

    def push_fix(self, potential_fixes) -> None:
        """
        Verifies fix matches raw board chess.Board object before pushing
        to pgn and chessboard chess.Board object.
        """
        goal, target = int(len(self.move_stack) / 2), int(len(self.move_stack) * 0.8 / 2)
        for move in potential_fixes:
            test1 = sum([move.from_square not in raw_board for raw_board in self.board_stack[goal:]])
            test2 = sum([move.to_square in raw_board for raw_board in self.board_stack[goal:]])
            if test1 >= target and test2 >= target:
                self.node = self.node.add_variation(move)
                self.chessboard.push(move)
                logging.info("ARTIFICIAL MOVE  %s PLAYED", move)
                break

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
