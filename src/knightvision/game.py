from typing import Optional, cast
from dataclasses import dataclass

import chess
import chess.svg
import chess.pgn


@dataclass
class _Mismatch:
    """
    Small dataclass for mismatched board states between frames.
    """
    location: chess.Square
    color: Optional[bool]
    old_color: Optional[bool]
    capture: bool
    type: Optional[chess.Piece]
    def __iter__(self):
        return iter((self.location, self.color, self.old_color, self.capture, self.type))


class StartChessGame:
    """
    Contains methods and variables to translate model to chess.pgn.Game.
    """
    def __init__(self, board_delay: int = 20) -> None:
        """
        Initializes game model, raw inputs, and stacks for in place mutation.
        """
        self.game: chess.pgn.Game = chess.pgn.Game.without_tag_roster()
        self.chessboard: chess.Board = chess.Board()
        self.old_chessboard: chess.Board = chess.Board()
        self.node: chess.pgn.GameNode = self.game
        self.delay: int = board_delay
        self.move_stack: list[list[chess.Move]] = [[] for _ in range(board_delay)]
        self.board_stack: list[dict[chess.Square, chess.Color]] = [dict() for _ in range(board_delay)]

    def board_has_changed(self) -> bool:
        """
        Outputs equality of new raw map and chessboard map.
        """
        return  self.board_stack[-1].keys() != self.chessboard.piece_map().keys()

    def update_move_stack(self) -> None:
        """
        Removes first index of move stack and adds empty list to top of stack.
        """
        self.move_stack.pop(0)
        self.move_stack.append([])

    def update_board_stack(self, raw_map) -> None:
        """
        Removes first index of board stack and adds latest chessboard to top of stack.
        """
        self.board_stack.pop(0)
        self.board_stack.append(raw_map)

    def find_mismatches(self) -> list[_Mismatch]:
        """
        Returns info of squares whose most recent map mismatch the
        current game state map.
        """
        mismatches: list[_Mismatch] = []
        chessboard_map: dict[chess.Square, chess.Piece] = self.chessboard.piece_map()
        raw_map: dict[chess.Square, chess.Color] = self.board_stack[-1]
        for square in chess.SQUARES:
            if square in raw_map and square in chessboard_map:
                if raw_map[square] != chessboard_map[square].color:
                    mismatches.append(_Mismatch(square, raw_map[square], chessboard_map[square].color, True, chessboard_map[square]))
            elif square in raw_map and square not in chessboard_map:
                mismatches.append(_Mismatch(square, raw_map[square], None, False, None))
            elif square not in raw_map and square in chessboard_map:
                mismatches.append(_Mismatch(square, None, chessboard_map[square].color, False, chessboard_map[square]))
        return mismatches

    def generate_moves_from_mismatches(self, mismatches: list[_Mismatch]) -> None:
        """
        Pairs mismatched squares into possible moves and adds
        them to the class instance's move_stack.
        """
        for x in mismatches:
            for y in mismatches:
                if x.color is None and y.color is not None:
                    white_pawn = x.type == chess.Piece.from_symbol('P')
                    white_rank = x.location // 8 == 6 and y.location // 8 == 7
                    black_pawn = x.type == chess.Piece.from_symbol('p')
                    black_rank = x.location // 8 == 1 and y.location // 8 == 0
                    if white_pawn and white_rank or black_pawn and black_rank:
                        self.move_stack[-1].append(chess.Move(from_square=x.location, to_square=y.location, promotion=chess.QUEEN))
                    elif (y.capture or x.color == y.old_color) and (y.color == x.old_color or x.capture):
                        self.move_stack[-1].append(chess.Move(from_square=x.location, to_square=y.location))

    def validate_moves_and_push(self) -> None:
        """
        Verifies moves are legal and observed multiple times then
        pushes move to chessboard and game.
        """
        goal, target = int(self.delay / 2), int(self.delay * 0.8 / 2) # magic number
        for move in self.move_stack[-1]:
            if sum([move in moves for moves in self.move_stack[goal:]]) >= target:
                if move in self.chessboard.legal_moves:
                    self.node = self.node.add_variation(move)
                    self.chessboard.push(move)

    def find_and_validate_takebacks_and_push(self) -> None:
        """
        Finds, validates, and pushes takebacks to chessboard and game.
        """
        goal, target = int(self.delay / 2), int(self.delay * 0.8 / 2) # magic number
        for move in self.move_stack[-1]:
            if sum([move in moves for moves in self.move_stack[goal:]]) >= target:
                if self.chessboard.fen() != chess.STARTING_FEN and self.node.parent is not None:
                    tri_source = self.chessboard.peek().uci()[2:4] == move.uci()[:2]
                    tri_target = self.chessboard.peek().uci()[:2] == move.uci()[2:4]
                    if move.uci()[:2] != self.chessboard.peek().uci()[2:4]:
                        target_check = sum([chess.Move.from_uci(move.uci()[:2] + self.chessboard.peek().uci()[2:4]) in moves for moves in self.move_stack])
                    else:
                        target_check = False
                    if tri_source or (tri_target and target_check):
                        self.node = cast(chess.pgn.ChildNode, self.node)
                        self.node.parent.variations.remove(self.node)
                        self.node = self.node.parent
                        self.chessboard.pop()

    def find_out_of_turn_moves(self) -> list[chess.Move]:
        """
        Identifies moves that likely skipped a turn.
        """
        future_moves: list[chess.Move] = []
        for move in self.move_stack[-1]:
            if sum([move in moves for moves in self.move_stack]) >= self.delay * .8:
                if self.chessboard.turn != self.chessboard.color_at(move.from_square):
                    future_moves.append(move)
        return future_moves

    def find_potentially_skipped_moves(self, future_moves) -> list[chess.Move]:
        """
        Finds all legal moves that pass the turn.
        """
        potential_fixes: list[chess.Move] = []
        for future_move in future_moves:
            for move in self.chessboard.legal_moves:
                dummy_board = self.chessboard.copy()
                dummy_board.push(move)
                if future_move in dummy_board.legal_moves:
                    potential_fixes.append(move)
        return potential_fixes

    def validate_potentially_skipped_moves_and_push(self, potential_fixes: list[chess.Move]) -> None:
        """
        Verifies fix matches raw board chess.Board object before pushing
        to class instance's chessboard and game.
        """
        goal, target = int(self.delay / 2), int(self.delay * 0.8 / 2)
        for move in potential_fixes:
            test1 = sum([move.from_square not in raw_board for raw_board in self.board_stack[goal:]])
            test2 = sum([move.to_square in raw_board for raw_board in self.board_stack[goal:]])
            if test1 >= target and test2 >= target:
                self.node = self.node.add_variation(move)
                self.chessboard.push(move)
                break

    def write_pgn(self, out_file) -> None:
        """
        Overwrites PGN to external file.
        """
        with open(out_file, 'w') as pgn_file:
            pgn_file.write(self.game.accept(chess.pgn.StringExporter(headers=True)))
