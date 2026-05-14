from dataclasses import dataclass

from board import Board, Move


@dataclass(frozen=True)
class BitBoard:
    x_bits: int = 0
    o_bits: int = 0
    heights: tuple = (0, 0, 0, 0, 0, 0, 0)

    COLUMNS = 7
    ROWS = 6
    BITS_PER_COLUMN = 7

    @classmethod
    def from_board(cls, board: Board):
        x_bits = 0
        o_bits = 0
        heights = [0] * cls.COLUMNS

        for column in range(cls.COLUMNS):
            pieces = board.columns[column].pieces
            heights[column] = len(pieces)
            for row in range(len(pieces)):
                piece = pieces[-1 - row]
                bit = 1 << cls._bit_index(column, row)
                if piece == "X":
                    x_bits |= bit
                else:
                    o_bits |= bit

        return cls(x_bits=x_bits, o_bits=o_bits, heights=tuple(heights))

    @classmethod
    def _bit_index(cls, column, row):
        return column * cls.BITS_PER_COLUMN + row

    @classmethod
    def _column_mask(cls, column):
        return ((1 << cls.BITS_PER_COLUMN) - 1) << (column * cls.BITS_PER_COLUMN)

    def key(self, current_player):
        return (self.x_bits, self.o_bits, current_player)

    def is_full(self):
        return all(height >= self.ROWS for height in self.heights)

    def piece_at(self, column, row):
        bit = 1 << self._bit_index(column, row)
        if self.x_bits & bit:
            return "X"
        if self.o_bits & bit:
            return "O"
        return None

    def poppable(self, player):
        return any(self._bottom_owned_by(column, player) for column in range(self.COLUMNS))

    def _player_bits(self, player):
        return self.x_bits if player == "X" else self.o_bits

    def _with_player_bits(self, player, bits):
        if player == "X":
            return bits, self.o_bits
        return self.x_bits, bits

    def drop(self, column, player):
        if self.heights[column] >= self.ROWS:
            return None

        bit = 1 << self._bit_index(column, self.heights[column])
        player_bits = self._player_bits(player) | bit
        x_bits, o_bits = self._with_player_bits(player, player_bits)
        heights = list(self.heights)
        heights[column] += 1
        return BitBoard(x_bits=x_bits, o_bits=o_bits, heights=tuple(heights))

    def pop(self, column, player):
        if not self._bottom_owned_by(column, player):
            return None

        mask = self._column_mask(column)
        base = column * self.BITS_PER_COLUMN

        local_x = (self.x_bits & mask) >> base
        local_o = (self.o_bits & mask) >> base

        local_x >>= 1
        local_o >>= 1

        x_bits = (self.x_bits & ~mask) | (local_x << base)
        o_bits = (self.o_bits & ~mask) | (local_o << base)

        heights = list(self.heights)
        heights[column] -= 1
        return BitBoard(x_bits=x_bits, o_bits=o_bits, heights=tuple(heights))

    def _bottom_owned_by(self, column, player):
        if self.heights[column] == 0:
            return False
        bit = 1 << self._bit_index(column, 0)
        return bool(self._player_bits(player) & bit)

    def possible_move_dict(self, player):
        pos = {"drop": [], "pop": []}
        for column in range(self.COLUMNS):
            if self.heights[column] < self.ROWS:
                pos["drop"].append(column)
            if self._bottom_owned_by(column, player):
                pos["pop"].append(column)
        return pos

    def possible_moves(self, player):
        moves = []
        for kind, columns in self.possible_move_dict(player).items():
            for column in columns:
                moves.append(Move(kind, column))
        return moves

    def is_win(self, player):
        bits = self._player_bits(player)
        for shift in (1, self.BITS_PER_COLUMN, self.BITS_PER_COLUMN - 1, self.BITS_PER_COLUMN + 1):
            matches = bits & (bits >> shift)
            if matches & (matches >> (2 * shift)):
                return True
        return False


class BitGameState:
    class _StateCountView:
        def __init__(self, counts):
            self._counts = counts

        def count(self, key):
            return self._counts.get(key, 0)

    def __init__(self, board=None, player_to_move="X", last_move=None, state_counts=None):
        self.board = board if board is not None else BitBoard()
        self.player_to_move = player_to_move
        self.last_move = last_move
        self.board_key = self.board.key(player_to_move)
        self.state_counts = dict(state_counts) if state_counts is not None else {}
        self.states = self._StateCountView(self.state_counts)

    @classmethod
    def from_game_state(cls, state):
        state_counts = getattr(state, "state_counts", None)
        if state_counts is None:
            counts = {}
        else:
            counts = dict(state_counts)
        return cls(
            board=BitBoard.from_board(state.board),
            player_to_move=state.player_to_move,
            last_move=state.last_move,
            state_counts=counts,
        )

    def is_drawn(self):
        return self.last_move is not None and self.last_move.kind == "draw"

    def repetition_count(self):
        return self.state_counts.get(self.board_key, 0)

    def draw_legal(self):
        return self.board.is_full() or self.repetition_count() >= 3

    def get_winner(self):
        win_x = self.board.is_win("X")
        win_o = self.board.is_win("O")
        if win_x and win_o and self.last_move and self.last_move.kind == "pop":
            return "O" if self.player_to_move == "X" else "X"
        if win_x:
            return "X"
        if win_o:
            return "O"
        return None

    def is_terminal(self):
        return self.get_winner() is not None or self.is_drawn()

    def legal_moves(self):
        if self.is_terminal():
            return []

        moves = self.board.possible_moves(self.player_to_move)
        if self.draw_legal():
            moves.append(Move("draw", None))
        return moves

    def apply_move(self, move):
        next_state_counts = self.state_counts.copy()
        next_state_counts[self.board_key] = next_state_counts.get(self.board_key, 0) + 1

        if move.kind == "draw":
            return BitGameState(
                board=self.board,
                player_to_move=self.player_to_move,
                last_move=move,
                state_counts=next_state_counts,
            )

        new_board = self.board.drop(move.column, self.player_to_move)
        if move.kind == "pop":
            new_board = self.board.pop(move.column, self.player_to_move)

        next_player = "O" if self.player_to_move == "X" else "X"
        return BitGameState(
            board=new_board,
            player_to_move=next_player,
            last_move=move,
            state_counts=next_state_counts,
        )
