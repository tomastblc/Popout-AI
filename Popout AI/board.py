import random
import math
from dataclasses import dataclass
from copy import deepcopy

@dataclass(frozen=True)
class Move:
    kind: str
    column: int

class Column:
    ROWS = 6
    def __init__(self, pieces = ()):
        self.pieces = pieces

    def __str__(self):
        return str(self.pieces)

    def __eq__(self, other):
        return self.pieces == other.pieces
    
    def last(self):
        if self.pieces:
            return self.pieces[-1]
        return None
    
    def is_full(self):
        return self.pieces and len(self.pieces) >= self.ROWS
    
    def poppable(self, player):
        return len(self.pieces) > 0 and self.pieces[-1] == player

    def drop(self, player):
        if not self.is_full():
            return Column((player,) + self.pieces)
    
    def pop(self, player):
        if self.pieces and self.pieces[-1] == player:
            return Column(self.pieces[:-1])

class Board:
    COLUMNS = 7

    def __init__(self, positions: list = None):
        if positions is None:
            self.columns = {i: Column() for i in range(self.COLUMNS)}
        else:
            self.columns = {i: p for i, p in enumerate(positions)}

    @classmethod
    def from_string(cls, string: str):
        rows = string.split("\n")
        cols = [Column() for _ in range(cls.COLUMNS)]
        for r in range(Column.ROWS - 1, -1, -1):
            i = 0
            for s in rows[r].strip():
                if s in ("X", "O"):
                    cols[i] = cols[i].drop(s)
                i += 1
        return cls(positions=cols)

    def __str__(self):
        s = ""
        for c in self.columns.keys():
            s += str(c) + " : " + str(self.columns[c]) + "\n"
        return s
    
    def copy(self):
        new_brd = Board()
        new_brd.columns = self.columns.copy()
        return new_brd

    def key(self, current_player):
        return (
            tuple(col.pieces for col in self.columns.values()),
            current_player
        )

    def is_full(self):
        return all(col.is_full() for col in self.columns.values())

    def apply_move(self, move, player):
        if move.kind == "drop":
            return self.make_drop(move.column, player)
        elif move.kind == "pop":
            return self.make_pop(move.column, player)
        else:
            raise ValueError("Invalid move kind")

    def make_drop(self, column, player):
        new_col = self.columns[column].drop(player)
        new_brd = self.copy()
        new_brd.columns[column] = new_col
        return new_brd

    def make_pop(self, column, player):
        new_col = self.columns[column].pop(player)
        new_brd = self.copy()
        new_brd.columns[column] = new_col
        return new_brd

    def is_win(self, player):
        return False
    
    def possible_move_dict(self, player):
        pos = {"drop": [], "pop": []}

        for num in self.columns.keys():
            if not self.columns[num].is_full():
                pos["drop"].append(num)
            if self.columns[num].poppable(player):
                pos["pop"].append(num)

        return pos
    
    def possible_moves(self, player):
        pos = self.possible_move_dict(player)
        moves = []
        for kind in pos.keys():
            for column in pos[kind]:
                moves.append(Move(kind, column))
        return moves

class GameState:
    def __init__(self, board, player_to_move='X', last_move = None, states = []):
        self.board = board
        self.board_key = board.key(player_to_move)
        self.player_to_move = player_to_move
        self.last_move = last_move
        self.states = states # when creating a new state, add +1 to the board state count by its key

    def draw_legal(self):
        if self.board.is_full() or self.states.count(self.board_key) >= 3:
            return True
        return False

    def legal_moves(self):
        moves = self.board.possible_moves(self.player_to_move)
        if self.draw_legal():
            moves.append(Move("draw", None))
        return moves

    def apply_move(self, move):
        if move.kind == "draw":
            return GameState(self.board, self.player_to_move, move, self.states + [self.board_key])
        new_board = self.board.apply_move(move, self.player_to_move)
        next_player = 'O' if self.player_to_move == 'X' else 'X'
        return GameState(new_board, next_player, move, self.states + [self.board_key])


class MCTS:
    def __init__(self, iterations=1000, exploration_weight=1.414):
        self.iterations = iterations
        self.exploration_weight = exploration_weight

    def search(self, initial_state: GameState):
        root = MCTSNode(state=initial_state)

        for _ in range(self.iterations):
            node = root

            while not node.untried_moves and node.children:
                node = node.select_child(self.exploration_weight)
            if node.untried_moves:
                move = random.choice(node.untried_moves)
                node = node.add_child(move)

            result = self.simulate(node.state)

            node.backpropagate(result)

        return sorted(root.children, key=lambda c: c.visits)[-1].move

    def simulate(self, state: GameState):
        """Simulação aleatória (Playout)."""
        current_state = state
        
        while True:
            if current_state.board.is_win('X'):
                return 1.0  
            if current_state.board.is_win('O'):
                return -1.0 
            if current_state.draw_legal():
                return 0.0  

            moves = current_state.legal_moves()
            if not moves:
                return 0.0
            
            move = random.choice(moves)
            current_state = current_state.apply_move(move)

class MCTSNode:
    def __init__(self, state: GameState, parent=None, move=None, player_to_move='X'):
        self.state = state
        self.parent = parent
        self.move = move

        self.children = []
        self.visits = 0
        self.wins = 0.0

        self.untried_moves = self.state.legal_moves()

    def select_child(self, exploration_weight):
        log_total_visits = math.log(self.visits)
        
        def ucb1(child):
            win_rate = child.wins / child.visits
            if self.state.player_to_move == 'O':
                win_rate = -win_rate
                
            return win_rate + exploration_weight * math.sqrt(log_total_visits / child.visits)

        return max(self.children, key=ucb1)

    def add_child(self, move):
        """Remove a jogada das não tentadas e cria um novo nó filho."""
        self.untried_moves.remove(move)
        new_state = self.state.apply_move(move)
        child_node = MCTSNode(state=new_state, parent=self, move=move)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, result):
        """Atualiza o nó atual e sobe para o pai."""
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(result)



