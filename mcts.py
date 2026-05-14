import math
import random

from bitboard import BitGameState
from board import Move


class MCTSNode:
    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.player_who_just_moved = "O" if state.player_to_move == "X" else "X"

        self.children = []
        self.visits = 0
        self.wins = 0.0
        self.untried_moves = self.state.legal_moves()

    def is_terminal(self):
        return self.state.is_terminal()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def expand(self):
        move = self.untried_moves.pop(0)
        new_state = self.state.apply_move(move)
        child = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child)
        return child

    def best_child(self, c=1.414):
        def ucb1(child):
            exploit = child.wins / child.visits
            explore = c * math.sqrt(math.log(self.visits) / child.visits)
            return exploit + explore

        return max(self.children, key=ucb1)

    def rollout(self):
        state = self.state

        while True:
            winner = state.get_winner()
            if winner:
                return winner
            if state.is_drawn():
                return None

            moves = state.board.possible_moves(state.player_to_move)
            if state.draw_legal():
                moves.append(Move("draw", None))
            if not moves:
                return None

            state = state.apply_move(random.choice(moves))

    def backpropagate(self, winner):
        self.visits += 1

        if winner is None:
            self.wins += 0.5
        elif winner == self.player_who_just_moved:
            self.wins += 1.0

        if self.parent:
            self.parent.backpropagate(winner)


class MCTS:
    def __init__(self, iterations=1000):
        self.iterations = iterations

    def search(self, initial_state):
        if not isinstance(initial_state, BitGameState):
            initial_state = BitGameState.from_game_state(initial_state)

        root = MCTSNode(initial_state)

        for _ in range(self.iterations):
            node = root

            while not node.is_terminal() and node.is_fully_expanded():
                if not node.children:
                    break
                node = node.best_child()

            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()

            winner = node.rollout()
            node.backpropagate(winner)

        if not root.children:
            return None

        return max(root.children, key=lambda c: c.visits).move
