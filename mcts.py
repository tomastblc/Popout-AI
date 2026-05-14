import math
import random
from dataclasses import dataclass

from bitboard import BitGameState
from board import Move


@dataclass(frozen=True)
class MCTSConfig:
    iterations: int = 1000
    exploration_c: float = 1.414
    expansion_policy: str = "all"
    expansion_k: int = 0
    rollout_policy: str = "random"
    rollout_depth_limit: int | None = None
    child_selection: str = "uct"
    draw_value: float = 0.5
    epsilon: float = 0.1
    random_seed: int | None = None


class MCTSNode:
    def __init__(self, state, config: MCTSConfig, rng: random.Random, parent=None, move=None):
        self.state = state
        self.config = config
        self.rng = rng
        self.parent = parent
        self.move = move
        self.player_who_just_moved = "O" if state.player_to_move == "X" else "X"

        self.children = []
        self.visits = 0
        self.wins = 0.0
        self._all_legal_moves = self.state.legal_moves()
        self._expandable_moves = self._select_expandable_moves()
        self.untried_moves = list(self._expandable_moves)

    def _select_expandable_moves(self):
        moves = list(self._all_legal_moves)
        if self.config.expansion_policy == "all" or self.config.expansion_k <= 0:
            return moves

        k = min(self.config.expansion_k, len(moves))
        if k >= len(moves):
            return moves

        if self.config.expansion_policy == "random_k":
            return self.rng.sample(moves, k)
        if self.config.expansion_policy == "heuristic_k":
            scored_moves = sorted(
                moves,
                key=lambda move: _move_priority(self.state, move),
                reverse=True,
            )
            return scored_moves[:k]

        raise ValueError(f"Unsupported expansion_policy: {self.config.expansion_policy}")

    def is_terminal(self):
        return self.state.is_terminal()

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def expand(self):
        move = self.untried_moves.pop(0)
        new_state = self.state.apply_move(move)
        child = MCTSNode(new_state, self.config, self.rng, parent=self, move=move)
        self.children.append(child)
        return child

    def best_child(self):
        if self.config.child_selection == "uct":
            return max(self.children, key=self._uct_score)
        if self.config.child_selection == "greedy":
            return max(self.children, key=self._greedy_score)
        if self.config.child_selection == "softmax":
            return self._softmax_child()
        raise ValueError(f"Unsupported child_selection: {self.config.child_selection}")

    def _uct_score(self, child):
        exploit = child.wins / child.visits
        explore = self.config.exploration_c * math.sqrt(math.log(self.visits) / child.visits)
        return exploit + explore

    def _greedy_score(self, child):
        return child.wins / child.visits

    def _softmax_child(self):
        scores = [self._uct_score(child) for child in self.children]
        max_score = max(scores)
        weights = [math.exp(score - max_score) for score in scores]
        total = sum(weights)
        pick = self.rng.random() * total
        cumulative = 0.0
        for child, weight in zip(self.children, weights):
            cumulative += weight
            if cumulative >= pick:
                return child
        return self.children[-1]

    def rollout(self):
        state = self.state
        depth = 0

        while True:
            winner = state.get_winner()
            if winner:
                return winner
            if state.is_drawn():
                return None

            if self.config.rollout_depth_limit is not None and depth >= self.config.rollout_depth_limit:
                return _evaluate_rollout_cutoff(state)

            moves = state.legal_moves()
            if not moves:
                return None

            move = _select_rollout_move(state, moves, self.config, self.rng)
            state = state.apply_move(move)
            depth += 1

    def backpropagate(self, winner):
        self.visits += 1

        if winner is None:
            self.wins += self.config.draw_value
        elif winner == self.player_who_just_moved:
            self.wins += 1.0

        if self.parent:
            self.parent.backpropagate(winner)


class MCTS:
    def __init__(self, iterations=1000, config: MCTSConfig | None = None):
        if config is None:
            config = MCTSConfig(iterations=iterations)
        elif iterations != 1000 and config.iterations == 1000:
            config = MCTSConfig(**{**config.__dict__, "iterations": iterations})

        self.config = config
        self.rng = random.Random(config.random_seed)

    def search(self, initial_state):
        if not isinstance(initial_state, BitGameState):
            initial_state = BitGameState.from_game_state(initial_state)

        root = MCTSNode(initial_state, self.config, self.rng)

        for _ in range(self.config.iterations):
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


def _select_rollout_move(state, moves, config: MCTSConfig, rng: random.Random):
    if config.rollout_policy == "random":
        return rng.choice(moves)
    if config.rollout_policy == "heuristic":
        return _select_heuristic_move(state, moves, rng)
    if config.rollout_policy == "epsilon_greedy":
        if rng.random() < config.epsilon:
            return rng.choice(moves)
        return _select_heuristic_move(state, moves, rng)
    raise ValueError(f"Unsupported rollout_policy: {config.rollout_policy}")


def _select_heuristic_move(state, moves, rng: random.Random):
    winning_moves = []
    blocking_moves = []

    opponent = "O" if state.player_to_move == "X" else "X"
    opponent_threats = _immediate_winning_moves_for_player(state, opponent)

    for move in moves:
        next_state = state.apply_move(move)
        if next_state.get_winner() == state.player_to_move:
            winning_moves.append(move)
        if move in opponent_threats:
            blocking_moves.append(move)

    if winning_moves:
        return rng.choice(winning_moves)
    if blocking_moves:
        return rng.choice(blocking_moves)
    return rng.choice(moves)


def _immediate_winning_moves_for_player(state, player):
    if state.player_to_move == player:
        candidate_state = state
    else:
        candidate_state = _state_with_player_to_move(state, player)

    winning_moves = set()
    for move in candidate_state.legal_moves():
        next_state = candidate_state.apply_move(move)
        if next_state.get_winner() == player:
            winning_moves.add(move)
    return winning_moves


def _state_with_player_to_move(state, player):
    return BitGameState(
        board=state.board,
        player_to_move=player,
        last_move=state.last_move,
        state_counts=state.state_counts,
    )


def _move_priority(state, move):
    next_state = state.apply_move(move)
    if next_state.get_winner() == state.player_to_move:
        return 3

    opponent = "O" if state.player_to_move == "X" else "X"
    opponent_threats = _immediate_winning_moves_for_player(state, opponent)
    if move in opponent_threats:
        return 2

    if move.kind == "draw":
        return 1

    return 0


def _evaluate_rollout_cutoff(state):
    current_player = state.player_to_move
    opponent = "O" if current_player == "X" else "X"

    current_wins = len(_immediate_winning_moves_for_player(state, current_player))
    opponent_wins = len(_immediate_winning_moves_for_player(state, opponent))

    if current_wins > opponent_wins:
        return current_player
    if opponent_wins > current_wins:
        return opponent
    return None
