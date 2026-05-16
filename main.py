import json
import time

from agents import AgentSpec, basic_mcts_specs, build_agent
from bitboard import BitBoard, BitGameState
from board import Move
from generate_dataset import state_to_dict
from id3_popout import classificar, move_from_classe_jogada
from mcts import MCTS, MCTSConfig

ID3_TREE_OPTIONS = [
    ("id3_v5", "arvore_id3_v5.json"),
    ("tour base", "arvore_tour_base.json"),
    ("tour adv", "arvore_tour_adv.json"),
    ("tour craz 1", "arvore_tour_craz_1.json"),
    ("tour craz 2", "arvore_tour_craz_2.json"),
    ("perturbed craz", "arvore_tour_craz_1_perturbed.json")
]


def piece_at(board, column, row):
    if hasattr(board, "piece_at"):
        return board.piece_at(column, row)

    pieces = board.columns[column].pieces
    if row < len(pieces):
        return pieces[-1 - row]
    return None


def print_board(state):
    print("\n" + "=" * 20)
    print("      POPOUT")
    print("=" * 20)

    grid = [["-" for _ in range(7)] for _ in range(6)]
    for c in range(7):
        for r in range(6):
            piece = piece_at(state.board, c, r)
            if piece is not None:
                grid[5 - r][c] = piece

    for row in grid:
        print(" ".join(row))
    print("0 1 2 3 4 5 6")
    print("=" * 20)


def get_human_move(state):
    valid_moves = state.legal_moves()
    while True:
        try:
            choice = input(
                f"Player {state.player_to_move}. Move ('drop 3', 'pop 3' or 'draw'): "
            ).strip().lower()
            parts = choice.split()

            if len(parts) == 1 and parts[0] == "draw":
                move = Move("draw", None)
            elif len(parts) == 2:
                kind = parts[0]
                col = int(parts[1])
                move = Move(kind, col)
            else:
                print("Invalid format. Use 'drop [column]', 'pop [column]' or 'draw'.")
                continue

            if any(m.kind == move.kind and m.column == move.column for m in valid_moves):
                return move

            print("Illegal move for the current state.")
        except ValueError:
            print("Column must be an integer between 0 and 6.")


def load_id3_tree(tree_file):
    try:
        with open(tree_file, encoding="utf-8") as f:
            return json.load(f), tree_file
    except FileNotFoundError:
        return None, None


def choose_id3_tree(player_label=None):
    if player_label:
        print(f"\nID3 tree for player {player_label}:")
    else:
        print("\nAvailable ID3 trees:")
    for i, (label, _) in enumerate(ID3_TREE_OPTIONS, start=1):
        print(f"  {i} - {label}")

    prompt = f"Choose tree (1-{len(ID3_TREE_OPTIONS)})"
    if player_label:
        prompt += f" for {player_label}"
    prompt += ": "

    while True:
        choice = input(prompt).strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(ID3_TREE_OPTIONS):
                break
        except ValueError:
            pass
        print("Invalid option.")

    label, tree_file = ID3_TREE_OPTIONS[idx]
    tree, path = load_id3_tree(tree_file)
    if tree is None:
        print(f"No ID3 JSON found at '{tree_file}'.")
        return None, None, None
    print(f"Loaded ID3 tree: {label} ({path})")
    return tree, label, path


def _move_is_legal(state, move):
    return any(
        m.kind == move.kind and m.column == move.column for m in state.legal_moves()
    )


def _spec_with_iterations(spec: AgentSpec, iterations: int) -> AgentSpec:
    return AgentSpec(
        name=spec.name,
        kind=spec.kind,
        config=MCTSConfig(**{**spec.config.__dict__, "iterations": iterations}),
        random_seed=spec.random_seed,
    )


def choose_mcts_opponent():
    specs = basic_mcts_specs()
    print("\nAvailable MCTS variants:")
    for i, spec in enumerate(specs, start=1):
        print(f"  {i} - {spec.name}")

    while True:
        choice = input(f"Choose variant (1-{len(specs)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(specs):
                break
        except ValueError:
            pass
        print("Invalid option.")

    while True:
        raw = input("MCTS iterations (e.g. 500): ").strip()
        try:
            iterations = int(raw)
            if iterations > 0:
                break
        except ValueError:
            pass
        print("Enter a positive integer.")

    spec = _spec_with_iterations(specs[idx], iterations)
    agent = build_agent(spec)
    label = f"{spec.name} ({iterations} iter)"
    print(f"Opponent: {label}")
    return agent, label


def pick_id3_move(state, tree, mcts_fallback):
    label = classificar(tree, state_to_dict(state))
    move = move_from_classe_jogada(label)
    if move is not None and _move_is_legal(state, move):
        return move, label
    fb = mcts_fallback.search(state)
    note = (
        f"ID3 illegal/unknown (label={label!r})"
        if move is None or not _move_is_legal(state, move)
        else "ID3 mismatch"
    )
    return fb, f"{note}; MCTS fallback -> {fb.kind}_{fb.column}"


def play_game(mode):
    state = BitGameState(BitBoard(), player_to_move="X")
    ia_mcts_fb = MCTS(iterations=200)

    arvore_id3 = None
    id3_tree_label = None
    arvore_id3_x = None
    id3_label_x = None
    arvore_id3_o = None
    id3_label_o = None
    if mode == "3":
        arvore_id3, id3_tree_label, _ = choose_id3_tree("X")
        if arvore_id3 is None:
            return
    elif mode == "4":
        arvore_id3_x, id3_label_x, _ = choose_id3_tree("X")
        if arvore_id3_x is None:
            return
        arvore_id3_o, id3_label_o, _ = choose_id3_tree("O")
        if arvore_id3_o is None:
            return

    mcts_opponent = None
    mcts_opponent_label = None
    if mode in ("2", "3"):
        mcts_opponent, mcts_opponent_label = choose_mcts_opponent()

    while True:
        print_board(state)

        winner = state.get_winner()
        if winner:
            print(f"\nPlayer {winner} wins.")
            break
        if state.is_drawn():
            print("\nThe game ended in a draw.")
            break

        if mode == "4" or (mode == "3" and state.player_to_move == "X"):
            if mode == "3":
                tree, label = arvore_id3, id3_tree_label
            else:
                tree = arvore_id3_x if state.player_to_move == "X" else arvore_id3_o
                label = id3_label_x if state.player_to_move == "X" else id3_label_o

            print(
                f"Computer ({state.player_to_move}) thinking via ID3 ({label})..."
            )
            move, info = pick_id3_move(state, tree, ia_mcts_fb)
            if "MCTS fallback" in info:
                print(f"  ({info})")
        elif state.player_to_move == "X":
            move = get_human_move(state)
        else:
            if mode == "1":
                move = get_human_move(state)
            elif mode in ("2", "3"):
                print(f"Computer (O) thinking via {mcts_opponent_label}...")
                start_time = time.time()
                move = mcts_opponent.choose_move(state)
                print(f"Took {round(time.time() - start_time, 2)}s")

        if move.kind == "draw":
            print("\n>>> Move played: DRAW")
        else:
            print(f"\n>>> Move played: {move.kind.upper()} in column {move.column}")
        state = state.apply_move(move)


if __name__ == "__main__":
    print("=" * 30)
    print(" POPOUT (IA 25/26)")
    print("=" * 30)
    print("1 - Human (X) vs Human (O)")
    print("2 - Human (X) vs MCTS (O)")
    print("3 - ID3 (X) vs MCTS (O)")
    print("4 - ID3 (X) vs ID3 (O)")
    print("=" * 30)

    while True:
        mode = input("Choose game mode (1/2/3/4): ")
        if mode in ("1", "2", "3", "4"):
            play_game(mode)
            break
        print("Invalid option.")
