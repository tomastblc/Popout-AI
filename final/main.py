import json
import time
from pathlib import Path

from libs.bitboard import BitBoard, BitGameState, Move
from libs.id3 import classificar, move_from_classe_jogada
from libs.mcts import MCTS
from workflow_support import FILES_DIR, workflow_specs
from generate_dataset_tournament import state_to_dict
from agents import build_agent


def available_id3_trees():
    trees = []
    for path in sorted(FILES_DIR.glob("*.json")):
        if "iris" in path.stem.lower():
            continue
        label = path.stem.replace("_", " ")
        trees.append((label, path))
    return trees


def print_board(state):
    print("\n" + "=" * 20)
    print("      POPOUT")
    print("=" * 20)

    grid = [["-" for _ in range(BitBoard.COLUMNS)] for _ in range(BitBoard.ROWS)]
    for column in range(BitBoard.COLUMNS):
        for row in range(BitBoard.ROWS):
            piece = state.board.piece_at(column, row)
            if piece is not None:
                grid[BitBoard.ROWS - 1 - row][column] = piece

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
                move = Move(parts[0], int(parts[1]))
            else:
                print("Invalid format. Use 'drop [column]', 'pop [column]' or 'draw'.")
                continue

            if any(m.kind == move.kind and m.column == move.column for m in valid_moves):
                return move

            print("Illegal move for the current state.")
        except ValueError:
            print("Column must be an integer between 0 and 6.")


def load_id3_tree(tree_path):
    try:
        with open(tree_path, encoding="utf-8") as f:
            return json.load(f), str(tree_path)
    except FileNotFoundError:
        return None, None


def choose_id3_tree(player_label=None):
    tree_options = available_id3_trees()
    if not tree_options:
        print("No PopOut ID3 JSON trees were found in final/files.")
        return None, None, None

    if player_label:
        print(f"\nID3 tree for player {player_label}:")
    else:
        print("\nAvailable ID3 trees:")

    for idx, (label, path) in enumerate(tree_options, start=1):
        print(f"  {idx} - {label} ({path.name})")

    prompt = f"Choose tree (1-{len(tree_options)})"
    if player_label:
        prompt += f" for {player_label}"
    prompt += ": "

    while True:
        choice = input(prompt).strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(tree_options):
                break
        except ValueError:
            pass
        print("Invalid option.")

    label, tree_path = tree_options[idx]
    tree, path = load_id3_tree(tree_path)
    if tree is None:
        print(f"No ID3 JSON found at '{tree_path}'.")
        return None, None, None
    print(f"Loaded ID3 tree: {label} ({path})")
    return tree, label, path


def _move_is_legal(state, move):
    return any(m.kind == move.kind and m.column == move.column for m in state.legal_moves())


def choose_mcts_opponent():
    while True:
        raw = input("MCTS iterations (e.g. 120): ").strip()
        try:
            iterations = int(raw)
            if iterations > 0:
                break
        except ValueError:
            pass
        print("Enter a positive integer.")

    specs = workflow_specs(iterations=iterations)
    print("\nAvailable MCTS variants:")
    for idx, spec in enumerate(specs, start=1):
        print(f"  {idx} - {spec.name}")

    while True:
        choice = input(f"Choose variant (1-{len(specs)}): ").strip()
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(specs):
                break
        except ValueError:
            pass
        print("Invalid option.")

    spec = specs[idx]
    agent = build_agent(spec)
    print(f"Opponent: {spec.name}")
    return agent, spec.name


def pick_id3_move(state, tree, fallback_engine):
    label = classificar(tree, state_to_dict(state))
    move = move_from_classe_jogada(label)
    if move is not None and _move_is_legal(state, move):
        return move, label

    fallback = fallback_engine.search(state)
    note = f"ID3 illegal/unknown (label={label!r})"
    return fallback, f"{note}; MCTS fallback -> {fallback.kind}_{fallback.column}"


def play_game(mode):
    state = BitGameState(BitBoard(), player_to_move="X")
    fallback_engine = MCTS(iterations=120)

    tree_x = None
    tree_o = None
    label_x = None
    label_o = None

    if mode == "3":
        tree_x, label_x, _ = choose_id3_tree("X")
        if tree_x is None:
            return
    elif mode == "4":
        tree_x, label_x, _ = choose_id3_tree("X")
        if tree_x is None:
            return
        tree_o, label_o, _ = choose_id3_tree("O")
        if tree_o is None:
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
            current_tree = tree_x if state.player_to_move == "X" else tree_o
            current_label = label_x if state.player_to_move == "X" else label_o

            print(f"Computer ({state.player_to_move}) thinking via ID3 ({current_label})...")
            move, info = pick_id3_move(state, current_tree, fallback_engine)
            if "MCTS fallback" in info:
                print(f"  ({info})")
        elif state.player_to_move == "X":
            move = get_human_move(state)
        else:
            if mode == "1":
                move = get_human_move(state)
            elif mode in ("2", "3"):
                print(f"Computer (O) thinking via {mcts_opponent_label}...")
                start_time = time.perf_counter()
                move = mcts_opponent.choose_move(state)
                print(f"Took {time.perf_counter() - start_time:.2f}s")

        if move.kind == "draw":
            print("\n>>> Move played: DRAW")
        else:
            print(f"\n>>> Move played: {move.kind.upper()} in column {move.column}")
        state = state.apply_move(move)


def main():
    print("=" * 30)
    print(" POPOUT (IA 25/26)")
    print("=" * 30)
    print("1 - Human (X) vs Human (O)")
    print("2 - Human (X) vs MCTS (O)")
    print("3 - ID3 (X) vs MCTS (O)")
    print("4 - ID3 (X) vs ID3 (O)")
    print("=" * 30)

    while True:
        mode = input("Choose game mode (1/2/3/4): ").strip()
        if mode in ("1", "2", "3", "4"):
            play_game(mode)
            return
        print("Invalid option.")


if __name__ == "__main__":
    main()
