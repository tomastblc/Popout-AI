import json
import time

from bitboard import BitBoard, BitGameState
from board import Move
from generate_dataset import state_to_dict
from id3_popout import classificar, move_from_classe_jogada
from mcts import MCTS

ID3_TREE_FILE = "arvore_id3_v5.json"


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


def load_id3_tree():
    try:
        with open(ID3_TREE_FILE, encoding="utf-8") as f:
            return json.load(f), ID3_TREE_FILE
    except FileNotFoundError:
        return None, None


def _move_is_legal(state, move):
    return any(
        m.kind == move.kind and m.column == move.column for m in state.legal_moves()
    )


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
    ia_mcts = MCTS(iterations=1000)
    ia_mcts_fb = MCTS(iterations=200)

    arvore_id3 = None
    if mode in ("2", "3"):
        arvore_id3, tree_path = load_id3_tree()
        if arvore_id3 is None:
            print(
                f"No ID3 JSON found at '{ID3_TREE_FILE}'. "
                "Generate it (e.g. run id3_popout.py writing to this path)."
            )
            return
        print(f"Loaded ID3 tree from {tree_path}")

    while True:
        print_board(state)

        winner = state.get_winner()
        if winner:
            print(f"\nPlayer {winner} wins.")
            break
        if state.is_drawn():
            print("\nThe game ended in a draw.")
            break

        if state.player_to_move == "X":
            if mode == "3":
                print("Computer (X) thinking via MCTS...")
                start_time = time.time()
                move = ia_mcts.search(state)
                print(f"Took {round(time.time() - start_time, 2)}s")
            else:
                move = get_human_move(state)
        else:
            if mode == "1":
                move = get_human_move(state)
            else:
                print("Computer (O) thinking via ID3...")
                move, info = pick_id3_move(state, arvore_id3, ia_mcts_fb)
                if "MCTS fallback" in info:
                    print(f"  ({info})")

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
    print("2 - Human (X) vs ID3 (O)")
    print("3 - MCTS (X) vs ID3 (O)")
    print("=" * 30)

    while True:
        mode = input("Choose game mode (1/2/3): ")
        if mode in ("1", "2", "3"):
            play_game(mode)
            break
        print("Invalid option.")
