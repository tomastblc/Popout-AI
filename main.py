from board import Board, GameState, Move
from mcts import MCTS
import time


def print_board(state):
    print("\n" + "=" * 20)
    print("      POPOUT")
    print("=" * 20)

    grid = [["-" for _ in range(7)] for _ in range(6)]
    for c in range(7):
        pieces = state.board.columns[c].pieces
        for r in range(len(pieces)):
            grid[5 - r][c] = pieces[-(r + 1)]

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


def play_game(mode):
    state = GameState(Board(), player_to_move="X")
    ia_mcts_x = MCTS(iterations=1000)
    ia_mcts_o = MCTS(iterations=1000)

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
            if mode in ["1", "2"]:
                move = get_human_move(state)
            else:
                print("Computer (X) thinking via MCTS...")
                start_time = time.time()
                move = ia_mcts_x.search(state)
                print(f"Took {round(time.time() - start_time, 2)}s")
        else:
            if mode == "1":
                move = get_human_move(state)
            else:
                print("Computer (O) thinking via MCTS...")
                start_time = time.time()
                move = ia_mcts_o.search(state)
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
    print("3 - MCTS (X) vs MCTS (O)")
    print("=" * 30)

    while True:
        mode = input("Choose game mode (1/2/3): ")
        if mode in ["1", "2", "3"]:
            play_game(mode)
            break
        print("Invalid option.")
