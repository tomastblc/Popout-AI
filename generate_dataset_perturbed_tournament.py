import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from agents import default_mcts_specs
from bitboard import BitBoard, BitGameState
from generate_dataset import state_to_dict
from agents import build_agent


INPUT_DATASET = "tour_adv_1.csv" #change to adv
OUTPUT_DATASET = "tour_adv_1_perturbed.csv"

# Use a range to keep the perturbation varied.
# Set both values to the same number if you want a fixed n.
MIN_RANDOM_MOVES = 2
MAX_RANDOM_MOVES = 6
MAX_PERTURBATION_ATTEMPTS = 3

ALLOW_RANDOM_DRAW = False
RANDOM_SEED = 42
NUM_WORKERS = None
CHUNK_SIZE = 80

# By default the script labels perturbed states with all available specs.
# Replace with a subset such as ["UCT_base_10k", "UCT_depth_limited_10k"] if needed.
SELECTED_SPEC_NAMES = ["UCT_heuristic_rollout_10k", "UCT_epsilon_greedy_10k", "UCT_depth_limited_2k"] #change to _2k

# Tournament bookkeeping is removed from the final dataset.
DROP_COLUMNS = {
    "tournament_name",
    "match_name",
    "game_id",
    "ply",
    "move_time",
}


def available_specs():
    return default_mcts_specs()


def selected_specs():
    specs = available_specs()
    if SELECTED_SPEC_NAMES is None:
        return specs

    spec_by_name = {spec.name: spec for spec in specs}
    missing = [name for name in SELECTED_SPEC_NAMES if name not in spec_by_name]
    if missing:
        raise ValueError(f"Unknown spec names: {missing}")

    return [spec_by_name[name] for name in SELECTED_SPEC_NAMES]


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() == "true"


def _row_to_state(row):
    x_bits = 0
    o_bits = 0
    heights = [0] * BitBoard.COLUMNS

    for column in range(BitBoard.COLUMNS):
        highest_piece = -1
        for row_idx in range(BitBoard.ROWS):
            key = f"c{column}_r{row_idx}"
            piece = row.get(key, "Vazio")
            if piece == "Vazio" or piece == "":
                continue

            highest_piece = row_idx
            bit = 1 << (column * BitBoard.BITS_PER_COLUMN + row_idx)
            if piece == "X":
                x_bits |= bit
            elif piece == "O":
                o_bits |= bit
            else:
                raise ValueError(f"Unexpected piece value {piece!r} in column {column}, row {row_idx}")

        heights[column] = highest_piece + 1

    player_to_move = row["player_to_move"]
    board = BitBoard(x_bits=x_bits, o_bits=o_bits, heights=tuple(heights))

    repetition_count = int(row.get("repetition_count", 0) or 0)
    # The dataset only stores the current repetition count, not the whole history.
    # Reconstruct just the current board key count so draw legality matches the row.
    state_counts = {}
    if repetition_count > 0:
        state_counts[board.key(player_to_move)] = repetition_count

    state = BitGameState(
        board=board,
        player_to_move=player_to_move,
        state_counts=state_counts,
    )

    expected_draw = _parse_bool(row.get("draw_legal", False))
    if state.draw_legal() != expected_draw:
        raise ValueError(
            "Reconstructed draw_legal does not match dataset row. "
            "The row likely depends on history that is not recoverable from the CSV alone."
        )

    return state


def _choose_random_move(state, rng):
    legal_moves = state.legal_moves()
    if not ALLOW_RANDOM_DRAW:
        non_draw_moves = [move for move in legal_moves if move.kind != "draw"]
        if non_draw_moves:
            legal_moves = non_draw_moves

    if not legal_moves:
        return None
    return rng.choice(legal_moves)


def perturb_state(state, rng):
    current_state = state
    applied_steps = 0

    for _ in range(MAX_PERTURBATION_ATTEMPTS):
        current_state = state
        target_steps = rng.randint(MIN_RANDOM_MOVES, MAX_RANDOM_MOVES)
        applied_steps = 0

        while applied_steps < target_steps and not current_state.is_terminal():
            move = _choose_random_move(current_state, rng)
            if move is None:
                break
            current_state = current_state.apply_move(move)
            applied_steps += 1

        if applied_steps > 0 and not current_state.is_terminal():
            return current_state, applied_steps

    return current_state, applied_steps


def _base_output_row(row):
    clean_row = {key: value for key, value in row.items() if key not in DROP_COLUMNS}
    clean_row["source"] = "tournament"
    clean_row["perturbation_steps"] = 0
    return clean_row


def _perturbed_output_row(state, move, agent_name, perturbation_steps):
    row = state_to_dict(state)
    row["classe_jogada"] = f"{move.kind}_{move.column}"
    row["agent_name"] = agent_name
    row["source"] = "perturbed_tournament"
    row["perturbation_steps"] = perturbation_steps
    return row


def _fieldnames():
    cells = [f"c{column}_r{row}" for column in range(BitBoard.COLUMNS) for row in range(BitBoard.ROWS)]
    return cells + [
        "player_to_move",
        "repetition_count",
        "draw_legal",
        "classe_jogada",
        "agent_name",
        "source",
        "perturbation_steps",
    ]


def load_rows(csv_path):
    with open(csv_path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _chunk_rows(rows, chunk_size):
    return [rows[idx:idx + chunk_size] for idx in range(0, len(rows), chunk_size)]


def initialize_output_dataset(rows, csv_path):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames())
        writer.writeheader()
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def append_rows(rows, csv_path):
    if not rows:
        return

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_fieldnames())
        writer.writerows(rows)
        f.flush()
        os.fsync(f.fileno())


def _process_chunk(rows, specs, seed):
    rng = random.Random(seed)
    agents = {spec.name: build_agent(spec) for spec in specs}

    output_rows = []
    perturbed_examples = 0
    skipped_rows = 0

    for row in rows:
        state = _row_to_state(row)
        perturbed_state, applied_steps = perturb_state(state, rng)

        if applied_steps == 0 or perturbed_state.is_terminal():
            skipped_rows += 1
            continue

        for spec in specs:
            move = agents[spec.name].choose_move(perturbed_state)
            if move is None:
                continue

            output_rows.append(
                _perturbed_output_row(
                    perturbed_state,
                    move,
                    spec.name,
                    applied_steps,
                )
            )
            perturbed_examples += 1

    return {
        "rows": output_rows,
        "perturbed_examples": perturbed_examples,
        "skipped_rows": skipped_rows,
        "processed_rows": len(rows),
    }


def augment_dataset(input_dataset, output_dataset):
    tournament_rows = load_rows(input_dataset)
    base_rows = [_base_output_row(row) for row in tournament_rows]

    specs = selected_specs()
    num_workers = NUM_WORKERS if NUM_WORKERS is not None else (os.cpu_count() or 1)
    num_workers = max(1, min(num_workers, len(tournament_rows) if tournament_rows else 1))
    chunks = _chunk_rows(tournament_rows, CHUNK_SIZE)

    print(f"Loaded {len(tournament_rows)} tournament rows from '{input_dataset}'.")
    print(f"Selected {len(specs)} specs for perturbed labels.")
    print(f"Using {num_workers} worker(s) across {len(chunks)} chunk(s) of up to {CHUNK_SIZE} rows.")

    initialize_output_dataset(base_rows, output_dataset)
    print(
        f"Initialized '{output_dataset}' with {len(base_rows)} original tournament rows.",
        flush=True,
    )

    perturbed_examples = 0
    skipped_rows = 0
    processed_rows = 0
    saved_rows = len(base_rows)
    started_at = time.perf_counter()

    if num_workers == 1:
        for chunk_idx, chunk in enumerate(chunks, start=1):
            result = _process_chunk(chunk, specs, RANDOM_SEED + chunk_idx)
            append_rows(result["rows"], output_dataset)
            perturbed_examples += result["perturbed_examples"]
            skipped_rows += result["skipped_rows"]
            processed_rows += result["processed_rows"]
            saved_rows += len(result["rows"])
            elapsed = time.perf_counter() - started_at
            print(
                f"Chunk {chunk_idx}/{len(chunks)} done; "
                f"processed {processed_rows}/{len(tournament_rows)} base rows, "
                f"generated {perturbed_examples} perturbed labels, "
                f"skipped {skipped_rows} terminal rows, "
                f"saved_rows {saved_rows}, "
                f"elapsed {elapsed:.1f}s.",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_process_chunk, chunk, specs, RANDOM_SEED + chunk_idx): chunk_idx
                for chunk_idx, chunk in enumerate(chunks, start=1)
            }

            for completed_idx, future in enumerate(as_completed(futures), start=1):
                chunk_idx = futures[future]
                result = future.result()
                append_rows(result["rows"], output_dataset)
                perturbed_examples += result["perturbed_examples"]
                skipped_rows += result["skipped_rows"]
                processed_rows += result["processed_rows"]
                saved_rows += len(result["rows"])
                elapsed = time.perf_counter() - started_at
                print(
                    f"Chunk {completed_idx}/{len(chunks)} finished "
                    f"(submitted chunk #{chunk_idx}); "
                    f"processed {processed_rows}/{len(tournament_rows)} base rows, "
                    f"generated {perturbed_examples} perturbed labels, "
                    f"skipped {skipped_rows} terminal rows, "
                    f"saved_rows {saved_rows}, "
                    f"elapsed {elapsed:.1f}s.",
                    flush=True,
                )

    elapsed = time.perf_counter() - started_at
    print(
        f"Saved {saved_rows} rows to '{output_dataset}' "
        f"({len(tournament_rows)} original + {perturbed_examples} perturbed) "
        f"in {elapsed:.1f}s."
    )


if __name__ == "__main__":
    augment_dataset(INPUT_DATASET, OUTPUT_DATASET)
