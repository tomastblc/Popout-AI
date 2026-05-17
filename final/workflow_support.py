from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from agents import AgentSpec, basic_mcts_specs, build_agent
from libs.bitboard import BitBoard, BitGameState, Move
from generate_dataset_tournament import (
    gerar_dataset_torneio,
    gerar_jogos,
    guardar_csv as guardar_selfplay_csv,
    guardar_csv as guardar_tournament_csv,
    state_to_dict,
)
from libs.id3 import (
    ALVO,
    atributos_treino_popout,
    classificar,
    move_from_classe_jogada,
    preparar_dados_popout,
    treinar_id3,
)
from iris import cross_validate_iris
from libs.mcts import MCTS, MCTSConfig
from tournament import play_game, round_robin

ROOT = Path(__file__).resolve().parent
FILES_DIR = ROOT / "files"
FILES_DIR.mkdir(exist_ok=True)


def ensure_local_paths():
    FILES_DIR.mkdir(exist_ok=True)
    return ROOT, FILES_DIR


def clone_spec(spec: AgentSpec, *, iterations: int | None = None, name_suffix: str = ""):
    if spec.kind != "mcts" or spec.config is None:
        return AgentSpec(
            name=f"{spec.name}{name_suffix}",
            kind=spec.kind,
            config=spec.config,
            random_seed=spec.random_seed,
        )

    config_dict = dict(spec.config.__dict__)
    if iterations is not None:
        config_dict["iterations"] = iterations

    return AgentSpec(
        name=f"{spec.name}{name_suffix}",
        kind=spec.kind,
        config=MCTSConfig(**config_dict),
        random_seed=spec.random_seed,
    )


def workflow_specs(iterations=120):
    selected_names = [
        "UCT_base",
        "UCT_low_c",
        "UCT_heuristic_rollout",
        "UCT_epsilon_greedy",
    ]
    base_specs = {spec.name: spec for spec in basic_mcts_specs()}
    return [
        clone_spec(base_specs[name], iterations=iterations, name_suffix=f"_{iterations}")
        for name in selected_names
    ]


def specs_to_frame(specs):
    rows = []
    for spec in specs:
        config = spec.config
        rows.append(
            {
                "name": spec.name,
                "kind": spec.kind,
                "iterations": getattr(config, "iterations", None),
                "exploration_c": getattr(config, "exploration_c", None),
                "expansion_policy": getattr(config, "expansion_policy", None),
                "expansion_k": getattr(config, "expansion_k", None),
                "rollout_policy": getattr(config, "rollout_policy", None),
                "rollout_depth_limit": getattr(config, "rollout_depth_limit", None),
                "child_selection": getattr(config, "child_selection", None),
                "draw_value": getattr(config, "draw_value", None),
                "epsilon": getattr(config, "epsilon", None),
            }
        )
    return pd.DataFrame(rows)


def format_board(state):
    grid = [["-" for _ in range(BitBoard.COLUMNS)] for _ in range(BitBoard.ROWS)]
    for column in range(BitBoard.COLUMNS):
        for row in range(BitBoard.ROWS):
            piece = state.board.piece_at(column, row)
            if piece is not None:
                grid[BitBoard.ROWS - 1 - row][column] = piece

    lines = [" ".join(row) for row in grid]
    lines.append("0 1 2 3 4 5 6")
    lines.append(f"player_to_move={state.player_to_move}")
    lines.append(f"legal_moves={[f'{m.kind}_{m.column}' for m in state.legal_moves()]}")
    return "\n".join(lines)


def board_demo():
    state = BitGameState(BitBoard(), player_to_move="X")
    scripted_moves = [
        Move("drop", 0),
        Move("drop", 1),
        Move("drop", 0),
        Move("drop", 1),
        Move("pop", 0),
        Move("drop", 2),
    ]

    snapshots = [
        {
            "step": 0,
            "move": "initial",
            "board": format_board(state),
            "winner": state.get_winner(),
        }
    ]

    for step, move in enumerate(scripted_moves, start=1):
        state = state.apply_move(move)
        snapshots.append(
            {
                "step": step,
                "move": f"{move.kind}_{move.column}",
                "board": format_board(state),
                "winner": state.get_winner(),
            }
        )

    return state, pd.DataFrame(snapshots)


def count_tree_nodes(tree):
    if not isinstance(tree, dict):
        return 1
    attribute = next(iter(tree))
    return 1 + sum(count_tree_nodes(branch) for branch in tree[attribute].values())


def tree_depth(tree):
    if not isinstance(tree, dict):
        return 0
    attribute = next(iter(tree))
    if not tree[attribute]:
        return 1
    return 1 + max(tree_depth(branch) for branch in tree[attribute].values())


def save_tree(tree, tree_path):
    tree_path = Path(tree_path)
    with tree_path.open("w", encoding="utf-8") as f:
        json.dump(tree, f, indent=4)
    return tree_path


def generate_selfplay_dataset(output_path, num_games=6, iterations=80):
    output_path = Path(output_path)
    rows = gerar_jogos(num_jogos=num_games, iteracoes_mcts=iterations, num_workers=1)
    guardar_selfplay_csv(rows, str(output_path))
    return rows


def train_popout_tree_from_csv(dataset_path, tree_path):
    dataset_path = Path(dataset_path)
    tree_path = Path(tree_path)
    rows = preparar_dados_popout(str(dataset_path))
    if not rows:
        raise ValueError(f"No rows found in dataset: {dataset_path}")

    attributes = atributos_treino_popout(rows, ALVO)
    tree = treinar_id3(rows, attributes, ALVO)
    save_tree(tree, tree_path)

    return {
        "tree": tree,
        "dataset_path": str(dataset_path),
        "tree_path": str(tree_path),
        "rows": len(rows),
        "attributes": len(attributes),
        "nodes": count_tree_nodes(tree),
        "depth": tree_depth(tree),
    }


def iris_workflow(csv_path="files/iris.csv", k=5, num_bins=4, seed=42):
    previous_cwd = Path.cwd()
    os.chdir(FILES_DIR)
    try:
        results = cross_validate_iris(caminho_csv=str(ROOT / csv_path), k=k, num_bins=num_bins, seed=seed)
    finally:
        os.chdir(previous_cwd)

    rows = []
    for fold_idx, result in enumerate(results, start=1):
        rows.append(
            {
                "fold": fold_idx,
                "accuracy": result["accuracy"],
                "unknown_rate": result["unknown_rate"],
                "tree_path": str(FILES_DIR / result["tree_path"]),
            }
        )
    return pd.DataFrame(rows)


def round_robin_workflow(specs, games_per_side=1, num_workers=1):
    summary = round_robin(specs, games_per_side=games_per_side, num_workers=num_workers)
    ranking_df = pd.DataFrame(summary["ranking"], columns=["agent", "score"])

    match_rows = []
    for match in summary["matches"]:
        result = match["result"]
        match_rows.append(
            {
                "agent_a": match["agent_a"],
                "agent_b": match["agent_b"],
                "score_a": result[match["agent_a"]],
                "score_b": result[match["agent_b"]],
                "draws": result["draws"],
                "games": result["games"],
                "avg_moves": result["avg_moves"],
                "avg_move_time_a": result["avg_move_time"][match["agent_a"]],
                "avg_move_time_b": result["avg_move_time"][match["agent_b"]],
            }
        )

    return summary, ranking_df, pd.DataFrame(match_rows)


def generate_tournament_dataset(output_path, specs, games_per_side=1):
    output_path = Path(output_path)
    tournament_specs = [
        {
            "name": "workflow_tournament",
            "specs": specs,
            "games_per_side": games_per_side,
        }
    ]
    rows = gerar_dataset_torneio(tournament_specs)
    guardar_tournament_csv(rows, str(output_path))
    return rows


def _is_legal_move(state, move):
    return any(candidate.kind == move.kind and candidate.column == move.column for candidate in state.legal_moves())


class ID3Agent:
    def __init__(self, tree, fallback_iterations=60):
        self.tree = tree
        self.fallback = MCTS(iterations=fallback_iterations)

    def choose_move(self, state):
        label = classificar(self.tree, state_to_dict(state))
        move = move_from_classe_jogada(label)
        if move is not None and _is_legal_move(state, move):
            return move
        return self.fallback.search(state)


def play_id3_vs_mcts(tree, spec, games_per_side=2, fallback_iterations=60):
    rows = []
    score_id3 = 0.0
    score_mcts = 0.0
    draws = 0

    for game_idx in range(1, games_per_side + 1):
        id3_agent = ID3Agent(tree, fallback_iterations=fallback_iterations)
        mcts_agent = build_agent(spec)
        result = play_game(id3_agent, mcts_agent)
        winner_label = "draw"
        if result["winner"] == "X":
            score_id3 += 1.0
            winner_label = "ID3"
        elif result["winner"] == "O":
            score_mcts += 1.0
            winner_label = spec.name
        else:
            score_id3 += 0.5
            score_mcts += 0.5
            draws += 1

        rows.append(
            {
                "game": game_idx,
                "id3_side": "X",
                "mcts_side": "O",
                "winner": winner_label,
                "moves": result["moves"],
            }
        )

    for game_idx in range(1, games_per_side + 1):
        id3_agent = ID3Agent(tree, fallback_iterations=fallback_iterations)
        mcts_agent = build_agent(spec)
        result = play_game(mcts_agent, id3_agent)
        winner_label = "draw"
        if result["winner"] == "X":
            score_mcts += 1.0
            winner_label = spec.name
        elif result["winner"] == "O":
            score_id3 += 1.0
            winner_label = "ID3"
        else:
            score_id3 += 0.5
            score_mcts += 0.5
            draws += 1

        rows.append(
            {
                "game": games_per_side + game_idx,
                "id3_side": "O",
                "mcts_side": "X",
                "winner": winner_label,
                "moves": result["moves"],
            }
        )

    games_total = games_per_side * 2
    summary = {
        "opponent": spec.name,
        "games": games_total,
        "id3_score": score_id3,
        "mcts_score": score_mcts,
        "draws": draws,
    }
    return summary, pd.DataFrame(rows)


def evaluate_id3_against_specs(tree, specs, games_per_side=2, fallback_iterations=60):
    summaries = []
    game_frames = []
    for spec in specs:
        summary, games_df = play_id3_vs_mcts(
            tree,
            spec,
            games_per_side=games_per_side,
            fallback_iterations=fallback_iterations,
        )
        summaries.append(summary)
        if not games_df.empty:
            game_frames.append(games_df.assign(opponent=spec.name))

    summary_df = pd.DataFrame(summaries)
    games_df = pd.concat(game_frames, ignore_index=True) if game_frames else pd.DataFrame()
    return summary_df, games_df
