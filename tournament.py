import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from bitboard import BitBoard, BitGameState
from agents import AgentSpec, build_agent, default_mcts_specs


def play_game(agent_x, agent_o):
    """
    Play one game and return the winner plus a few tournament metrics.
    """
    return _play_game_internal(agent_x, agent_o, collect_history=False)


def play_game_with_history(agent_x, agent_o, agent_x_name, agent_o_name):
    """
    Play one game and return both summary metrics and a move-by-move history.
    """
    return _play_game_internal(
        agent_x,
        agent_o,
        collect_history=True,
        agent_x_name=agent_x_name,
        agent_o_name=agent_o_name,
    )


def _play_game_internal(agent_x, agent_o, collect_history, agent_x_name=None, agent_o_name=None):
    state = BitGameState(BitBoard(), player_to_move="X")
    move_count = 0
    move_counts = {"X": 0, "O": 0}
    total_move_time = {"X": 0.0, "O": 0.0}
    move_history = [] if collect_history else None

    while not state.is_terminal():
        agent = agent_x if state.player_to_move == "X" else agent_o
        player = state.player_to_move
        agent_name = agent_x_name if player == "X" else agent_o_name
        start_time = time.perf_counter()
        move = agent.choose_move(state)
        elapsed = time.perf_counter() - start_time
        total_move_time[player] += elapsed
        move_counts[player] += 1
        if move is None:
            break
        if collect_history:
            move_history.append(
                {
                    "state": state,
                    "move": move,
                    "player": player,
                    "agent_name": agent_name,
                    "move_time": elapsed,
                }
            )
        state = state.apply_move(move)
        move_count += 1

    result = {
        "winner": state.get_winner(),
        "is_draw": state.is_drawn() or state.get_winner() is None,
        "moves": move_count,
        "move_counts": move_counts,
        "total_move_time": total_move_time,
        "avg_move_time": {
            player: (total_move_time[player] / move_counts[player]) if move_counts[player] else 0.0
            for player in ("X", "O")
        },
    }
    if collect_history:
        result["history"] = move_history
    return result


def play_match(spec_a: AgentSpec, spec_b: AgentSpec, games_per_side=10):
    """
    Play both color assignments to reduce first-player bias.
    """
    results = {
        spec_a.name: 0.0,
        spec_b.name: 0.0,
        "draws": 0,
        "games": 0,
        "avg_moves": 0.0,
        "avg_move_time": {
            spec_a.name: 0.0,
            spec_b.name: 0.0,
        },
    }

    total_moves = 0
    total_move_time = {
        spec_a.name: 0.0,
        spec_b.name: 0.0,
    }
    total_agent_moves = {
        spec_a.name: 0,
        spec_b.name: 0,
    }

    for _ in range(games_per_side):
        agent_a = build_agent(spec_a)
        agent_b = build_agent(spec_b)
        game = play_game(agent_a, agent_b)
        _record_game(results, spec_a.name, spec_b.name, game)
        total_moves += game["moves"]
        total_move_time[spec_a.name] += game["total_move_time"]["X"]
        total_move_time[spec_b.name] += game["total_move_time"]["O"]
        total_agent_moves[spec_a.name] += game["move_counts"]["X"]
        total_agent_moves[spec_b.name] += game["move_counts"]["O"]

    for _ in range(games_per_side):
        agent_a = build_agent(spec_a)
        agent_b = build_agent(spec_b)
        game = play_game(agent_b, agent_a)
        _record_game(results, spec_b.name, spec_a.name, game)
        total_moves += game["moves"]
        total_move_time[spec_b.name] += game["total_move_time"]["X"]
        total_move_time[spec_a.name] += game["total_move_time"]["O"]
        total_agent_moves[spec_b.name] += game["move_counts"]["X"]
        total_agent_moves[spec_a.name] += game["move_counts"]["O"]

    if results["games"]:
        results["avg_moves"] = total_moves / results["games"]
    for name in (spec_a.name, spec_b.name):
        if total_agent_moves[name]:
            results["avg_move_time"][name] = total_move_time[name] / total_agent_moves[name]

    return results


def round_robin(specs, games_per_side=10, num_workers=1):
    table = {spec.name: 0.0 for spec in specs}
    detailed_results = []
    pairings = [
        (specs[idx], specs[jdx], games_per_side)
        for idx in range(len(specs))
        for jdx in range(idx + 1, len(specs))
    ]

    if num_workers is None:
        num_workers = os.cpu_count() or 1
    num_workers = max(1, min(num_workers, len(pairings) if pairings else 1))

    if num_workers == 1:
        for spec_a, spec_b, games in pairings:
            result = play_match(spec_a, spec_b, games_per_side=games)
            _record_match_result(table, detailed_results, spec_a, spec_b, result)
    else:
        # Each pairing is independent, so matchup-level parallelism is safe.
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_play_match_task, spec_a, spec_b, games): (spec_a, spec_b)
                for spec_a, spec_b, games in pairings
            }
            for future in as_completed(futures):
                spec_a, spec_b = futures[future]
                result = future.result()
                _record_match_result(table, detailed_results, spec_a, spec_b, result)

    ranking = sorted(table.items(), key=lambda item: item[1], reverse=True)
    return {
        "scores": table,
        "ranking": ranking,
        "matches": detailed_results,
    }


def _record_game(results, x_name, o_name, game):
    results["games"] += 1
    if game["winner"] == "X":
        results[x_name] += 1.0
    elif game["winner"] == "O":
        results[o_name] += 1.0
    else:
        results[x_name] += 0.5
        results[o_name] += 0.5
        results["draws"] += 1


def _play_match_task(spec_a, spec_b, games_per_side):
    return play_match(spec_a, spec_b, games_per_side=games_per_side)


def _record_match_result(table, detailed_results, spec_a, spec_b, result):
    table[spec_a.name] += result[spec_a.name]
    table[spec_b.name] += result[spec_b.name]
    detailed_results.append(
        {
            "agent_a": spec_a.name,
            "agent_b": spec_b.name,
            "result": result,
        }
    )
    print(
        f"{spec_a.name} vs {spec_b.name}: "
        f"{result[spec_a.name]:.1f} - {result[spec_b.name]:.1f}, "
        f"draws={result['draws']}, "
        f"avg_moves={result['avg_moves']:.1f}, "
        f"avg_move_time=({spec_a.name}: {result['avg_move_time'][spec_a.name]:.4f}s, "
        f"{spec_b.name}: {result['avg_move_time'][spec_b.name]:.4f}s)"
    )


if __name__ == "__main__":
    specs = default_mcts_specs()
    summary = round_robin(specs, games_per_side=4, num_workers=os.cpu_count() or 1)

    print("\nRanking:")
    for position, (name, score) in enumerate(summary["ranking"], start=1):
        print(f"{position}. {name}: {score:.1f}")
