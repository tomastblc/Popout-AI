from bitboard import BitBoard, BitGameState
from agents import AgentSpec, build_agent, default_mcts_specs


def play_game(agent_x, agent_o):
    """
    Play one game and return the winner plus a few tournament metrics.
    """
    state = BitGameState(BitBoard(), player_to_move="X")
    move_count = 0

    while not state.is_terminal():
        agent = agent_x if state.player_to_move == "X" else agent_o
        move = agent.choose_move(state)
        if move is None:
            break
        state = state.apply_move(move)
        move_count += 1

    return {
        "winner": state.get_winner(),
        "is_draw": state.is_drawn() or state.get_winner() is None,
        "moves": move_count,
    }


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
    }

    total_moves = 0

    for _ in range(games_per_side):
        agent_a = build_agent(spec_a)
        agent_b = build_agent(spec_b)
        game = play_game(agent_a, agent_b)
        _record_game(results, spec_a.name, spec_b.name, game)
        total_moves += game["moves"]

    for _ in range(games_per_side):
        agent_a = build_agent(spec_a)
        agent_b = build_agent(spec_b)
        game = play_game(agent_b, agent_a)
        _record_game(results, spec_b.name, spec_a.name, game)
        total_moves += game["moves"]

    if results["games"]:
        results["avg_moves"] = total_moves / results["games"]

    return results


def round_robin(specs, games_per_side=10):
    table = {spec.name: 0.0 for spec in specs}
    detailed_results = []

    for idx in range(len(specs)):
        for jdx in range(idx + 1, len(specs)):
            spec_a = specs[idx]
            spec_b = specs[jdx]
            result = play_match(spec_a, spec_b, games_per_side=games_per_side)
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
                f"draws={result['draws']}, avg_moves={result['avg_moves']:.1f}"
            )

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


if __name__ == "__main__":
    specs = default_mcts_specs()
    summary = round_robin(specs, games_per_side=4)

    print("\nRanking:")
    for position, (name, score) in enumerate(summary["ranking"], start=1):
        print(f"{position}. {name}: {score:.1f}")
