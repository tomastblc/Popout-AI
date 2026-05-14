import csv

from agents import AgentSpec, default_mcts_specs, basic_mcts_specs, up2k_mcts_specs, up10k_mcts_specs, build_agent
from generate_dataset import state_to_dict
from tournament import play_game_with_history


def gerar_dataset_torneio(tournament_specs):
    """
    Run one or more tournament definitions and collect every move into
    a dataset compatible with popout_dataset plus an agent_name column.
    """
    dados_totais = []

    for tournament_idx, tournament_spec in enumerate(tournament_specs, start=1):
        tournament_name = tournament_spec["name"]
        specs = tournament_spec["specs"]
        games_per_side = tournament_spec.get("games_per_side", 1)

        print(
            f"A correr torneio {tournament_idx}/{len(tournament_specs)}: "
            f"{tournament_name} com {len(specs)} agentes."
        )

        dados_totais.extend(
            _gerar_dados_round_robin(
                tournament_name=tournament_name,
                specs=specs,
                games_per_side=games_per_side,
            )
        )

    return dados_totais


def _gerar_dados_round_robin(tournament_name, specs, games_per_side):
    dados = []
    game_id = 0

    for idx in range(len(specs)):
        for jdx in range(idx + 1, len(specs)):
            spec_a = specs[idx]
            spec_b = specs[jdx]

            # Play both color assignments to keep the generated dataset balanced.
            for _ in range(games_per_side):
                game_id += 1
                dados.extend(
                    _gerar_dados_jogo(
                        tournament_name=tournament_name,
                        match_name=f"{spec_a.name}_vs_{spec_b.name}",
                        game_id=game_id,
                        spec_x=spec_a,
                        spec_o=spec_b,
                    )
                )

            for _ in range(games_per_side):
                game_id += 1
                dados.extend(
                    _gerar_dados_jogo(
                        tournament_name=tournament_name,
                        match_name=f"{spec_b.name}_vs_{spec_a.name}",
                        game_id=game_id,
                        spec_x=spec_b,
                        spec_o=spec_a,
                    )
                )

    return dados


def _gerar_dados_jogo(tournament_name, match_name, game_id, spec_x: AgentSpec, spec_o: AgentSpec):
    agent_x = build_agent(spec_x)
    agent_o = build_agent(spec_o)
    resultado = play_game_with_history(agent_x, agent_o, spec_x.name, spec_o.name)

    dados_jogo = []
    for ply_idx, item in enumerate(resultado["history"], start=1):
        linha = state_to_dict(item["state"])
        linha["classe_jogada"] = f"{item['move'].kind}_{item['move'].column}"
        linha["agent_name"] = item["agent_name"]
        linha["tournament_name"] = tournament_name
        linha["match_name"] = match_name
        linha["game_id"] = game_id
        linha["ply"] = ply_idx
        linha["move_time"] = item["move_time"]
        dados_jogo.append(linha)

    return dados_jogo


def guardar_csv(dados, nome_ficheiro="popout_dataset_tournament.csv"):
    if not dados:
        print("Nao ha dados para guardar.")
        return

    cabecalho = list(dados[0].keys())

    with open(nome_ficheiro, mode="w", newline="") as ficheiro:
        escritor = csv.DictWriter(ficheiro, fieldnames=cabecalho)
        escritor.writeheader()
        escritor.writerows(dados)

    print(f"Dataset guardado com sucesso em '{nome_ficheiro}' com {len(dados)} linhas!")


def baseline_tournament_specs():
    return [
        {
            "name": "baseline_tournament",
            "specs": basic_mcts_specs(),
            "games_per_side": 2,
        }
    ]

def advanced_tournament_specs():
    return [
        {
            "name": "advanced_tournament",
            "specs": up2k_mcts_specs(),
            "games_per_side": 2,
        }
    ]

def crazy_tournament_specs():
    return [
        {
            "name": "crazy_tournament",
            "specs": up10k_mcts_specs(),
            "games_per_side": 5
        }
    ]


if __name__ == "__main__":
    tournament_specs = baseline_tournament_specs()
    dataset = gerar_dataset_torneio(tournament_specs)
    guardar_csv(dataset, "tour_base_1.csv")
    tournament_specs = advanced_tournament_specs()
    dataset = gerar_dataset_torneio(tournament_specs)
    guardar_csv(dataset, "tour_adv_1.csv")
    tournament_specs = crazy_tournament_specs()
    dataset = gerar_dataset_torneio(tournament_specs)
    guardar_csv(dataset, "tour_craz_1.csv")
    tournament_specs = crazy_tournament_specs()
    dataset = gerar_dataset_torneio(tournament_specs)
    guardar_csv(dataset, "tour_craz_2.csv")
