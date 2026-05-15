import json
import pprint

import pandas as pd

from board import Board, GameState
from generate_dataset import state_to_dict
from id3_popout import (
    classificar,
    construir_exemplo_estado,
    treinar_id3,
)

FICHEIRO_DATASET = "tour_craz_2.csv"
FICHEIRO_ARVORE = "arvore_tour_craz_2.json"
ALVO = "classe_jogada"

# Metadata from tournament generation — not available when predicting from board state alone.
COLUNAS_EXCLUIDAS = {
    ALVO,
    "agent_name",
    "tournament_name",
    "match_name",
    "game_id",
    "ply",
    "move_time",
}


def preparar_dados_torneio(caminho_csv):
    try:
        df = pd.read_csv(caminho_csv)
        return df.to_dict(orient="records")
    except FileNotFoundError:
        print(f"Erro: o ficheiro {caminho_csv} nao foi encontrado.")
        return []


def atributos_treino(dados):
    return [chave for chave in dados[0].keys() if chave not in COLUNAS_EXCLUIDAS]


if __name__ == "__main__":
    print(f"A preparar os dados do ficheiro '{FICHEIRO_DATASET}'...")

    dados_treino = preparar_dados_torneio(FICHEIRO_DATASET)

    if not dados_treino:
        print("Corre primeiro o 'generate_dataset_tournament.py' para criar os dados.")
    else:
        atributos = atributos_treino(dados_treino)

        print(f"\nA treinar a Arvore de Decisao ID3 ({len(dados_treino)} exemplos)...")
        arvore_gerada = treinar_id3(dados_treino, atributos, ALVO)

        with open(FICHEIRO_ARVORE, "w") as f:
            json.dump(arvore_gerada, f, indent=4)

        print(f"\nArvore guardada em '{FICHEIRO_ARVORE}'.")
        print("\n=== ARVORE DE DECISAO GERADA ===")
        pprint.pprint(arvore_gerada, width=80, depth=3)

        print("\n=== TESTAR COM O ESTADO INICIAL ===")
        estado_inicial = GameState(Board(), player_to_move="X")
        exemplo_inicial = construir_exemplo_estado(estado_inicial)
        previsao = classificar(arvore_gerada, exemplo_inicial)
        print(f"Com o tabuleiro vazio, a Arvore preve a jogada: {previsao}")
