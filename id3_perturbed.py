import json
import pprint
import random

import pandas as pd

from board import Board, GameState
from id3_popout import (
    avaliar_arvore,
    classificar,
    construir_exemplo_estado,
    estatisticas_arvore,
    podar_arvore,
    treinar_id3,
)

FICHEIRO_DATASET = "tour_craz_1_perturbed.csv"
FICHEIRO_ARVORE = "arvore_tour_craz_1_perturbed.json"
ALVO = "classe_jogada"
AGENTE_ALVO = "UCT_heuristic_rollout_10k"
RANDOM_SEED = 42
FRACAO_VALIDACAO = 0.2

# Limita overfitting: folhas maiores, arvore mais rasa, corta splits fracos.
MIN_SAMPLES_LEAF = 6
MAX_DEPTH = 12
MIN_GAIN = 0.005

COLUNAS_EXCLUIDAS = {
    ALVO,
    "agent_name",
    "source",
    "perturbation_steps",
}


def preparar_dados_perturbados(caminho_csv, agente_alvo=AGENTE_ALVO):
    try:
        df = pd.read_csv(caminho_csv)

        if "agent_name" in df.columns:
            df = df[df["agent_name"] == agente_alvo]
            print(f"Dataset filtrado: a usar {len(df)} exemplos do agente '{agente_alvo}'.")

        return df.to_dict(orient="records")
    except FileNotFoundError:
        print(f"Erro: o ficheiro {caminho_csv} nao foi encontrado.")
        return []


def atributos_treino(dados):
    return [chave for chave in dados[0].keys() if chave not in COLUNAS_EXCLUIDAS]


def assinatura_tabuleiro(linha):
    celulas = tuple(linha[f"c{col}_r{row}"] for col in range(7) for row in range(6))
    draw = linha["draw_legal"]
    if isinstance(draw, str):
        draw = draw.strip().lower() == "true"
    rep = linha["repetition_count"]
    if isinstance(rep, float) and rep == int(rep):
        rep = int(rep)
    return celulas + (linha["player_to_move"], rep, draw)


def deduplicar_por_tabuleiro(dados):
    grupos = {}
    for linha in dados:
        grupos.setdefault(assinatura_tabuleiro(linha), []).append(linha)

    unicos = []
    for grupo in grupos.values():
        exemplo = dict(grupo[0])
        rotulos = [g[ALVO] for g in grupo]
        exemplo[ALVO] = max(set(rotulos), key=rotulos.count)
        unicos.append(exemplo)

    print(
        f"Deduplicacao por tabuleiro: {len(dados)} -> {len(unicos)} exemplos "
        f"({len(dados) - len(unicos)} repetidos removidos)."
    )
    return unicos


def dividir_treino_validacao(dados, fracao_val=FRACAO_VALIDACAO, seed=RANDOM_SEED):
    por_tabuleiro = {}
    for linha in dados:
        por_tabuleiro.setdefault(assinatura_tabuleiro(linha), []).append(linha)

    chaves = list(por_tabuleiro.keys())
    rng = random.Random(seed)
    rng.shuffle(chaves)

    n_val = max(1, int(len(chaves) * fracao_val))
    val_keys = set(chaves[:n_val])

    treino, validacao = [], []
    for chave, linhas in por_tabuleiro.items():
        destino = validacao if chave in val_keys else treino
        destino.extend(linhas)

    print(
        f"Split por tabuleiro: {len(treino)} treino, {len(validacao)} validacao "
        f"({len(val_keys)} tabuleiros em validacao)."
    )
    return treino, validacao


def _imprimir_metricas(rotulo, arvore, dados):
    acc, certos, desconhecidos, errados = avaliar_arvore(arvore, dados, ALVO)
    print(
        f"  {rotulo}: {certos}/{len(dados)} ({acc * 100:.1f}%) "
        f"| desconhecidos={desconhecidos} errados={errados}"
    )


def treinar_arvore_perturbada(dados):
    dados = deduplicar_por_tabuleiro(dados)
    treino, validacao = dividir_treino_validacao(dados)
    atributos = atributos_treino(treino)

    print(
        f"\nA treinar ID3 (min_leaf={MIN_SAMPLES_LEAF}, "
        f"max_depth={MAX_DEPTH}, min_gain={MIN_GAIN})..."
    )
    arvore = treinar_id3(
        treino,
        atributos,
        ALVO,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_depth=MAX_DEPTH,
        min_gain=MIN_GAIN,
    )

    nos, prof = estatisticas_arvore(arvore)
    print(f"  Antes da poda: {nos} nos, profundidade {prof}")
    _imprimir_metricas("Treino", arvore, treino)
    _imprimir_metricas("Validacao", arvore, validacao)

    arvore = podar_arvore(arvore, validacao, ALVO)
    nos, prof = estatisticas_arvore(arvore)
    print(f"\nApos poda com validacao: {nos} nos, profundidade {prof}")
    _imprimir_metricas("Treino", arvore, treino)
    _imprimir_metricas("Validacao", arvore, validacao)

    return arvore


if __name__ == "__main__":
    print(f"A preparar os dados do ficheiro '{FICHEIRO_DATASET}'...")

    dados = preparar_dados_perturbados(FICHEIRO_DATASET)

    if not dados:
        print(
            "Corre primeiro o 'generate_dataset_perturbed_tournament.py' "
            "para criar os dados."
        )
    else:
        arvore_gerada = treinar_arvore_perturbada(dados)

        with open(FICHEIRO_ARVORE, "w", encoding="utf-8") as f:
            json.dump(arvore_gerada, f, indent=4)

        print(f"\nArvore guardada em '{FICHEIRO_ARVORE}'.")
        print("\n=== ARVORE DE DECISAO GERADA ===")
        pprint.pprint(arvore_gerada, width=80, depth=3)

        print("\n=== TESTAR COM O ESTADO INICIAL ===")
        estado_inicial = GameState(Board(), player_to_move="X")
        exemplo_inicial = construir_exemplo_estado(estado_inicial)
        previsao = classificar(arvore_gerada, exemplo_inicial)
        print(f"Com o tabuleiro vazio, a Arvore preve a jogada: {previsao}")
