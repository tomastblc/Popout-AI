"""ID3 training and inference helpers used by the final PopOut notebook."""

import json
import math
import pprint
from collections import Counter

import pandas as pd

from libs.bitboard import BitBoard, BitGameState, Move
from generate_dataset_tournament import state_to_dict

ALVO = "classe_jogada"
COLUNAS_EXCLUIDAS = {
    ALVO,
    "agent_name",
    "tournament_name",
    "match_name",
    "game_id",
    "ply",
    "move_time",
    "source",
    "perturbation_steps",
}


def entropia(dados, atributo_alvo):
    """Compute the Shannon entropy of the target attribute in ``dados``."""
    frequencias = Counter(linha[atributo_alvo] for linha in dados)
    total = len(dados)
    ent = 0.0
    for contagem in frequencias.values():
        prob = contagem / total
        ent -= prob * math.log2(prob)
    return ent


def ganho_informacao(dados, atributo_divisao, atributo_alvo):
    """Measure how much a split on one attribute reduces target entropy."""
    entropia_total = entropia(dados, atributo_alvo)
    valores_atributo = set(linha[atributo_divisao] for linha in dados)

    entropia_ponderada = 0.0
    for valor in valores_atributo:
        subconjunto = [linha for linha in dados if linha[atributo_divisao] == valor]
        prob_subconjunto = len(subconjunto) / len(dados)
        entropia_ponderada += prob_subconjunto * entropia(subconjunto, atributo_alvo)

    return entropia_total - entropia_ponderada


def treinar_id3(dados, atributos, atributo_alvo):
    """Train an ID3 decision tree from categorical examples."""
    valores_alvo = [linha[atributo_alvo] for linha in dados]

    if len(set(valores_alvo)) == 1:
        return valores_alvo[0]

    if not atributos:
        return Counter(valores_alvo).most_common(1)[0][0]

    ganhos = [(ganho_informacao(dados, attr, atributo_alvo), attr) for attr in atributos]
    _, melhor_atributo = max(ganhos)

    arvore = {melhor_atributo: {}}
    atributos_restantes = [a for a in atributos if a != melhor_atributo]
    valores_possiveis = set(linha[melhor_atributo] for linha in dados)

    for valor in valores_possiveis:
        subconjunto = [linha for linha in dados if linha[melhor_atributo] == valor]
        if not subconjunto:
            arvore[melhor_atributo][valor] = Counter(valores_alvo).most_common(1)[0][0]
        else:
            arvore[melhor_atributo][valor] = treinar_id3(
                subconjunto,
                atributos_restantes,
                atributo_alvo,
            )

    return arvore


def _valor_chave_arvore_id3(valor):
    """Normalize runtime values so they match JSON string keys in saved trees."""
    if isinstance(valor, bool):
        return "true" if valor else "false"
    if isinstance(valor, int):
        return str(valor)
    if isinstance(valor, float) and valor == int(valor):
        return str(int(valor))
    return valor


def classificar(arvore, exemplo):
    """Classify one example by recursively traversing the trained tree."""
    if not isinstance(arvore, dict):
        return arvore

    atributo_raiz = list(arvore.keys())[0]
    raw = exemplo.get(atributo_raiz)
    norm = _valor_chave_arvore_id3(raw)
    ramos = arvore[atributo_raiz]
    sub_arvore = ramos.get(norm)
    if sub_arvore is None:
        sub_arvore = ramos.get(raw)

    if sub_arvore is None:
        return "Desconhecido"

    return classificar(sub_arvore, exemplo)


def move_from_classe_jogada(label):
    """Convert a dataset label such as ``drop_3`` into a ``Move`` object."""
    if not label or label == "Desconhecido" or not isinstance(label, str):
        return None
    if label.startswith("draw"):
        return Move("draw", None)

    parts = label.split("_", 1)
    if len(parts) != 2:
        return None

    kind, col_s = parts
    try:
        col = int(col_s)
    except ValueError:
        return None
    return Move(kind, col)


def preparar_dados_popout(caminho_csv):
    """Load a PopOut dataset CSV and return it as a list of row dictionaries."""
    try:
        df = pd.read_csv(caminho_csv)
        return df.to_dict(orient="records")
    except FileNotFoundError:
        print(f"Erro: o ficheiro {caminho_csv} nao foi encontrado.")
        return []


def atributos_treino_popout(dados, atributo_alvo=ALVO):
    """List the input feature columns that should be used for ID3 training."""
    colunas_excluidas = set(COLUNAS_EXCLUIDAS)
    colunas_excluidas.add(atributo_alvo)
    return [chave for chave in dados[0].keys() if chave not in colunas_excluidas]


def construir_exemplo_estado(state):
    """Convert a game state into the flat feature dictionary used by ID3."""
    return state_to_dict(state)
