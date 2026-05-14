import math
import pprint
from collections import Counter
import json

import pandas as pd

from board import Board, GameState
from generate_dataset import state_to_dict


def entropia(dados, atributo_alvo):
    frequencias = Counter(linha[atributo_alvo] for linha in dados)
    total = len(dados)
    ent = 0.0
    for contagem in frequencias.values():
        prob = contagem / total
        ent -= prob * math.log2(prob)
    return ent


def ganho_informacao(dados, atributo_divisao, atributo_alvo):
    entropia_total = entropia(dados, atributo_alvo)
    valores_atributo = set(linha[atributo_divisao] for linha in dados)

    entropia_ponderada = 0.0
    for valor in valores_atributo:
        subconjunto = [linha for linha in dados if linha[atributo_divisao] == valor]
        prob_subconjunto = len(subconjunto) / len(dados)
        entropia_ponderada += prob_subconjunto * entropia(subconjunto, atributo_alvo)

    return entropia_total - entropia_ponderada


def treinar_id3(dados, atributos, atributo_alvo):
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


def classificar(arvore, exemplo):
    if not isinstance(arvore, dict):
        return arvore

    atributo_raiz = list(arvore.keys())[0]
    valor_exemplo = exemplo.get(atributo_raiz)
    sub_arvore = arvore[atributo_raiz].get(valor_exemplo)

    if sub_arvore is None:
        return "Desconhecido"

    return classificar(sub_arvore, exemplo)


def preparar_dados_popout(caminho_csv):
    try:
        df = pd.read_csv(caminho_csv)
        return df.to_dict(orient="records")
    except FileNotFoundError:
        print(f"Erro: o ficheiro {caminho_csv} nao foi encontrado.")
        return []


def construir_exemplo_estado(state):
    return state_to_dict(state)


if __name__ == "__main__":
    ficheiro_dataset = "popout_dataset.csv"
    print(f"A preparar os dados do ficheiro '{ficheiro_dataset}'...")

    dados_treino = preparar_dados_popout(ficheiro_dataset)

    if not dados_treino:
        print("Corre primeiro o 'generate_dataset.py' para criar os dados.")
    else:
        alvo = "classe_jogada"
        atributos = [chave for chave in dados_treino[0].keys() if chave != alvo]

        print("\nA treinar a Arvore de Decisao ID3...")
        arvore_gerada = treinar_id3(dados_treino, atributos, alvo)

        with open('arvore_id3_v2.json', 'w') as f:
            json.dump(arvore_gerada, f, indent=4)
            
        print("\n=== ARVORE DE DECISAO GERADA ===")
        pprint.pprint(arvore_gerada, width=80, depth=3)

        print("\n=== TESTAR COM O ESTADO INICIAL ===")
        estado_inicial = GameState(Board(), player_to_move="X")
        exemplo_inicial = construir_exemplo_estado(estado_inicial)
        previsao = classificar(arvore_gerada, exemplo_inicial)
        print(f"Com o tabuleiro vazio, a Arvore preve a jogada: {previsao}")
