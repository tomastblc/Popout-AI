import json
import pprint

import pandas as pd

from board import Board, GameState
from id3_popout import (
    classificar,
    construir_exemplo_estado,
    treinar_id3,
)

# ATENÇÃO: Confirma se os nomes destes ficheiros correspondem aos que geraste!
FICHEIRO_DATASET = "tour_craz_2_perturbed.csv" 
FICHEIRO_ARVORE = "arvore_tour_craz_2_perturbed.json"
ALVO = "classe_jogada"

# Escolhe qual o agente que vai "ensinar" a árvore. 
# Deve ser um dos que definiste no SELECTED_SPEC_NAMES do gerador.
AGENTE_ALVO = "UCT_low_c_10k" 

COLUNAS_EXCLUIDAS = {
    ALVO,
    "agent_name",
    "source",
    "perturbation_steps",
}

def preparar_dados_perturbados(caminho_csv, agente_alvo=AGENTE_ALVO):
    try:
        df = pd.read_csv(caminho_csv)
        
        # Filtra apenas as jogadas sugeridas pelo nosso agente de elite
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
    """Cria uma assinatura única para o tabuleiro para podermos encontrar repetidos."""
    celulas = tuple(linha[f"c{col}_r{row}"] for col in range(7) for row in range(6))
    draw = linha["draw_legal"]
    if isinstance(draw, str):
        draw = draw.strip().lower() == "true"
    rep = linha["repetition_count"]
    if isinstance(rep, float) and rep == int(rep):
        rep = int(rep)
    return celulas + (linha["player_to_move"], rep, draw)

def deduplicar_por_tabuleiro(dados):
    """
    Como injetámos caos, podemos ter o mesmo tabuleiro gerado várias vezes, 
    por vezes com respostas diferentes. A árvore base não lida bem com contradições.
    Esta função junta-os e escolhe a jogada mais frequente para esse tabuleiro.
    """
    grupos = {}
    for linha in dados:
        grupos.setdefault(assinatura_tabuleiro(linha), []).append(linha)

    unicos = []
    for grupo in grupos.values():
        exemplo = dict(grupo[0])
        # Qual foi a jogada mais recomendada para este tabuleiro exato?
        rotulos = [g[ALVO] for g in grupo]
        exemplo[ALVO] = max(set(rotulos), key=rotulos.count)
        unicos.append(exemplo)

    print(f"Deduplicacao por tabuleiro: {len(dados)} -> {len(unicos)} exemplos "
          f"({len(dados) - len(unicos)} repetidos removidos para evitar contradicoes).")
    return unicos

if __name__ == "__main__":
    print(f"A preparar os dados do ficheiro '{FICHEIRO_DATASET}'...")

    dados = preparar_dados_perturbados(FICHEIRO_DATASET)

    if not dados:
        print("Erro: Nao ha dados. Corre primeiro o 'generate_dataset_perturbed_tournament.py'.")
    else:
        # 1. Limpar os dados antes de treinar a árvore base
        dados_limpos = deduplicar_por_tabuleiro(dados)
        atributos = atributos_treino(dados_limpos)

        print(f"\nA treinar a Arvore de Decisao ID3 com o algoritmo base ({len(dados_limpos)} exemplos unicos)...")
        
        # 2. Usar a função treinar_id3 original sem parâmetros extra
        arvore_gerada = treinar_id3(dados_limpos, atributos, ALVO)

        with open(FICHEIRO_ARVORE, "w", encoding="utf-8") as f:
            json.dump(arvore_gerada, f, indent=4)

        print(f"\nArvore guardada com sucesso em '{FICHEIRO_ARVORE}'.")
        print("\n=== ARVORE DE DECISAO GERADA ===")
        pprint.pprint(arvore_gerada, width=80, depth=3)

        print("\n=== TESTAR COM O ESTADO INICIAL ===")
        estado_inicial = GameState(Board(), player_to_move="X")
        exemplo_inicial = construir_exemplo_estado(estado_inicial)
        previsao = classificar(arvore_gerada, exemplo_inicial)
        print(f"Com o tabuleiro vazio, a Arvore preve a jogada: {previsao}")
