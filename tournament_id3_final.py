import json
import itertools
import time

from bitboard import BitBoard, BitGameState
from generate_dataset import state_to_dict
from id3_popout import classificar, move_from_classe_jogada
from mcts import MCTS

def play_match(name_x, tree_x, name_o, tree_o, fallback_agent):
    """
    Joga uma partida entre duas árvores ID3. 
    Usa um MCTS leve como fallback caso a árvore encontre um estado desconhecido.
    """
    state = BitGameState(BitBoard(), player_to_move="X")
    
    while True:
        winner = state.get_winner()
        if winner:
            return winner
        if state.is_drawn():
            return "Draw"

        if state.player_to_move == "X":
            tree = tree_x
        else:
            tree = tree_o

        # 1. Pedir a jogada à Árvore de Decisão
        label = classificar(tree, state_to_dict(state))
        move = move_from_classe_jogada(label)

        # 2. Verificar se a jogada é legal (ou se a árvore respondeu "Desconhecido")
        legal_moves = state.legal_moves()
        is_legal = False
        if move is not None:
            for m in legal_moves:
                if m.kind == move.kind and m.column == move.column:
                    is_legal = True
                    break

        # 3. Fallback System (Solução para o problema de Generalização do ID3)
        if not is_legal:
            move = fallback_agent.search(state)

        # Aplicar jogada
        state = state.apply_move(move)

if __name__ == "__main__":
    # Dicionário com todas as árvores pedidas
    TREES = {
        "ID3_v5": "arvore_id3_v5.json",
        "Tour_Base": "arvore_tour_base.json",
        "Tour_Adv": "arvore_tour_adv.json",
        "Tour_Craz_1": "arvore_tour_craz_1.json",
        "Tour_Craz_2_Perturbed": "arvore_tour_craz_2_perturbed.json"
    }

    print("A carregar as árvores de decisão para o Torneio...")
    loaded_trees = {}
    for name, path in TREES.items():
        try:
            with open(path, 'r', encoding='utf-8') as f:
                loaded_trees[name] = json.load(f)
            print(f" - {name} carregada com sucesso.")
        except FileNotFoundError:
            print(f" [AVISO] {path} não encontrado no diretório. Será ignorada.")

    # Inicializar o painel de pontuações (Vitória=3pts, Empate=1pt)
    scores = {name: {"wins": 0, "losses": 0, "draws": 0, "points": 0} for name in loaded_trees}
    
    # Sistema Híbrido: Fallback MCTS para lidar com falhas de "Desconhecido"
    print("\nA inicializar MCTS Fallback de Segurança (200 iter)...")
    fallback = MCTS(iterations=200)

    print("\n" + "="*40)
    print("=== O GRANDE TORNEIO DE ÁRVORES ID3 ===")
    print("="*40)
    
    match_count = 1
    # Criar todas as combinações de 2 equipas
    for a, b in itertools.combinations(loaded_trees.keys(), 2):
        print(f"\n--- Ronda: {a} vs {b} ---")
        
        # Jogo 1: A joga com o 'X' (Primeiro), B joga com o 'O' (Segundo)
        print(f" Jogo {match_count}: {a} (X) vs {b} (O)", end=" -> ")
        start_time = time.time()
        res1 = play_match(a, loaded_trees[a], b, loaded_trees[b], fallback)
        dur = round(time.time() - start_time, 2)
        print(f"Vencedor: {res1} ({dur}s)")
        
        if res1 == "X":
            scores[a]["wins"] += 1; scores[a]["points"] += 3
            scores[b]["losses"] += 1
        elif res1 == "O":
            scores[b]["wins"] += 1; scores[b]["points"] += 3
            scores[a]["losses"] += 1
        else:
            scores[a]["draws"] += 1; scores[a]["points"] += 1
            scores[b]["draws"] += 1; scores[b]["points"] += 1
        match_count += 1

        # Jogo 2: B joga com o 'X' (Primeiro), A joga com o 'O' (Segundo)
        print(f" Jogo {match_count}: {b} (X) vs {a} (O)", end=" -> ")
        start_time = time.time()
        res2 = play_match(b, loaded_trees[b], a, loaded_trees[a], fallback)
        dur = round(time.time() - start_time, 2)
        print(f"Vencedor: {res2} ({dur}s)")
        
        if res2 == "X":
            scores[b]["wins"] += 1; scores[b]["points"] += 3
            scores[a]["losses"] += 1
        elif res2 == "O":
            scores[a]["wins"] += 1; scores[a]["points"] += 3
            scores[b]["losses"] += 1
        else:
            scores[b]["draws"] += 1; scores[b]["points"] += 1
            scores[a]["draws"] += 1; scores[a]["points"] += 1
        match_count += 1

    print("\n===========================================")
    print("            CLASSIFICAÇÃO FINAL            ")
    print("===========================================")
    print(f"{'Agente (Árvore)':<23} | Pts | V - E - D")
    print("-" * 43)
    
    # Ordenar por pontos (do maior para o menor)
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]['points'], reverse=True)
    for i, (name, data) in enumerate(sorted_scores, 1):
        print(f"{i}. {name:<20} | {data['points']:>3} | {data['wins']} - {data['draws']} - {data['losses']}")
