import csv
from board import Board, GameState
from mcts import MCTS

def state_to_dict(state):
    board = state.board
    linha = {}
    for c in range(board.COLUMNS):
        pecas = list(reversed(board.columns[c].pieces))

        for r in range(6):
            nome_coluna = f"c{c}_r{r}"
            if r < len(pecas):
                linha[nome_coluna] = pecas[r]
            else:
                linha[nome_coluna] = 'Vazio'

    linha["player_to_move"] = state.player_to_move
    linha["repetition_count"] = state.states.count(state.board_key)
    linha["draw_legal"] = state.draw_legal()

    return linha

def gerar_jogos(num_jogos=10, iteracoes_mcts=100):
    print(f"A gerar {num_jogos} jogos para o dataset...")
    dados_totais = []
    
    
    ia = MCTS(iterations=iteracoes_mcts)
    
    for jogo_idx in range(num_jogos):
        print(f"A simular jogo {jogo_idx + 1}/{num_jogos}...")
        
        estado = GameState(Board(), player_to_move='X')
        
        
        while not estado.is_terminal():
            
            melhor_jogada = ia.search(estado)
            
            
            if melhor_jogada is None:
                break
                
            
            linha_dataset = state_to_dict(estado)
            
            
            linha_dataset['classe_jogada'] = f"{melhor_jogada.kind}_{melhor_jogada.column}"
            
            
            dados_totais.append(linha_dataset)
            
            
            estado = estado.apply_move(melhor_jogada)
            
    return dados_totais

def guardar_csv(dados, nome_ficheiro="popout_dataset.csv"):
    if not dados:
        print("Não há dados para guardar.")
        return
        
    
    cabecalho = list(dados[0].keys())
    
    with open(nome_ficheiro, mode='w', newline='') as ficheiro:
        escritor = csv.DictWriter(ficheiro, fieldnames=cabecalho)
        escritor.writeheader()
        escritor.writerows(dados)
        
    print(f"Dataset guardado com sucesso em '{nome_ficheiro}' com {len(dados)} linhas!")

if __name__ == "__main__":
    dataset = gerar_jogos(num_jogos=5, iteracoes_mcts=200)
    guardar_csv(dataset)
