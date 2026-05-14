from board import Board, GameState, Move
from mcts import MCTS
import time

def print_board(state):
    print("\n" + "="*20)
    print("      POPOUT")
    print("="*20)
    
    # Criar uma representação visual limpa do tabuleiro (7x6)
    grid = [['-' for _ in range(7)] for _ in range(6)]
    for c in range(7):
        pecas = state.board.columns[c].pieces
        # Colocamos as peças de baixo (índice 0 da grelha é o topo no terminal, por isso invertemos a lógica visual)
        for r in range(len(pecas)):
            grid[5 - r][c] = pecas[-(r + 1)] # As peças mais recentes ficam no topo
            
    for row in grid:
        print(" ".join(row))
    print("0 1 2 3 4 5 6") # Números das colunas para facilitar
    print("="*20)

def get_human_move(state):
    valid_moves = state.legal_moves()
    while True:
        try:
            choice = input(f"Vez do {state.player_to_move}. Jogada (ex: 'drop 3' ou 'pop 3'): ").strip().lower()
            parts = choice.split()
            
            if len(parts) != 2:
                print("❌ Formato inválido. Usa 'drop [coluna]' ou 'pop [coluna]'.")
                continue
            
            kind = parts[0]
            col = int(parts[1])
            move = Move(kind, col)
            
            # Verificar se a jogada gerada existe na lista de jogadas legais
            if any(m.kind == move.kind and m.column == move.column for m in valid_moves):
                return move
            else:
                print("🚫 Jogada ilegal! Verifica se a coluna está cheia ou se tens peças tuas na base para fazer pop.")
        except ValueError:
            print("❌ A coluna deve ser um número inteiro entre 0 e 6.")

def play_game(mode):
    # Inicializar o estado do jogo
    state = GameState(Board(), player_to_move='X')
    
    # Inicializar as Inteligências Artificiais
    # (Podes ajustar o número de iterações. 1000 é forte, mas pode demorar uns segundos)
    ia_mcts_x = MCTS(iterations=1000) 
    ia_mcts_o = MCTS(iterations=1000) 
    
    # Para o modo PC vs PC, poderias usar aqui a tua Árvore de Decisão ID3 para um dos lados
    
    while True:
        print_board(state)
        
        # 1. Verificar Condições de Fim de Jogo (Usando a lógica de correção que discutimos)
        winner = state.get_winner() 
        if winner:
            print(f"\n🏆 O JOGADOR {winner} VENCEU!")
            break
        if state.draw_legal():
            print("\n🤝 O JOGO TERMINOU EMPATADO!")
            break
            
        # 2. Decidir de quem é a vez e como escolhe a jogada
        if state.player_to_move == 'X':
            if mode in ['1', '2']: # Humano joga
                move = get_human_move(state)
            else: # Computador joga
                print("🤖 Computador (X) a pensar via MCTS...")
                start_time = time.time()
                move = ia_mcts_x.search(state)
                print(f"Demorou {round(time.time() - start_time, 2)}s")
        else: # Vez do 'O'
            if mode == '1': # Humano joga
                move = get_human_move(state)
            else: # Computador joga
                print("🤖 Computador (O) a pensar...")
                start_time = time.time()
                # Aqui o ideal será eventualmente substituires o MCTS pelo teu modelo ID3
                # se o objetivo for testar as duas IAs uma contra a outra.
                move = ia_mcts_o.search(state)
                print(f"Demorou {round(time.time() - start_time, 2)}s")
        
        # 3. Aplicar a jogada no tabuleiro
        print(f"\n>>> Jogada efetuada: {move.kind.upper()} na coluna {move.column}")
        state = state.apply_move(move)

if __name__ == "__main__":
    print("="*30)
    print(" BEM-VINDO AO POPOUT (IA 25/26)")
    print("="*30)
    print("1 - Humano (X) vs Humano (O)")
    print("2 - Humano (X) vs Computador MCTS (O)")
    print("3 - Computador MCTS (X) vs Computador MCTS (O)")
    print("="*30)
    
    while True:
        modo = input("Escolhe o modo de jogo (1/2/3): ")
        if modo in ['1', '2', '3']:
            play_game(modo)
            break
        print("Opção inválida.")