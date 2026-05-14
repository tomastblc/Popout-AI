import random
import math
from copy import deepcopy
# Importamos o que é necessário do teu ficheiro original
from board import Move, GameState

class MCTSNode:
    def __init__(self, state: GameState, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        
        # Guardamos quem fez a jogada para chegar a este nó
        # (Se agora é a vez do 'O', quem jogou antes foi o 'X')
        self.player_who_just_moved = 'O' if state.player_to_move == 'X' else 'X'

        self.children = []
        self.visits = 0
        self.wins = 0.0

        # Obtemos as jogadas legais a partir do estado do jogo
        self.untried_moves = self.state.legal_moves()
    
    def is_terminal(self):
        return self.state.is_terminal()
    
    def is_fully_expanded(self):
        return len(self.untried_moves) == 0
    
    def expand(self):
        """Escolhe uma jogada não tentada e cria um novo nó filho."""
        move = self.untried_moves.pop(0)
        new_state = self.state.apply_move(move)
        child = MCTSNode(new_state, parent=self, move=move)
        self.children.append(child)
        return child
    
    def best_child(self, c=1.414):
        """Usa a fórmula UCT para selecionar o melhor filho."""
        def ucb1(child):
            exploit = child.wins / child.visits
            explore = c * math.sqrt(math.log(self.visits) / child.visits)
            return exploit + explore
        
        return max(self.children, key=ucb1)

    def rollout(self):
        """Simulação aleatória até ao fim do jogo."""
        state = deepcopy(self.state)

        while True:
            winner = state.get_winner()
            if winner: return winner
            if state.is_drawn():
                return None 
            
            moves = state.legal_moves()
            if not moves:
                return None
            
            move = random.choice(moves)
            state = state.apply_move(move)
    
    def backpropagate(self, winner):
        """Atualiza as estatísticas subindo na árvore."""
        self.visits += 1
        
        if winner is None:
            self.wins += 0.5 # Empate
        elif winner == self.player_who_just_moved:
            self.wins += 1.0 # Vitória
            
        if self.parent:
            self.parent.backpropagate(winner)

class MCTS:
    def __init__(self, iterations=1000):
        self.iterations = iterations

    def search(self, initial_state: GameState):
        """Executa a pesquisa e devolve a melhor jogada encontrada."""
        root = MCTSNode(initial_state)

        for _ in range(self.iterations):
            node = root

            # 1. Seleção
            while not node.is_terminal() and node.is_fully_expanded():
                if not node.children: break
                node = node.best_child()

            # 2. Expansão
            if not node.is_terminal() and not node.is_fully_expanded():
                node = node.expand()
            
            # 3. Simulação
            winner = node.rollout()
            
            # 4. Retropropagação
            node.backpropagate(winner)

        if not root.children:
            return None
            
        # Retorna a jogada do filho com mais visitas (mais robusto)
        return max(root.children, key=lambda c: c.visits).move

