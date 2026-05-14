import pandas as pd
import math
from collections import Counter
import pprint

# ==========================================
# 1. FUNÇÕES DO ALGORITMO ID3 (Matemática Pura)
# ==========================================

def entropia(dados, atributo_alvo):
    frequencias = Counter([linha[atributo_alvo] for linha in dados])
    total = len(dados)
    ent = 0.0
    for contagem in frequencias.values():
        prob = contagem / total
        ent -= prob * math.log2(prob)
    return ent

def ganho_informacao(dados, atributo_divisao, atributo_alvo):
    entropia_total = entropia(dados, atributo_alvo)
    
    # Agrupar os dados pelos valores possíveis do atributo
    valores_atributo = set(linha[atributo_divisao] for linha in dados)
    
    entropia_ponderada = 0.0
    for valor in valores_atributo:
        subconjunto = [linha for linha in dados if linha[atributo_divisao] == valor]
        prob_subconjunto = len(subconjunto) / len(dados)
        entropia_ponderada += prob_subconjunto * entropia(subconjunto, atributo_alvo)
        
    return entropia_total - entropia_ponderada

def treinar_id3(dados, atributos, atributo_alvo):
    valores_alvo = [linha[atributo_alvo] for linha in dados]
    
    # Condição de paragem 1: Se todas as flores forem da mesma espécie
    if len(set(valores_alvo)) == 1:
        return valores_alvo[0]
        
    # Condição de paragem 2: Se não houver mais atributos para analisar
    if not atributos:
        return Counter(valores_alvo).most_common(1)[0][0] # Retorna a mais comum
        
    # Escolher o melhor atributo usando o Ganho de Informação
    ganhos = [(ganho_informacao(dados, attr, atributo_alvo), attr) for attr in atributos]
    melhor_ganho, melhor_atributo = max(ganhos)
    
    # Inicializar o nó da árvore com o melhor atributo
    arvore = {melhor_atributo: {}}
    
    # Atributos restantes para os próximos nós
    atributos_restantes = [a for a in atributos if a != melhor_atributo]
    valores_possiveis = set(linha[melhor_atributo] for linha in dados)
    
    # Dividir e continuar recursivamente
    for valor in valores_possiveis:
        subconjunto = [linha for linha in dados if linha[melhor_atributo] == valor]
        
        if not subconjunto:
            arvore[melhor_atributo][valor] = Counter(valores_alvo).most_common(1)[0][0]
        else:
            arvore[melhor_atributo][valor] = treinar_id3(subconjunto, atributos_restantes, atributo_alvo)
            
    return arvore

def classificar(arvore, exemplo):
    # Se não for um dicionário, chegámos à folha (nome da classe)
    if not isinstance(arvore, dict):
        return arvore
        
    atributo_raiz = list(arvore.keys())[0]
    valor_exemplo = exemplo.get(atributo_raiz)
    
    sub_arvore = arvore[atributo_raiz].get(valor_exemplo)
    
    if sub_arvore is None:
        return "Desconhecido" # Caso o valor nunca tenha sido visto no treino
        
    return classificar(sub_arvore, exemplo)

# (Mantém toda a Secção 1 do teu código: entropia, ganho_informacao, treinar_id3, classificar)

# ==========================================
# 2. PRÉ-PROCESSAMENTO DOS DADOS PARA POPOUT
# ==========================================

def preparar_dados_popout(caminho_csv):
    """
    Carrega o dataset gerado pelo MCTS.
    Como os dados do tabuleiro ('X', 'O', 'Vazio') já são categóricos,
    não precisamos de usar o qcut para discretizar.
    """
    try:
        # Carregar os dados
        df = pd.read_csv(caminho_csv)
        
        # O Pandas é excelente para ler, mas para o nosso ID3 recursivo,
        # uma lista de dicionários é muito mais eficiente e fácil de iterar.
        return df.to_dict(orient='records')
    except FileNotFoundError:
        print(f"Erro: O ficheiro {caminho_csv} não foi encontrado.")
        return []

# ==========================================
# 3. EXECUTAR O SCRIPT COM O DATASET POPOUT
# ==========================================
if __name__ == "__main__":
    ficheiro_dataset = 'popout_dataset.csv'
    print(f"A preparar os dados do ficheiro '{ficheiro_dataset}'...")
    
    dados_treino = preparar_dados_popout(ficheiro_dataset)
    
    if not dados_treino:
        print("Por favor, corre primeiro o 'generate_dataset.py' para criares os dados.")
    else:
        # 1. Definir o alvo: queremos prever a jogada que o MCTS escolheu
        alvo = 'classe_jogada'
        
        # 2. Definir os atributos dinamicamente:
        # São todas as colunas do dataset, exceto a coluna alvo (as 42 casas do tabuleiro)
        atributos = [chave for chave in dados_treino[0].keys() if chave != alvo]
        
        print("\nA treinar a Árvore de Decisão ID3...")
        arvore_gerada = treinar_id3(dados_treino, atributos, alvo)
        
        print("\n=== ÁRVORE DE DECISÃO GERADA ===")
        # Como o tabuleiro tem 42 posições, a árvore pode ficar gigante.
        # Usamos depth=3 para imprimir apenas os primeiros níveis e não encher a consola toda.
        pprint.pprint(arvore_gerada, width=80, depth=3)
        
        print("\n=== TESTAR COM O ESTADO INICIAL ===")
        # Vamos criar um tabuleiro completamente vazio para testar o que a árvore prevê
        tabuleiro_vazio = {}
        for c in range(7):
            for r in range(6):
                tabuleiro_vazio[f"c{c}_r{r}"] = 'Vazio'
                
        previsao = classificar(arvore_gerada, tabuleiro_vazio)
        print(f"Com o tabuleiro vazio, a Árvore prevê a jogada: {previsao}")