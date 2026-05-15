import csv
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

from bitboard import BitBoard, BitGameState
from mcts import MCTS


def piece_at(board, column, row):
    if hasattr(board, "piece_at"):
        return board.piece_at(column, row)

    pieces = board.columns[column].pieces
    if row < len(pieces):
        return pieces[-1 - row]
    return None


def state_to_dict(state):
    board = state.board
    linha = {}
    for c in range(board.COLUMNS):
        for r in range(6):
            nome_coluna = f"c{c}_r{r}"
            piece = piece_at(board, c, r)
            linha[nome_coluna] = piece if piece is not None else "Vazio"

    linha["player_to_move"] = state.player_to_move
    linha["repetition_count"] = state.repetition_count()
    linha["draw_legal"] = state.draw_legal()
    return linha


def _simular_jogo(ia):
    """Generate training rows for one self-play game."""
    estado = BitGameState(BitBoard(), player_to_move="X")
    dados_jogo = []

    while not estado.is_terminal():
        melhor_jogada = ia.search(estado)
        if melhor_jogada is None:
            break

        linha_dataset = state_to_dict(estado)
        linha_dataset["classe_jogada"] = f"{melhor_jogada.kind}_{melhor_jogada.column}"
        dados_jogo.append(linha_dataset)
        estado = estado.apply_move(melhor_jogada)

    return dados_jogo


def _gerar_lote(num_jogos, iteracoes_mcts):
    """
    Each worker owns its own MCTS instance.
    That keeps process communication small and lets workers run independently.
    """
    ia = MCTS(iterations=iteracoes_mcts)
    dados_totais = []

    for _ in range(num_jogos):
        dados_totais.extend(_simular_jogo(ia))

    return dados_totais


def _dividir_trabalho(num_jogos, num_workers):
    base = num_jogos // num_workers
    resto = num_jogos % num_workers
    return [base + (1 if idx < resto else 0) for idx in range(num_workers) if base + (1 if idx < resto else 0) > 0]


def gerar_jogos(num_jogos=10, iteracoes_mcts=100, num_workers=None):
    print(f"A gerar {num_jogos} jogos para o dataset...")

    if num_jogos <= 0:
        return []

    if num_workers is None:
        num_workers = os.cpu_count() or 1

    # There is no benefit in starting more processes than games.
    num_workers = max(1, min(num_workers, num_jogos))
    lotes = _dividir_trabalho(num_jogos, num_workers)

    if num_workers == 1:
        print("A usar 1 processo.")
        return _gerar_lote(num_jogos, iteracoes_mcts)

    print(f"A usar {num_workers} processos.")

    dados_totais = []

    # Split the games across processes because each game is independent.
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(_gerar_lote, jogos_no_lote, iteracoes_mcts): jogos_no_lote
            for jogos_no_lote in lotes
        }

        jogos_concluidos = 0
        for future in as_completed(futures):
            jogos_no_lote = futures[future]
            dados_totais.extend(future.result())
            jogos_concluidos += jogos_no_lote
            print(f"Lote concluido: {jogos_concluidos}/{num_jogos} jogos.")

    return dados_totais


def guardar_csv(dados, nome_ficheiro="popout_dataset_v4.csv"):
    if not dados:
        print("Nao ha dados para guardar.")
        return

    cabecalho = list(dados[0].keys())

    with open(nome_ficheiro, mode="w", newline="") as ficheiro:
        escritor = csv.DictWriter(ficheiro, fieldnames=cabecalho)
        escritor.writeheader()
        escritor.writerows(dados)

    print(f"Dataset guardado com sucesso em '{nome_ficheiro}' com {len(dados)} linhas!")


if __name__ == "__main__":
    dataset = gerar_jogos(num_jogos=750, iteracoes_mcts=1000, num_workers=os.cpu_count() or 1)
    guardar_csv(dataset)
