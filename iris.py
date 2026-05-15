import json
import random
from collections import defaultdict

import pandas as pd

from id3_popout import classificar, treinar_id3


TARGET_COLUMN = "class"
ID_COLUMN = "ID"
NUM_BINS = 4
NUM_FOLDS = 5
RANDOM_SEED = 42
TREE_FILE_TEMPLATE = "arvore_iris_fold_{fold}.json"


def carregar_iris(caminho_csv):
    df = pd.read_csv(caminho_csv)
    if ID_COLUMN in df.columns:
        df = df.drop(columns=[ID_COLUMN])
    return df


def criar_folds_estratificados(df, alvo, k, seed):
    """
    Keep class balance roughly stable in every fold.
    Iris is balanced, but stratification avoids accidental skew.
    """
    rng = random.Random(seed)
    folds = [[] for _ in range(k)]

    for classe, grupo in df.groupby(alvo):
        indices = list(grupo.index)
        rng.shuffle(indices)
        for posicao, indice in enumerate(indices):
            folds[posicao % k].append(indice)

    return folds


def discretizar_fold(train_df, test_df, feature_cols, num_bins):
    """
    ID3 in this project expects categorical values, so numeric Iris features
    are discretized using train-fold quantile bins and then applied to test.
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    binning_info = {}

    for coluna in feature_cols:
        # Build bins only on the training fold to avoid data leakage.
        _, bins = pd.qcut(
            train_df[coluna],
            q=num_bins,
            retbins=True,
            duplicates="drop",
        )

        bins = sorted(set(float(valor) for valor in bins))

        # If a column collapses to a single value, keep it as one category.
        if len(bins) <= 1:
            categoria = f"{coluna}_single"
            train_df[coluna] = categoria
            test_df[coluna] = categoria
            continue

        bins[0] = float("-inf")
        bins[-1] = float("inf")
        labels = [f"{coluna}_bin_{idx}" for idx in range(len(bins) - 1)]

        train_df[coluna] = pd.cut(
            train_df[coluna],
            bins=bins,
            labels=labels,
            include_lowest=True,
        ).astype(str)
        test_df[coluna] = pd.cut(
            test_df[coluna],
            bins=bins,
            labels=labels,
            include_lowest=True,
        ).astype(str)
        binning_info[coluna] = labels

    return train_df, test_df, binning_info


def avaliar_fold(train_df, test_df, target_col):
    atributos = [coluna for coluna in train_df.columns if coluna != target_col]
    dados_treino = train_df.to_dict(orient="records")
    dados_teste = test_df.to_dict(orient="records")

    arvore = treinar_id3(dados_treino, atributos, target_col)

    corretos = 0
    desconhecidos = 0
    previsoes = []

    for exemplo in dados_teste:
        previsao = classificar(arvore, exemplo)
        previsoes.append(previsao)
        if previsao == "Desconhecido":
            desconhecidos += 1
        if previsao == exemplo[target_col]:
            corretos += 1

    total = len(dados_teste)
    accuracy = corretos / total if total else 0.0
    unknown_rate = desconhecidos / total if total else 0.0

    return {
        "accuracy": accuracy,
        "unknown_rate": unknown_rate,
        "tree": arvore,
        "predictions": previsoes,
    }


def cross_validate_iris(caminho_csv="iris.csv", k=NUM_FOLDS, num_bins=NUM_BINS, seed=RANDOM_SEED):
    df = carregar_iris(caminho_csv)
    folds = criar_folds_estratificados(df, TARGET_COLUMN, k, seed)
    feature_cols = [coluna for coluna in df.columns if coluna != TARGET_COLUMN]

    resultados = []

    for fold_idx, test_indices in enumerate(folds, start=1):
        test_df = df.loc[test_indices].reset_index(drop=True)
        train_df = df.drop(index=test_indices).reset_index(drop=True)

        train_df, test_df, _ = discretizar_fold(train_df, test_df, feature_cols, num_bins)
        resultado = avaliar_fold(train_df, test_df, TARGET_COLUMN)

        tree_path = TREE_FILE_TEMPLATE.format(fold=fold_idx)
        with open(tree_path, "w", encoding="utf-8") as f:
            json.dump(resultado["tree"], f, indent=4)
        resultado["tree_path"] = tree_path
        resultados.append(resultado)

        print(
            f"Fold {fold_idx}/{k}: "
            f"accuracy={resultado['accuracy']:.3f}, "
            f"unknown_rate={resultado['unknown_rate']:.3f}, "
            f"tree={tree_path}"
        )

    media_accuracy = sum(r["accuracy"] for r in resultados) / len(resultados)
    media_unknown = sum(r["unknown_rate"] for r in resultados) / len(resultados)

    print("\nResumo:")
    print(f"Accuracy media: {media_accuracy:.3f}")
    print(f"Unknown rate medio: {media_unknown:.3f}")

    return resultados


if __name__ == "__main__":
    cross_validate_iris()
