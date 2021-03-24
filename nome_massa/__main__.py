import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from metaAga import generalFuncs, genetic

class Ponto:
    def __init__(self, components, identif):
        self.components = components
        self.identif = identif

    def __repr__(self):
        return f"<id: {self.identif}, componentes: {self.components}>"


def calcula_euclidiana(linha: pd.Series, centroid: np.array):
    return (linha - centroid)**2


if __name__ == '__main__':

    populacao = 10
    k = 5
    cross_ratio = 0.8
    m = 1.0
    iris = load_iris()
    iris_df = (pd.DataFrame(data=iris.data, columns=iris.feature_names))

    estados = []
    sse_list = np.array([])
    sse_total = 0
    for _ in range(populacao):
        state = generalFuncs.random_state(iris_df, k)
        estados.append(state.copy())
        sse_total, sse_list = generalFuncs.append_sse(
            sse_total, state, sse_list
            )

    print(sse_list)
    sse_list = (1 - sse_list/sse_total)/(populacao - 1)
    print(sse_list)
    fittados = []
    for _ in range(200):
        nova_gen, sse_total, sse_list = genetic.pega_nova_geracao(
            populacao, sse_total, sse_list,
            cross_ratio, k, estados, m
            )
        print(f"Mais fittado: {sse_list.min()}")
        sse_list = (1 - sse_list/sse_total)/(populacao - 1)
