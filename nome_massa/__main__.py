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
    # print(linha)
    # print(centroid)
    return (linha - centroid)**2


if __name__ == '__main__':

    populacao = 10
    k = 5
    cross_ratio = 0.75
    m = 1.0
    iris = load_iris()
    iris_df = (pd.DataFrame(data=iris.data, columns=iris.feature_names))

    estados = []
    sse_list = np.array([])
    sse_total = 0
    for _ in range(populacao):
        state = generalFuncs.random_state(iris_df, k)
        estados.append(state.copy())

        list_by_group = [*state.groupby('C-grupo')]
        sse = generalFuncs.sse_estado(list_by_group)
        sse_total += sse
        sse_list = np.append(sse_list, sse)
    print(sse_list)
    sse_list = (1 - sse_list/sse_total)/(populacao - 1)
    print(sse_list)
    for _ in range(50):
        nova_gen, sse_total, sse_list = genetic.pega_nova_geracao(
            populacao, sse_total, sse_list,
            cross_ratio, k, estados, m
            )
        print(f"Mais fittado: {sse_list.min()}")
        sse_list = (1 - sse_list/sse_total)/(populacao - 1)
