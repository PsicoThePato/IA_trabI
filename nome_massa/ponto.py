import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from metaAga import generalFuncs
import metaAga.genetic


class Ponto:
    def __init__(self, components, identif):
        self.components = components
        self.identif = identif

    def __repr__(self):
        return f"<id: {self.identif}, componentes: {self.components}>"


def calcula_euclidiana(linha: pd.Series, centroid: np.array):
    print(linha)
    print(centroid)
    return (linha - centroid)**2


populacao = 10
k = 5
iris = load_iris()
iris_df = (pd.DataFrame(data=iris.data, columns=iris.feature_names))
print(iris_df.head())

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

sse_list = (1 - sse_list/sse_total)/(populacao - 1)
choice = np.random.choice(range(populacao), 2, p=sse_list, replace=False)
pai1 = estados[choice[0]]
pai2 = estados[choice[1]]
novo_estado = pai1.copy()
novo_estado[145:] = pai2[145:]
m = 0.1
m_lines = np.random.choice(
    novo_estado.index, int(len(novo_estado)*m), replace=False)
new_groups = np.random.choice(range(k), len(m_lines))
