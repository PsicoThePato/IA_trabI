from math import exp
import random

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

import generalFuncs


def step(estado_atual, temp):
    factor = exp(- len(estado_atual)/temp)
    m_lines = np.random.choice(
        estado_atual.index,
        (int(len(estado_atual) * factor) + 60),
        replace=True
    )
    teste = estado_atual.iloc[m_lines]['C-grupo']
    np.random.shuffle(teste.values)
    estado_atual.loc[m_lines, "C-grupo"] = teste
    return estado_atual


if __name__ == '__main__':
    k = 10
    nIter = 350
    temp = 500
    a = 0.95
    iris = load_iris()
    iris_df = (pd.DataFrame(data=iris.data, columns=iris.feature_names))
    state = generalFuncs.random_state(iris_df, k)
    sse = generalFuncs.sse_estado(*[state.copy().groupby('C-grupo')])
    best_estado = (state, sse)
    estado_comparacao = state
    sse_comparacao = sse

    for _ in range(nIter):
        state_candidato = step(state, temp)
        sse_candidato = generalFuncs.sse_estado(*[state_candidato.copy().groupby('C-grupo')])
        if(sse_candidato < best_estado[1]):
            best_estado = (state_candidato, sse_candidato)
            print(f"Estamos na iteracao {_} e o melhor estado tem sse {sse_candidato}")
        delta = sse_candidato - sse_comparacao

        if(delta < 0 or random.uniform(0, 1) < exp(-delta/(temp+1))):
            estado_comparacao, sse_comparacao = state_candidato, sse_candidato
            temp = temp * a
