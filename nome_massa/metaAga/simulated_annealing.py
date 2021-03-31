from math import exp
import random
from operator import itemgetter
from time import time

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

from . import generalFuncs


def step(estado_atual, temp, k):
    factor = exp(- len(estado_atual)/temp)
    kuga = []
    sse_list = np.array([])
    sse_total = 0
    # for _ in range(10):
    potential_state = estado_atual.copy()
    m_lines = np.random.choice(potential_state.index, 10, replace=False)
    teste = (potential_state.iloc[m_lines]['C-grupo']).copy()
    pior_grp = generalFuncs.pior_grupo([*potential_state.copy().groupby('C-grupo')])
    pior_ponto = generalFuncs.calcula_pior_ponto(potential_state.loc[potential_state['C-grupo'] == pior_grp])
    ponto = np.random.choice(range(k))
    potential_state.at[pior_ponto, 'C-grupo'] = ponto
    #return potential_state
    for idx, _ in enumerate(teste):
        teste[idx] = np.random.choice(range(k))
    potential_state.loc[m_lines, "C-grupo"] = teste
    #     sse_total, sse_list = generalFuncs.append_sse(sse_total, potential_state.copy(), sse_list)
    #     kuga.append(potential_state)

    # sse_list = (1 - sse_list/sse_total)/(len(sse_list) - 1)
    # idx = np.random.choice(range(len(sse_list)), p=sse_list)
    # best_state = kuga[idx]
    return potential_state


def run_sa(dataset, k, nIter, temp, a):
    state = generalFuncs.random_state(dataset, k)
    sse = generalFuncs.sse_estado(*[state.copy().groupby('C-grupo')])
    best_estado = (state, sse)
    estado_comparacao = state
    sse_comparacao = sse
    inicio = time()
    for i in range(5):
        #print("Temp desceu")
        for _ in range(nIter):
            if (time() - inicio >= 1):
                return best_estado[0]
            state_candidato = step(estado_comparacao, temp, k)
            sse_candidato = generalFuncs.sse_estado(*[state_candidato.copy().groupby('C-grupo')])

            delta = sse_candidato - sse_comparacao
            if(delta < 0 or random.uniform(0, 1) < exp(-delta/(temp+1))):
                #print("aceitei")
                estado_comparacao, sse_comparacao = state_candidato, sse_candidato

            if(sse_candidato < best_estado[1]):
                best_estado = (state_candidato, sse_candidato)
                #print(f"Estamos na iteracao {_} e o melhor estado tem sse {sse_candidato}")
        temp = temp * a


if __name__ == '__main__':
    k = 10
    nIter = 350
    temp = 500
    a = 0.7
    iris = load_iris()
    iris_df = (pd.DataFrame(data=iris.data, columns=iris.feature_names))
    run_sa(iris_df, k, nIter, temp, a)
