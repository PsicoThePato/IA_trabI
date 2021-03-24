import numpy as np
from typing import List
import copy

from metaAga import generalFuncs


def pega_nova_geracao(
    populacao: int,
    sse_total: float,
    sse_list: np.array,
    cross_ratio: float,
    k: int,
    estados: List,
    mutation_ratio: float,
):
    #sse_list = (1 - sse_list/sse_total)/(populacao - 1)
    sse_teste = sse_list.copy()
    nova_geracao = []
    sse_list_return = np.array([])
    sse_total = 0
    for _ in range(int(populacao/2)):
        choice = np.random.choice(range(populacao), 2, p=sse_teste, replace=False)
        pai = estados[choice[0]]
        mae = estados[choice[1]]
        novo_estado_paiD = pai.copy()
        novo_estado_maeD = mae.copy()
        novo_estado_paiD[int(cross_ratio * len(novo_estado_paiD)):] = \
            mae[int(cross_ratio * len(novo_estado_paiD)):]
        novo_estado_maeD[int(cross_ratio * len(novo_estado_maeD)):] = \
            pai[int(cross_ratio * len(novo_estado_maeD)):]
        if(np.random.rand() < mutation_ratio):
            m_lines = np.random.choice(
                novo_estado_paiD.index, int(len(novo_estado_paiD) * mutation_ratio),
                replace=False
            )
            mutated_linha = np.random.choice(len(novo_estado_paiD))
            novo_estado_paiD.loc[[mutated_linha], ['C-grupo']] = np.random.choice(k)
            # new_groups = np.random.choice(range(k), len(m_lines))
            # novo_estado_paiD.loc[m_lines, "C-grupo"] = new_groups
        nova_geracao.append(novo_estado_paiD.copy())
        nova_geracao.append(novo_estado_maeD.copy())

        list_by_group = [*novo_estado_paiD.groupby('C-grupo')]
        sse = generalFuncs.sse_estado(list_by_group)
        sse_total += sse
        sse_list_return = np.append(sse_list_return, sse)

        list_by_group = [*novo_estado_maeD.groupby('C-grupo')]
        sse = generalFuncs.sse_estado(list_by_group)
        sse_total += sse
        sse_list_return = np.append(sse_list_return, sse)
    return nova_geracao, sse_total, sse_list_return
