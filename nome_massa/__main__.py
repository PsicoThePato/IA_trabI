import random
import itertools
from time import time

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris, load_wine
import scipy.stats

from metaAga import (
    generalFuncs, genetic, simulated_annealing, grasp
)

class Ponto:
    def __init__(self, components, identif):
        self.components = components
        self.identif = identif

    def __repr__(self):
        return f"<id: {self.identif}, componentes: {self.components}>"


def calcula_euclidiana(linha: pd.Series, centroid: np.array):
    return (linha - centroid)**2


if __name__ == '__main__':
    #Base de dados Iris - k = [3, 7, 10, 13, 22]
    #Base de dados Wine - k = [2, 6, 9, 11, 33]

    # To, alfa, numIter
    sa_hipP = [[350, 500], [500, 100, 50], [0.95, 0.85, 0.7]]
    # População, Taxa de crossover, taxa de mutação
    genetic_hipP = [[10, 30, 50], [0.75, 0.85, 0.95], [0.1, 0.2]]
    #genetic_hipP = [[10], [0.75], [0.1]]
    # nIter
    grasp_hipP = [[2, 5, 10, 20, 35, 50]]

    sa_params = [*itertools.product(*sa_hipP)]
    genetic_params = [*itertools.product(*genetic_hipP)]
    grasp_params = [*itertools.product(*grasp_hipP)]
    ks = [5]
    iris = load_iris()
    wine = load_wine()

    iris_df = (pd.DataFrame(data=iris.data, columns=iris.feature_names))
    wine_df = (pd.DataFrame(data=wine.data, columns=wine.feature_names))
    iris_df = iris_df[0:40]

    dict_params = {'sa': sa_params, 'genetic': genetic_params, 'grasp': grasp_params}
    #  debug
    #dict_params = {'genetic': genetic_params}
    dict_func = {'sa': simulated_annealing.run_sa, 'genetic': genetic.run_ag, 'grasp': grasp.run_grasp}
    k_iris = [3, 7, 10, 13, 22]
    k_wine = [2, 6, 9, 11, 33]
    dict_datasets = {'iris': {'data': iris_df, 'k': k_iris}, 'wine': {'data': wine_df, 'k': k_wine} }
    #  debug
    #dict_datasets = {'iris': {'data': iris_df, 'k': k_iris}}
    dict_saida = {}
    for method in dict_params.keys():
        print(f"Examinando o método {method}")
        n_exec = len(dict_params[method]) * len(k_iris)  #+ len(k_wine))
        name_problema = np.array([])
        sse_zscorado = np.array([])
        mean_tempo_list = np.array([])
        for dataset in dict_datasets.keys():
            print(f"Examinando o dataset {dataset}")
            for k in dict_datasets[dataset]['k']:
                print(f"Examinando {k} grupos")
                mean_sse_list = np.array([])
                for params in dict_params[method]:
                    tempo_list = np.array([])
                    sse_list = np.array([])
                    for _ in range(10):
                        tempo = time()
                        state = dict_func[method](dict_datasets[dataset]['data'], k, *params)
                        delta_t = time() - tempo
                        sse_list = np.append(
                            sse_list,
                            generalFuncs.sse_estado(*[state.groupby('C-grupo')])
                            )
                        tempo_list = np.append(tempo_list, delta_t)
                    mean_sse_list = np.append(mean_sse_list, np.mean(sse_list))
                    mean_tempo_list = np.append(mean_tempo_list, np.mean(tempo_list))
                    name_problema = np.append(name_problema, dataset+str(k))
                sse_zscorado = np.append(sse_zscorado, scipy.stats.zscore(mean_sse_list))
        resultado = [*zip(sse_zscorado, dict_params[method] *n_exec, name_problema, mean_tempo_list)]
        resultado.sort(key=lambda x: x[0])
        df = pd.DataFrame(resultado, columns=['zscore', 'params', 'name_problema', 'mean_tempo'])
        df.to_csv('output/'+f'{method}/'+'tabela')
