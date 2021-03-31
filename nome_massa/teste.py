import functools
from time import time

import pandas as pd
from sklearn.datasets import load_iris, load_wine
import numpy as np
import scipy.stats
import sklearn.cluster

from metaAga import generalFuncs, genetic, simulated_annealing, grasp

if __name__ == "__main__":
    run_ag_best_hipP = functools.partial(genetic.run_ag, 10, 0.85, 0.1)
    best_param_genetic = [10, 0.85, 0.1]
    best_param_sa = [350, 500, 0.9]
    methods_params = {"genetic": genetic.run_ag}

    iris = load_iris()
    wine = load_wine()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
    k_iris = [3, 7, 10, 13, 22]
    k_wine = [2, 6, 9, 11, 33]
    dict_datasets = {
        "iris": {"data": iris_df, "k": k_iris},
        #"wine": {"data": wine_df, "k": k_wine},
    }

    name_problema_list = np.array([])
    method_name = np.array([])
    mean_tempo_list = np.array([])
    zscore_list = np.array([])

    for dataset in dict_datasets.keys():
        for k in dict_datasets[dataset]["k"]:
            mean_sse_list = np.array([])
            for method in methods_params.keys():
                method_name = np.append(method_name, method)
                sse_list = np.array([])
                tempo_list = np.array([])
                for _ in range(2):
                    inicio = time()
                    best_state_method = methods_params[method](
                        dict_datasets[dataset]["data"],
                        k,
                        *best_param_genetic,
                    )

                    tempo_list = np.append(tempo_list, time() - inicio)
                    sse_list = np.append(sse_list, generalFuncs.sse_estado([*best_state_method.copy().groupby("C-grupo")]))

                name_problema_list = np.append(name_problema_list, dataset + str(k))
                mean_sse_list = np.append(mean_sse_list, sse_list.mean())
                mean_tempo_list = np.append(mean_tempo_list, tempo_list.mean())
            
            #kminhos
            method_name = np.append(method_name, "kmeans")
            for _ in range(2):
                sse_list = np.array([])
                tempo_list = np.array([])
                state_kmeans = dict_datasets[dataset]["data"].copy()
                inicio = time()

                kminhos = sklearn.cluster.KMeans(n_clusters=k, random_state=0).fit(iris_df)
                state_kmeans['C-grupo'] = kminhos.labels_

                tempo_list = np.append(tempo_list, time() - inicio)
                sse_list = np.append(sse_list, generalFuncs.sse_estado([*state_kmeans.groupby("C-grupo")]))
            
            name_problema_list = np.append(name_problema_list, dataset + str(k))
            mean_tempo_list = np.append(mean_tempo_list, tempo_list.mean())
            mean_sse_list = np.append(mean_sse_list, sse_list.mean())
            zscore_list = np.append(
                zscore_list, scipy.stats.zscore(mean_sse_list)
            )

    df_results = pd.DataFrame(
        data={
            "zscore": zscore_list,
            "tempo_medio": mean_tempo_list,
            "method_name": method_name,
            "problema": name_problema_list,
        }
    )
    breakpoint()
    
