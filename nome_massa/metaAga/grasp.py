import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

import generalFuncs


@jit(nopython=True)
def cij_calc(linha, centroid_dict, wi):
    dj = linha.loc[centroid_dict.keys()].min()
    cij = linha.iloc[wi]
    return max(dj - cij, 0)


@jit(nopython=True)
def kaufman_centroid(linha, centroids_dict, dist_m):
    x = dist_m.apply(cij_calc, axis=1, args=(centroids_dict, linha.name))
    suma = x.sum()
    return (suma, linha.name)


@jit(nopython=True)
def local_search(estado, k):
    comp_estado = estado.copy()
    pior_grp = generalFuncs.pior_grupo([*comp_estado.copy().groupby('C-grupo')])
    pior_ponto = generalFuncs.calcula_pior_ponto(comp_estado.loc[comp_estado['C-grupo'] == pior_grp])
    ponto = np.random.choice(range(k))
    comp_estado.at[pior_ponto, 'C-grupo'] = ponto
    if(generalFuncs.sse_estado([*comp_estado.copy().groupby('C-grupo')]) < generalFuncs.sse_estado([*estado.copy().groupby('C-grupo')])):
        return comp_estado
    return estado


@jit(nopython=True)
def initial_greedy_state(iris_df, k):
    dist_m = generalFuncs.faz_matriz_distancias(iris_df)
    primeiro_centroid = dist_m.sum(axis=1).idxmin()
    centroids_dict = {}
    centroids_dict[primeiro_centroid] = 0
    dist_m.drop(primeiro_centroid, inplace=True)
    for iteracao in range(1, k):
        result = dist_m.apply(kaufman_centroid, args=(centroids_dict, dist_m))
        novo_centroid = np.random.choice(result.loc[1, :], p=result.loc[0, :]/result.loc[0,:].sum())
        centroids_dict[novo_centroid] = iteracao
        dist_m.drop(novo_centroid, inplace=True)
        #print(iteracao)
    points_centroids = dist_m.loc[:, centroids_dict.keys()].idxmin(axis=1)
    points_centroids = points_centroids.apply(lambda x: centroids_dict[x])
    points_centroids = points_centroids.append(pd.Series(centroids_dict.values(), index=centroids_dict.keys()))
    iris_df["C-grupo"] = points_centroids
    return iris_df


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_df = (pd.DataFrame(data=iris.data, columns=iris.feature_names))
    breakpoint()
    k = 5

    for _ in range(10):
        iris_df = initial_greedy_state(iris_df, k)
        if(_ == 0):
            print(generalFuncs.sse_estado([*iris_df.copy().groupby('C-grupo')]))
        iris_df = local_search(iris_df, k)
    print(generalFuncs.sse_estado([*iris_df.copy().groupby('C-grupo')]))
    threedeeplt = plt.figure().gca(projection="3d")
    ax = threedeeplt.scatter(
        iris_df["sepal length (cm)"],
        iris_df["sepal width (cm)"],
        iris_df["petal length (cm)"],
        c=iris_df["C-grupo"],
    )
    cb = plt.colorbar(ax)
    plt.show()