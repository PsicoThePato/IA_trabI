import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numba import jit

# from . import generalFuncs
import generalFuncs


# @jit(nopython=True)
def cij_calc(linha, centroid_dict, wi):
    dj = linha[[*centroid_dict.keys()]].min()
    cij = linha[wi]
    return max(dj - cij, 0)


# @jit(nopython=True)
def kaufman_centroid(linha, centroids_dict, dist_m):
    x = np.apply_along_axis(
        cij_calc, 1, dist_m[:, :-1], centroids_dict, int(linha[-1])
    )
    suma = x.sum()
    return (suma, linha[-1])


# @jit(nopython=True)
def local_search(estado, k):
    comp_estado = estado.copy()
    pior_grp = generalFuncs.pior_grupo([*comp_estado.copy().groupby("C-grupo")])
    pior_ponto = generalFuncs.calcula_pior_ponto(
        comp_estado.loc[comp_estado["C-grupo"] == pior_grp]
    )
    ponto = np.random.choice(range(k))
    comp_estado.at[pior_ponto, "C-grupo"] = ponto
    if generalFuncs.sse_estado(
        [*comp_estado.copy().groupby("C-grupo")]
    ) < generalFuncs.sse_estado([*estado.copy().groupby("C-grupo")]):
        return comp_estado
    return estado


# @jit(nopython=True)
def initial_greedy_state(iris_df, k):
    dist_m = generalFuncs.faz_matriz_distancias(iris_df)
    primeiro_centroid = dist_m.sum(axis=1).idxmin()
    centroids_dict = {}
    centroids_dict[primeiro_centroid] = 0
    dist_m.drop(primeiro_centroid, inplace=True)
    dist_m = dist_m.to_numpy()
    index_points = np.array(range(len(dist_m) + 1))

    dist_m = np.append(dist_m, index_points.reshape(1, 150), 0)
    dist_m = np.append(dist_m, index_points.reshape(150, 1), 1)

    for iteracao in range(1, k):
        result = np.apply_along_axis(
            kaufman_centroid, 1, dist_m[0:-1], centroids_dict, dist_m
        )
        novo_centroid = int(
            np.random.choice(result[:, 1], p=result[:, 0] / result[:, 0].sum())
        )
        centroids_dict[novo_centroid] = iteracao
        dist_m = np.delete(dist_m, novo_centroid, 0)

    points_centroids = np.argmin(dist_m[:-1, [*centroids_dict.keys()]], axis=1)
    points_centroids = np.insert(
        points_centroids, [*centroids_dict.keys()], [*centroids_dict.values()]
    )

    iris_df["C-grupo"] = points_centroids
    return iris_df


def run_grasp(dataset, k, nIter):
    iris_df = initial_greedy_state(dataset, k)
    for _ in range(nIter):
        iris_df = local_search(iris_df, k)
    return iris_df


if __name__ == "__main__":
    from sklearn.datasets import load_iris

    iris = load_iris()
    iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    k = 5
    numIter = 10
    run_grasp(iris_df, k, numIter)
    threedeeplt = plt.figure().gca(projection="3d")
    ax = threedeeplt.scatter(
        iris_df["sepal length (cm)"],
        iris_df["sepal width (cm)"],
        iris_df["petal length (cm)"],
        c=iris_df["C-grupo"],
    )
    cb = plt.colorbar(ax)
    plt.show()
