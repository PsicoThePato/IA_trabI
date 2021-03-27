from typing import List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt


def random_state(dataset: pd.DataFrame, k: int) -> pd.DataFrame:
    grupos = np.append(
        np.random.randint(0, k, size=len(dataset) - k),
        range(k)
        )
    np.random.shuffle(grupos)
    dataset['C-grupo'] = grupos
    return dataset


def sse_estado(listGroups: List[Tuple[int, pd.DataFrame]]):
    sse = 0
    for _, df in listGroups:
        centroid = (df.iloc[:, 0:-1].sum(axis=0))/len(df)
        sse += ((df.iloc[:, 0:-1] - centroid) ** 2).sum(axis=1).sum(axis=0)
    return sse


def append_sse(sse_total, estado, sse_list_return):
    list_by_group = [*estado.groupby('C-grupo')]
    sse = sse_estado(list_by_group)
    sse_total += sse
    return sse_total, np.append(sse_list_return, sse)


def calcula_pior_ponto(grupo: pd.DataFrame):
    centroid = (grupo.iloc[:, 0:-1].sum(axis=0))/len(grupo)
    dist_pontos = ((grupo.iloc[:, 0:-1] - centroid) ** 2).sum(axis=1)
    pior_ponto = dist_pontos.idxmax()
    return pior_ponto


def pior_grupo(listGroups):
    #fazer no calcula sse
    pior_sse = 0
    pior_grp = -1
    for _, df in listGroups:
        centroid = (df.iloc[:, 0:-1].sum(axis=0))/len(df)
        sse = ((df.iloc[:, 0:-1] - centroid) ** 2).sum(axis=1).sum(axis=0)
        if(sse > pior_sse):
            pior_sse = sse
            pior_grp = _
    return pior_grp


def faz_matriz_distancias(dataset):
    dist_m = pd.DataFrame(distance_matrix(dataset.values, dataset.values), index=dataset.index, columns=dataset.index)
    return dist_m


def cij_calc(linha, centroid_dict, wi):
    dj = linha.loc[centroid_dict.keys()].min()
    cij = linha.iloc[wi]
    #print("#####")
    #print(dj)
    #print(cij)
    #print("#####")
    return max(dj - cij, 0)


if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    iris_df = (pd.DataFrame(data=iris.data, columns=iris.feature_names))
    # estado = random_state(iris_df, 5)
    # pior_grp = pior_grupo([*estado.copy().groupby('C-grupo')])
    # print(pior_grp)
    # #print(estado.loc[estado['C-grupo'] == pior_grp])
    # pior_ponto = calcula_pior_ponto(estado.loc[estado['C-grupo'] == pior_grp])
    # print(estado)
    # print(estado[pior_ponto])
    k = 5
    dist_m = faz_matriz_distancias(iris_df)
    primeiro_centroid = dist_m.sum(axis=1).idxmin()
    centroids_dict = {}
    centroids_dict[primeiro_centroid] = 0
    dist_m.drop(primeiro_centroid, inplace=True)
    for iteracao in range(1, k):
        centroid_candidates_points = np.array([])
        centroid_candidates_idx = np.array([])
        total_sum = 0
        for idx, _ in dist_m.iterrows():
            x = dist_m.apply(cij_calc, axis=1, args=(centroids_dict, idx))
            suma = x.sum()
            total_sum += suma
            centroid_candidates_points = np.append(centroid_candidates_points, suma)
            centroid_candidates_idx = np.append(centroid_candidates_idx, idx)
            print(f"MEU IDX eh {idx}")
            #print(f"To na iteração {iteracao, idx}")
        centroid_candidates_points = centroid_candidates_points/total_sum
        #print(centroid_candidates_idx)
        novo_centroid = np.random.choice(centroid_candidates_idx, p=centroid_candidates_points)
        centroids_dict[novo_centroid] = iteracao
        print(f"novo_centroid é {novo_centroid}")
        dist_m.drop(novo_centroid, inplace=True)
        print(len(dist_m))
    points_centroids = dist_m.loc[:, centroids_dict.keys()].idxmin(axis=1)
    points_centroids = points_centroids.apply(lambda x: centroids_dict[x])
    points_centroids = points_centroids.append(pd.Series(centroids_dict.values(), index=centroids_dict.keys()))
    iris_df["C-grupo"] = points_centroids
    threedeeplt = plt.figure().gca(projection="3d")
    ax = threedeeplt.scatter(
        iris_df["sepal length (cm)"],
        iris_df["sepal width (cm)"],
        iris_df["petal length (cm)"],
        c=iris_df["C-grupo"],
    )
    cb = plt.colorbar(ax)
    plt.show()

    breakpoint()
