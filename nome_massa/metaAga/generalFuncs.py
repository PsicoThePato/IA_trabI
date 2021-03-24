from typing import List, Tuple

import numpy as np
import pandas as pd


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
        df['centroide'] = centroid
    return sse
