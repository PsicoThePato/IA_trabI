import random

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

from metaAga import generalFuncs, genetic

class Ponto:
    def __init__(self, components, identif):
        self.components = components
        self.identif = identif

    def __repr__(self):
        return f"<id: {self.identif}, componentes: {self.components}>"


def calcula_euclidiana(linha: pd.Series, centroid: np.array):
    return (linha - centroid)**2


if __name__ == '__main__':
    pass
