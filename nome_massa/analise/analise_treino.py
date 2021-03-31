import pandas as pd

data = pd.read_csv("../output/genetic/tabela")

#  Parece que eu gerei uma coluna de lixo sem querer teehee
data.drop(data.columns[0], axis=1, inplace=True)

#  Tabela com rank médio e zscore médio de cada configuração
data['rank'] = data['zscore'].rank(method='min', ascending=True)
mean_rank_table = data.groupby("params").mean()

#  Melhores
best_by_score = mean_rank_table.sort_values("zscore").iloc[0].drop("rank")
best_by_rank = mean_rank_table.sort_values("rank").iloc[0].drop("zscore")

#  Melhores 5
best_5_score = mean_rank_table.sort_values("zscore").drop("rank")[0:5]
best_5_rank = mean_rank_table.sort_values("rank").drop("zscore")[0:5]

