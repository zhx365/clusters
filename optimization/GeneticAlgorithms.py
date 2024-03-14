import geneticalgorithm as ga
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score  # 例如使用 geneticalgorithm 库


# features 是数据集的 numpy 数组
# data = pd.read_csv(r"D:\Documents\数据集\支持向量机(SVM)分类.csv", encoding='GB2312')
# features = data.iloc[:, [1, 2, 3, 4]].to_numpy()


data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
features = data.iloc[:, [1, 2, 3, 4]].to_numpy()

def fitness_function(solution):
    eps, min_samples = solution
    dbscan = DBSCAN(eps=eps, min_samples=int(min_samples))
    clusters = dbscan.fit_predict(features)
    if len(set(clusters)) > 1:
        score = silhouette_score(features, clusters)
    else:
        score = -1
    return -score  # 由于库寻找的是最小值，我们通过取反来寻找最大轮廓系数

varbound = np.array([[0.1, 1.0],  # eps范围
                     [2, 10]])    # min_samples范围

algorithm_param = {'max_num_iteration': 100,
                   'population_size':10,
                   'mutation_probability':0.1,
                   'elit_ratio': 0.01,
                   'crossover_probability': 0.5,
                   'parents_portion': 0.3,
                   'crossover_type':'uniform',
                   'max_iteration_without_improv':None}

model = ga.geneticalgorithm(function=fitness_function,
                            dimension=2,
                            variable_type='real',
                            variable_boundaries=varbound,
                            algorithm_parameters=algorithm_param)

model.run()
