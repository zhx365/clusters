import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from numpy.random import rand, uniform

def bat_algorithm(X, n_bat, n_iter, freq_min, freq_max, A, r, bounds):
    """
    X: 数据集
    n_bat: 蝙蝠数量
    n_iter: 迭代次数
    freq_min, freq_max: 频率最小值和最大值
    A: 响度
    r: 脉冲率
    bounds: 参数的界限，如[(0.1, 1.0), (2, 20)]对应eps和min_samples的范围
    """
    dim = len(bounds)  # 维度
    v = np.zeros((n_bat, dim))  # 速度
    Q = np.zeros(n_bat)  # 频率
    solutions = np.zeros((n_bat, dim))  # 解
    fitness = np.array([float('inf')] * n_bat)  # 适应度
    best = np.min(fitness)  # 最好的适应度
    best_solution = np.zeros(dim)  # 最好的解

    # 初始化解
    for i in range(n_bat):
        solutions[i, :] = [uniform(bounds[d][0], bounds[d][1]) for d in range(dim)]
    
    for t in range(n_iter):
        for i in range(n_bat):
            Q[i] = freq_min + (freq_max - freq_min) * rand()
            v[i, :] += (solutions[i, :] - best_solution) * Q[i]
            new_solution = np.clip(solutions[i, :] + v[i, :], [b[0] for b in bounds], [b[1] for b in bounds])

            if rand() > r:
                new_solution = best_solution + 0.001 * np.random.randn(dim)

            new_solution = np.clip(new_solution, [b[0] for b in bounds], [b[1] for b in bounds])
            clusters = DBSCAN(eps=new_solution[0], min_samples=int(new_solution[1])).fit_predict(X)
            
            # 检查聚类结果，避免单一聚类导致的错误
            if len(set(clusters)) <= 1 or len(set(clusters)) == len(X):
                new_fitness = -1  # 给出一个默认的低适应度值
            else:
                new_fitness = -silhouette_score(X, clusters)

            if (new_fitness < fitness[i]) and (rand() < A):
                solutions[i, :] = new_solution
                fitness[i] = new_fitness

            if new_fitness < best:
                best_solution = new_solution
                best = new_fitness

        # 更新响度和脉冲率
        A *= 0.9
        r = r * (1 - np.exp(-0.9 * t))

    return best_solution

# features 是数据集的 numpy 数组
# data = pd.read_csv(r"D:\Documents\数据集\支持向量机(SVM)分类.csv", encoding='GB2312')
# features = data.iloc[:, [1, 2, 3, 4]].to_numpy()


data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
features = data.iloc[:, [1, 2, 3, 4]].to_numpy()

X = features
n_bat = 50
n_iter = 100
freq_min = 0
freq_max = 1
A = 0.5
r = 0.5
bounds = [(0.01, 1.0), (4, 20)]  # 参数范围

best_solution = bat_algorithm(X, n_bat, n_iter, freq_min, freq_max, A, r, bounds)
print("Optimized eps and min_samples:", best_solution)
