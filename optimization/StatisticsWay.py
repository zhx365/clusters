import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np

"""
这个方法不合适
这个方法适合密度差异较大的数据集
"""

data = pd.read_csv(r"D:\Documents\数据集\支持向量机(SVM)分类.csv", encoding='GB2312')
features = data.iloc[:, [1, 2, 3, 4]].to_numpy()


# data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
# features = data.iloc[:, [1, 2, 3, 4]].to_numpy()

# 计算数据点的两两欧式距离
distances = euclidean_distances(features, features)

# 排除距离为0的情况（即点到其本身的距离）
mean_distances = np.mean(distances + np.eye(distances.shape[0]) * np.max(distances), axis=1)

# 计算所有点的平均距离
overall_avg_distance = np.mean(mean_distances)

# 计算标准差
std_dev = np.std(mean_distances)

# 设定eps为平均距离加上一个或两个标准差（这取决于你想有多少点被划为核心点）
eps = overall_avg_distance + std_dev

print(f"Estimated eps: {eps}")
