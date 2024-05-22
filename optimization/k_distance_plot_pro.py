import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import pandas as pd


eps, min_sample = 0.05, 2
data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
features1 = data.iloc[:, [1, 2, 3, 4]].to_numpy()

pca = PCA(n_components=2)
features = pca.fit_transform(features1)

# 使用PCA降维后的数据
neigh = NearestNeighbors()
nbrs = neigh.fit(features)
distances, indices = nbrs.kneighbors(features)

# 对距离进行排序
distances = np.sort(distances, axis=0)
distances = distances[:,1]

plt.figure(figsize=(12, 8))

# 不同min_samples值对应的曲线
for min_sample in [3, 5, 9, 18]:
    neigh = NearestNeighbors(n_neighbors=min_sample)
    nbrs = neigh.fit(features)
    distances, indices = nbrs.kneighbors(features)

    # 对距离进行排序
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]  # 选取每个点到其最近邻的距离

    plt.plot(distances, label=f'MinPts={min_sample}')

plt.title('K-Distance Graph for Various MinPts')
plt.xlabel('Points sorted by their nearest neighbor distance')
plt.ylabel('Distance to the MinPts-th Nearest Neighbor')
plt.axhline(y=eps, color='r', linestyle='--')
plt.legend()
plt.show()
