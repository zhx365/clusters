import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
"""
K 距离图
用于判断寻找最合适的eps
通用数据集的
eps = [0.4, 0.5]

车站数据集,(这个数据集是经过Min-Max标准化的)
eps = [0.04, 0.05]
"""
# eps = 0.5
# min_sample = 5
# data = pd.read_csv(r"D:\Documents\数据集\支持向量机(SVM)分类.csv", encoding='GB2312')
# features = data.iloc[:, [1, 2, 3, 4]].to_numpy()

eps, min_sample = 0.03, 18
data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
features1 = data.iloc[:, [1, 2, 3, 4]].to_numpy()

pca = PCA(n_components=2)
features = pca.fit_transform(features1)

neigh = NearestNeighbors(n_neighbors=min_sample)
nbrs = neigh.fit(features)
distances, indices = nbrs.kneighbors(features)

# 对距离进行排序并绘制
distances = np.sort(distances, axis=0)
distances = distances[:,1]
k = 5
diff = distances[k:] - distances[:-k]
# diff = np.diff(distances, 2)
knee_point_index = np.where(diff == np.max(diff))[0][0]
eps = distances[knee_point_index]

plt.plot(distances)
plt.title('K-Distance Graph')
plt.xlabel('Points sorted by distance to {}-th nearest neighbor'.format(min_sample))
plt.ylabel('{}-th nearest neighbor distance'.format(min_sample))
plt.axhline(y=eps, color='r', linestyle='--')
plt.show()
