import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# 示例数据集
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
# X1 = data.iloc[:, [1, 2, 3, 4]].to_numpy()
#
# pca = PCA(n_components=2)
# X = pca.fit_transform(X1)

# SWO优化算法简化版
def SWO_optimize_DBSCAN(X, num_agents=10, max_iter=100):
    bounds = np.array([[0.01, 2], [2, 20]])  # eps 和 min_samples 的搜索范围
    positions = np.random.uniform(low=bounds[:, 0], high=bounds[:, 1], size=(num_agents, 2))

    best_score = -np.inf
    best_params = None

    for _ in range(max_iter):
        for position in positions:
            eps, min_samples = position
            clustering = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(X)
            labels = clustering.labels_

            # 忽略没有聚类或只有一个聚类的情况
            if len(set(labels)) <= 1 or -1 in labels:
                continue

            silhouette_avg = silhouette_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)

            # 将三个评估指标进行组合作为适应度分数
            # 注意：Davies-Bouldin指数越小越好，所以使用其倒数
            score = silhouette_avg + calinski_harabasz - davies_bouldin

            if score > best_score:
                best_score = score
                best_params = position

        # 简化的位置更新过程，实际应用中需要根据SWO算法细节设计
        positions += np.random.uniform(-0.1, 0.1, positions.shape)
        positions = np.clip(positions, bounds[:, 0], bounds[:, 1])

    return best_params, best_score


best_params, best_score = SWO_optimize_DBSCAN(X)
print(f"Optimized Parameters: eps={best_params[0]:.2f}, min_samples={int(best_params[1])}")
print(f"Best Score: {best_score:.2f}")
