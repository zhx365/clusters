import numpy as np
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# 生成示例数据
# X, _ = make_moons(n_samples=300, noise=0.1, random_state=42)

data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
X1 = data.iloc[:, [1, 2, 3, 4]].to_numpy()
# X1 = data.iloc[:, [5, 6, 7]].to_numpy()

pca = PCA(n_components=2)
X = pca.fit_transform(X1)

tsne = TSNE(n_components=2, random_state=42)

# SWO优化DBSCAN参数的函数
def swo_optimize_dbscan(X, num_agents=10, max_iter=50):
    # 初始化eps和min_samples的搜索范围
    eps_range = (0.01, 1)
    min_samples_range = (2, 10)

    # 随机初始化蜘蛛蜂的位置
    eps_positions = np.random.uniform(*eps_range, num_agents)
    min_samples_positions = np.random.randint(*min_samples_range, num_agents)

    best_score = -1
    best_eps = None
    best_min_samples = None
    calinski_harabasz = None
    davies_bouldin = None

    for _ in range(max_iter):
        for i in range(num_agents):
            # 应用DBSCAN并计算轮廓系数作为适应度
            clustering = DBSCAN(eps=eps_positions[i], min_samples=min_samples_positions[i]).fit(X)
            labels = clustering.labels_
            # 确保找到的聚类数大于1小于样本数，以便计算轮廓系数
            if len(set(labels)) > 1:
                score = silhouette_score(X, labels)
                calinski_harabasz = calinski_harabasz_score(X, labels)
                davies_bouldin = davies_bouldin_score(X, labels)


                # 更新最佳参数
                if score > best_score:
                    best_score = score
                    best_eps = eps_positions[i]
                    best_min_samples = min_samples_positions[i]

            # 更新位置（这里简单地使用随机扰动）
            eps_positions[i] += np.random.uniform(-0.1, 0.1)
            min_samples_positions[i] += np.random.randint(-2, 3)

            # 确保更新后的位置在搜索范围内
            eps_positions[i] = np.clip(eps_positions[i], *eps_range)
            min_samples_positions[i] = np.clip(min_samples_positions[i], *min_samples_range)

    return best_eps, best_min_samples, best_score, calinski_harabasz, davies_bouldin


# 使用SWO优化DBSCAN参数
best_eps, best_min_samples, best_score, calinski_harabasz, davies_bouldin  = swo_optimize_dbscan(X)

# 展示最优参数下的DBSCAN聚类结果
print(
    f"Optimized eps: {best_eps}, "
    f"min_samples: {best_min_samples}, "
    f"silhouette score: {best_score},"
    f"calinski_harabasz: {calinski_harabasz},"
    f"davies_bouldin: {davies_bouldin}"
)
clustering = DBSCAN(eps=best_eps, min_samples=best_min_samples).fit(X)
plt.scatter(X[:, 0], X[:, 1], c=clustering.labels_)
plt.title("DBSCAN Clustering with Optimized Parameters")
plt.show()
