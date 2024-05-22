import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# 数据集
data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
features1 = data.iloc[:, [1, 2, 3, 4]].to_numpy()
pca = PCA(n_components=2)
X = pca.fit_transform(features1)


# 参数范围
eps_values = np.linspace(0.01, 2, 50)
min_samples_values = range(2, 20)
scores = []

# 参数敏感度分析
for eps in eps_values:
    for min_samples in min_samples_values:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = clustering.labels_
        # 忽略噪声点的聚类结果
        if len(set(labels)) > 1 and -1 in labels:
            silhouette_avg = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            scores.append((eps, min_samples, silhouette_avg, davies_bouldin, calinski_harabasz))
        else:
            scores.append((eps, min_samples, np.nan, np.nan, np.nan))

# 转换为Numpy数组方便绘图
scores_np = np.array(scores)

# 绘制图形
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# 轮廓系数
sc = axes[0].scatter(scores_np[:,0], scores_np[:,1], c=scores_np[:,2], cmap='viridis')
plt.colorbar(sc, ax=axes[0])
axes[0].set_title('Silhouette Score')
axes[0].set_xlabel('eps')
axes[0].set_ylabel('min_samples')

# 戴维森堡丁指数
db = axes[1].scatter(scores_np[:,0], scores_np[:,1], c=scores_np[:,3], cmap='viridis')
plt.colorbar(db, ax=axes[1])
axes[1].set_title('Davies-Bouldin Index')
axes[1].set_xlabel('eps')
axes[1].set_ylabel('min_samples')

# Calinski-Harabasz指数
ch = axes[2].scatter(scores_np[:,0], scores_np[:,1], c=scores_np[:,4], cmap='viridis')
plt.colorbar(ch, ax=axes[2])
axes[2].set_title('Calinski-Harabasz Index')
axes[2].set_xlabel('eps')
axes[2].set_ylabel('min_samples')
plt.tight_layout()
plt.savefig(r"C:\Users\努力学习呀\Desktop\plt\pca\image.png", format='png', dpi=300)
plt.show()
