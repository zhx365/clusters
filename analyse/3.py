import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import DBSCAN

# 假定这是CSV数据的正确路径
csv_file_path = r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv"

# 读取数据
data = pd.read_csv(csv_file_path)
features1 = data.iloc[:, [1, 2, 3, 4]].values

# 应用PCA
pca = PCA(n_components=2)
X = pca.fit_transform(features1)

# 参数范围
eps_values = np.linspace(0.01, 0.12, 20)
min_samples_values = np.arange(2, 20)
best_silhouette_score = -1
best_davies_bouldin_score = np.inf
best_calinski_harabasz_score = -1
best_params_silhouette = None
best_params_db = None
best_params_ch = None
silhouette_avg = None
# 参数敏感度分析
for eps in eps_values:
    for min_samples in min_samples_values:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = clustering.labels_
        # 忽略噪声点的聚类结果
        if len(set(labels)) > 1 and -1 not in labels:
            silhouette_avg = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski_harabasz = calinski_harabasz_score(X, labels)
            # 更新最佳分数和参数
            if silhouette_avg > best_silhouette_score:
                best_silhouette_score = silhouette_avg
                best_params_silhouette = (eps, min_samples)
            if davies_bouldin < best_davies_bouldin_score:
                best_davies_bouldin_score = davies_bouldin
                best_params_db = (eps, min_samples)
            if calinski_harabasz > best_calinski_harabasz_score:
                best_calinski_harabasz_score = calinski_harabasz
                best_params_ch = (eps, min_samples)

# 绘制图形并保存
plt.figure(figsize=(10, 8))
plt.scatter(eps_values, min_samples_values, c=[silhouette_avg], cmap='viridis')
plt.colorbar()
plt.title('Silhouette Score')
plt.xlabel('eps')
plt.ylabel('min_samples')
plt.scatter(best_params_silhouette[0], best_params_silhouette[1], c='red', label='Best Parameters')
plt.legend()
plt.savefig('/mnt/data/silhouette_score.png')

plt.figure(figsize=(10, 8))
plt.scatter(eps_values, min_samples_values, c=[davies_bouldin], cmap='viridis')
plt.colorbar()
plt.title('Davies-Bouldin Index')
plt.xlabel('eps')
plt.ylabel('min_samples')
plt.scatter(best_params_db[0], best_params_db[1], c='red', label='Best Parameters')
plt.legend()
plt.savefig('/mnt/data/davies_bouldin_index.png')

plt.figure(figsize=(10, 8))
plt.scatter(eps_values, min_samples_values, c=[calinski_harabasz], cmap='viridis')
plt.colorbar()
plt.title('Calinski-Harabasz Index')
plt.xlabel('eps')
plt.ylabel('min_samples')
plt.scatter(best_params_ch[0], best_params_ch[1], c='red', label='Best Parameters')
plt.legend()
plt.savefig('/mnt/data/calinski_harabasz_index.png')

# 返回保存的图片路径和最优参数
best_params = {
    'silhouette': best_params_silhouette,
    'davies_bouldin': best_params_db,
    'calinski_harabasz': best_params_ch
}

best_params
