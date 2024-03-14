import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# features 是数据集的 numpy 数组
# data = pd.read_csv(r"D:\Documents\数据集\支持向量机(SVM)分类.csv", encoding='GB2312')
# features = data.iloc[:, [1, 2, 3, 4]].to_numpy()


data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
features = data.iloc[:, [1, 2, 3, 4]].to_numpy()

# 定义可能的参数范围
eps_values = np.arange(0.01, 1.0, 0.01)
min_samples_values = range(2, 20)



# 用于记录最好的参数组合和对应的轮廓系数
best_eps = None
best_min_samples = None
best_score = -1

# 网格搜索
for eps in eps_values:
    for min_samples in min_samples_values:
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        # 注意：在这里应该捕获当eps值太小，导致所有点都是噪声的情况
        clusters = dbscan.fit_predict(features)
        if len(set(clusters)) > 1:  # 至少要有一个簇和噪声点
            score = silhouette_score(features, clusters)
            if score > best_score:
                best_eps = eps
                best_min_samples = min_samples
                best_score = score

print(f"Best eps: {best_eps}")
print(f"Best min_samples: {best_min_samples}")
print(f"Best Silhouette Score: {best_score}")
