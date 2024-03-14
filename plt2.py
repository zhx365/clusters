import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dkImplication import dbscan, kmeans, srcd_dbscan
"""
过程可视化
"""

# 已经正确设置了eps和min_sample，以及成功读取了数据集到data变量中
eps, min_sample = 0.5, 9
# features已经根据您的数据集选择了正确的列
data = pd.read_csv(r"D:\Documents\数据集\支持向量机(SVM)分类.csv", encoding='GB2312')
features = data.iloc[:, [1, 2, 3, 4]].to_numpy()


# 保持之前定义的函数euclidean_distance, find_neighbors, dbscan不变

# DBSCAN 聚类
dbscan_labels = dbscan(features, eps, min_sample)

# SRCD-DBSCAN 以及中心点初始化
srcd_dbscan_labels, init_centers = srcd_dbscan(features, eps, min_sample)

# K-means 聚类
kmeans_labels, kmeans_centers = kmeans(features, init_centers)

# 可视化
def plot_clusters_subplot(points, labels_list, centers_list=None, titles=None, plot_shape=(1, 3)):
    """
    修改版的可视化函数，支持在一个窗口中绘制不同步骤的聚类结果。
    """
    plt.figure(figsize=(18, 6))
    for i, (labels, centers) in enumerate(zip(labels_list, centers_list)):
        plt.subplot(plot_shape[0], plot_shape[1], i+1)
        unique_labels = set(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        for k, col in zip(unique_labels, colors):
            if k < 0:  # 噪声点
                col = 'k'
            class_member_mask = (labels == k)
            xy = points[class_member_mask]
            plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=8)
        if centers is not None:
            plt.scatter(centers[:, 0], centers[:, 1], s=300, c='white', alpha=0.6, edgecolor='red', linewidth=2)
            for i, center in enumerate(centers):
                plt.text(center[0], center[1], str(i), color='red', fontsize=14)
        plt.title(titles[i] if titles else f'Step {i+1}')
    plt.show()

# 准备数据以供可视化
labels_list = [np.array(dbscan_labels), np.array(srcd_dbscan_labels), kmeans_labels]
centers_list = [None, np.array(init_centers), kmeans_centers]
titles = ['DBSCAN Clustering Result', 'Initial Centers for K-means', 'Final K-means Clustering Result']

# 绘制子图
plot_clusters_subplot(features, labels_list, centers_list, titles, plot_shape=(1, 3))
