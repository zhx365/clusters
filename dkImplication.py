import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# eps, min_sample = 0.4, 9
# data = pd.read_csv(r"D:\Documents\数据集\支持向量机(SVM)分类.csv", encoding='GB2312')
eps, min_sample = 0.03, 9
data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")

features1 = data.iloc[:, [1, 2, 3, 4]].to_numpy()
features = PCA(n_components=2).fit_transform(features1)

# eps, min_sample = 2, 10
# features1 = data.iloc[:, [1, 2, 3, 4, 5, 6, 7]].to_numpy()
# tsne = TSNE(n_components=2, random_state=42)
# features = tsne.fit_transform(features1)

def euclidean_distance(point1, point2):
    """计算两点的欧式距离"""
    return np.sqrt(np.sum((point1 - point2) ** 2))


def find_neighbors(points, point_idx, eps):
    """寻找核心点周围的邻居点 返回的是索引"""
    neighbors = []
    for i, point in enumerate(points):
        if euclidean_distance(points[point_idx], point) < eps:
            neighbors.append(i)
    return neighbors


def dbscan(points, eps, min_samples):
    """DBSCAN实现"""
    # 初始化所有的标签为-1
    labels = [-1] * len(points)
    # labels = -np.ones(len(points))
    cluster_id = 0
    # 如果某一个点被标记为噪声，则跳过
    for point_idx in range(len(points)):
        if labels[point_idx] != -1:
            continue

        # 寻找核心点数据
        neighbors = find_neighbors(points, point_idx, eps)

        # 标记为噪声
        if len(neighbors) < min_samples:
            labels[point_idx] = -2
            continue

        # 创建一个新的簇
        labels[point_idx] = cluster_id
        seeds = set(neighbors)
        seeds.discard(point_idx)

        while seeds:
            current_point = seeds.pop()
            if labels[current_point] == -2:
                labels[current_point] = cluster_id
            if labels[current_point] != -1:
                continue

            labels[current_point] = cluster_id
            current_neighbors = find_neighbors(points, current_point, eps)
            if len(current_neighbors) >= min_samples:
                seeds.update(current_neighbors)

        # 准备下一个簇
        cluster_id += 1
    return labels


dbscan_labels = dbscan(features, eps, min_sample)
dbscan_label_set = set(dbscan_labels)
print('DBSCAN:', dbscan_labels)
print('DBSCAN中有几个标签', dbscan_label_set)


def srcd_dbscan(points, eps, min_sample):
    labels = dbscan(points, eps, min_sample)

    # 分析每个簇，找到密度最高的区域(简化版)
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(points[idx])

    # 假设选择所有簇的中心作为K-means初始化
    init_centers = []
    for label, cluster_points in clusters.items():
        # 忽略噪声点
        if label == -2:
            continue
        # 计算每一列的平均值
        cluster_center = np.mean(cluster_points, axis=0)
        init_centers.append(cluster_center)

    return labels, init_centers


srcd_dbscan_labels, init_centers = srcd_dbscan(features, eps, min_sample)
print('初始中心点点', init_centers)
print('srcd_dbscan的labels', srcd_dbscan_labels)


def kmeans(points, init_centers, max_iterations=100, tolerance=0.0001):
    """简单的k-means聚类实现"""
    # 初始化
    centers = np.array(init_centers)
    # 创建一个与centers相同的数组，用于存储迭代前一个过程的中心点，以便检查中心点是否发生变化
    prev_centers = np.zeros(centers.shape)
    # 用于存储每个数据点的簇标签，会被赋值相应的索引
    labels = np.zeros((len(points),))
    # 用于存储每个数据点到中心点的距离
    distances = np.zeros((len(points), len(centers)))

    # 当前中心与前一次迭代的中心偏差
    center_shift = np.linalg.norm(centers - prev_centers, axis=1)

    iteration = 0

    while max(center_shift) > tolerance and iteration < max_iterations:
        # 计算每个点到各中心的距离
        for i in range(len(centers)):
            distances[:, i] = np.linalg.norm(points - centers[i], axis=1)

        # 将每个点分配给最近的中心
        labels = np.argmin(distances, axis=1)

        prev_centers = np.copy(centers)

        # 更新每个簇的中心点
        for i in range(len(centers)):
            points_in_cluster = points[labels == i]
            if points_in_cluster.any():
                centers[i] = np.mean(points_in_cluster, axis=0)

        center_shift = np.linalg.norm(centers - prev_centers, axis=1)

    return labels, centers


# 结合SRCD-DBSCAN和k-means
def srcd_dbscan_kmeans(points, eps, min_sample, max_iterations=100, tolerance=0.001):
    _, init_centers = srcd_dbscan(points, eps, min_sample)

    labels, centers = kmeans(points, init_centers, max_iterations, tolerance)

    return labels, centers


srcd_dbscan_kmeans_labels, centers = srcd_dbscan_kmeans(features, eps, min_sample)
print('最终:', srcd_dbscan_kmeans_labels)
print('中心:', centers)


# 可视化
def plot_clusters(points, labels, centers=None, title='Cluster Visualization'):
    plt.figure(figsize=(10, 6))
    unique_label = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_label)))

    for k, col in zip(unique_label, colors):
        if k == -2:
            # 使用黑色表示噪声
            col = 'k'

        class_member_mask = (labels == k)

        xy = points[class_member_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=8)

    if centers is not None:
        plt.scatter(centers[:, 0], centers[:, 1], s=300, c='k', alpha=0.6, edgecolor='red', linewidth=2)
        for i, center in enumerate(centers):
            plt.text(center[0], center[1], str(i), color='red', fontsize=14)

    plt.title(title)
    plt.show()


# # 这里labels是DBSCAN的输出
# plot_clusters(features, np.array(dbscan_labels), title='SRCD_DBSCAN Cluster Result')

# # 这里init_centers是srcd_dbscan的输出之一
# plot_clusters(features, np.array(srcd_dbscan_labels), np.array(init_centers), title='Initial Centers for K-means')

# # 这里labels和centers是srcd_dbscan_kmeans的输出
plot_clusters(features, srcd_dbscan_kmeans_labels, np.array(centers), title='Final SWO_SRCD_DBSCAN_K-means Clustering Result')
