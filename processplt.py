from dkImplication import *


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def jitter(points, jitter_amount=0.001):
    return points + np.random.normal(0, jitter_amount, points.shape)

def plot_clusters(points, labels, centers=None, title='Cluster Visualization'):
    plt.figure(figsize=(12, 8), dpi=150)
    unique_label = set(labels)
    colors = plt.cm.get_cmap('tab20', len(unique_label))
    markers = ['o', 's', 'D', '^', 'v', '>', '<', 'p', 'H', '*', 'X']

    jittered_points = jitter(points)  # 对数据点进行抖动处理

    for k in unique_label:
        if k == -2:
            col = 'k'
            marker = 'x'
        else:
            col = colors(k)
            marker = markers[k % len(markers)]

        class_member_mask = (labels == k)
        xy = jittered_points[class_member_mask]
        plt.scatter(xy[:, 0], xy[:, 1], c=[col], marker=marker, label=f'Cluster {k+1}' if k != -2 else 'Noise', 
                    edgecolor='k', s=50)  

    if centers is not None:
        jittered_centers = jitter(centers)  # 对中心点也进行抖动处理
        plt.scatter(jittered_centers[:, 0], jittered_centers[:, 1], s=300, c='k', alpha=0.6, edgecolor='red', linewidth=2)
        for i, center in enumerate(jittered_centers):
            plt.text(center[0], center[1], str(i), color='red', fontsize=14)

    plt.title(title)
    plt.legend(loc='best', markerscale=1.5)
    plt.grid(True)
    plt.show()


# # 这里labels和centers是srcd_dbscan_kmeans的输出
plot_clusters(features, srcd_dbscan_kmeans_labels, np.array(centers), title='Final SWO_SRCD_DBSCAN_K-means Clustering Result')
