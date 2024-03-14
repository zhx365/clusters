import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dkImplication import dbscan, kmeans, srcd_dbscan

eps, min_sample = 0.05, 5
data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
features = data.iloc[:, [1, 2, 3, 4]].to_numpy()

def op_plt(points, labels):
    plt.figure(figsize=(10, 6))
    unique_label = set(labels)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_label)))
    for k, col in zip(unique_label, colors):
        if k == -2:
            # 使用黑色表示噪声
            col = 'k'
        
        class_member_mask = (labels == k)

        xy = points[class_member_mask]
        plt.scatter(xy[:,0], xy[:, 1], c=col, marker='o', edgecolor='k', s=80)

    plt.title('Basic Scatter Plot')
    plt.show()


labels = dbscan(features, eps, min_sample)
print('Original Data:', features[:5])
print('标签', labels)
op_plt(features, labels)