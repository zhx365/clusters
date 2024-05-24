import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")

features1 = data.iloc[:, [1, 2, 3, 4]].to_numpy()
features1 = data.iloc[:, [1, 2, 3, 4, 5, 6, 7]].to_numpy()
tsne = TSNE(n_components=2, random_state=42)
features = tsne.fit_transform(features1)

kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(features)

plt.figure(figsize=(10, 8))
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='viridis')
# plt.colorbar(label='Cluster Label')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE Visualization with K-Means Clustering')
plt.show()