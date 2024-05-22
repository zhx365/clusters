import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

from dkImplication import srcd_dbscan_kmeans

eps_values = np.linspace(0.01, 2, 200)
min_samples_values = range(2, 20)
scores = []
data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
features1 = data.iloc[:, [1, 2, 3, 4]].to_numpy()
pca = PCA(n_components=2)
features = pca.fit_transform(features1)

labels, centers = srcd_dbscan_kmeans(features, eps_values, min_samples_values)

silhouette_avg = silhouette_score(features, labels)
calinski_harabasz = calinski_harabasz_score(features, labels)
davies_bouldin = davies_bouldin_score(features, labels)
