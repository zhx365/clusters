import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import const

from metrics import metrics

data = pd.read_csv(const.path)

features1 = data.iloc[:, [1, 2, 3, 4]].to_numpy()

features = PCA(n_components=2).fit_transform(features1)

kmeans = KMeans(n_clusters=const.K, random_state=42)

kmeans_labels = kmeans.fit_predict(X=features)

SCS, CHI, DBI = metrics(X=features, labels=kmeans_labels)

print(f"SCS:{SCS:.3f}\nCHI:{CHI:.3f}\nDBI:{DBI:.3f}")









