
import pandas as pd

from sklearn.cluster import KMeans

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import const

from metrics import metrics

data = pd.read_csv(const.path)

features1 = data.iloc[:, [1, 2, 3, 4]].to_numpy()

features = PCA(n_components=2).fit_transform(features1)

kemans = KMeans(init='k-means++', n_clusters=const.K, n_init='auto')

kmeans_plus_labels = kemans.fit_predict(features)

SCS, CHI, DBI = metrics(features, kmeans_plus_labels)

print(f"SCS:{SCS:.3f}\nCHI:{CHI:.3f}\nDBI:{DBI:.3f}")