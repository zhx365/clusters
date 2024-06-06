import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA


from metrics import metrics
import const

data = pd.read_csv(const.path)

features1 = data.iloc[:, [1, 2, 3, 4]].to_numpy()

features = PCA(n_components=2).fit_transform(features1)

dbscan = DBSCAN(eps=const.eps, min_samples=const.min_sample)

dbscan_labels = dbscan.fit_predict(X=features)

SCS, CHI, DBI = metrics(X=features, labels=dbscan_labels)

print(f"SCS:{SCS}\nCHI:{CHI}\nDBI:{DBI}")