

import const
import dkImplication as dk
from metrics import metrics

labels, centers = dk.srcd_dbscan_kmeans(dk.features, dk.eps, const.min_sample)

SCS, CHI, DBI = metrics(X=dk.features, labels=labels)

print(f"SCS:{SCS}\nCHI:{CHI}\nDBI:{DBI}")