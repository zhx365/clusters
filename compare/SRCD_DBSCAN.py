
import const
import dkImplication as dk
from metrics import metrics

labels, centers = dk.srcd_dbscan(dk.features, const.eps, const.min_sample)

SCS, CHI, DBI = metrics(X=dk.features, labels=labels)

print(f"SCS:{SCS:.3f}\nCHI:{CHI:.3f}\nDBI:{DBI:.3f}")