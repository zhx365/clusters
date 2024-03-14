import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args

# 定义搜索空间
space  = [Real(0.01, 1.0, name='eps'),
          Integer(2, 10, name='min_samples')]

# features 是数据集的 numpy 数组
data = pd.read_csv(r"D:\Documents\数据集\支持向量机(SVM)分类.csv", encoding='GB2312')
features = data.iloc[:, [1, 2, 3, 4]].to_numpy()


# data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
# features = data.iloc[:, [1, 2, 3, 4]].to_numpy()


# 定义目标函数
@use_named_args(space)
def objective(**params):
    dbscan = DBSCAN(**params)
    clusters = dbscan.fit_predict(features)
    if len(set(clusters)) > 1:
        return -silhouette_score(features, clusters)
    return 1  # 处理所有点都是噪声的情况

# 执行贝叶斯优化
res_gp = gp_minimize(objective, space, n_calls=50, random_state=0)

print("Best parameters: {}".format(res_gp.x))
