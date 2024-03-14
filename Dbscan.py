import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

# 读取数据
data = pd.read_csv(r"D:\Documents\数据集\支持向量机(SVM)分类.csv", encoding='GB2312')
features = data.iloc[:, [1, 2, 3, 4]]

# 特征标准化
scaler = StandardScaler()
features_standardized = scaler.fit_transform(features)

def evaluate_dbscan(eps, min_samples):
    try:
        print(f"Trying parameters: eps={eps}, min_samples={min_samples}")

        # 检查 features 类型
        print(f"Features type: {type(features)}")

        # 检查目标变量是否有NaN值或无穷大值
        if isinstance(features, pd.DataFrame) and (features.isnull().values.any() or np.isinf(features).any()):
            return -100000  # 返回一个大的负数，而不是负无穷

        dbscan = DBSCAN(eps=eps, min_samples=int(min_samples))

        # 检查评分是否有NaN值或无穷大值
        scores = cross_val_score(dbscan, features, cv=5, scoring='neg_mean_squared_error')

        if np.isnan(np.sum(scores)):
            return -100000  # 返回一个大的负数，而不是负无穷

        return np.nanmean(scores)  # 使用nanmean处理NaN值
    except Exception as e:
        print(f"An error occurred: {e}")
        return -100000  # 返回一个大的负数，而不是负无穷

# 定义参数空间
param_grid = {'eps': np.arange(0.05, 1.1, 0.05), 'min_samples': np.arange(5, 21)}

# 创建DBSCAN模型
dbscan = DBSCAN()

# 创建适用于无监督学习的scorer
scorer = make_scorer(lambda estimator, X: -np.mean(np.square(estimator.fit_predict(X))))

# 创建GridSearchCV对象
grid_search = GridSearchCV(estimator=dbscan, param_grid=param_grid, cv=5, scoring=scorer)

# 拟合网格搜索到数据
grid_search.fit(features_standardized)

# 获取最佳参数和相应的目标值
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Target Value:", -best_score)  # 注意：对于均方误差要取负号
