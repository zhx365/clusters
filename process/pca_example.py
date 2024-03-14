from matplotlib import pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA

# 初始化PCA，设置降维后的维数，例如降到2维
pca = PCA(n_components=2)


eps, min_sample = 0.05, 18
data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
features = data.iloc[:, [1, 2, 3, 4]].to_numpy()
# 对数据进行降维
pca_result = pca.fit_transform(features)

# 可视化降维结果
plt.figure(figsize=(8, 5))
plt.scatter(pca_result[:, 0], pca_result[:, 1])
plt.title('PCA Result')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
