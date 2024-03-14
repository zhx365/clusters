from matplotlib import pyplot as plt
import pandas as pd
from sklearn.manifold import TSNE

# 初始化t-SNE，设置降维后的维数和随机种子以确保可复现性
tsne = TSNE(n_components=2, random_state=42)



eps, min_sample = 0.05, 18
data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
features = data.iloc[:, [1, 2, 3, 4, 5, 6, 7]].to_numpy()

# 对数据进行降维
tsne_result = tsne.fit_transform(features)

# 可视化降维结果
plt.figure(figsize=(8, 5))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1])
plt.title('t-SNE Result')
plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.show()
