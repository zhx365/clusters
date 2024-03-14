from matplotlib import pyplot as plt
import pandas as pd
import umap

# 初始化UMAP，设置降维后的维数
reducer = umap.UMAP(n_components=2, random_state=42)


data = pd.read_csv(r"D:\Documents\数据集\车站\车站分类-加上休-线路-处理后.csv")
features = data.iloc[:, [1, 2, 3, 4]].to_numpy()
# 对数据进行降维
umap_result = reducer.fit_transform(features)

# 可视化降维结果
plt.figure(figsize=(8, 5))
plt.scatter(umap_result[:, 0], umap_result[:, 1])
plt.title('UMAP Result')
plt.xlabel('UMAP Feature 1')
plt.ylabel('UMAP Feature 2')
plt.show()
