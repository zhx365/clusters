import matplotlib.pyplot as plt
import numpy as np

# 假设有4个独特的簇标签
unique_label = [0, 1, 2, -1]  # 假设-1代表噪声

# 生成颜色
colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_label)))

# 显示颜色
plt.figure(figsize=(8, 1))
for i, color in enumerate(colors):
    plt.fill_between([i, i+1], 0, 1, color=color)
plt.xlim(0, len(unique_label))
plt.gca().yaxis.set_visible(False)
plt.gca().xaxis.set_visible(False)
plt.title("Colors for Unique Labels")
plt.show()
