#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

pca = plt.figure().add_subplot(111, projection='3d')
pca.set_xlabel("U1")
pca.set_ylabel("U2")
pca.set_zlabel("U3")
pca.scatter(xs=pca_data[:, 0], ys=pca_data[:, 1],
            zs=pca_data[:, 2], c=labels, cmap=plt.cm.plasma)
plt.title("PCA of Iris Dataset")
plt.show()
