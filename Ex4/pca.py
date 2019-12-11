import numpy as np
import pandas as pd
f = pd.read_csv("pca_dataset.txt", names= ['x', 'y'], delimiter= ' ')

data = f.values
mean_data = data.mean(axis=0)
data -= mean_data
n ,m = data.shape
# Compute covariance matrix
C = np.dot(data.T, data) / (n-1)
# Eigen decomposition
eigen_vals, eigen_vecs = np.linalg.eig(C)
# SVD
U, Sigma, Vh = np.linalg.svd(data,
    full_matrices=False,
    compute_uv=True)
# Relationship between singular values and eigen values:
print(np.diag(Sigma)) # True