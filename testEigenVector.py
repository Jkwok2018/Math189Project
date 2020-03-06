from sklearn.decomposition import PCA
import numpy as np
import pandas as pd

d  = pd.read_csv("test1_change_only.csv")
JAPAN = d['JAPAN']
HK = d['HK']
dataframe = d.loc[:, ['JAPAN', 'HK']]
# X = np.array(dataframe.to_numpy())
X = np.array([[1, 1], [4, 1]])
pca = PCA()
pca.fit(X)
# print(pca.components_)

# centered_matrix = X - X.mean(axis=1)[:, np.newaxis]
# cov = np.dot(centered_matrix, centered_matrix.T)
cov = np.dot(X, X.T)
print(cov)
eigvals, eigvecs = np.linalg.eig(cov)
print(eigvals)
print(eigvecs)