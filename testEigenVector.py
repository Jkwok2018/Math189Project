from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

d  = pd.read_csv("test1_change_only.csv")
JAPAN = d['JAPAN']
HK = d['HK']
dataframe = d.loc[:, ['JAPAN', 'HK']]
X = np.array(dataframe.to_numpy())
pca = PCA()
# Covariance matrix
# [[0.67028827 0.08640905]
# [0.08640905 1.6896526 ]]

pca.fit(X)
# print(pca.get_covariance())
# print(pca.components_)
# print(pca.explained_variance_)
# plt.scatter(JAPAN, HK, color='r')
# plt.xlabel('JAPAN')
# plt.ylabel('HK')
# plt.savefig("Japan VS HK")


def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    color='b',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(JAPAN, HK, color='r')
for length, vector in zip(pca.explained_variance_, pca.components_):
    print(vector)
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)


plt.axis('equal')
plt.show()


# centered_matrix = X - X.mean(axis=1)[:, np.newaxis]
# cov = np.dot(centered_matrix.T, centered_matrix)
# # print(cov)
# eigvals, eigvecs = np.linalg.eig(cov)
# print(eigvecs)
# print((eigvecs.real)[1])