from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import math as math

# read in the data, X will be in the form [[x1,y1],[..],...]
d  = pd.read_csv("test1_change_only.csv")
JAPAN = d['JAPAN']
HK = d['HK']
dataframe = d.loc[:, ['JAPAN', 'HK']]
X = np.array(dataframe.to_numpy())

# Perform PCA
pca = PCA()
pca.fit(X)
# Covariance matrix
# [[0.67028827 0.08640905]
# [0.08640905 1.6896526 ]]

y = pca.components_[0][1]
x = pca.components_[0][0]
evals = pca.explained_variance_
# [1.6969254  0.66301547]

# Draw the eignevectors
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    color='b',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# plot data
plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
for length, vector in zip(pca.explained_variance_, pca.components_):
    v = vector * 3 * np.sqrt(length)
    draw_vector(pca.mean_, pca.mean_ + v)
plt.axis('equal')
plt.margins(1,1)
plt.scatter(JAPAN, HK, color='r')
plt.show()

#  Calculate theta in degrees
theta = math.atan(y/x)*180/(math.pi)