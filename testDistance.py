from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import math as math
import torch
from statistics import mean

# read in the data, X will be in the form [[x1,y1],[..],...]
d  = pd.read_csv("test1_change_only.csv")
JAPAN = d['JAPAN']
HK = d['HK']
SHANGHAI = d['SHANGHAI']
dataframe = d.loc[:, ['JAPAN', 'HK']]
dataframe2 = d.loc[:, ['JAPAN', 'SHANGHAI']]
X = np.array(dataframe.to_numpy())
X2 = np.array(dataframe2.to_numpy())

# center: return the mean
def center(df):
    df_L = df.tolist()
    return mean(df_L)
JAPAN_mean = center(JAPAN)
HK_mean = center(HK)
SHANGHAI_mean = center(SHANGHAI)

# check_center: return true if the distance between the centers is
# smaller than the threshold
def check_center(df, df2, threshold):
    diff = math.abs(center(df) - center(df2))
    return diff < threshold

# np.array(JAPAN.to_numpy())
print(JAPAN_mean)

# Perform PCA on JAPAN vs HK
pca = PCA()
pca.fit(X)
# Eigenvectors
y = pca.components_[0][1]
x = pca.components_[0][0]
#  Calculate theta in degrees
theta = math.atan(y/x)*180/(math.pi)
# Eigenvalues
evals = pca.explained_variance_
print(evals) 
# [2.58450984 2.11708665]


# Perform PCA on JAPAN vs SHANGHAI
pca2 = PCA()
pca2.fit(X2)
# Eigenvectors
y2 = pca2.components_[0][1]
x2 = pca2.components_[0][0]
#  Calculate theta in degrees
theta2 = math.atan(y2/x2)*180/(math.pi)
# Eigenvalues
evals2 = pca2.explained_variance_
print(evals2)
# [6.71260216 2.4525767 ]

def distance(x,y,x2,y2,theta,theta2):
    # m = M[0].item()
    # n = 1-x
    m = 0.5
    n = 0.5
    lambda1 = torch.tensor([x,y])
    lambda2 = torch.tensor([x2,y2])
    output = torch.tensor(m * (lambda1 - lambda2) + n * (theta - theta2))
    return output

# # Calculate the mimunum of the rosenbrock function
# M = torch.tensor([0.5], requires_grad=True)
# alpha = 0.01
# for i in range(5000):
#     z = distance(M)
#     z.backward()
#     m = m - alpha * m.grad
#     m = torch.tensor(m, requires_grad=True)
#     print('i=',i,', m=', n, 'value=',z)


# distance_min = z.item()
# print('The minimum of the distance function is ', distance_min)
