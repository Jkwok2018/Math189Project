from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import math as math
import torch
from statistics import mean
from numpy.linalg import norm

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

def distance(p1, p2):
    """
    p1 and p2 each have 5 elements
    #1. mean of the price
    #2. mean of the volume
    #3. principal eigen value
    #4. the other eigenvalue
    $5 theta
    """
   
    h1 = 0.5       # Weight for center
    m1 = 0.3    # Weight for Prinipal Eigenvector
    m2 = 0.1      # Weight for the smaller eigenvector
    n = 0.1      # Weight of theta
   
    center1 = p1[0:2]
    center2 = p2[0:2]
    evalue1 = p1[2:4]
    evalue2 = p2[2:4]
    theta1 = p1[-1]
    theta2 = p2[-1]
  
    center_dis = numpy.linalg.norm(center2-center1)
    evalue_dis = math.abs(evalue2[0]-evalue1[0])
    evalue_dis2 = math.abs(evalue2[1]-evalue1[1])
    theta_dis = math.abs(theta2-theta1)
    
    return h1*center_dis + m1*evalue_dis + m2*evalue_dis2 + n*theta_dis




     #output = torch.tensor(m * (lambda1 - lambda2) + n * (theta - theta2))
    #output = h1 * #put the norm for vectors (c1-c2) * m1* math.abs(lambda11-lambda21) + m2 * math.abs(Lambda12 - lambda22) + n * (theta-theta2) 
    # output = m1* (math.abs(lambda11-lambda21))^2 + m2 * math.abs(Lambda12 - lambda22) # also can use squared
