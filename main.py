from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import math as math
import torch
from statistics import mean

def main()
    # read in the data, X will be in the form [[x1,y1],[..],...]
    d  = pd.read_csv("test1_change_only.csv")
    PercentChange = d['PercentChange']
    Volumn = d['Volume']
    dataframe = d.loc[:, ['PercentChange', 'Volumn']]
    X = np.array(dataframe.to_numpy())
    
    pca = PCA()
    # Perform PCA on every 30 data points
    i = 0
    while (i<len(X)):
        oneMonth = X[i:i+30]
        pca.fit(oneMonth)
        # Eigenvectors
        v = pca.components_[0]
        #  Calculate theta in degrees
        theta = math.atan(v[1]/v[2])*180/(math.pi)
        # Eigenvalues in the form [lambda1, lambda2]
        evals = pca.explained_variance_
        #TODO: Format the 4 values into one array and append them into a matrix
        i = i+30


def distance(x,y,x2,y2,theta,theta2):
    # TODO: write a function that determines the best m and n
    # TODO: need to update the distance function
    m = 0.5
    n = 0.5
    lambda1 = torch.tensor([x,y])
    lambda2 = torch.tensor([x2,y2])
    #output = torch.tensor(m * (lambda1 - lambda2) + n * (theta - theta2))
    #output = h1 * #put the norm for vectors (c1-c2) * m1* math.abs(lambda11-lambda21) + m2 * math.abs(Lambda12 - lambda22) + n * (theta-theta2) 
    # output = m1* (math.abs(lambda11-lambda21))^2 + m2 * math.abs(Lambda12 - lambda22) # also can use squared
    return output

#TODO: k-mean clustering
#TODO: write a function that tests what k would yield the sum of the smallest distances 