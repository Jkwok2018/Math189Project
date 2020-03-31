from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
from statistics import mean 

# Take in a list of 30 elements
list1 = [1,2,3,4,5]
list2 = [-1, -2, -3, -4, -5]

d  = pd.read_csv("test1_change_only.csv")
JAPAN = d['JAPAN'].tolist()
HK = d['HK'].tolist()

def PCAappend(PriceList, VolumeList):
    returnList=[]

    # Append mean of price and volume
    returnList.append(mean(PriceList))
    returnList.append(mean(VolumeList))

    dataframe = pd.DataFrame(list(zip(PriceList,VolumeList)),
                    columns = ['PriceChange', 'VolumeChange'])
    X = np.array(dataframe.to_numpy())
    pca = PCA()
    pca.fit(X)

    # Append the two eigenvalues, the bigger eigenvalues go first
    # and hold more weight
    evals = pca.explained_variance_
    returnList.append(evals[0])
    returnList.append(evals[1])

    print(evals)
    y = pca.components_[0][1]
    x = pca.components_[0][0]
    theta = math.atan(y/x)*180/(math.pi)

    returnList.append(theta)

    return returnList

print(PCAappend(JAPAN,HK))