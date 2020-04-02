from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
# import seaborn as sns; sns.set()
from statistics import mean
import cluster as cluster
from matplotlib.patches import Ellipse

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->', color='b', linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    print(v0)

def draw_ellipse(data, cluster):
    
    plt.figure()
    ax = plt.gca()
    
    for i in range(len(data)):
        if cluster[i] == 0:
            color='r' 
            # print("r")
        elif cluster[i] == 1:
            color='g'
            # print("g")
        else:
            color='b'
            # print("b")
        L = data[i]
        print(1/math.sqrt(L[3]))
        ellipse = Ellipse(xy=(L[0], L[1]), width=1/math.sqrt(L[2]), height=1/math.sqrt(L[3]),angle=L[4], 
                                edgecolor=color, fc='None', lw=2)
        ax.add_patch(ellipse)
        # ax.set_xlim(-2, 2)
        # ax.set_ylim(0, 5)
        plt.margins(1,1)
    plt.show()

def draw_single_ellipse(X, L):
    plt.figure()
    ax = plt.gca()

    ellipse = Ellipse(xy=(L[0], L[1]), width=L[3], height=L[2],angle=L[4], 
                            edgecolor='r', fc='None', lw=2)
    ax.add_patch(ellipse)
    plt.scatter(X[:30, 0], X[:30, 1], color='r')
    plt.margins(1,1)
    plt.show()

def main():
    
    # read in the data, X will be in the form [[x1,y1],[..],...]
    d  = pd.read_csv("newProcessed.csv")
    # PercentChange = d['PriceChange']
    # Volume = d['VolumeChange']

    # PercentChange = [1,2,3, 4]
    # Volume = [-1, -2, -3, 4]
    # dataframe = pd.DataFrame(list(zip(PercentChange,Volume)),
    #                 columns = ['PriceChange', 'VolumeChange'])
    dataframe = d.loc[:, ['PriceChange', 'VolumeChange']]
    X = np.array(dataframe.to_numpy())

    #pca = PCA()

    # data is the matrix that holds all the pca 5-elements lists
    # it has a dimension of (n, 5) where n is the number of pcas we have
    data = []
    # Perform PCA on every 30 data points
    i = 0
    while (i<200):
        pca = PCA()
        oneMonth = X[i:i+30]
        returnList=[]

        # Append mean of price and volume
        returnList.append(mean(oneMonth[:, 0]))
        returnList.append(mean(oneMonth[:, 1]))
        
        # PCA
        pca.fit(oneMonth)
       
        # Append the two eigenvalues, the bigger eigenvalues go first
        # and hold more weight
        evals = pca.explained_variance_
        returnList.append(evals[0])
        returnList.append(evals[1])

        # TODO: delete later test only
        print("eigenvectors \n")
        print(pca.components_)

        # [[-0.02825844 -0.99960065]
        # [-0.99960065  0.02825844]]

        # calculate theta
        y = pca.components_[0][1]
        x = pca.components_[0][0]
        theta = math.atan(y/x)*180/(math.pi)

        # append theta
        returnList.append(theta)

        # plots the pca and its eigenvectors
        plt.scatter(oneMonth[:, 0], oneMonth[:, 1], color='r')
        for length, vector in zip(pca.explained_variance_, pca.components_):
            v = vector * length
            draw_vector(pca.mean_, pca.mean_ + v)
        ax = plt.gca()
        ellipse = Ellipse(xy=(returnList[0], returnList[1]), width=2*returnList[2], 
                            height=2*returnList[3],angle=returnList[4], 
                            edgecolor='r', fc='None', lw=2)
        ax.add_patch(ellipse)
        # plt.scatter(JAPAN, HK, color='r')
        plt.margins(1,1)
        plt.figure()
        plt.show()

        # increment i 
        i = i+30

        # append returnList to data
        data.append(returnList)

    # Draw ellipse
    # print(data[:5])
    # draw_single_ellipse(X,data[0])
    # draw_ellipse(X,data[0])
  


    
    # print("\nBegin k-means clustering demo \n")
    # np.set_printoptions(precision=4, suppress=True)
    # np.random.seed(2)

    # # convert data to np.array
    # raw_data =  np.asarray(data, dtype=np.float32)
    # # normalize the raw data so that they are all in the range of (0,1)
    # (norm_data, mins, maxs) = cluster.mm_normalize(raw_data)
    
    # # TODO: delete this later
    # # printing out the first 5 norm and raw data for testing
    # for i in range(5):
    #     print('raw')
    #     print("%4d " % i, end=""); print(raw_data[i])
    #     print('norm')  
    #     print("%4d " % i, end=""); print(norm_data[i])  

    # # define the number of clusters 
    # k = 3

    # # # perform clustering
    # # print("\nClustering normalized data with k=" + str(k))
    # # clustering = cluster.cluster(norm_data, k)
    
    # # # print results
    # # print("\nDone. Clustering:")
    # # print(clustering)
    # # print("\nRaw data grouped by cluster: ")
    # # cluster.display(raw_data, clustering, k)
    # # print("\nEnd k-means demo ")
    # # # draw_ellipse(data, clustering)

if __name__ == "__main__":
    main()


#TODO: write a function that determines the weights
#TODO: write a function that tests what k would yield the sum of the smallest distances 