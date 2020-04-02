from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
import random
# import seaborn as sns; sns.set()
from statistics import mean
import cluster as cluster
from matplotlib.patches import Ellipse

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->', color='b', linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

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
        ellipse = Ellipse(xy=(L[0], L[1]), width=2*L[2], height=2*L[3],angle=L[4], 
                                edgecolor=color, fc='None', lw=2)
        ax.add_patch(ellipse)
        # ax.set_xlim(-2, 2)
        # ax.set_ylim(0, 5)
        plt.margins(1,1)
    plt.show()

# def draw_single_ellipse(X, L):
#     plt.figure()
#     ax = plt.gca()

#     ellipse = Ellipse(xy=(L[0], L[1]), width=L[3], height=L[2],angle=L[4], 
#                             edgecolor='r', fc='None', lw=2)
#     ax.add_patch(ellipse)
#     plt.scatter(X[:30, 0], X[:30, 1], color='r')
#     plt.margins(1,1)
#     plt.show()

def cluster_distance(clusters):
    """ Sum the average distance in each cluster, and then take the average
    Input: clusters is a array in an array, saving all the eclipses by its cluster
    [[ellipses in cluster 0]
     [ellipses in cluster 1] 
     ... 
     [ellipses in cluster k-1]]
    This array is the output of the cluster.display function
    output: floating point number that is the average distance between all clusters
    """
    clusters_avg = 0
    total = 0.0
    for i in range(len(clusters)):
        c = clusters[i]
        mean = np.mean(c, axis=0)
        # r = random.sample(range(len(c)), 10)
        # for j in range(len(r)):
        #     total += cluster.distance(c[r[j]], mean)
        # clusters_avg += total*1.0/len(r)
        for j in range(len(c)):
            total += cluster.distance(c[j], mean)
        clusters_avg += total*1.0/len(c)
    return float(clusters_avg/len(clusters))
    
            
def threshold_plot(L):
    """
    Using the output of random_distance, plot distance vs. k for lists of k values
    input: L is a list of paired k values and distances
    """
    # Plot the distances against k
    k = [i[0] for i in L]
    distances = [i[1] for i in L]
    plt.scatter(k, distances)
    plt.show()

    

def main():
    # read in preprocessed values
    d  = pd.read_csv("newProcessed.csv")
    dataframe = d.loc[:, ['PriceChange', 'VolumeChange']]
    X = np.array(dataframe.to_numpy())

    # data is the matrix that holds all the pca 5-elements lists
    # it has a dimension of (n, 5) where n is the number of pcas we have
    data = []

    # Perform PCA on every 30 data points using the shifting strategy
    i = 0
    pca = PCA() # declare PCA object with constructor

    '''
    Summarizing 30 days data in the following format:
        [mean of percent price change, mean of percent volume change change,
        principal eigenvalue, secondary principal eigenvalue, theta]
    '''
    while (i<len(X)-30):
        oneMonth = X[i:i+30]
        returnList=[]

        # Append mean of % change price and % change volume
        returnList.append(mean(oneMonth[:, 0]))
        returnList.append(mean(oneMonth[:, 1]))
        
        # Append the two eigenvalues, the bigger eigenvalues go first
        # and hold more weight
        pca.fit(oneMonth)
        evals = pca.explained_variance_
        returnList.append(evals[0])
        returnList.append(evals[1])

        # calculate theta
        y = pca.components_[0][1]
        x = pca.components_[0][0]
        theta = math.atan(y/x)*180/(math.pi)

        # append theta
        returnList.append(theta)

        # increment i 
        i = i+1

        # append returnList to data
        data.append(returnList)

  
    print("\nBegin k-means clustering demo \n")
    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(2)

    # convert data to np.array
    raw_data =  np.asarray(data, dtype=np.float32)
    # normalize the raw data so that they are all in the range of (0,1)
    (norm_data, mins, maxs) = cluster.mm_normalize(raw_data)
    # print(norm_data[:5])
    # # define the number of clusters 
    # k = 3

    # # perform clustering
    # print("\nClustering normalized data with k=" + str(k))
    # clustering = cluster.cluster(norm_data, k)
    
    # print results
    # print("\nDone. Clustering:")
    # print(clustering)
    # print("\nRaw data grouped by cluster: ")
    # clusters = cluster.display(raw_data, clustering, k)
    # draw_ellipse(data, clustering)
    # print("\nEnd k-means demo ")

    # Find the optimal k value by calculating the average distance associated with each
    distance_L = []
    for  k in range(1, 4):
        print("k = "+ str(k))
        clustering = cluster.cluster(norm_data, k)
        clusters = cluster.display(norm_data, clustering, k)
        distance = cluster_distance(clusters)
        distance_L.append([k, distance])
    threshold_plot(distance_L)

if __name__ == "__main__":
    main()



