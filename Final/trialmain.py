# main.py
# Perform k-means clustering and Markov Model

from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
import random
from statistics import mean
import cluster2 as cluster
from sklearn import metrics
from matplotlib.patches import Ellipse
from scipy.optimize import minimize
from optimization import opt_helper
from scipy.optimize import NonlinearConstraint

###### FUNCTIONS #########

""" Visualizing Data"""

def draw_vector(v0, v1, ax=None):
    ''' draws eigen vectors
    '''
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->', color='b', linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

def draw_ellipse(data, cluster):
    '''Plot all of the eclipses, and plot each cluster in different color
    '''
    plt.figure()
    ax = plt.gca()
    for i in range(len(data)):
        if cluster[i] == 0:
            color='r' 
        elif cluster[i] == 1:
            color='g'
        elif cluster[i] == 2:
            color = 'c'
        elif cluster[i] == 3:
            color = 'y'
        elif cluster[i] == 4:
            color = 'k'
        elif cluster[i] == 5:
            color = 'm'
        else:
            color='b'
        L = data[i]
        ellipse = Ellipse(xy=(L[0], L[1]), width=2*L[2], height=2*L[3],angle=L[4], 
                                edgecolor=color, fc='None', lw=2)
        ax.add_patch(ellipse)

        # Uncomment if want to see the scales on both axis are the same
        #
        # plt.xlim(-6000,6000)
        # plt.ylim(-6000,6000)
        plt.margins(1,1)
    plt.show()


""" Functions used for determining the value of k """

def cluster_distance(clusters,weight):
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
        for j in range(len(c)):
            total += cluster.distance(weight,c[j], mean)
        #clusters_avg += total*1.0/len(c)
        clusters_avg += total*1.0
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
    plt.xlabel('k')
    plt.ylabel('distance')
    plt.show()

""" Functions Used to Determine Cluster of unused testing data """
def cluster_mean(clusters):
    meanList = []
    for i in range(len(clusters)):
        c = clusters[i]
        mean = np.mean(c, axis = 0)
        meanList.append(mean)
    return meanList

def assign_clusters(weight, meanList, dataToBeAssigned):
    dataCluster = []
    for data in dataToBeAssigned:
        distanceList = []
        for clusterMean in meanList:
            distanceList += [cluster.distance(weight, data, clusterMean)]
        dataCluster += [np.argmin(distanceList)]
    return dataCluster

""" Training and Testing the Mark Model"""

def get_cluster_dict(L):
    """ 
    input: a list of cluster labels in time series
    output: an dictionary where the keys are chunks of 4 clusters 
    and the values is [the most probable fifth cluster, probability of the most probable]
    """
    items = dict()
    for i in range(len(L)-5):
        four_label = ''.join(map(str, L[i:i+4] )) 
        if four_label not in items:
            items.update({four_label: [0,0,0,0,0,0,0]})
        old_value = items.get(four_label)
        index = L[i + 4]
        old_value[index] += 1
        items.update({four_label: old_value})
            
    for key in items:
        max_cluster = np.argmax(items.get(key))
        prob = max(items.get(key))/sum(items.get(key))
        items[key] = [max_cluster, prob]

    return items

def test_markov(dict, testL):
    '''
    input: dictionary generated from running get_cluster_dict, and testL is the test set
    output: the accuracy of prediction and the number of cases in which the testing set's
    shift window was not found in the training set. 
    '''
    not_found = 0
    failed_cases = 0
    for i in range(len(testL)-5):
        four_label = ''.join(map(str, testL[i:i+4] )) 
        if four_label not in dict:
            not_found += 1
        elif (dict[four_label][0] != testL[i+4]):
            failed_cases += 1
    correctness = 1-((failed_cases + not_found)* 1.0 /(len(testL)))
    return correctness, not_found

"""Helper Function for Main"""

'''
    Summarizing 30 days data in the following format:
        [mean of percent price change, mean of percent volume change change,
        principal eigenvalue, secondary principal eigenvalue, theta]
'''
def sum30Day(dataframe):

    # data is the matrix that holds all the pca 5-elements lists
    # it has a dimension of (n, 5) where n is the number of pcas we have
    data = []

    # Perform PCA on every 30 data points using the shifting strategy
    i = 0
    pca = PCA() # declare PCA object with constructor

    while (i<len(dataframe)-30):
        oneMonth = dataframe[i:i+30]
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

    return data    

###### MAIN ########

def main():
    # read in preprocessed values
    d  = pd.read_csv("SandPMarch31.csv")
    dataframe = d.loc[:, ['PriceChange', 'frac']]
    X = np.array(dataframe.to_numpy())

    """ summarize 30 day data and put into matrix"""
    data = sum30Day(X)

    print("\nBegin k-means clustering demo \n")
    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(2)
    
    # convert data to np.array
    raw_data =  np.asarray(data, dtype=np.float32)

    def rosen(weight):
        return opt_helper(weight, raw_data)

    # # weight0 = [0.2, 0.78, 0.015, 0.005]
    # weight0 = [0.05, 0.45, 0.45, 0.05]
    weight0 = [0.1, 0.4, 0.4, 0.1]
    # print(weight0)
    
    constr = {'type':'eq',
              'fun': lambda x: 1-sum(x)}
    # # bounds = tuple(((0,1) for x in weight0))
    bounds = [(0, 0.1), (0.1,1), (0.1,1), (0, 0.1)]

    # # constraint = NonlinearConstraint(sum, 1, 1)
    # Nelder-Mead, BFGS
    ans = minimize(rosen, weight0, method='Nelder-Mead', 
                 options={'maxiter':10})
    # ans = minimize(rosen, weight0, method='SLSQP', 
    #                  constraints=[constr],bounds=bounds,
    #                 options={'maxiter':10,'ftol':0.001})
    # ans = minimize(rosen, weight0, method='COBYLA', 
    #                  constraints=[constr],
    #                 options={'tol':0.1,'maxfev':2})
    
    # weight0 = [0.1, 0.1, 0.1, 0.7]
    # bounds = tuple((0.1,1) for x in weight0)
    # # print(weight0)
    # constraint = NonlinearConstraint(sum, 1, 1)
    # ans = minimize(rosen, weight0, method='trust-constr', 
    #                 bounds=bounds,constraints=[constraint], options={'maxiter':3})
    print(ans)


    # testing_weight = [[0.3,0.3,0.2,0.2],
    #                   [0.5, 0.1, 0.1, 0.3],
    #                   [0.3, 0.2, 0.2, 0.3],
    #                   [0.2, 0.2, 0.3 , 0.3],
    #                   [0.1, 0.1, 0.1, 0.7],
    #                   [0.2, 0.3, 0.3, 0.2],
    #                   [0.1, 0.3, 0.3, 0.3],
    #                   [0.1, 0.2, 0.2, 0.5]
    #                   ]
    # #                   # highest: [0.2, 0.5, 0.2, 0.1]

    # for weight_ in testing_weight:
    #     score = rosen(weight_)
    #     print("weight, score:", weight_, score)
    # normalize the raw data so that they are all in the range of (0,1)
    (norm_data, mins, maxs) = cluster.mm_normalize(raw_data)
    # define the number of clusters 
    # k = 7

    # perform clustering
    # print("\nClustering normalized data with k=" + str(k))
    # weight: a list of 4 items that contained the weight of the center, the principal and the secondary eigenvalue, and theta
    # weight = [0.1, 0.2, 0.1, 0.7]

    def distance(item1, item2):
        return cluster.distance(weight0,item1, item2)

    # clustering = cluster.cluster(weight, norm_data, k)
    
    # # print results
    # print("\nDone. Clustering:")
    # print(clustering)
    # print("\nRaw data grouped by cluster: ")
    # clusters = cluster.display(norm_data, clustering, k)

    # result = metrics.silhouette_score(norm_data, clustering, metric = distance, sample_size=50)
    # result2 = metrics.silhouette_score(raw_data, clustering, metric = distance,sample_size=50)
    # print("smthing score")
    # print(result)
    # print("Secod score")
    # print(result2)

    # Uncomment in order to visualize the result by plotting all elicpses on the same plot
    #
    # draw_ellipse(data, clustering)

    ### Uncoment if want to visualize the optimal k value, must comment out the block above
    # Find the optimal k value by calculating the average distance associated with each
    #
    # weight = [0.05, 0.45, 0.45, 0.05]
    # #weight = [0.10, 0.45, 0.40, 0.05]
    # distance_L = []
    # for  k in range(3, 9):
    #     print("k = "+ str(k))
    #     clustering = cluster.cluster(weight,norm_data, k)
    #     clusters = cluster.display(norm_data, clustering, k)
    #     distance0 = cluster_distance(clusters, weight)
    #     distance_L.append([k, distance0])
    #     result = metrics.silhouette_score(raw_data, clustering, 
    #                 metric = distance, sample_size=500)
    #     print(result)
    # threshold_plot(distance_L)


    print("\nEnd k-means demo ")

"""
    # Split the eclipses in 9 Xtrain: 1 Xtest for the Markov Model
    Xtrain = clustering

    # Obtain testData
    dTest  = pd.read_csv("Processed95-00.csv")
    dataframeTest = dTest.loc[:, ['PriceChange', 'VolumeChange']]
    Y = np.array(dataframeTest.to_numpy())
    testData = sum30Day(Y)

    # Get cluster mean
    meanCluster = cluster_mean(clusters)
    # Normalize TestData
    raw_testdata =  np.asarray(testData, dtype=np.float32)
    (norm_testdata, mins, maxs) = cluster.mm_normalize(raw_testdata)
    testDataClustering = assign_clusters(meanCluster,norm_testdata)

    Xtest = testDataClustering
    
    # Training and Testing Markov Model
    dictionary = get_cluster_dict(Xtrain)
    correctness, not_found = test_markov(dictionary, Xtest)
    print("Accuracy is " + str(correctness*100) + "%")
    print("Cases not found: ", not_found)
    
    metrics.silhouette_score(norm_data, clustering)

"""
if __name__ == "__main__":
    main()



