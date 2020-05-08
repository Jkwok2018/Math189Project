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
import rnn
from sklearn import metrics
from matplotlib.patches import Ellipse
from scipy.optimize import minimize
from optimization import opt_helper
from scipy.optimize import NonlinearConstraint
import tensorflow as tf

###### FUNCTIONS #########

#-----------------------Visualizing Data-------------------------
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


#-----------------------Determining K-------------------------
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
    plt.xlabel('k')
    plt.ylabel('distance')
    plt.show()

""" Functions Used to Determine Cluster of unused testing data """
def cluster_mean(clusters):
    meanList = []
    for i in range(len(clusters)):
        c = clusters[i]
        mean = np.mean(c, axis = 0)
        # print(mean)
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

#--------------------------------------Markov Chain---------------

def get_cluster_dict(L, seq_length):
    """ 
    input: a list of cluster labels in time series
    output: an dictionary where the keys are chunks of seq_length clusters 
    and the values is [the most probable fifth cluster, probability of the most probable]
    """
    items = dict()
    for i in range(len(L)-seq_length):
        labels = ''.join(map(str, L[i:i+seq_length] )) 
        if labels not in items:
            items.update({labels: [0,0,0,0,0,0,0]})
        old_value = items.get(labels)
        index = L[i + seq_length]
        old_value[index] += 1
        items.update({labels: old_value})
            
    for key in items:
        max_cluster = np.argmax(items.get(key))
        prob = max(items.get(key))/sum(items.get(key))
        items[key] = [max_cluster, prob]

    return items

def test_markov(dict, testL, seq_length):
    '''
    input: dictionary generated from running get_cluster_dict, testL is the test set, 
            seq_length is the length of the key
    output: the accuracy of prediction and the number of cases in which the testing set's
    shift window was not found in the training set. 
    '''
    not_found = 0
    failed_cases = 0
    for i in range(len(testL)-seq_length-1):
        labels = ''.join(map(str, testL[i:i+seq_length])) 
        if labels not in dict:
            not_found += 1
        elif (dict[labels][0] != testL[i+seq_length]):
            failed_cases += 1
    correctness = 1-((failed_cases + not_found)* 1.0 /(len(testL)))
    return correctness, not_found


def dictionary(Xtrain, Xtest, seq_length):
    """
    Returns the accuracy for a specific sequence length 
    Input: Xtrain, the training data (list)
           Xtest, the testing data (list)
           seq_length, the sequence length of the key (integer)
    """
    # Training and Testing Markov Model
    dictionary = get_cluster_dict(Xtrain, seq_length)
    accuracy, not_found = test_markov(dictionary, Xtest, seq_length)
    # print("Accuracy is " + str(accuracy*100) + "%")
    # print("Cases not found: ", not_found)
    return accuracy

def preprocess_test_data(weight, clusters):
    """
    Pre-processing the testing data
    Input: clusters - an array of time-series label of the training data
    Note: Called in accuracy_vs_length
    """
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
    Xtest = assign_clusters(weight, meanCluster,norm_testdata)
    return Xtest
    

def accuracy_vs_length(weight, clusters, Xtrain):
    print("accuracy vs length")
    """
    Plot the sequence legnth vs accuracy plot
    Input: clusters - a list of time-series label of the training data
    """
    # Pre-processes the test data and return a list of time-series label
    # of the testing data
    Xtest = preprocess_test_data(weight, clusters)

    # generate a list of different sequence lengths
    seq_length_L = [i for i in range(4, 10)]
    accuracy_L = []

    # For each sequence length, determines the accuracy by calling the 
    # dictionary function. Then, append the accuracy to a list
    for seq_length in seq_length_L:
        accuracy = dictionary(Xtrain, Xtest, seq_length)
        accuracy_L.append(accuracy)
    
    # plot sequence legnth vs accuracy
    plt.scatter(seq_length_L, accuracy_L)
    plt.xlabel('seq_length')
    plt.ylabel('accuracy')
    plt.show()


"""Helper Function for Main"""
def sum30Day(dataframe):
    '''
    Summarizing 30 days data in the following format:
        [mean of percent price change, mean of percent volume change change,
        principal eigenvalue, secondary principal eigenvalue, theta]
    '''
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

    # def rosen(weight):
    #     return opt_helper(weight, raw_data)

    # # weight0 = [0.2, 0.78, 0.015, 0.005]
    # weight0 = [0.05, 0.45, 0.45, 0.05]
    # weight0 = [0.08, 0.25, 0.2, 0.47]
    # print(weight0)
    
    # constr = {'type':'eq',
    #           'fun': lambda x: 1-sum(x)}
    # bounds = tuple(((0,1) for x in weight0))
    # bounds = [(0, 0.1), (0.1,1), (0.1,1), (0, 0.1)]

    # # # constraint = NonlinearConstraint(sum, 1, 1)
    # # Nelder-Mead, BFGS
    # ans = minimize(rosen, weight0, method='Nelder-Mead', 
    #              options={'maxiter':10})
    # ans = minimize(rosen, weight0, method='SLSQP', 
    #                  constraints=[constr],bounds=bounds,
    #                 options={'maxiter':10,'ftol':0.001})
    # ans = minimize(rosen, weight0, method='COBYLA', 
    #                  constraints=[constr],
    #                 options={'tol':0.1,'maxfev':2})
    
    
    # bounds = tuple((0.1,1) for x in weight0)
    # # print(weight0)
    # constraint = NonlinearConstraint(sum, 1, 1)
    # ans = minimize(rosen, weight0, method='trust-constr', 
    #                 bounds=bounds,constraints=[constraint], options={'maxiter':3})
    # print(ans)

    # for weight_ in testing_weight:
    #     score = rosen(weight_)
    #     print("weight, score:", weight_, score)
    # normalize the raw data so that they are all in the range of (0,1)
    (norm_data, mins, maxs) = cluster.mm_normalize(raw_data)
    # # define the number of clusters 
    k = 7

    # perform clustering
    print("\nClustering normalized data with k=" + str(k))
    # # weight: a list of 4 items that contained the weight of the center, the principal and the secondary eigenvalue, and theta
    # weight = [0.16, 0.74, 0.05, 0.05]
    # weight = [0.15, 0.75, 0.05, 0.05]
    weight = [0.09, 0.1, 0.1, 0.71]
    
    def distance(item1, item2):
        return cluster.distance(weight,item1, item2)

    clustering = cluster.cluster(weight, norm_data, k)
    
    # print results
    print("\nDone. Clustering:")
    # print(clustering)
    print("\nRaw data grouped by cluster: ")
    clusters = cluster.display(norm_data, clustering, k)

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
    # distance_L = []
    # for  k in range(1, 8):
    #     print("k = "+ str(k))
    #     clustering = cluster.cluster(norm_data, k)
    #     clusters = cluster.display(norm_data, clustering, k)
    #     distance = cluster_distance(clusters)
    #     distance_L.append([k, distance])
    # threshold_plot(distance_L)


    print("\nEnd k-means demo ")

    # # Prediction
    # # rnn.rnn(clustering, seq_length, k)
    # checkpoint_dir = './training_checkpoints'
    # # Restore the latest checkpoint
    # tf.train.latest_checkpoint(checkpoint_dir)
    
    # accuracy_L = []
    # for seq_length in range(8,11):
    #     embedding_dim = 256
    #     rnn_units = 1024
    #     model = rnn.build_model(k, embedding_dim, rnn_units, batch_size=1)
    #     model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
    #     model.build(tf.TensorShape([1, None]))

    #     predictions_L = rnn.predict(model, clustering)
    #     actual_L = clustering[4:]
    #     accuracy = rnn.accuracy(predictions_L, actual_L)
    #     accuracy_L.append(accuracy)
    #     print("accuracy of seq_length ", seq_length, " is ", str(accuracy))
    # print(accuracy_L)
    
    # prediction = rnn.predict(model, [0, 0, 1, 1])
    # print("prediction: ", prediction)
    # seq_length = 4, accuracy 0.8086890243902439
    # [0.14, 0.74, 0.15, 0.05]

    # Setting the trained data to Xtrain
    Xtrain = clustering
    # create the accuracy vs sequence length plot
    accuracy_vs_length(weight, clusters, Xtrain)


if __name__ == "__main__":
    main()



