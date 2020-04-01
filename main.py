from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math as math
# import seaborn as sns; sns.set()
from statistics import mean
import cluster as cluster

def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->', color='b', linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)
    print(v0)

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

    print("HI")
    pca = PCA()
    data = []
    # Perform PCA on every 30 data points
    i = 0
    
    while (i<150):
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

        y = pca.components_[0][1]
        x = pca.components_[0][0]
        theta = math.atan(y/x)*180/(math.pi)
        returnList.append(theta)
        plt.scatter(oneMonth[:, 0], oneMonth[:, 1], color='r')
        for length, vector in zip(pca.explained_variance_, pca.components_):
            v = vector * 3 * np.sqrt(length)
            draw_vector(pca.mean_, pca.mean_ + v)
        # plt.scatter(JAPAN, HK, color='r')
        plt.margins(1,1)
        # plt.show()
        i = i+30
        data.append(returnList)
    
    print("\nBegin k-means clustering demo \n")
    np.set_printoptions(precision=4, suppress=True)
    np.random.seed(2)

    raw_data =  np.asarray(data, dtype=np.float32)
    # print(data[:10])

    (norm_data, mins, maxs) = cluster.mm_normalize(raw_data)
    print(norm_data[:5])
    
   
    for i in range(5):
        print('raw')
        print("%4d " % i, end=""); print(raw_data[i])
        print('norm')  
        print("%4d " % i, end=""); print(norm_data[i])  
        
    
    k = 3
    print("\nClustering normalized data with k=" + str(k))
    clustering = cluster.cluster(norm_data, k)

    print("\nDone. Clustering:")
    print(clustering)

    print("\nRaw data grouped by cluster: ")
    cluster.display(raw_data, clustering, k)

    print("\nEnd k-means demo ")
    # plot data
    
    

       
if __name__ == "__main__":
    main()


#TODO: k-mean clustering
#TODO: write a function that tests what k would yield the sum of the smallest distances 