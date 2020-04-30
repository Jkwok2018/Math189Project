import cluster2 as cluster
from sklearn import metrics


def opt_helper(weight, raw_data):
    print("weight:",weight)
    # normalize the raw data so that they are all in the range of (0,1)
    (norm_data, mins, maxs) = cluster.mm_normalize(raw_data)
    # define the number of clusters 
    k = 7

    def distance(item1, item2):
        return cluster.distance(weight,item1, item2)

    clustering = cluster.cluster(weight, norm_data, k)
    # clusters = cluster.display(norm_data, clustering, k)

    result = -metrics.silhouette_score(norm_data, clustering, metric = distance, sample_size=50)
    print("result: ",result)
    return result


