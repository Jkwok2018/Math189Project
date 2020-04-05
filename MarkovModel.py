
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
        items[four_label] = old_value
            
    for key in items:
        max_cluster = np.argmax(items.get(key))
        prob = max(items.get(key))/sum(items.get(key))
        items[key] = [max_cluster, prob]

    return items

def test_markov(dict, testL):
    not_found = 0
    failed_cases = 0
    for i in range(len(testL)-5):
        four_label = ''.join(map(str, L[i:i+4] )) 
        if four_label not in dict:
            not_found += 1
        elif (dict[four_label][0] != testL[i+4]):
            failed_cases += 1
    correctness = 1-((failed_cases + not_found)* 1.0 /(len(testL)))
    print("Accuracy is " + str(correctness) + "%")
    print("Cases not found: ", not_found)
    return correctness, not_found