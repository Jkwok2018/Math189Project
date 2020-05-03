import numpy as np
import matplotlib.pyplot as plt

seq_L = [4,5,6,7,8,9,10]
accuracy_L = [0.8119283536585366, 0.8147865853658537, 0.8071646341463414, 0.8111661585365854, 
0.8054496951219512, 0.8084984756097561, 0.8071646341463414]
plt.scatter(seq_L, accuracy_L)
plt.xlabel("sequence length")
plt.ylabel("accuracy")
plt.show()