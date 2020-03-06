import pandas as pd
import matplotlib.pyplot as plt


# reading the % change data
d  = pd.read_csv("test1_change_only.csv")
JAPAN = d['JAPAN']
HK = d['HK']

# plotting 
plt.scatter(JAPAN, HK, color='r')
plt.xlabel('JAPAN')
plt.ylabel('HK')
plt.savefig("Japan VS HK")
plt.show()

# calculate eigenvalue
