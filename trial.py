import numpy as np

raw_data = np.loadtxt("trial.txt", dtype=np.float32,
    delimiter=",", skiprows=0, usecols=[0,1,2,3])

print(raw_data)
print(raw_data[0])
print(type(raw_data))