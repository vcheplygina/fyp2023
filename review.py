import numpy as np

metdata = np.loadtxt("metadata.csv", delimiter=',', dtype=str)
print(np.unique(metdata[:, 17])) # print all rows and 17th column
