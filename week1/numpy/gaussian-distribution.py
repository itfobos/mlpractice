import numpy as np

srcMatrix = np.random.normal(loc=1, scale=10, size=(1000, 50))

columnMeans = np.mean(srcMatrix, axis=0)

standardDeviation = np.std(srcMatrix, axis=0)

srcNormedMatrix = (srcMatrix - columnMeans) / standardDeviation
