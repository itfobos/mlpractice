import numpy as np

from common.Axis import Axis

srcMatrix = np.random.normal(loc=1, scale=10, size=(1000, 50))

columnMeans = np.mean(srcMatrix, axis=Axis.COLUMN)

standardDeviation = np.std(srcMatrix, axis=Axis.COLUMN)

srcNormedMatrix = (srcMatrix - columnMeans) / standardDeviation
