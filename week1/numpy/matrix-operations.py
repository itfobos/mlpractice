import numpy as np

from common.Axis import Axis

ROW_SUM_THRESHOLD = 10

srcMatrix = np.array([[4, 5, 0],
                      [1, 9, 3],
                      [5, 1, 1],
                      [3, 3, 3],
                      [9, 9, 9],
                      [4, 7, 1]])

sumsColumn = np.sum(srcMatrix, axis=Axis.ROW)

thresholdExceedRows = np.nonzero(sumsColumn > ROW_SUM_THRESHOLD)

print(thresholdExceedRows)
