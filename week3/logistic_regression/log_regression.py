from math import exp

import numpy as np
import pandas

from common.Axis import Axis
from week3.logistic_regression.data_index import DataIndex

DATA = pandas.read_csv('./data/data-logistic.csv', header=None, index_col=False)
ROWS = DATA.values


def samples_sum(rows, w, xIndex):
    w1, w2 = w
    return np.apply_along_axis(
        lambda row: row[DataIndex.Y] * row[xIndex] * (
                1 - 1 / (1 + exp(-row[DataIndex.Y] * (w1 * row[DataIndex.X1] + w2 * row[DataIndex.X2])))
        ),
        axis=Axis.ROW,
        arr=rows).mean()


def fit_weights(rows, k, w, c, accuracy, iterations_limit):
    w1, w2 = w

    i = 0
    while i < iterations_limit:
        i = i + 1
        w1_new = w1 + k * samples_sum(rows, w, DataIndex.X1) - k * c * w1
        w2_new = w2 + k * samples_sum(rows, w, DataIndex.X2) - k * c * w2
        # print(w1_new, w2_new)

        distance = np.sqrt((w1 - w1_new) ** 2 + (w2 - w2_new) ** 2)
        if distance < accuracy:
            print('Distance break')
            break

        w1, w2 = w1_new, w2_new

    if i == iterations_limit:
        print('Iterations limit break')

    return [w1, w2]


# With normalization
print(fit_weights(rows=ROWS, k=0.1, c=0, accuracy=0.00001, iterations_limit=10, w=[0.0, 0.0]))

# Without normalization
# print(fit_weights(rows=ROWS, k=0.1, c=0, accuracy=0.00001, iterations_limit=10000, w=[0.0, 0.0]))
