from math import exp

import numpy as np
import pandas
from sklearn.metrics import roc_auc_score

from common.Axis import Axis
from week3.logistic_regression.data_index import DataIndex

DATA = pandas.read_csv('./data/data-logistic.csv', header=None, index_col=False)
ROWS = DATA.values


class LogRegressionClassifier:
    def __init__(self):
        self.__w = [0.0, 0.0]

    @staticmethod
    def __samples_sum(rows, w, x_index):
        w1, w2 = w
        return np.apply_along_axis(
            lambda row: row[DataIndex.Y] * row[x_index] * (
                    1 - 1 / (1 + exp(-row[DataIndex.Y] * (w1 * row[DataIndex.X1] + w2 * row[DataIndex.X2])))
            ),
            axis=Axis.ROW,
            arr=rows).mean()

    def fit(self, rows, k, w, c, accuracy, iterations_limit):
        w1, w2 = w

        i = 0
        while i < iterations_limit:
            i = i + 1
            x1_sum = self.__samples_sum(rows, [w1, w2], DataIndex.X1)
            x2_sum = self.__samples_sum(rows, [w1, w2], DataIndex.X2)

            w1_new = w1 + k * x1_sum - k * c * w1
            w2_new = w2 + k * x2_sum - k * c * w2

            # print('w1 w2: %.5f %.5f      sums %.5f %.5f' % (w1_new, w2_new, x1_sum, x2_sum))
            # print(w1_new, w2_new)

            distance = np.sqrt((w1 - w1_new) ** 2 + (w2 - w2_new) ** 2)
            if distance < accuracy:
                print('Distance break')
                break

            w1, w2 = w1_new, w2_new

        if i == iterations_limit:
            print('Iterations limit break')

        self.__w = [w1, w2]
        print('w1: %.5f,  w2: %.5f' % (w1, w2))
        return self.__w

    def predict(self, rows):
        w1, w2 = self.__w
        # a = lambda row: 1 / (1 + exp(-1 * w1 * row[DataIndex.X1] - w2 * row[DataIndex.X2]))

        return np.apply_along_axis(a(self.__w), axis=Axis.ROW, arr=rows)


def a(w):
    w1, w2 = w

    def a_internal(row):
        c_pow = (-1 * w1 * row[DataIndex.X1] - w2 * row[DataIndex.X2])
        # print('(x1 x2) = (%.4f %.4f)  c_pow: %.4f' % (row[DataIndex.X1], row[DataIndex.X2], c_pow))
        return 1 / (1 + exp(c_pow))

    return a_internal


y_true = ROWS[:, 0]
clf = LogRegressionClassifier()

# Without normalization
clf.fit(rows=ROWS, k=0.1, c=0, accuracy=0.00001, iterations_limit=10_000, w=[0.0, 0.0])
y_scores = clf.predict(ROWS)
print('Without normalization %.3f' % roc_auc_score(y_true, y_scores))

# With normalization
clf.fit(rows=ROWS, k=0.1, c=10, accuracy=0.00001, iterations_limit=10000, w=[0.0, 0.0])
y_scores = clf.predict(ROWS)
print('With normalization %.3f' % roc_auc_score(y_true, y_scores))
