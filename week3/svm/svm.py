import pandas

from sklearn.svm import SVC

DATA = pandas.read_csv('./data/svm-data.csv', header=None, index_col=False)
Y = DATA[0]
X = DATA[[1, 2]]

clf = SVC(C=100000, random_state=241, kernel='linear')
clf.fit(X, Y)

print('Indexes', clf.support_ + 1)
