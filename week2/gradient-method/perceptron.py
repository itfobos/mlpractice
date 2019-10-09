import pandas
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

DATA = pandas.read_csv('./data/perceptron-train.csv', header=None, index_col=False)
Y = DATA[0]
X = DATA[[1, 2]]

TEST_DATA = pandas.read_csv('./data/perceptron-test.csv', header=None, index_col=False)
Y_test = TEST_DATA[0]
X_test = TEST_DATA[[1, 2]]

# clf = Perceptron(random_state=241)
clf = Perceptron()
clf.fit(X, Y)

Y_predicted = clf.predict(X_test)

not_scaled_accuracy = accuracy_score(Y_test, Y_predicted)
print("not_scaled_accuracy %.3f" % not_scaled_accuracy)

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

clf.fit(X_scaled, Y)

Y_predicted = clf.predict(X_test_scaled)

scaled_accuracy = accuracy_score(Y_test, Y_predicted)
print("scaled_accuracy %.3f" % scaled_accuracy)

print("Diff is %.3f" % (scaled_accuracy - not_scaled_accuracy))
