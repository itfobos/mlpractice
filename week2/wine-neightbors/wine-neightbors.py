import pandas
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import scale

FEATURE_COLUMNS = ['Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                   'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 'Proline']

ANSWER_COLUMN = 'class'

DATA = pandas.read_csv('./data/wine.data', header=None, index_col=False,
                       names=[ANSWER_COLUMN] + FEATURE_COLUMNS)

X = DATA[FEATURE_COLUMNS]
X_scaled = scale(X)
Y = DATA[ANSWER_COLUMN]

kf = KFold(n_splits=5, shuffle=True, random_state=42)

neighborsScores = []
for n_neighbors in range(1, 50):
    neighboursClf = KNeighborsClassifier(n_neighbors=n_neighbors)

    scores = cross_val_score(neighboursClf, X, Y, cv=kf)
    neighborsScores.append([n_neighbors, scores.mean()])
    # print("n_neighbors=%d Accuracy: %0.3f (+/- %0.2f)" % (n_neighbors, scores.mean(), scores.std() * 2))

neighborsScoresDataFrame = pandas.DataFrame(neighborsScores, columns=['n_neighbors', 'accuracy'])
print('Non normalized data. Best result is: \n', neighborsScoresDataFrame.nlargest(1, 'accuracy').iloc[0])

# ------------------------Scaled data------------------------------
neighborsScores = []
for n_neighbors in range(1, 50):
    neighboursClf = KNeighborsClassifier(n_neighbors=n_neighbors)

    scores = cross_val_score(neighboursClf, X_scaled, Y, cv=kf)
    neighborsScores.append([n_neighbors, scores.mean()])
    # print("n_neighbors=%d Accuracy: %0.3f (+/- %0.2f)" % (n_neighbors, scores.mean(), scores.std() * 2))

neighborsScoresDataFrame = pandas.DataFrame(neighborsScores, columns=['n_neighbors', 'accuracy'])
answer = neighborsScoresDataFrame.nlargest(1, 'accuracy').iloc[0]

print('\n\nScaled data.\n')
print('    Best result is: for %d neighbors with accuracy %.2f\n' % (answer['n_neighbors'], answer['accuracy']))
